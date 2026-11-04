"""
episodes.py – Episodic sampler for N-way K-shot few-shot learning.

This module provides the data-loading infrastructure for episodic training
and evaluation of few-shot models.  It constructs *episodes* (also called
*tasks*) — small labelled subsets drawn from the full dataset — that consist
of a support set and a query set.

Classes
-------
EpisodeDataset
    Wraps a manifest DataFrame and provides per-class image access.
EpisodicSampler
    Draws random N-way K-shot episodes from an EpisodeDataset.
EpisodicBatchSampler
    Adapter that yields episodes via the standard ``torch.utils.data.Sampler``
    interface so it can be used with ``DataLoader``.

Functions
---------
collate_episodes
    Stack a list of episode dicts into a batched dict.
create_episode_loader
    Convenience factory that returns an iterable of episode dicts.
"""

from __future__ import annotations

import logging
import random
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
import torch
import torch.utils.data

from src.preprocessing import preprocess_for_model

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# 1. EpisodeDataset
# ─────────────────────────────────────────────────────────────────────


class EpisodeDataset(torch.utils.data.Dataset):
    """Dataset that organises preprocessed images by class for episodic
    sampling.

    Parameters
    ----------
    manifest_df : pd.DataFrame
        Must contain at least the columns ``[file_path, class_name]``.
        Optional columns: ``split``, ``source_group``.
    class_to_idx : dict[str, int]
        Mapping from NFC-normalised class name → integer index.
    image_dir : str | Path
        Root directory for processed images (paths in ``manifest_df`` are
        resolved relative to this directory).
    transform : callable | None
        Optional deterministic transform applied when loading an image
        (e.g., resize / normalise).  Applied **before** augmentation.
    augmentor : callable | None
        Optional stochastic augmentation applied per-episode at train time.
    """

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        class_to_idx: Dict[str, int],
        image_dir: Union[str, Path],
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        augmentor: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.augmentor = augmentor
        self.class_to_idx = class_to_idx

        # Build per-class image lists -----------------------------------------
        # class_images: {class_idx: [Path, ...]}
        self.class_images: Dict[int, List[Path]] = defaultdict(list)
        # source_map: {class_idx: {path: source_group}}
        self.source_map: Dict[int, Dict[Path, str]] = defaultdict(dict)

        has_source = "source_group" in manifest_df.columns

        for _, row in manifest_df.iterrows():
            class_name = unicodedata.normalize("NFC", str(row["class_name"]))
            if class_name not in class_to_idx:
                continue
            cidx = class_to_idx[class_name]
            # Support processed_path (full path) or relative file_path
            raw_path = str(row.get("processed_path", row.get("file_path", "")))
            p = Path(raw_path)
            fpath = p if p.is_absolute() else (self.image_dir / p)
            self.class_images[cidx].append(fpath)
            if has_source:
                self.source_map[cidx][fpath] = str(row["source_group"])

        # Filter to classes with ≥ 1 image ------------------------------------
        empty = [c for c, imgs in self.class_images.items() if len(imgs) == 0]
        for c in empty:
            del self.class_images[c]
            self.source_map.pop(c, None)

        logger.info(
            "EpisodeDataset initialised: %d classes, %d total images",
            len(self.class_images),
            sum(len(v) for v in self.class_images.values()),
        )

    # -- Public API --------------------------------------------------------

    def get_classes(self) -> List[int]:
        """Return sorted list of class indices that have images."""
        return sorted(self.class_images.keys())

    def get_class_images(self, class_idx: int) -> List[Path]:
        """Return list of image paths for *class_idx*."""
        return self.class_images.get(class_idx, [])

    def load_image(self, path: Path) -> torch.Tensor:
        """Load a preprocessed image and return tensor of shape ``(1, H, W)``.

        Supports ``.npy`` files (pre-computed float32 arrays) and standard
        image formats (``.png``, ``.jpg``, etc.) via
        :func:`src.preprocessing.preprocess_for_model`.
        """
        path = Path(path)
        if path.suffix == ".npy":
            arr = np.load(str(path))
            if arr.ndim == 2:
                tensor = torch.from_numpy(arr).unsqueeze(0).float()
            elif arr.ndim == 3 and arr.shape[0] == 1:
                tensor = torch.from_numpy(arr).float()
            else:
                # Assume (H, W, C) → take first channel
                tensor = torch.from_numpy(arr[:, :, 0]).unsqueeze(0).float()
        else:
            # First try direct reading of preprocessed 8-bit grayscale image
            try:
                raw_bytes = np.fromfile(str(path), dtype=np.uint8)
                im = cv2.imdecode(raw_bytes, cv2.IMREAD_GRAYSCALE)
                if im is not None:
                    arr = im.astype(np.float32) / 255.0
                    tensor = torch.from_numpy(arr).unsqueeze(0).float()
                else:
                    result = preprocess_for_model(path)
                    tensor = result if result is not None else torch.zeros(1, 64, 64, dtype=torch.float32)
            except Exception:
                result = preprocess_for_model(path)
                tensor = result if result is not None else torch.zeros(1, 64, 64, dtype=torch.float32)

        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor

    # Required by Dataset, but we don't use index-based access ---------------
    def __len__(self) -> int:  # pragma: no cover
        return sum(len(v) for v in self.class_images.values())

    def __getitem__(self, index: int) -> Any:  # pragma: no cover
        raise NotImplementedError(
            "EpisodeDataset is accessed through EpisodicSampler, not by index."
        )


# ─────────────────────────────────────────────────────────────────────
# 2. EpisodicSampler
# ─────────────────────────────────────────────────────────────────────


class EpisodicSampler:
    """Draw N-way K-shot episodes from an :class:`EpisodeDataset`.

    Parameters
    ----------
    dataset : EpisodeDataset
        Source of per-class image lists.
    n_way : int
        Number of classes per episode.
    k_shot : int
        Number of support examples per class.
    q_query : int
        Number of query examples per class.
    num_episodes : int
        Number of episodes to generate per epoch.
    source_aware : bool
        If ``True``, try to assign support and query from different source
        groups (best-effort; falls back to random when diversity is
        insufficient).
    min_samples_per_class : int | None
        Minimum images a class must have to be eligible. Defaults to
        ``k_shot + q_query``.
    """

    def __init__(
        self,
        dataset: EpisodeDataset,
        n_way: int,
        k_shot: int,
        q_query: int,
        num_episodes: int,
        source_aware: bool = False,
        min_samples_per_class: Optional[int] = None,
        query_dataset: Optional[EpisodeDataset] = None,
    ) -> None:
        self.dataset = dataset
        self.query_dataset = query_dataset if query_dataset is not None else dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.num_episodes = num_episodes
        self.source_aware = source_aware

        if min_samples_per_class is None:
            self.min_samples_per_class = k_shot if query_dataset is not None else (k_shot + q_query)
        else:
            self.min_samples_per_class = min_samples_per_class

        all_classes = dataset.get_classes()
        if query_dataset is not None:
            # Dual dataset: support from dataset, query from query_dataset
            sup_classes = set(all_classes)
            qry_classes = set(self.query_dataset.get_classes())
            common = sup_classes.intersection(qry_classes)
            self.valid_classes = [
                c for c in common
                if len(self.dataset.get_class_images(c)) >= self.min_samples_per_class
                and len(self.query_dataset.get_class_images(c)) >= 1
            ]
        else:
            self.valid_classes = [
                c
                for c in all_classes
                if len(dataset.get_class_images(c)) >= self.min_samples_per_class
            ]

        skipped = len(all_classes) - len(self.valid_classes)
        if skipped > 0:
            logger.warning(
                "%d class(es) skipped: fewer than %d samples (K=%d + Q=%d)",
                skipped,
                self.min_samples_per_class,
                self.k_shot,
                self.q_query,
            )

        # Reduce n_way if not enough valid classes --------------------------
        if len(self.valid_classes) < self.n_way:
            logger.warning(
                "Only %d valid classes available, reducing n_way from %d → %d",
                len(self.valid_classes),
                self.n_way,
                len(self.valid_classes),
            )
            self.n_way = len(self.valid_classes)

        if self.n_way == 0:
            raise ValueError(
                "No classes have enough samples for episodic sampling. "
                f"Need at least {self.min_samples_per_class} images per class."
            )

        logger.info(
            "EpisodicSampler: %d-way %d-shot %d-query, %d episodes, "
            "%d eligible classes",
            self.n_way,
            self.k_shot,
            self.q_query,
            self.num_episodes,
            len(self.valid_classes),
        )

    # ------------------------------------------------------------------
    # Core episode sampling
    # ------------------------------------------------------------------

    def sample_episode(self) -> Dict[str, Any]:
        """Sample a single N-way K-shot episode.

        Returns
        -------
        dict
            ``support_images``  – ``(n_way*k_shot, 1, H, W)``
            ``support_labels``  – ``(n_way*k_shot,)``   labels in ``0..n_way-1``
            ``query_images``    – ``(n_way*q_query, 1, H, W)``
            ``query_labels``    – ``(n_way*q_query,)``  labels in ``0..n_way-1``
            ``class_indices``   – list of original class indices selected
        """
        # 1. Select n_way classes ------------------------------------------
        selected_classes: List[int] = random.sample(
            self.valid_classes, self.n_way
        )

        support_images: List[torch.Tensor] = []
        support_labels: List[int] = []
        query_images: List[torch.Tensor] = []
        query_labels: List[int] = []

        for episode_label, class_idx in enumerate(selected_classes):
            if self.query_dataset is not self.dataset:
                # Dual dataset sampling: support from dataset, query from query_dataset
                sup_paths = self.dataset.get_class_images(class_idx)
                qry_paths = self.query_dataset.get_class_images(class_idx)

                k = min(self.k_shot, len(sup_paths))
                q = min(self.q_query, len(qry_paths))

                sampled_sup = random.sample(sup_paths, k)
                if k < self.k_shot:
                    sampled_sup += random.choices(sup_paths, k=self.k_shot - k)

                sampled_qry = random.sample(qry_paths, q)
                if q < self.q_query:
                    sampled_qry += random.choices(qry_paths, k=self.q_query - q)

                self._append_images(sampled_sup, episode_label, support_images, support_labels)
                self._append_images(sampled_qry, episode_label, query_images, query_labels)
                continue

            all_paths = self.dataset.get_class_images(class_idx)
            n_available = len(all_paths)

            # Determine effective k and q ----------------------------------
            k = self.k_shot
            q = self.q_query

            if n_available < k + q:
                # Auto-reduce q while keeping k intact if possible
                if n_available > k:
                    q = n_available - k
                    logger.debug(
                        "Class %d: reduced q_query to %d (only %d samples)",
                        class_idx,
                        q,
                        n_available,
                    )
                else:
                    # Not enough even for k: sample support WITH replacement
                    logger.warning(
                        "Class %d has only %d images (< k_shot=%d). "
                        "Sampling support WITH replacement.",
                        class_idx,
                        n_available,
                        k,
                    )
                    support_paths = random.choices(all_paths, k=k)
                    remaining = [p for p in all_paths if p not in support_paths]
                    if len(remaining) == 0:
                        # Extremely small class — use same images for query
                        query_paths = random.choices(all_paths, k=q)
                    else:
                        query_paths = random.choices(remaining, k=q)

                    self._append_images(
                        support_paths,
                        episode_label,
                        support_images,
                        support_labels,
                    )
                    self._append_images(
                        query_paths,
                        episode_label,
                        query_images,
                        query_labels,
                    )
                    continue

            # Normal path: sample k + q without replacement ----------------
            if self.source_aware:
                support_paths, query_paths = self._source_aware_split(
                    class_idx, all_paths, k, q
                )
            else:
                chosen = random.sample(all_paths, k + q)
                support_paths = chosen[:k]
                query_paths = chosen[k:]

            self._append_images(
                support_paths, episode_label, support_images, support_labels
            )
            self._append_images(
                query_paths, episode_label, query_images, query_labels
            )

        # 2. Stack tensors -------------------------------------------------
        support_img_t = torch.stack(support_images, dim=0)
        query_img_t = torch.stack(query_images, dim=0)
        support_lbl_t = torch.tensor(support_labels, dtype=torch.long)
        query_lbl_t = torch.tensor(query_labels, dtype=torch.long)

        # 3. Apply augmentor (training-time) --------------------------------
        if self.dataset.augmentor is not None:
            aug = self.dataset.augmentor
            aug_support = []
            for img in support_img_t:
                im_np = img.squeeze(0).cpu().numpy()
                im_aug = aug(im_np)
                aug_support.append(torch.from_numpy(im_aug).unsqueeze(0).float())
            support_img_t = torch.stack(aug_support, dim=0)

            aug_query = []
            for img in query_img_t:
                im_np = img.squeeze(0).cpu().numpy()
                im_aug = aug(im_np)
                aug_query.append(torch.from_numpy(im_aug).unsqueeze(0).float())
            query_img_t = torch.stack(aug_query, dim=0)

        return {
            "support_images": support_img_t,
            "support_labels": support_lbl_t,
            "query_images": query_img_t,
            "query_labels": query_lbl_t,
            "class_indices": selected_classes,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _append_images(
        self,
        paths: List[Path],
        label: int,
        image_list: List[torch.Tensor],
        label_list: List[int],
    ) -> None:
        """Load images from *paths* and append to accumulators."""
        for p in paths:
            img = self.dataset.load_image(p)
            image_list.append(img)
            label_list.append(label)

    def _source_aware_split(
        self,
        class_idx: int,
        all_paths: List[Path],
        k: int,
        q: int,
    ) -> Tuple[List[Path], List[Path]]:
        """Split images into support and query favouring different source
        groups.

        Falls back to random split when source diversity is insufficient.

        Parameters
        ----------
        class_idx : int
            Class index (used to look up source map).
        all_paths : list[Path]
            All available image paths for this class.
        k : int
            Number of support images to select.
        q : int
            Number of query images to select.

        Returns
        -------
        tuple[list[Path], list[Path]]
            ``(support_paths, query_paths)`` with no overlap.
        """
        source_info = self.dataset.source_map.get(class_idx, {})
        if not source_info:
            # No source information — fall back to random
            chosen = random.sample(all_paths, k + q)
            return chosen[:k], chosen[k:]

        # Group paths by source -------------------------------------------
        groups: Dict[str, List[Path]] = defaultdict(list)
        for p in all_paths:
            grp = source_info.get(p, "unknown")
            groups[grp].append(p)

        group_keys = list(groups.keys())
        if len(group_keys) < 2:
            # Only one source group — fall back to random
            chosen = random.sample(all_paths, k + q)
            return chosen[:k], chosen[k:]

        # Try to fill support from one half and query from the other -------
        random.shuffle(group_keys)
        support_pool: List[Path] = []
        query_pool: List[Path] = []

        mid = max(1, len(group_keys) // 2)
        for gi in group_keys[:mid]:
            support_pool.extend(groups[gi])
        for gi in group_keys[mid:]:
            query_pool.extend(groups[gi])

        random.shuffle(support_pool)
        random.shuffle(query_pool)

        # If either pool is too small, spill from the other ----------------
        if len(support_pool) < k:
            shortfall = k - len(support_pool)
            extra = query_pool[:shortfall]
            support_pool.extend(extra)
            query_pool = query_pool[shortfall:]

        if len(query_pool) < q:
            shortfall = q - len(query_pool)
            extra = support_pool[k : k + shortfall]
            query_pool.extend(extra)

        # Final safety: if still not enough, fall back to random -----------
        if len(support_pool) < k or len(query_pool) < q:
            logger.debug(
                "Source-aware split failed for class %d; falling back to "
                "random.",
                class_idx,
            )
            chosen = random.sample(all_paths, k + q)
            return chosen[:k], chosen[k:]

        support_paths = support_pool[:k]
        query_paths = query_pool[:q]

        # Guarantee no overlap ---------------------------------------------
        support_set = set(id(p) for p in support_paths)
        query_paths = [p for p in query_paths if id(p) not in support_set]
        if len(query_paths) < q:
            # Fill from remaining paths not already in support
            used = set(id(p) for p in support_paths + query_paths)
            for p in all_paths:
                if id(p) not in used:
                    query_paths.append(p)
                    used.add(id(p))
                    if len(query_paths) >= q:
                        break

        return support_paths, query_paths[:q]


# ─────────────────────────────────────────────────────────────────────
# 3. EpisodicBatchSampler
# ─────────────────────────────────────────────────────────────────────


class EpisodicBatchSampler(torch.utils.data.Sampler):
    """Wraps :class:`EpisodicSampler` to yield episode dicts.

    This makes episodic training compatible with ``DataLoader``-style
    iteration loops::

        sampler = EpisodicBatchSampler(dataset, n_way=5, k_shot=1, ...)
        for episode in sampler:
            support = episode["support_images"]
            ...

    Parameters
    ----------
    dataset : EpisodeDataset
        Source dataset.
    n_way, k_shot, q_query, num_episodes : int
        Episode parameters (forwarded to :class:`EpisodicSampler`).
    source_aware : bool
        Forward to :class:`EpisodicSampler`.
    min_samples_per_class : int | None
        Forward to :class:`EpisodicSampler`.
    """

    def __init__(
        self,
        dataset: EpisodeDataset,
        n_way: int,
        k_shot: int,
        q_query: int,
        num_episodes: int,
        source_aware: bool = False,
        min_samples_per_class: Optional[int] = None,
    ) -> None:
        super().__init__(data_source=None)  # type: ignore[arg-type]
        self.sampler = EpisodicSampler(
            dataset=dataset,
            n_way=n_way,
            k_shot=k_shot,
            q_query=q_query,
            num_episodes=num_episodes,
            source_aware=source_aware,
            min_samples_per_class=min_samples_per_class,
        )
        self.num_episodes = num_episodes

    def __len__(self) -> int:
        return self.num_episodes

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for _ in range(self.num_episodes):
            yield self.sampler.sample_episode()


# ─────────────────────────────────────────────────────────────────────
# 4. collate_episodes
# ─────────────────────────────────────────────────────────────────────


def collate_episodes(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate a list of episode dicts into a batched dict.

    Each tensor field is stacked along a new leading (batch) dimension.
    Non-tensor fields (``class_indices``) are collected into a list.

    Parameters
    ----------
    episodes : list[dict]
        Individual episode dicts as returned by
        :meth:`EpisodicSampler.sample_episode`.

    Returns
    -------
    dict
        Batched episode dict with keys identical to the input episodes.
        Tensor shapes gain a leading ``B`` dimension.
    """
    if not episodes:
        return {}

    batch: Dict[str, Any] = {}
    tensor_keys = [
        "support_images",
        "support_labels",
        "query_images",
        "query_labels",
    ]
    for key in tensor_keys:
        batch[key] = torch.stack([ep[key] for ep in episodes], dim=0)

    batch["class_indices"] = [ep["class_indices"] for ep in episodes]
    return batch


# ─────────────────────────────────────────────────────────────────────
# 5. create_episode_loader
# ─────────────────────────────────────────────────────────────────────


class _EpisodeIterator:
    """Lightweight iterable wrapper around :class:`EpisodicSampler`."""

    def __init__(self, sampler: EpisodicSampler) -> None:
        self._sampler = sampler

    def __len__(self) -> int:
        return self._sampler.num_episodes

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for _ in range(self._sampler.num_episodes):
            yield self._sampler.sample_episode()


def create_episode_loader(
    manifest_df: pd.DataFrame,
    class_to_idx: Dict[str, int],
    processed_dir: Union[str, Path],
    n_way: int,
    k_shot: int,
    q_query: int,
    num_episodes: int,
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    augmentor: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    source_aware: bool = False,
    num_workers: int = 0,
    min_samples_per_class: Optional[int] = None,
    query_manifest_df: Optional[pd.DataFrame] = None,
) -> _EpisodeIterator:
    dataset = EpisodeDataset(
        manifest_df=manifest_df,
        class_to_idx=class_to_idx,
        image_dir=processed_dir,
        transform=transform,
        augmentor=augmentor,
    )

    query_dataset = None
    if query_manifest_df is not None:
        query_dataset = EpisodeDataset(
            manifest_df=query_manifest_df,
            class_to_idx=class_to_idx,
            image_dir=processed_dir,
            transform=transform,
            augmentor=None,
        )

    sampler = EpisodicSampler(
        dataset=dataset,
        n_way=n_way,
        k_shot=k_shot,
        q_query=q_query,
        num_episodes=num_episodes,
        source_aware=source_aware,
        min_samples_per_class=min_samples_per_class,
        query_dataset=query_dataset,
    )

    if num_workers > 0:
        logger.info(
            "num_workers=%d requested but episodic loading is single-process. "
            "Ignoring.",
            num_workers,
        )

    return _EpisodeIterator(sampler)
