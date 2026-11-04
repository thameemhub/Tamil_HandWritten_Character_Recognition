"""Siamese Network with contrastive / triplet loss for Tamil few-shot HTR.

Provides a Siamese twin-tower architecture that shares a single backbone
for embedding pairs or triplets of images, plus associated loss functions,
pair/triplet samplers, k-NN / prototype prediction heads, and training
utilities.

Usage
-----
>>> from src.backbones import get_backbone
>>> from src.fewshot.siamese import SiameseNetwork, contrastive_loss
>>> backbone = get_backbone("conv4", in_channels=1, embedding_dim=64)
>>> model = SiameseNetwork(backbone)
>>> emb1, emb2 = model(img_batch_a, img_batch_b)
>>> loss = contrastive_loss(emb1, emb2, labels)
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)

__all__ = [
    "SiameseNetwork",
    "contrastive_loss",
    "triplet_loss",
    "PairSampler",
    "TripletSampler",
    "knn_predict",
    "prototype_predict",
    "train_siamese_epoch",
    "evaluate_siamese_fewshot",
]


# ======================================================================
# Siamese Network
# ======================================================================


class SiameseNetwork(nn.Module):
    """Twin-tower Siamese network with a shared embedding backbone.

    The same backbone is applied to each input independently (weight
    sharing), producing two embeddings that can be compared with
    :func:`contrastive_loss` or :func:`triplet_loss`.

    Parameters
    ----------
    backbone : nn.Module
        Embedding network that maps ``(batch, 1, H, W)`` to
        ``(batch, embed_dim)``.  Typically a :class:`src.backbones.Conv4`
        or :class:`src.backbones.ResNet12Lite`.
    """

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    # ------------------------------------------------------------------
    # forward helpers
    # ------------------------------------------------------------------

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        """Embed a single image batch through the shared backbone.

        Parameters
        ----------
        x : torch.Tensor
            Input images of shape ``(batch, 1, H, W)``.

        Returns
        -------
        torch.Tensor
            Embedding of shape ``(batch, embed_dim)``.
        """
        return self.backbone(x)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed two image batches through the shared backbone.

        Parameters
        ----------
        x1 : torch.Tensor
            First batch of images, shape ``(batch, 1, H, W)``.
        x2 : torch.Tensor
            Second batch of images, shape ``(batch, 1, H, W)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(embedding1, embedding2)`` each of shape ``(batch, embed_dim)``.
        """
        emb1 = self.forward_once(x1)
        emb2 = self.forward_once(x2)
        return emb1, emb2


# ======================================================================
# Loss functions
# ======================================================================


def contrastive_loss(
    emb1: torch.Tensor,
    emb2: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Contrastive loss for Siamese pairs (Hadsell et al., 2006).

    .. math::

        L = y \\cdot d^2 + (1 - y) \\cdot \\max(0,\\; \\text{margin} - d)^2

    where :math:`d` is the Euclidean distance and :math:`y = 1` when the
    pair belongs to the **same** class.

    Parameters
    ----------
    emb1, emb2 : torch.Tensor
        Embeddings of shape ``(batch, embed_dim)``.
    labels : torch.Tensor
        Float tensor of shape ``(batch,)``.  ``1`` if same class, ``0``
        if different class.
    margin : float
        Distance margin for negative pairs.

    Returns
    -------
    torch.Tensor
        Scalar mean contrastive loss.
    """
    distances = F.pairwise_distance(emb1, emb2, p=2)  # (batch,)
    labels = labels.float()

    pos_loss = labels * distances.pow(2)
    neg_loss = (1.0 - labels) * F.relu(margin - distances).pow(2)

    loss = (pos_loss + neg_loss).mean()
    return loss


def triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Triplet loss with squared Euclidean distances.

    .. math::

        L = \\max(0,\\; d(a, p)^2 - d(a, n)^2 + \\text{margin})

    Parameters
    ----------
    anchor, positive, negative : torch.Tensor
        Embeddings of shape ``(batch, embed_dim)``.
    margin : float
        Triplet margin.

    Returns
    -------
    torch.Tensor
        Scalar mean triplet loss.
    """
    d_ap = F.pairwise_distance(anchor, positive, p=2).pow(2)
    d_an = F.pairwise_distance(anchor, negative, p=2).pow(2)

    loss = F.relu(d_ap - d_an + margin).mean()
    return loss


# ======================================================================
# Pair / Triplet Samplers
# ======================================================================


def _build_class_index(
    dataset: Any,
) -> Dict[str, List[str]]:
    """Build a mapping from class label → list of image paths.

    Parameters
    ----------
    dataset : object
        Must expose either a ``class_to_images`` dict attribute
        (``{label: [path, ...]}``) or ``samples`` / ``imgs`` list of
        ``(path, label)`` tuples (torchvision ImageFolder style).

    Returns
    -------
    dict[str, list[str]]
        Mapping from class label to image paths.
    """
    if hasattr(dataset, "class_to_images"):
        return dict(dataset.class_to_images)

    if hasattr(dataset, "samples"):
        items = dataset.samples
    elif hasattr(dataset, "imgs"):
        items = dataset.imgs
    else:
        raise TypeError(
            "dataset must have 'class_to_images', 'samples', or 'imgs' "
            f"attribute, got {type(dataset).__name__}"
        )

    index: Dict[str, List[str]] = defaultdict(list)
    for path, label in items:
        index[str(label)].append(str(path))
    return dict(index)


class PairSampler:
    """Sample positive and negative image pairs for contrastive learning.

    Produces roughly 50 % positive (same class) and 50 % negative
    (different class) pairs per epoch.

    Parameters
    ----------
    dataset : object
        Dataset with class-to-image mapping (see :func:`_build_class_index`).
    num_pairs_per_epoch : int
        Total number of pairs to sample each time :meth:`sample_pairs` is
        called.
    """

    def __init__(
        self,
        dataset: Any,
        num_pairs_per_epoch: int = 1000,
    ) -> None:
        self.class_to_images = _build_class_index(dataset)
        self.num_pairs_per_epoch = num_pairs_per_epoch

        # Only keep classes with at least 2 images (for positive pairs)
        self._classes_with_pairs = [
            c for c, imgs in self.class_to_images.items() if len(imgs) >= 2
        ]
        self._all_classes = list(self.class_to_images.keys())

        if len(self._classes_with_pairs) == 0:
            raise ValueError(
                "No class has ≥ 2 images; cannot form positive pairs."
            )
        if len(self._all_classes) < 2:
            raise ValueError(
                "Need at least 2 classes to form negative pairs."
            )

        logger.info(
            "PairSampler: %d classes, %d with ≥2 images, %d pairs/epoch",
            len(self._all_classes),
            len(self._classes_with_pairs),
            self.num_pairs_per_epoch,
        )

    def sample_pairs(
        self,
    ) -> List[Tuple[str, str, int]]:
        """Sample pairs for one epoch.

        Returns
        -------
        list[tuple[str, str, int]]
            Each element is ``(img1_path, img2_path, same_class_label)``
            where ``same_class_label`` is ``1`` for positive and ``0`` for
            negative pairs.
        """
        num_pos = self.num_pairs_per_epoch // 2
        num_neg = self.num_pairs_per_epoch - num_pos
        pairs: List[Tuple[str, str, int]] = []

        # --- Positive pairs ------------------------------------------------
        for _ in range(num_pos):
            cls = random.choice(self._classes_with_pairs)
            img1, img2 = random.sample(self.class_to_images[cls], 2)
            pairs.append((img1, img2, 1))

        # --- Negative pairs ------------------------------------------------
        for _ in range(num_neg):
            cls1, cls2 = random.sample(self._all_classes, 2)
            img1 = random.choice(self.class_to_images[cls1])
            img2 = random.choice(self.class_to_images[cls2])
            pairs.append((img1, img2, 0))

        random.shuffle(pairs)
        return pairs


class TripletSampler:
    """Sample anchor-positive-negative triplets for triplet learning.

    Parameters
    ----------
    dataset : object
        Dataset with class-to-image mapping (see :func:`_build_class_index`).
    num_triplets_per_epoch : int
        Total number of triplets per call to :meth:`sample_triplets`.
    mining : str
        Mining strategy — ``'random'``, ``'hard'``, or ``'semi-hard'``.

        * ``'random'``: uniformly random negative.
        * ``'hard'``: closest negative in embedding space (falls back to
          random when no embeddings are available).
        * ``'semi-hard'``: negative farther than the positive but within
          the margin (falls back to random when not possible).
    """

    VALID_MINING = {"random", "hard", "semi-hard"}

    def __init__(
        self,
        dataset: Any,
        num_triplets_per_epoch: int = 1000,
        mining: str = "semi-hard",
    ) -> None:
        if mining not in self.VALID_MINING:
            raise ValueError(
                f"mining must be one of {self.VALID_MINING}, got '{mining}'"
            )
        self.class_to_images = _build_class_index(dataset)
        self.num_triplets_per_epoch = num_triplets_per_epoch
        self.mining = mining

        self._classes_with_pairs = [
            c for c, imgs in self.class_to_images.items() if len(imgs) >= 2
        ]
        self._all_classes = list(self.class_to_images.keys())

        if len(self._classes_with_pairs) == 0:
            raise ValueError(
                "No class has ≥ 2 images; cannot form positive pairs."
            )
        if len(self._all_classes) < 2:
            raise ValueError(
                "Need at least 2 classes to form negative triplets."
            )

        # Embedding cache for hard / semi-hard mining (set externally).
        self._embeddings: Optional[Dict[str, np.ndarray]] = None
        self._margin: float = 1.0

        logger.info(
            "TripletSampler: %d classes, mining='%s', %d triplets/epoch",
            len(self._all_classes),
            self.mining,
            self.num_triplets_per_epoch,
        )

    # ------------------------------------------------------------------
    # Public API to update embeddings for online mining
    # ------------------------------------------------------------------

    def update_embeddings(
        self,
        embeddings: Dict[str, np.ndarray],
        margin: float = 1.0,
    ) -> None:
        """Cache current embeddings for hard / semi-hard mining.

        Parameters
        ----------
        embeddings : dict[str, np.ndarray]
            Mapping from image path to its embedding vector.
        margin : float
            Current triplet margin.
        """
        self._embeddings = embeddings
        self._margin = margin

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _random_negative(self, anchor_cls: str) -> str:
        """Pick a random image from a different class."""
        neg_cls = anchor_cls
        while neg_cls == anchor_cls:
            neg_cls = random.choice(self._all_classes)
        return random.choice(self.class_to_images[neg_cls])

    def _mine_negative(
        self,
        anchor_path: str,
        positive_path: str,
        anchor_cls: str,
    ) -> str:
        """Mine a negative using the current strategy and cached embeddings.

        Falls back to random if embeddings are unavailable or no suitable
        negative is found.
        """
        if self._embeddings is None or anchor_path not in self._embeddings:
            return self._random_negative(anchor_cls)

        anchor_emb = self._embeddings[anchor_path]
        pos_emb = self._embeddings.get(positive_path)
        d_ap = (
            float(np.linalg.norm(anchor_emb - pos_emb))
            if pos_emb is not None
            else None
        )

        # Collect candidate negatives with distances
        candidates: List[Tuple[str, float]] = []
        for cls in self._all_classes:
            if cls == anchor_cls:
                continue
            for neg_path in self.class_to_images[cls]:
                neg_emb = self._embeddings.get(neg_path)
                if neg_emb is None:
                    continue
                d_an = float(np.linalg.norm(anchor_emb - neg_emb))
                candidates.append((neg_path, d_an))

        if not candidates:
            return self._random_negative(anchor_cls)

        if self.mining == "hard":
            # Closest (hardest) negative
            candidates.sort(key=lambda t: t[1])
            return candidates[0][0]

        if self.mining == "semi-hard" and d_ap is not None:
            # Farther than positive but within margin
            semi_hard = [
                (p, d) for p, d in candidates
                if d_ap < d < d_ap + self._margin
            ]
            if semi_hard:
                return random.choice(semi_hard)[0]

        # Fallback to random
        return self._random_negative(anchor_cls)

    def sample_triplets(
        self,
    ) -> List[Tuple[str, str, str]]:
        """Sample triplets for one epoch.

        Returns
        -------
        list[tuple[str, str, str]]
            Each element is ``(anchor_path, positive_path, negative_path)``.
        """
        triplets: List[Tuple[str, str, str]] = []
        use_mining = self.mining in {"hard", "semi-hard"}

        for _ in range(self.num_triplets_per_epoch):
            cls = random.choice(self._classes_with_pairs)
            anchor, positive = random.sample(self.class_to_images[cls], 2)

            if use_mining:
                negative = self._mine_negative(anchor, positive, cls)
            else:
                negative = self._random_negative(cls)

            triplets.append((anchor, positive, negative))

        return triplets


# ======================================================================
# k-NN and Prototype prediction heads
# ======================================================================


@torch.no_grad()
def knn_predict(
    model: nn.Module,
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    query_images: torch.Tensor,
    k: int = 1,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """k-Nearest-Neighbour prediction in embedding space.

    Parameters
    ----------
    model : nn.Module
        Siamese or backbone model with a ``forward_once`` or direct
        ``forward`` method that maps ``(batch, 1, H, W)`` → ``(batch, D)``.
    support_images : torch.Tensor
        Shape ``(n_support, 1, H, W)``.
    support_labels : torch.Tensor
        Shape ``(n_support,)`` integer class labels.
    query_images : torch.Tensor
        Shape ``(n_query, 1, H, W)``.
    k : int
        Number of neighbours.
    device : str
        Torch device string.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(predicted_labels, confidences)`` each of shape ``(n_query,)``.
        Confidence is the fraction of *k* neighbours that agree on the
        predicted label.
    """
    model.eval()
    support_images = support_images.to(device)
    query_images = query_images.to(device)
    support_labels = support_labels.to(device)

    # Embed
    embed_fn = getattr(model, "forward_once", model)
    support_emb = embed_fn(support_images)  # (n_support, D)
    query_emb = embed_fn(query_images)      # (n_query, D)

    # Pairwise Euclidean distances: (n_query, n_support)
    dists = torch.cdist(query_emb, support_emb, p=2)

    # Top-k nearest
    k = min(k, support_emb.size(0))
    _, topk_idx = dists.topk(k, dim=1, largest=False)  # (n_query, k)
    topk_labels = support_labels[topk_idx]  # (n_query, k)

    # Majority vote
    preds: List[int] = []
    confs: List[float] = []
    for i in range(query_emb.size(0)):
        labels_i = topk_labels[i]
        unique, counts = labels_i.unique(return_counts=True)
        best_idx = counts.argmax()
        preds.append(int(unique[best_idx].item()))
        confs.append(float(counts[best_idx].item()) / k)

    predicted = torch.tensor(preds, dtype=torch.long, device=device)
    confidences = torch.tensor(confs, dtype=torch.float32, device=device)
    return predicted, confidences


@torch.no_grad()
def prototype_predict(
    model: nn.Module,
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    query_images: torch.Tensor,
    n_way: int,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prototype-based prediction in embedding space.

    Computes the mean embedding (prototype) per class from the support
    set, then assigns each query to the nearest prototype.

    Parameters
    ----------
    model : nn.Module
        Embedding model (Siamese or backbone).
    support_images : torch.Tensor
        Shape ``(n_support, 1, H, W)``.
    support_labels : torch.Tensor
        Shape ``(n_support,)`` integer class labels.
    query_images : torch.Tensor
        Shape ``(n_query, 1, H, W)``.
    n_way : int
        Number of distinct classes in the episode.
    device : str
        Torch device string.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(predicted_labels, confidences)`` each of shape ``(n_query,)``.
        Confidence is ``softmax(-distances)`` evaluated at the predicted
        class.
    """
    model.eval()
    support_images = support_images.to(device)
    query_images = query_images.to(device)
    support_labels = support_labels.to(device)

    # Embed
    embed_fn = getattr(model, "forward_once", model)
    support_emb = embed_fn(support_images)  # (n_support, D)
    query_emb = embed_fn(query_images)      # (n_query, D)

    # Compute class prototypes
    unique_classes = support_labels.unique(sorted=True)  # (n_way,)
    prototypes = torch.stack(
        [support_emb[support_labels == c].mean(dim=0) for c in unique_classes]
    )  # (n_way, D)

    # Euclidean distances query → prototypes: (n_query, n_way)
    dists = torch.cdist(query_emb, prototypes, p=2)

    # Softmax over negative distances → confidence
    log_probs = F.log_softmax(-dists, dim=1)  # (n_query, n_way)
    probs = log_probs.exp()

    # Predictions: index into unique_classes
    best_indices = dists.argmin(dim=1)  # (n_query,)
    predicted = unique_classes[best_indices]

    # Confidence for the predicted class
    confidences = probs[torch.arange(query_emb.size(0), device=device), best_indices]

    return predicted, confidences


# ======================================================================
# Training helpers
# ======================================================================


def train_siamese_epoch(
    model: nn.Module,
    pair_or_triplet_loader: Any = None,
    optimizer: torch.optim.Optimizer = None,
    device: Any = "cpu",
    loss_fn: Any = None,
    epoch: int = 1,
    episode_loader: Any = None,
    margin: float = 1.0,
) -> Dict[str, float]:
    loader = episode_loader if episode_loader is not None else pair_or_triplet_loader
    if loader is None:
        raise ValueError("Must provide either episode_loader or pair_or_triplet_loader")
    """Train a Siamese model for one epoch.

    Supports both pair-based (contrastive) and triplet-based loaders.

    * **Pair loader** yields ``(img1, img2, labels)`` batches.
    * **Triplet loader** yields ``(anchor, positive, negative)`` batches.

    The ``loss_fn`` signature must match the loader type:

    * ``contrastive_loss(emb1, emb2, labels, ...)``
    * ``triplet_loss(anchor_emb, pos_emb, neg_emb, ...)``

    Parameters
    ----------
    model : nn.Module
        A :class:`SiameseNetwork` (or any module with ``forward_once``).
    pair_or_triplet_loader : DataLoader
        Iterable of batches.
    optimizer : torch.optim.Optimizer
        Optimiser bound to model parameters.
    device : str
        Torch device string.
    loss_fn : callable
        Loss function matching the loader type.
    epoch : int
        Current epoch number (for logging / progress bar).

    Returns
    -------
    dict[str, float]
        ``{'loss': avg_loss}`` over the epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    embed_fn = getattr(model, "forward_once", model)

    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch}",
        leave=False,
    )

    for batch in pbar:
        optimizer.zero_grad()

        if isinstance(batch, dict):
            # Compute triplet loss from episode batch
            sup_x = batch["support_images"].to(device)
            sup_y = batch["support_labels"].to(device)
            qry_x = batch["query_images"].to(device)
            qry_y = batch["query_labels"].to(device)

            sup_emb = embed_fn(sup_x)
            qry_emb = embed_fn(qry_x)

            # For each query, positive is support of same class, negative is random support of different class
            losses = []
            for q_idx in range(len(qry_y)):
                q_label = qry_y[q_idx].item()
                q_vec = qry_emb[q_idx:q_idx+1]

                pos_mask = (sup_y == q_label)
                neg_mask = (sup_y != q_label)
                if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                    pos_vec = sup_emb[pos_mask][0:1]
                    neg_vec = sup_emb[neg_mask][0:1]
                    d_pos = torch.norm(q_vec - pos_vec, p=2)
                    d_neg = torch.norm(q_vec - neg_vec, p=2)
                    l = torch.clamp(d_pos - d_neg + margin, min=0.0)
                    losses.append(l)

            if losses:
                loss = torch.stack(losses).mean()
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)

        elif len(batch) == 3:
            t1, t2, t3 = batch
            t1 = t1.to(device)
            t2 = t2.to(device)
            t3_dev = t3.to(device)

            emb1 = embed_fn(t1)
            emb2 = embed_fn(t2)

            # Distinguish pair vs triplet by whether t3 looks like labels
            # Labels are 1-D integer/float, images are 4-D tensors
            if t3_dev.dim() >= 2:
                # Triplet: t3 is negative images
                emb3 = embed_fn(t3_dev)
                loss = loss_fn(emb1, emb2, emb3)
            else:
                # Pair: t3 is labels
                loss = loss_fn(emb1, emb2, t3_dev)
        else:
            raise ValueError(
                f"Expected batches of 3 elements (pair or triplet) or episode dict, "
                f"got {type(batch)}"
            )

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = running_loss / max(num_batches, 1)
    logger.info("Epoch %d – avg loss: %.4f", epoch, avg_loss)
    return {"loss": avg_loss}


def evaluate_siamese_fewshot(
    model: nn.Module,
    episode_loader: Any,
    device: str,
    n_way: int,
    k_shot: int,
    num_episodes: int = 600,
) -> Dict[str, float]:
    """Evaluate a Siamese model using prototype-based few-shot episodes.

    Each episode from ``episode_loader`` must yield a tuple of
    ``(support_images, support_labels, query_images, query_labels)``.

    Parameters
    ----------
    model : nn.Module
        Embedding model.
    episode_loader : iterable
        Yields ``(support_imgs, support_labels, query_imgs, query_labels)``
        tensors per episode.
    device : str
        Torch device string.
    n_way : int
        Number of classes per episode.
    k_shot : int
        Number of support examples per class.
    num_episodes : int
        Number of episodes to evaluate (loader may have more; we stop
        after this many).

    Returns
    -------
    dict[str, float]
        ``{'accuracy': mean_acc, 'ci95': confidence_interval_95}``.
    """
    model.eval()
    accuracies: List[float] = []

    pbar = tqdm(
        episode_loader,
        total=num_episodes,
        desc=f"Eval {n_way}-way {k_shot}-shot",
        leave=False,
    )

    for i, episode in enumerate(pbar):
        if i >= num_episodes:
            break

        if isinstance(episode, dict):
            support_imgs = episode["support_images"]
            support_labels = episode["support_labels"]
            query_imgs = episode["query_images"]
            query_labels = episode["query_labels"]
        else:
            support_imgs, support_labels, query_imgs, query_labels = episode

        support_imgs = support_imgs.to(device)
        support_labels = support_labels.to(device)
        query_imgs = query_imgs.to(device)
        query_labels = query_labels.to(device)

        preds, _ = prototype_predict(
            model,
            support_imgs,
            support_labels,
            query_imgs,
            n_way=n_way,
            device=device,
        )

        correct = (preds == query_labels).float().mean().item()
        accuracies.append(correct)
        pbar.set_postfix(acc=f"{np.mean(accuracies):.4f}")

    mean_acc = float(np.mean(accuracies)) if accuracies else 0.0
    ci95 = (
        1.96 * float(np.std(accuracies)) / np.sqrt(len(accuracies))
        if len(accuracies) > 1
        else 0.0
    )

    logger.info(
        "Few-shot evaluation (%d-way %d-shot, %d episodes): "
        "accuracy=%.4f ± %.4f",
        n_way,
        k_shot,
        len(accuracies),
        mean_acc,
        ci95,
    )
    return {"accuracy": mean_acc, "ci95": ci95}
