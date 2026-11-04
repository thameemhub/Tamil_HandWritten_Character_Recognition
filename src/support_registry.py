"""
src/support_registry.py – Class prototype registry for zero-retraining deployment.

Allows users to manage few-shot reference examples per letter in data/support_set/<label>/,
compute class prototypes, and enroll brand new letters (e.g. empty classes such as ச, சா, ழ)
by simply dropping 3-5 images into a folder and refreshing prototypes without any model retraining.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.backbones import get_backbone
from src.config import get_config
from src.preprocessing import preprocess_for_model, preprocess_image, read_image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("support_registry")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def normalize_nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


class SupportRegistry:
    """
    Manages class prototype embeddings built from reference images in a support directory.
    Provides fast nearest-prototype classification and open-set unknown rejection.
    """

    def __init__(
        self,
        support_dir: Optional[Union[str, Path]] = None,
        model: Optional[torch.nn.Module] = None,
        model_path: Optional[Union[str, Path]] = None,
        device: str = "cpu",
        cfg: Any = None
    ):
        if cfg is None:
            cfg = get_config()
        self.cfg = cfg
        self.device = torch.device(device)
        self.support_dir = Path(support_dir) if support_dir else Path(cfg.paths._support_set)
        self.support_dir.mkdir(parents=True, exist_ok=True)

        self.prototypes: Dict[str, np.ndarray] = {}  # class_name -> (D,)
        self.class_names: List[str] = []
        self.rejection_threshold: float = 1.25  # Default threshold for normalized Euclidean distance

        # Load or initialize model backbone
        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = self._load_backbone(model_path)
        self.model.eval()

        # Try loading existing cached prototypes
        self.cache_file = self.support_dir / "prototypes.npz"
        self.metadata_file = self.support_dir / "registry_classes.json"
        if self.cache_file.exists():
            self.load_prototypes()

    def _load_backbone(self, model_path: Optional[Union[str, Path]]) -> torch.nn.Module:
        backbone_name = getattr(self.cfg.fewshot, "backbone", "conv4")
        embed_dim = getattr(self.cfg.fewshot, "embedding_dim", 64)
        backbone = get_backbone(backbone_name, in_channels=1, embedding_dim=embed_dim)

        if model_path is None:
            model_path = Path(self.cfg.paths._models) / "protonet_best.pth"
        else:
            model_path = Path(model_path)

        if model_path.exists():
            logger.info(f"Loading backbone weights from checkpoint: {model_path}")
            ckpt = torch.load(model_path, map_location=self.device)
            state_dict = ckpt.get("backbone_state", ckpt.get("model_state", ckpt))
            # Clean state dict keys if needed
            cleaned_state = {}
            for k, v in state_dict.items():
                k_clean = k.replace("backbone.", "")
                cleaned_state[k_clean] = v
            try:
                backbone.load_state_dict(cleaned_state, strict=False)
                logger.info("Successfully loaded backbone weights.")
            except Exception as e:
                logger.warning(f"Could not load state dict strictly: {e}")
        else:
            logger.warning(f"No checkpoint found at {model_path}. Using initialized backbone.")

        return backbone.to(self.device)

    def scan_support_set(self) -> Dict[str, List[Path]]:
        """
        Recursively scan support_dir for folders containing images.
        Folder name = NFC-normalized class label.
        """
        enrolled: Dict[str, List[Path]] = {}
        for root, dirs, files in os.walk(self.support_dir):
            p_root = Path(root)
            if p_root == self.support_dir:
                continue
            images = [p_root / f for f in files if any(f.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)]
            if images:
                label = normalize_nfc(p_root.name)
                enrolled[label] = images
        return enrolled

    def build_prototypes(self) -> Dict[str, np.ndarray]:
        """
        Compute mean embedding vector for each class folder in the support set.
        """
        class_images = self.scan_support_set()
        if not class_images:
            logger.warning(f"No support image folders found in {self.support_dir}")
            return {}

        logger.info(f"Building prototypes for {len(class_images)} classes...")
        new_prototypes = {}

        self.model.eval()
        with torch.no_grad():
            for cname, paths in class_images.items():
                embeddings = []
                for p in paths:
                    tensor = preprocess_for_model(p, self.cfg)
                    if tensor is None:
                        continue
                    tensor = tensor.unsqueeze(0).to(self.device)  # (1, 1, H, W)
                    emb = self.model(tensor)                       # (1, D), L2-normalized
                    embeddings.append(emb.cpu().numpy().flatten())

                if embeddings:
                    # Class prototype = mean vector, L2 re-normalized
                    mean_emb = np.mean(embeddings, axis=0)
                    norm = np.linalg.norm(mean_emb)
                    if norm > 0:
                        mean_emb = mean_emb / norm
                    new_prototypes[cname] = mean_emb

        self.prototypes = new_prototypes
        self.class_names = sorted(new_prototypes.keys())
        self.save_prototypes()
        logger.info(f"Built and cached {len(self.prototypes)} prototypes.")
        return self.prototypes

    def save_prototypes(self, path: Optional[Path] = None):
        """Save prototypes to .npz and class list to .json."""
        if path is None:
            path = self.cache_file
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save numpy archive
        np.savez_compressed(str(path), **self.prototypes)

        # Save metadata
        metadata = {
            "classes": self.class_names,
            "rejection_threshold": float(self.rejection_threshold),
            "num_classes": len(self.class_names)
        }
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def load_prototypes(self, path: Optional[Path] = None) -> bool:
        """Load prototypes from .npz and metadata from .json."""
        if path is None:
            path = self.cache_file
        if not path.exists():
            return False

        try:
            with np.load(str(path)) as data:
                self.prototypes = {k: data[k] for k in data.files}
            self.class_names = sorted(self.prototypes.keys())

            if self.metadata_file.exists():
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    self.rejection_threshold = meta.get("rejection_threshold", 1.25)

            logger.info(f"Loaded {len(self.prototypes)} prototypes from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading prototypes from {path}: {e}")
            return False

    def refresh(self):
        """Rescan support set and recompute all prototypes."""
        self.build_prototypes()

    def enroll_class(self, class_name: str, image_paths: List[Union[str, Path]]) -> bool:
        """
        Enroll a new letter or add examples to an existing letter.
        Copies images into data/support_set/<class_name>/ and updates prototypes.
        """
        cname_nfc = normalize_nfc(class_name)
        target_dir = self.support_dir / cname_nfc
        target_dir.mkdir(parents=True, exist_ok=True)

        added = 0
        for p in image_paths:
            p = Path(p)
            if p.exists() and any(p.suffix.lower() == ext for ext in IMAGE_EXTENSIONS):
                dest = target_dir / p.name
                shutil.copy2(p, dest)
                added += 1

        logger.info(f"Enrolled class '{cname_nfc}' with {added} images in {target_dir}")
        self.refresh()
        return True

    def enroll_from_canvas_image(self, class_name: str, img_array: np.ndarray) -> bool:
        """
        Enroll an image directly provided as a numpy array (e.g. from Tkinter canvas).
        """
        cname_nfc = normalize_nfc(class_name)
        target_dir = self.support_dir / cname_nfc
        target_dir.mkdir(parents=True, exist_ok=True)

        existing = list(target_dir.glob("sample_*.png"))
        idx = len(existing) + 1
        out_path = target_dir / f"sample_{idx:03d}.png"

        # Safe save with cv2.imencode for non-ASCII path
        success, encoded = cv2.imencode(".png", img_array)
        if success:
            encoded.tofile(str(out_path))
        else:
            cv2.imwrite(str(out_path), img_array)

        logger.info(f"Saved drawn sample to {out_path}")
        self.refresh()
        return True

    def calibrate_threshold(self, val_features: np.ndarray, val_labels: List[str], percentile: float = 95.0):
        """
        Calibrate rejection distance threshold based on validation distances.
        """
        if not self.prototypes:
            return

        distances = []
        for feat, label in zip(val_features, val_labels):
            if label in self.prototypes:
                proto = self.prototypes[label]
                # Euclidean distance
                d = float(np.linalg.norm(feat - proto))
                distances.append(d)

        if distances:
            self.rejection_threshold = float(np.percentile(distances, percentile))
            self.save_prototypes()
            logger.info(f"Calibrated open-set rejection threshold to {self.rejection_threshold:.4f} (at {percentile}th percentile)")

    def predict(
        self,
        img_or_tensor: Union[np.ndarray, torch.Tensor, str, Path],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Predict nearest classes for an input image.
        Returns sorted list of top_k results with class, confidence, distance, and rejection status.
        """
        if not self.prototypes:
            return [{"class": "None Enrolled", "confidence": 0.0, "distance": 999.0, "rejected": True}]

        # Prepare tensor
        if isinstance(img_or_tensor, (str, Path)):
            tensor = preprocess_for_model(img_or_tensor, self.cfg)
        elif isinstance(img_or_tensor, np.ndarray):
            if img_or_tensor.dtype != np.float32 or img_or_tensor.max() > 1.0:
                pre = preprocess_image(img_or_tensor, self.cfg.preprocessing.image_size)
            else:
                pre = img_or_tensor
            tensor = torch.from_numpy(pre).unsqueeze(0).float() if pre is not None else None
        elif isinstance(img_or_tensor, torch.Tensor):
            tensor = img_or_tensor
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(0)
            elif tensor.dim() == 4:
                tensor = tensor.squeeze(0)
        else:
            tensor = None

        if tensor is None:
            return [{"class": "Invalid Input", "confidence": 0.0, "distance": 999.0, "rejected": True}]

        # Extract embedding
        self.model.eval()
        with torch.no_grad():
            t_input = tensor.unsqueeze(0).to(self.device)  # (1, 1, H, W)
            emb = self.model(t_input).cpu().numpy().flatten()  # (D,)

        # Compute Euclidean distances to all prototypes
        scores = []
        for cname in self.class_names:
            proto = self.prototypes[cname]
            dist = float(np.linalg.norm(emb - proto))
            scores.append((cname, dist))

        # Sort by distance ascending
        scores.sort(key=lambda x: x[1])

        # Convert top distances to softmax confidence
        top_scores = scores[:top_k]
        dists = np.array([s[1] for s in top_scores])
        # Temperature scaling
        neg_dists = -dists
        exp_dists = np.exp(neg_dists - np.max(neg_dists))
        probs = exp_dists / np.sum(exp_dists)

        results = []
        for i, (cname, dist) in enumerate(top_scores):
            is_rejected = (i == 0 and dist > self.rejection_threshold)
            results.append({
                "class": cname,
                "confidence": float(probs[i]),
                "distance": dist,
                "rejected": is_rejected
            })

        return results


def main():
    parser = argparse.ArgumentParser(description="Tamil Support Registry CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    subparsers.add_parser("build", help="Build and cache prototypes from support_set directory")
    subparsers.add_parser("list", help="List all enrolled classes")

    enroll_p = subparsers.add_parser("enroll", help="Enroll a new letter or additional images")
    enroll_p.add_argument("--class_name", required=True, help="Tamil letter label")
    enroll_p.add_argument("--images", nargs="+", required=True, help="Paths to image files")

    pred_p = subparsers.add_parser("predict", help="Predict class of an image")
    pred_p.add_argument("--image", required=True, help="Path to image file")
    pred_p.add_argument("--top_k", type=int, default=5, help="Number of predictions")

    args = parser.parse_args()
    reg = SupportRegistry()

    if args.command == "build":
        reg.build_prototypes()
    elif args.command == "list":
        print(f"Total enrolled classes: {len(reg.class_names)}")
        for idx, c in enumerate(reg.class_names, 1):
            print(f"  {idx:3d}. {c}")
    elif args.command == "enroll":
        reg.enroll_class(args.class_name, args.images)
        print(f"Successfully enrolled '{args.class_name}' with {len(args.images)} images.")
    elif args.command == "predict":
        res = reg.predict(args.image, top_k=args.top_k)
        print("\nPredictions:")
        for r in res:
            status = "⚠ UNKNOWN (REJECTED)" if r["rejected"] else "✓"
            print(f"  {r['class']:10s} | Conf: {r['confidence']:.4f} | Dist: {r['distance']:.4f} | {status}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

