"""
src/predict.py – CLI inference tool for Tamil Few-Shot Character Recognition.

Features:
  - Predict on single image or whole directory
  - Display top-K letters with Unicode character, category, confidence, and Euclidean distance
  - Open-set rejection: flags low-confidence or distant samples as 'unknown / not enrolled'
  - Optional CSV export for batch evaluation
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from src.config import get_config
from src.support_registry import SupportRegistry
from src.tamil_text import find_tamil_font

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("predict")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def predict_single(
    image_path: Path,
    registry: SupportRegistry,
    category_map: Dict[str, str],
    top_k: int = 5,
    threshold: Optional[float] = None
) -> List[Dict[str, Any]]:
    """Predict top_k candidates for a single image."""
    if threshold is not None:
        registry.rejection_threshold = threshold

    results = registry.predict(image_path, top_k=top_k)
    for r in results:
        r["category"] = category_map.get(r["class"], "தெரியாதது")
    return results


def print_prediction_table(image_path: Path, results: List[Dict[str, Any]]):
    """Pretty-print formatted prediction table."""
    print("\n" + "=" * 65)
    print(f"Image: {image_path.name}")
    print("=" * 65)
    print(f"{'Rank':4s} | {'Letter':6s} | {'Category':14s} | {'Confidence':10s} | {'Distance':8s}")
    print("-" * 65)

    is_unknown = results[0]["rejected"] if results else True

    for i, r in enumerate(results, 1):
        cname = r["class"]
        cat = r.get("category", "")
        conf = f"{r['confidence'] * 100:.1f}%"
        dist = f"{r['distance']:.4f}"
        print(f"{i:4d} | {cname:6s} | {cat:14s} | {conf:10s} | {dist:8s}")

    print("-" * 65)
    if is_unknown:
        print("Status: ⚠ UNKNOWN / NOT ENROLLED (Exceeds distance threshold)")
    else:
        top_cls = results[0]["class"]
        top_conf = results[0]["confidence"] * 100
        print(f"Status: ✓ RECOGNIZED as '{top_cls}' ({top_conf:.1f}% confidence)")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Tamil Few-Shot Character Recognition CLI")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to a single image file")
    group.add_argument("--folder", type=str, help="Path to a folder of images")

    parser.add_argument("--model", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--prototypes", type=str, default=None, help="Path to prototypes.npz")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top predictions to display")
    parser.add_argument("--threshold", type=float, default=None, help="Rejection distance threshold")
    parser.add_argument("--output_csv", type=str, default=None, help="Export batch predictions to CSV")

    args = parser.parse_args()
    cfg = get_config()

    # Load category map if available
    class_names_path = Path(cfg.paths._processed) / "class_names.json"
    category_map = {}
    if class_names_path.exists():
        import json
        with open(class_names_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            category_map = meta.get("class_categories", {})

    # Initialize registry
    registry = SupportRegistry(
        model_path=args.model,
        cfg=cfg
    )
    if args.prototypes:
        registry.load_prototypes(Path(args.prototypes))

    if not registry.prototypes:
        print("Warning: No class prototypes loaded! Rebuilding from support set...")
        registry.build_prototypes()

    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"Error: Image not found at {img_path}")
            sys.exit(1)
        res = predict_single(img_path, registry, category_map, top_k=args.top_k, threshold=args.threshold)
        print_prediction_table(img_path, res)

    elif args.folder:
        folder_path = Path(args.folder)
        if not folder_path.exists():
            print(f"Error: Folder not found at {folder_path}")
            sys.exit(1)

        image_files = [p for p in folder_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        print(f"Found {len(image_files)} images in {folder_path}")

        records = []
        for img_p in image_files:
            res = predict_single(img_p, registry, category_map, top_k=args.top_k, threshold=args.threshold)
            print_prediction_table(img_p, res)

            top = res[0]
            records.append({
                "filename": img_p.name,
                "top1_class": top["class"],
                "top1_category": top.get("category", ""),
                "top1_confidence": top["confidence"],
                "top1_distance": top["distance"],
                "rejected": top["rejected"]
            })

        if args.output_csv:
            csv_out = Path(args.output_csv)
            csv_out.parent.mkdir(parents=True, exist_ok=True)
            with open(csv_out, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["filename", "top1_class", "top1_category", "top1_confidence", "top1_distance", "rejected"])
                writer.writeheader()
                writer.writerows(records)
            print(f"Batch predictions saved to {csv_out}")


if __name__ == "__main__":
    main()

