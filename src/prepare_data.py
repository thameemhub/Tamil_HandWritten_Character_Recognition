"""
src/prepare_data.py – Data preparation pipeline for Tamil Few-Shot HTR.

Key steps:
  1. Extract zip-only class folders (ப, பா, பி, பீ) into data/interim/extracted/<class>/
     (The original dataset is strictly READ-ONLY).
  2. Recursively discover all active leaf image folders with NFC normalization.
  3. Detect near/exact duplicates via perceptual hash (imagehash.phash) and exclude them.
  4. Preprocess all unique images using the unified preprocessing module (64x64, polarity normalized,
     grayscale, cropped & padded to square).
  5. Save preprocessed images to data/processed/images/<class>/<id>.png.
  6. Generate and save class_names.json (mapping index <-> label <-> category).
  7. Construct Protocol A image-level splits (train 60%, val 20%, test 20%) stratified by source group.
  8. Construct Protocol B class-level splits (base 70%, val 15%, novel 15%) stratified by category.
  9. Save manifest.csv with columns: image_id, class_name, class_idx, category, source_group,
     processed_path, split_protocol_a, split_protocol_b.
  10. Generate reports/preprocess_samples.png displaying 24 random preprocessed samples.
"""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
import sys
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import imagehash
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.config import get_config, seed_everything
from src.preprocessing import preprocess_image, read_image
from src.tamil_text import configure_matplotlib_tamil, find_tamil_font

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("prepare_data")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def normalize_nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def determine_category(relative_path: Path) -> str:
    parts = relative_path.parts
    if not parts:
        return "தெரியாதது"
    top = normalize_nfc(parts[0])
    if "ஆய்த" in top:
        return "ஆய்தம"
    elif "உயிர்மெய்" in top:
        return "உயிர்மெய்"
    elif "உயிர்" in top:
        return "உயிர்"
    elif "மெய்" in top:
        return "மெய்"
    return "பிற"


def determine_source_group(filename: str) -> str:
    fn_lower = filename.lower()
    if "whatsapp" in fn_lower:
        return "WhatsApp"
    elif "screenshot" in fn_lower:
        return "Screenshot"
    elif "scanned" in fn_lower:
        return "Scanned"
    elif "resized" in fn_lower:
        return "resized"
    elif filename.startswith("PXL_") or "pxl_" in fn_lower:
        return "PXL"
    elif filename.startswith("IMG_") or "img_" in fn_lower or filename.startswith("IMG"):
        return "IMG"
    elif "copy" in fn_lower:
        return "Copy"
    else:
        return "numeric"


def extract_zips(dataset_root: Path, interim_dir: Path) -> Dict[str, List[Path]]:
    """
    Extract zip files found in zip-only class folders into interim_dir/extracted/<class>/.
    Does not touch the dataset root.
    """
    extracted_classes: Dict[str, List[Path]] = defaultdict(list)
    extracted_base = interim_dir / "extracted"
    extracted_base.mkdir(parents=True, exist_ok=True)

    for root, dirs, files in os.walk(dataset_root):
        zips = [f for f in files if f.lower().endswith(".zip")]
        images = [f for f in files if any(f.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)]
        if zips and not images:
            folder_path = Path(root)
            class_name = normalize_nfc(folder_path.name)
            target_class_dir = extracted_base / class_name
            target_class_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Extracting {len(zips)} zip(s) for class '{class_name}' into {target_class_dir}")
            for zfile in zips:
                zip_path = folder_path / zfile
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zf:
                        for member in zf.namelist():
                            if any(member.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
                                # Extract directly into class dir without subfolder nesting
                                filename = Path(member).name
                                if not filename:
                                    continue
                                out_file = target_class_dir / filename
                                with zf.open(member) as source, open(out_file, "wb") as target:
                                    shutil.copyfileobj(source, target)
                                extracted_classes[class_name].append(out_file)
                except Exception as e:
                    logger.error(f"Error extracting zip {zip_path}: {e}")

    logger.info(f"Extracted zip classes: {list(extracted_classes.keys())} with counts: {[len(v) for v in extracted_classes.values()]}")
    return extracted_classes


def discover_all_images(dataset_root: Path, extracted_classes: Dict[str, List[Path]]) -> Tuple[Dict[str, List[Path]], Dict[str, str]]:
    """
    Discover all images from dataset root + interim extracted zips.
    Returns:
      class_images: {class_name: [image_paths]}
      class_categories: {class_name: category}
    """
    class_images: Dict[str, List[Path]] = defaultdict(list)
    class_categories: Dict[str, str] = {}

    for root, dirs, files in os.walk(dataset_root):
        folder_path = Path(root)
        if folder_path == dataset_root:
            continue
        rel = folder_path.relative_to(dataset_root)
        cat = determine_category(rel)

        images = [folder_path / f for f in files if any(f.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)]
        if images:
            class_name = normalize_nfc(folder_path.name)
            class_images[class_name].extend(images)
            class_categories[class_name] = cat

    # Add extracted classes (all under 'உயிர்மெய்')
    for class_name, img_paths in extracted_classes.items():
        class_images[class_name].extend(img_paths)
        class_categories[class_name] = "உயிர்மெய்"

    return class_images, class_categories


def deduplicate_images(class_images: Dict[str, List[Path]], max_distance: int = 2) -> Tuple[Dict[str, List[Path]], int]:
    """
    Compute perceptual hashes per class and remove near/exact duplicates.
    Keeps the first occurrence.
    """
    unique_class_images: Dict[str, List[Path]] = {}
    total_removed = 0

    logger.info("Computing perceptual hashes for deduplication...")
    for class_name, paths in class_images.items():
        hashes: List[imagehash.ImageHash] = []
        unique_paths: List[Path] = []

        for p in paths:
            img = read_image(p)
            if img is None:
                continue

            try:
                pil_im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                h = imagehash.phash(pil_im)
            except Exception:
                # If phash fails, keep the image
                unique_paths.append(p)
                continue

            is_dup = False
            for existing_h in hashes:
                if h - existing_h <= max_distance:
                    is_dup = True
                    break

            if not is_dup:
                hashes.append(h)
                unique_paths.append(p)
            else:
                total_removed += 1

        unique_class_images[class_name] = unique_paths

    logger.info(f"Deduplication complete: removed {total_removed} duplicate/near-duplicate images.")
    return unique_class_images, total_removed


def create_protocol_splits(
    manifest_records: List[Dict[str, Any]],
    class_names: List[str],
    class_categories: Dict[str, str],
    cfg: Any
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """
    Construct:
      1. Protocol A: within-class splits (train 60%, val 20%, test 20%), stratified by source_group where feasible.
      2. Protocol B: class-level splits (base 70%, val 15%, novel 15%), stratified by category.
    """
    # ── Protocol B: Class-level split ─────────────────────────────
    # Stratify classes by category
    cat_to_classes: Dict[str, List[str]] = defaultdict(list)
    for c in sorted(class_names):
        cat_to_classes[class_categories[c]].append(c)

    base_classes: List[str] = []
    val_classes: List[str] = []
    novel_classes: List[str] = []

    b_ratio = cfg.splits.protocol_b.base_ratio
    v_ratio = cfg.splits.protocol_b.val_ratio

    for cat, cls_list in sorted(cat_to_classes.items()):
        # Deterministic shuffle
        rng = random.Random(cfg.seed)
        shuffled = list(cls_list)
        rng.shuffle(shuffled)

        n = len(shuffled)
        if n == 1:
            # Single class (e.g. ஆய்தம ஃ): place in base
            base_classes.extend(shuffled)
            continue

        n_base = max(1, int(round(n * b_ratio)))
        n_val = max(1, int(round(n * v_ratio)))
        # Ensure at least 1 novel class if n >= 3
        if n >= 3 and (n - n_base - n_val) < 1:
            n_base = n - 2
            n_val = 1

        base = shuffled[:n_base]
        val = shuffled[n_base:n_base + n_val]
        novel = shuffled[n_base + n_val:]

        base_classes.extend(base)
        val_classes.extend(val)
        novel_classes.extend(novel)

    class_split_dict = {
        "base": sorted(base_classes),
        "val": sorted(val_classes),
        "novel": sorted(novel_classes)
    }

    class_b_assignment = {}
    for c in base_classes:
        class_b_assignment[c] = "base"
    for c in val_classes:
        class_b_assignment[c] = "val"
    for c in novel_classes:
        class_b_assignment[c] = "novel"

    # ── Protocol A: Per-class image splits ────────────────────────
    # Group records by class
    class_to_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in manifest_records:
        class_to_records[r["class_name"]].append(r)

    a_train_ratio = cfg.splits.protocol_a.train_ratio
    a_val_ratio = cfg.splits.protocol_a.val_ratio

    for cname, recs in class_to_records.items():
        rng = random.Random(cfg.seed + hash(cname) % 10000)
        # Group by source_group for stratified split
        src_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in recs:
            src_groups[r["source_group"]].append(r)

        train_set: List[Dict[str, Any]] = []
        val_set: List[Dict[str, Any]] = []
        test_set: List[Dict[str, Any]] = []

        for sgrp, s_recs in src_groups.items():
            s_shuffled = list(s_recs)
            rng.shuffle(s_shuffled)
            n_s = len(s_shuffled)

            if n_s == 1:
                train_set.append(s_shuffled[0])
            elif n_s == 2:
                train_set.append(s_shuffled[0])
                test_set.append(s_shuffled[1])
            else:
                n_tr = max(1, int(round(n_s * a_train_ratio)))
                n_v = max(1, int(round(n_s * a_val_ratio)))
                train_set.extend(s_shuffled[:n_tr])
                val_set.extend(s_shuffled[n_tr:n_tr + n_v])
                test_set.extend(s_shuffled[n_tr + n_v:])

        for r in train_set:
            r["split_protocol_a"] = "train"
        for r in val_set:
            r["split_protocol_a"] = "val"
        for r in test_set:
            r["split_protocol_a"] = "test"

        for r in recs:
            r["split_protocol_b"] = class_b_assignment.get(cname, "base")

    return manifest_records, class_split_dict


def save_preprocess_samples(manifest_records: List[Dict[str, Any]], output_path: Path, num_samples: int = 24):
    """Save a 4x6 grid of preprocessed character images to reports/preprocess_samples.png."""
    configure_matplotlib_tamil()
    rng = random.Random(42)
    sample_records = rng.sample(manifest_records, min(num_samples, len(manifest_records)))

    rows, cols = 4, 6
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    fig.patch.set_facecolor("#1e1e1e")

    for i, ax in enumerate(axes.flat):
        if i < len(sample_records):
            rec = sample_records[i]
            # Use np.fromfile + cv2.imdecode for Windows non-ASCII path support
            try:
                raw_bytes = np.fromfile(str(rec["processed_path"]), dtype=np.uint8)
                img = cv2.imdecode(raw_bytes, cv2.IMREAD_GRAYSCALE)
            except Exception:
                img = None
            if img is not None:
                ax.imshow(img, cmap="gray")
            ax.set_title(f"{rec['class_name']}\n({rec['category']})", color="white", fontsize=11, fontname="Nirmala UI")
        ax.axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none", dpi=150)
    plt.close()
    logger.info(f"Saved 24 preprocessed sample grid to {output_path}")


def prepare_data():
    """Main data preparation execution."""
    cfg = get_config()
    seed_everything(cfg.seed)

    dataset_root = Path(cfg.paths.dataset_root)
    interim_dir = Path(cfg.paths._interim)
    processed_dir = Path(cfg.paths._processed)
    images_dir = processed_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(cfg.paths._reports)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract zip folders
    extracted_classes = extract_zips(dataset_root, interim_dir)

    # 2. Discover all images
    class_images, class_categories = discover_all_images(dataset_root, extracted_classes)

    # 3. Deduplicate
    unique_class_images, num_dups = deduplicate_images(class_images, max_distance=2)

    # Filter out empty classes
    active_class_images = {k: v for k, v in unique_class_images.items() if len(v) > 0}
    sorted_class_names = sorted(active_class_images.keys())
    class_to_idx = {name: idx for idx, name in enumerate(sorted_class_names)}
    idx_to_class = {idx: name for idx, name in enumerate(sorted_class_names)}

    logger.info(f"Total active classes for processing: {len(sorted_class_names)}")

    # 4. Save class_names.json
    class_metadata = {
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "class_categories": {k: class_categories[k] for k in sorted_class_names},
        "num_classes": len(sorted_class_names)
    }
    class_names_path = processed_dir / "class_names.json"
    with open(class_names_path, "w", encoding="utf-8") as f:
        json.dump(class_metadata, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved class metadata to {class_names_path}")

    # 5. Preprocess all images and write to data/processed/images/<class>/
    manifest_records: List[Dict[str, Any]] = []
    img_counter = 0

    logger.info("Preprocessing all unique images...")
    for cname in tqdm(sorted_class_names, desc="Preprocessing classes"):
        c_idx = class_to_idx[cname]
        cat = class_categories[cname]
        c_dir = images_dir / cname
        c_dir.mkdir(parents=True, exist_ok=True)

        for p in active_class_images[cname]:
            # Unified preprocessing
            preprocessed = preprocess_image(
                p,
                image_size=cfg.preprocessing.image_size,
                use_binarization=cfg.preprocessing.use_binarization,
                denoise_strength=cfg.preprocessing.denoise_strength,
            )
            if preprocessed is None:
                continue

            img_id = f"{c_idx:03d}_{img_counter:05d}"
            img_counter += 1
            out_img_path = c_dir / f"{img_id}.png"

            # Save as 8-bit PNG
            save_img = (preprocessed * 255.0).astype(np.uint8)
            # Use cv2.imencode + tofile for Windows non-ASCII path safety
            success, encoded_img = cv2.imencode(".png", save_img)
            if success:
                encoded_img.tofile(str(out_img_path))
            else:
                cv2.imwrite(str(out_img_path), save_img)

            src_grp = determine_source_group(p.name)

            manifest_records.append({
                "image_id": img_id,
                "class_name": cname,
                "class_idx": c_idx,
                "category": cat,
                "source_group": src_grp,
                "processed_path": str(out_img_path),
                "original_path": str(p),
            })

    logger.info(f"Preprocessed {len(manifest_records)} images across {len(sorted_class_names)} classes.")

    # 6. Create Protocol A and B splits
    manifest_records, class_split_dict = create_protocol_splits(
        manifest_records, sorted_class_names, class_categories, cfg
    )

    # 7. Save class_split.json
    class_split_path = processed_dir / "class_split.json"
    with open(class_split_path, "w", encoding="utf-8") as f:
        json.dump(class_split_dict, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved class split to {class_split_path}")

    # 8. Save manifest.csv
    manifest_df = pd.DataFrame(manifest_records)
    manifest_path = processed_dir / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False, encoding="utf-8")
    logger.info(f"Saved manifest to {manifest_path}")

    # 9. Save preprocessed sample grid
    save_preprocess_samples(manifest_records, reports_dir / "preprocess_samples.png", 24)

    # 10. Summary prints
    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETED")
    print("=" * 60)
    print(f"Total Classes Processed        : {len(sorted_class_names)}")
    print(f"Total Images in Manifest       : {len(manifest_df)}")
    print("\nProtocol A (Image-level split counts):")
    for s, cnt in manifest_df["split_protocol_a"].value_counts().items():
        print(f"  {s:10s}: {cnt} ({cnt/len(manifest_df)*100:.1f}%)")
    print("\nProtocol B (Class-level split counts):")
    print(f"  Base Classes  : {len(class_split_dict['base'])}")
    print(f"  Val Classes   : {len(class_split_dict['val'])}")
    print(f"  Novel Classes : {len(class_split_dict['novel'])}")
    print("=" * 60)


if __name__ == "__main__":
    prepare_data()
