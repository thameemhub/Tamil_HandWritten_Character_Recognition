"""
src/dataset_audit.py – Comprehensive audit of the Tamil Letter Dataset.

Performs a recursive scan across the dataset root, discovering all classes,
handling nested directories, empty folders, zip-only folders, Unicode NFC normalization,
source-group categorization, image corruption checks, resolution statistics,
and perceptual hash duplicate detection.

Outputs:
  - reports/dataset_audit.md
  - reports/dataset_audit.csv
"""

from __future__ import annotations

import csv
import logging
import os
import re
import sys
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import imagehash
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.config import get_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("dataset_audit")

# Supported image extensions
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def normalize_nfc(text: str) -> str:
    """Normalize unicode strings to NFC standard."""
    return unicodedata.normalize("NFC", text)


def determine_category(relative_path: Path) -> str:
    """
    Determine the category of a class folder based on its top-level path component.
    Categories:
      - ஆய்தம (ஃ)
      - உயிர் (vowels)
      - உயிர்மெய் (consonants + vowel compounds)
      - மெய் (pure consonants with pulli)
    """
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
    """
    Derive coarse source_group from filename pattern.
    Priority order: WhatsApp, Screenshot, Scanned, resized, PXL, IMG, Copy, numeric.
    """
    fn_lower = filename.lower()
    fn_orig = filename
    if "whatsapp" in fn_lower:
        return "WhatsApp"
    elif "screenshot" in fn_lower:
        return "Screenshot"
    elif "scanned" in fn_lower:
        return "Scanned"
    elif "resized" in fn_lower:
        return "resized"
    elif fn_orig.startswith("PXL_") or "pxl_" in fn_lower:
        return "PXL"
    elif fn_orig.startswith("IMG_") or "img_" in fn_lower or fn_orig.startswith("IMG"):
        return "IMG"
    elif "copy" in fn_lower:
        return "Copy"
    else:
        return "numeric"


def read_image_safe(path: Path) -> Optional[np.ndarray]:
    """
    Read an image safely on Windows with Unicode path support using np.fromfile + cv2.imdecode.
    Falls back to PIL. Returns None if corrupt.
    """
    try:
        # np.fromfile handles non-ASCII paths properly on Windows
        raw_bytes = np.fromfile(str(path), dtype=np.uint8)
        if len(raw_bytes) == 0:
            return None
        img = cv2.imdecode(raw_bytes, cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img
    except Exception:
        pass

    try:
        with Image.open(path) as pil_img:
            pil_img.load()
            return np.array(pil_img)
    except Exception:
        return None


def run_audit(
    dataset_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    phash_threshold: int = 4,
) -> Dict[str, Any]:
    """
    Execute full dataset audit and write reports.
    """
    cfg = get_config()
    if dataset_root is None:
        dataset_root = Path(cfg.paths.dataset_root)
    if output_dir is None:
        output_dir = Path(cfg.paths._reports)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    logger.info(f"Starting audit on dataset root: {dataset_root}")

    # 1. Walk entire directory tree
    # Discover all directories, check for images, zips, empty
    all_dirs: List[Path] = []
    for root, dirs, _ in os.walk(dataset_root):
        all_dirs.append(Path(root))

    class_image_files: Dict[str, List[Path]] = defaultdict(list)
    class_category_map: Dict[str, str] = {}
    class_path_map: Dict[str, Path] = {}
    empty_classes: List[Tuple[str, str, Path]] = []
    zip_classes: Dict[str, List[Path]] = defaultdict(list)
    corrupt_files: List[Path] = []
    format_counts: Counter = Counter()
    source_group_counts: Counter = Counter()
    image_sizes: List[Tuple[int, int]] = []  # (width, height)
    image_hashes: Dict[str, List[Tuple[Path, str]]] = defaultdict(list)  # phash -> [(path, class)]

    # Map leaf directory containing image files directly
    for d in all_dirs:
        # Skip root itself
        if d == dataset_root:
            continue
        rel = d.relative_to(dataset_root)
        cat = determine_category(rel)

        # Check files directly in d
        try:
            entries = list(d.iterdir())
        except Exception as e:
            logger.warning(f"Could not read directory {d}: {e}")
            continue

        images = [p for p in entries if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        zips = [p for p in entries if p.is_file() and p.suffix.lower() == ".zip"]
        subdirs = [p for p in entries if p.is_dir()]

        # Class label is the name of the folder that directly contains image files
        class_label = normalize_nfc(d.name)

        if len(images) > 0:
            class_image_files[class_label].extend(images)
            class_category_map[class_label] = cat
            class_path_map[class_label] = d
        elif len(zips) > 0 and len(subdirs) == 0:
            zip_classes[class_label].extend(zips)
        elif len(subdirs) == 0 and len(images) == 0 and len(zips) == 0:
            # Empty leaf directory
            empty_classes.append((class_label, cat, d))

    logger.info(f"Discovered {len(class_image_files)} active classes with direct images.")
    logger.info(f"Discovered {len(zip_classes)} zip-only classes: {list(zip_classes.keys())}")
    logger.info(f"Discovered {len(empty_classes)} empty leaf directories.")

    # 2. Inspect all images (formats, corruptions, sizes, perceptual hashes, source groups)
    class_stats: Dict[str, Dict[str, Any]] = {}
    all_valid_images_count = 0

    logger.info("Inspecting image files, computing p-hashes and dimensions...")
    for class_name, files in tqdm(class_image_files.items(), desc="Auditing classes"):
        cat = class_category_map[class_name]
        class_formats = Counter()
        class_sources = Counter()
        class_sizes = []
        valid_files = []

        for fpath in files:
            ext = fpath.suffix.lower()
            class_formats[ext] += 1
            format_counts[ext] += 1

            src_grp = determine_source_group(fpath.name)
            class_sources[src_grp] += 1
            source_group_counts[src_grp] += 1

            img = read_image_safe(fpath)
            if img is None:
                corrupt_files.append(fpath)
                continue

            h, w = img.shape[:2]
            class_sizes.append((w, h))
            image_sizes.append((w, h))
            valid_files.append(fpath)

            # Perceptual hash for near-duplicate detection
            try:
                pil_im = Image.fromarray(img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                ph = str(imagehash.phash(pil_im))
                image_hashes[ph].append((fpath, class_name))
            except Exception:
                pass

        all_valid_images_count += len(valid_files)
        widths = [s[0] for s in class_sizes]
        heights = [s[1] for s in class_sizes]

        class_stats[class_name] = {
            "category": cat,
            "total_files": len(files),
            "valid_files": len(valid_files),
            "formats": dict(class_formats),
            "sources": dict(class_sources),
            "min_w": min(widths) if widths else 0,
            "max_w": max(widths) if widths else 0,
            "mean_w": float(np.mean(widths)) if widths else 0.0,
            "min_h": min(heights) if heights else 0,
            "max_h": max(heights) if heights else 0,
            "mean_h": float(np.mean(heights)) if heights else 0.0,
        }

    # 3. Duplicate and near-duplicate analysis
    duplicate_groups: List[List[Tuple[Path, str]]] = []
    for ph, file_list in image_hashes.items():
        if len(file_list) > 1:
            duplicate_groups.append(file_list)

    total_duplicate_files = sum(len(grp) - 1 for grp in duplicate_groups)

    # 4. Check zip details
    zip_summary = {}
    for zcls, zpaths in zip_classes.items():
        zsummary = []
        for zp in zpaths:
            try:
                with zipfile.ZipFile(zp, 'r') as zf:
                    flist = [name for name in zf.namelist() if any(name.lower().endswith(e) for e in IMAGE_EXTENSIONS)]
                    zsummary.append((zp.name, len(flist)))
            except Exception as e:
                zsummary.append((zp.name, f"Error: {e}"))
        zip_summary[zcls] = zsummary

    # 5. Episodic eligibility analysis for various K+Q thresholds
    # e.g. thresholds = [2, 4, 6, 8, 10, 15, 20, 25]
    eligibility = {}
    counts = [stats["valid_files"] for stats in class_stats.values()]
    for threshold in [2, 4, 6, 8, 10, 15, 20, 25]:
        qualifying = [cls for cls, stats in class_stats.items() if stats["valid_files"] >= threshold]
        eligibility[threshold] = {
            "count": len(qualifying),
            "percentage": (len(qualifying) / len(class_stats) * 100) if class_stats else 0,
            "disqualified": [cls for cls, stats in class_stats.items() if stats["valid_files"] < threshold]
        }

    # 6. Generate CSV Report
    csv_path = output_dir / "dataset_audit.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "class_name", "category", "image_count", "valid_count",
            "formats", "source_groups",
            "min_resolution", "max_resolution", "mean_resolution"
        ])
        for cls_name, s in sorted(class_stats.items(), key=lambda x: (x[1]["category"], x[0])):
            formats_str = "; ".join(f"{k}:{v}" for k, v in s["formats"].items())
            sources_str = "; ".join(f"{k}:{v}" for k, v in s["sources"].items())
            writer.writerow([
                cls_name,
                s["category"],
                s["total_files"],
                s["valid_files"],
                formats_str,
                sources_str,
                f"{s['min_w']}x{s['min_h']}",
                f"{s['max_w']}x{s['max_h']}",
                f"{int(s['mean_w'])}x{int(s['mean_h'])}"
            ])

    # 7. Generate Markdown Report
    md_path = output_dir / "dataset_audit.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Tamil Letter Dataset – Comprehensive Audit Report\n\n")
        f.write("## 1. Executive Summary\n\n")
        f.write(f"- **Dataset Root**: `{dataset_root}`\n")
        f.write(f"- **Total Active Image Classes**: {len(class_stats)}\n")
        f.write(f"- **Total Valid Images**: {all_valid_images_count}\n")
        f.write(f"- **Zip-Only Classes**: {len(zip_classes)} ({', '.join(zip_classes.keys())})\n")
        f.write(f"- **Empty Classes (Enrollable Later)**: {len(empty_classes)}\n")
        f.write(f"- **Corrupt / Unreadable Files**: {len(corrupt_files)}\n")
        f.write(f"- **Exact Hash Duplicate Groups**: {len(duplicate_groups)} ({total_duplicate_files} redundant files)\n")
        f.write(f"- **Image Formats**: {dict(format_counts)}\n\n")

        # Category Breakdown
        cat_class_count = Counter(s["category"] for s in class_stats.values())
        cat_img_count = defaultdict(int)
        for s in class_stats.values():
            cat_img_count[s["category"]] += s["valid_files"]

        f.write("## 2. Category Breakdown\n\n")
        f.write("| Category | Class Count | Total Images | Min Images/Class | Max Images/Class | Mean Images/Class |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|\n")
        for cat in ["ஆய்தம", "உயிர்", "உயிர்மெய்", "மெய்"]:
            c_classes = [s for s in class_stats.values() if s["category"] == cat]
            if c_classes:
                img_counts = [s["valid_files"] for s in c_classes]
                f.write(f"| {cat} | {len(c_classes)} | {sum(img_counts)} | {min(img_counts)} | {max(img_counts)} | {np.mean(img_counts):.1f} |\n")
        f.write("\n")

        # Source Groups
        f.write("## 3. Source Group Distribution\n\n")
        f.write("| Source Group | Count | Percentage |\n")
        f.write("|:---|:---:|:---:|\n")
        for src, cnt in source_group_counts.most_common():
            f.write(f"| {src} | {cnt} | {(cnt / all_valid_images_count * 100):.1f}% |\n")
        f.write("\n")

        # Zip Classes
        f.write("## 4. Zip-Only Classes (To Be Extracted to `data/interim/extracted/`)\n\n")
        f.write("| Class | Zip Filename | Internal Images |\n")
        f.write("|:---|:---|:---:|\n")
        for zcls, zinfo in zip_summary.items():
            for zname, zcount in zinfo:
                f.write(f"| {zcls} | `{zname}` | {zcount} |\n")
        f.write("\n")

        # Empty Classes
        f.write("## 5. Empty Classes (Record as Enrollable Later)\n\n")
        f.write("The following folders contain 0 images in the source dataset and will not be trained on directly, but are marked for interactive enrollment:\n\n")
        f.write("| Folder Name | Category | Path |\n")
        f.write("|:---|:---|:---|\n")
        for cname, cat, p in empty_classes:
            f.write(f"| {cname} | {cat} | `{p.relative_to(dataset_root)}` |\n")
        f.write("\n")

        # Episodic Thresholds
        f.write("## 6. Episodic Eligibility by (K + Q) Threshold\n\n")
        f.write("An episode requires at least $K + Q$ samples per sampled class:\n\n")
        f.write("| $K+Q$ Threshold | Eligible Classes | % of Total Classes | Disqualified Classes |\n")
        f.write("|:---:|:---:|:---:|:---|\n")
        for th, el in eligibility.items():
            disq = ", ".join(el["disqualified"]) if el["disqualified"] else "None"
            f.write(f"| >= {th} | {el['count']} / {len(class_stats)} | {el['percentage']:.1f}% | {disq} |\n")
        f.write("\n")

        # Image Size Stats
        if image_sizes:
            widths = [s[0] for s in image_sizes]
            heights = [s[1] for s in image_sizes]
            f.write("## 7. Image Resolution Statistics\n\n")
            f.write(f"- **Width**: Min={min(widths)}px, Max={max(widths)}px, Median={int(np.median(widths))}px, Mean={np.mean(widths):.1f}px\n")
            f.write(f"- **Height**: Min={min(heights)}px, Max={max(heights)}px, Median={int(np.median(heights))}px, Mean={np.mean(heights):.1f}px\n\n")

        # Top Duplicate Groups
        f.write("## 8. Near/Exact Perceptual Duplicate Groups (Samples)\n\n")
        f.write(f"Found {len(duplicate_groups)} duplicate p-hash groups. First 5 groups:\n\n")
        for i, grp in enumerate(duplicate_groups[:5]):
            f.write(f"**Group {i+1}** ({len(grp)} images):\n")
            for fp, cls_name in grp:
                f.write(f"- Class `{cls_name}`: `{fp.name}`\n")
            f.write("\n")

    logger.info(f"Audit completed successfully! Reports written to:\n  - {md_path}\n  - {csv_path}")

    return {
        "class_stats": class_stats,
        "empty_classes": empty_classes,
        "zip_classes": zip_classes,
        "zip_summary": zip_summary,
        "corrupt_files": corrupt_files,
        "format_counts": format_counts,
        "source_group_counts": source_group_counts,
        "duplicate_groups": duplicate_groups,
        "eligibility": eligibility,
        "all_valid_images_count": all_valid_images_count,
    }


if __name__ == "__main__":
    audit_res = run_audit()
    print("\n" + "=" * 60)
    print("TAMIL LETTER DATASET AUDIT SUMMARY")
    print("=" * 60)
    print(f"Total Active Classes (with images) : {len(audit_res['class_stats'])}")
    print(f"Total Valid Images Found           : {audit_res['all_valid_images_count']}")
    print(f"Zip-Only Classes                   : {list(audit_res['zip_classes'].keys())}")
    print(f"Empty Leaf Folders                 : {len(audit_res['empty_classes'])}")
    print(f"Corrupt Files                      : {len(audit_res['corrupt_files'])}")
    print(f"Duplicate Groups (p-hash)          : {len(audit_res['duplicate_groups'])}")
    print("\nSource Group Breakdown:")
    for k, v in audit_res["source_group_counts"].most_common():
        print(f"  {k:15s}: {v}")
    print("\nEpisodic Eligibility:")
    for k, v in audit_res["eligibility"].items():
        print(f"  K+Q >= {k:2d}: {v['count']:2d} classes ({v['percentage']:.1f}%)")
    print("=" * 60)

