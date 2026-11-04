"""
Run test on C:/Tamil_Dataset/test.png, generate output, visual artifacts,
and store in folder 'testing'.
"""

import json
import shutil
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.config import get_config
from src.support_registry import SupportRegistry
from src.preprocessing import preprocess_image
from src.tamil_text import configure_matplotlib_tamil


def main():
    cfg = get_config()
    registry = SupportRegistry(cfg=cfg)

    input_path = Path(r"C:\Tamil_Dataset\test.png")
    out_dir = Path(r"C:\Tamil_Dataset\testing")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_dir_internal = Path(r"C:\Tamil_Dataset\tamil-fewshot-htr\testing")
    out_dir_internal.mkdir(parents=True, exist_ok=True)

    # 1. Copy original
    shutil.copy2(input_path, out_dir / "original_test.png")
    shutil.copy2(input_path, out_dir_internal / "original_test.png")

    # 2. Preprocess
    raw_bgr = cv2.imread(str(input_path))
    prep = preprocess_image(input_path)  # returns [64, 64] float32 in [0, 1]
    prep_uint8 = (prep * 255).astype(np.uint8)
    cv2.imwrite(str(out_dir / "preprocessed_64x64.png"), prep_uint8)
    cv2.imwrite(str(out_dir_internal / "preprocessed_64x64.png"), prep_uint8)

    # 3. Predict Top 5
    results = registry.predict(prep, top_k=5)

    with open(r"C:\Tamil_Dataset\tamil-fewshot-htr\data\processed\class_names.json", "r", encoding="utf-8") as f:
        class_meta = json.load(f)
    category_map = class_meta.get("class_categories", {})

    for i, r in enumerate(results, 1):
        r["rank"] = i
        r["category"] = category_map.get(r["class"], "உயிர்மெய்")

    # 4. JSON result
    is_rejected = results[0].get("rejected", False)
    output_data = {
        "input_image": str(input_path),
        "status": "RECOGNIZED" if not is_rejected else "UNKNOWN / NOT ENROLLED",
        "top_prediction": results[0]["class"],
        "confidence": float(results[0]["confidence"]),
        "distance": float(results[0]["distance"]),
        "category": results[0]["category"],
        "top_k_candidates": results
    }

    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    with open(out_dir_internal / "result.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    # 5. Text report
    lines = [
        "=================================================================",
        "TAMIL CHARACTER RECOGNITION REPORT - FEW-SHOT LEARNING",
        "=================================================================",
        f"Input Image  : {input_path}",
        f"Image Size   : {raw_bgr.shape[1]}x{raw_bgr.shape[0]} (Channels: {raw_bgr.shape[2]})",
        f"Result Status: {output_data['status']}",
        f"Top Class    : {output_data['top_prediction']}",
        f"Category     : {output_data['category']}",
        f"Confidence   : {output_data['confidence']*100:.2f}%",
        f"Distance     : {output_data['distance']:.4f}",
        "-----------------------------------------------------------------",
        "TOP-5 CANDIDATES:",
        "Rank | Character | Category              | Confidence | Distance",
        "-----+-----------+-----------------------+------------+---------"
    ]

    for r in results:
        lines.append(
            f"{r['rank']:4d} | {r['class']:9s} | {r['category']:21s} | {r['confidence']*100:6.2f}%     | {r['distance']:8.4f}"
        )
    lines.append("=================================================================")

    report_text = "\n".join(lines) + "\n"
    with open(out_dir / "result.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    with open(out_dir_internal / "result.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    # 6. Annotated Visualization
    configure_matplotlib_tamil()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Original
    axes[0].imshow(cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Original Input ({raw_bgr.shape[1]}x{raw_bgr.shape[0]})", fontsize=12)
    axes[0].axis("off")

    # Preprocessed
    axes[1].imshow(prep_uint8, cmap="gray")
    axes[1].set_title("Normalized 64x64 Input", fontsize=12)
    axes[1].axis("off")

    # Predictions Bar Chart
    chars = [r["class"] for r in reversed(results)]
    confs = [r["confidence"] * 100 for r in reversed(results)]
    y_pos = np.arange(len(chars))

    colors = ["#4C72B0" if i < len(chars)-1 else "#2CA02C" for i in range(len(chars))]
    bars = axes[2].barh(y_pos, confs, color=colors)
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels(chars, fontsize=18)
    axes[2].set_xlabel("Confidence (%)", fontsize=12)
    axes[2].set_xlim(0, max(confs) * 1.35)
    axes[2].set_title(f"Predicted: '{results[0]['class']}' ({results[0]['confidence']*100:.1f}%)", fontsize=14, fontweight="bold")

    for bar, conf in zip(bars, confs):
        axes[2].text(
            bar.get_width() + 0.6,
            bar.get_y() + bar.get_height() / 2,
            f"{conf:.1f}%",
            va="center",
            ha="left",
            fontsize=12,
            fontweight="bold"
        )

    plt.tight_layout()
    plt.savefig(str(out_dir / "result_visualization.png"), dpi=200)
    plt.savefig(str(out_dir_internal / "result_visualization.png"), dpi=200)
    plt.close()

    print(report_text)
    print(f"Results stored in:\n  - {out_dir}\n  - {out_dir_internal}")


if __name__ == "__main__":
    main()
