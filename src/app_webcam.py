"""
src/app_webcam.py – Real-Time Webcam Tamil Character Recognition.

Features:
  - Video stream capture with automatic character ROI detection using contours
  - Extracts bounding box, crops, binarizes, and normalizes polarity
  - Fast nearest-prototype recognition via SupportRegistry
  - Pillow-based Unicode Tamil text overlay onto OpenCV video frames
  - Graceful fallback and clear diagnostic error if camera is unavailable
  - Keyboard shortcuts:
      'q' : Quit
      'c' : Snapshot & detailed Top-5 prediction
      'r' : Reset / clear detection
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from src.config import get_config
from src.preprocessing import preprocess_image
from src.support_registry import SupportRegistry
from src.tamil_text import find_tamil_font, put_tamil_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("app_webcam")


def find_character_roi(frame: np.ndarray, min_area: int = 600) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect character contour on white paper or dark surface.
    Returns bounding box (x, y, w, h) of largest qualifying contour or None.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold to isolate ink strokes
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 6
    )

    # Find external contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    valid_boxes = []
    h_f, w_f = frame.shape[:2]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > (h_f * w_f * 0.75):
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter extreme thin lines
        aspect = w / float(h)
        if 0.15 <= aspect <= 6.0:
            valid_boxes.append((area, (x, y, w, h)))

    if not valid_boxes:
        return None

    # Pick largest contour
    valid_boxes.sort(key=lambda item: item[0], reverse=True)
    return valid_boxes[0][1]


def run_webcam(
    camera_index: int = 0,
    model_path: Optional[str] = None,
    prototypes_path: Optional[str] = None,
    min_contour_area: int = 600,
    threshold: Optional[float] = None
):
    cfg = get_config()
    registry = SupportRegistry(model_path=model_path, cfg=cfg)

    if prototypes_path:
        registry.load_prototypes(Path(prototypes_path))
    if not registry.prototypes:
        logger.info("Building prototypes from support set...")
        registry.build_prototypes()

    if threshold is not None:
        registry.rejection_threshold = threshold

    logger.info(f"Opening camera index {camera_index}...")
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("\n" + "=" * 60)
        print(f"ERROR: Could not open camera (index {camera_index}).")
        print("Please verify your webcam is connected and not in use by another app.")
        print("You can test single image prediction instead with:")
        print("  python -m src.predict --image <path_to_image>")
        print("=" * 60 + "\n")
        return

    logger.info("Webcam started successfully! Press 'q' to exit, 'c' to snapshot.")

    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to grab camera frame.")
            break

        # Calculate FPS
        cur_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(cur_time - prev_time, 1e-5))
        prev_time = cur_time

        display_frame = frame.copy()
        h_f, w_f = frame.shape[:2]

        # Detect character ROI
        roi_box = find_character_roi(frame, min_area=min_contour_area)

        pred_text = "Searching..."
        status_color = (200, 200, 200)

        if roi_box is not None:
            x, y, w, h = roi_box
            # Draw ROI bounding box
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Crop with small margin
            pad = int(min(w, h) * 0.1)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w_f, x + w + pad)
            y2 = min(h_f, y + h + pad)

            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                preprocessed = preprocess_image(crop, image_size=cfg.preprocessing.image_size)
                if preprocessed is not None:
                    res = registry.predict(preprocessed, top_k=3)
                    if res:
                        top = res[0]
                        if top["rejected"]:
                            pred_text = "Unknown / Not Enrolled"
                            status_color = (0, 0, 255)
                        else:
                            pred_text = f"{top['class']} ({top['confidence']*100:.0f}%)"
                            status_color = (0, 255, 0)

        # Overlay Info Panel at top
        cv2.rectangle(display_frame, (0, 0), (w_f, 55), (30, 30, 30), -1)

        # Draw Tamil text onto frame using Pillow
        display_frame = put_tamil_text(
            display_frame,
            text=f"Letter: {pred_text}  |  FPS: {fps:.1f}",
            position=(15, 12),
            font_size=22,
            color=status_color
        )

        cv2.imshow("Tamil Few-Shot Recognition (Webcam)", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            print(f"\n[Snapshot] Detection: {pred_text}")

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Webcam window closed.")


def main():
    parser = argparse.ArgumentParser(description="Webcam Tamil Character Recognition")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index")
    parser.add_argument("--model", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--prototypes", type=str, default=None, help="Prototypes npz path")
    parser.add_argument("--min_area", type=int, default=600, help="Min contour area for detection")
    parser.add_argument("--threshold", type=float, default=None, help="Rejection distance threshold")

    args = parser.parse_args()
    run_webcam(
        camera_index=args.camera,
        model_path=args.model,
        prototypes_path=args.prototypes,
        min_contour_area=args.min_area,
        threshold=args.threshold
    )


if __name__ == "__main__":
    main()

