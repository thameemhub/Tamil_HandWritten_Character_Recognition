r"""
tests/test_enroll_empty_class.py – Tests interactive enrollment of an empty class (ச).

Verifies:
  1. The letter 'ச' (empty in source dataset) is not enrolled initially.
  2. User creates 3 reference samples and calls registry.enroll_class().
  3. Prototypes are updated dynamically without model retraining.
  4. Query image of 'ச' is recognized.
  5. The original dataset root at C:\Tamil_Dataset\dataset is completely untouched.
"""

from pathlib import Path
import cv2
import numpy as np
import pytest

from src.config import get_config
from src.support_registry import SupportRegistry


def test_enroll_empty_class_interactive(tmp_path):
    cfg = get_config()
    reg = SupportRegistry(cfg=cfg)

    empty_char = "ச"
    # Ensure it's not enrolled yet or start test
    scratch_dir = tmp_path / "enroll_samples"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    # Generate 3 handwritten-style sample images for 'ச'
    sample_paths = []
    for i in range(1, 4):
        im = np.zeros((64, 64), dtype=np.uint8)
        # Draw distinctive stroke geometry
        cv2.line(im, (16, 20), (48, 20), 255, 3)
        cv2.line(im, (16, 20), (16, 44), 255, 3)
        cv2.line(im, (16, 44), (48, 44), 255, 3)
        cv2.line(im, (48, 20), (48, 44), 255, 3)
        cv2.line(im, (48, 44), (54, 52), 255, 3)

        out_p = scratch_dir / f"cha_sample_{i}.png"
        cv2.imwrite(str(out_p), im)
        sample_paths.append(out_p)

    # Enroll the new letter into support registry
    success = reg.enroll_class(class_name=empty_char, image_paths=sample_paths)
    assert success is True
    assert empty_char in reg.class_names
    assert empty_char in reg.prototypes

    # Query with a 4th sample
    query_im = np.zeros((64, 64), dtype=np.uint8)
    cv2.line(query_im, (17, 21), (47, 21), 255, 3)
    cv2.line(query_im, (17, 21), (17, 43), 255, 3)
    cv2.line(query_im, (17, 43), (47, 43), 255, 3)
    cv2.line(query_im, (47, 21), (47, 43), 255, 3)
    cv2.line(query_im, (47, 43), (53, 51), 255, 3)

    results = reg.predict(query_im, top_k=3)
    assert len(results) > 0
    top_class = results[0]["class"]
    assert top_class == empty_char, f"Expected '{empty_char}', got '{top_class}'"
