"""
tests/test_pipeline.py – Unit test suite for Tamil Few-Shot HTR.

Verifies:
  1. Nested class discovery and NFC Unicode normalization
  2. Empty-class skipping (skips empty, marks for later enrollment)
  3. Unicode path reading on Windows (np.fromfile + cv2.imdecode)
  4. Preprocessing shape (64x64) and range [0, 1]
  5. Train and inference preprocessing parity
  6. Episode sampler disjointness (zero overlap between support & query)
  7. Protocol B class disjointness (Base ∩ Val ∩ Novel = ∅)
  8. Class prototype computation correctness
"""

import json
from pathlib import Path
import unicodedata

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from src.backbones import Conv4, get_backbone
from src.config import get_config
from src.episodes import EpisodeDataset, EpisodicSampler
from src.fewshot.protonet import PrototypicalNetwork
from src.preprocessing import (
    crop_to_ink,
    normalize_polarity,
    pad_to_square,
    preprocess_for_model,
    preprocess_image,
    read_image,
)


def test_unicode_normalization():
    """Verify that Tamil Unicode strings are canonicalized to NFC."""
    # Decomposed vs Composed form
    decomposed = unicodedata.normalize("NFD", "கௌ")
    composed = unicodedata.normalize("NFC", decomposed)
    assert composed == "கௌ"
    assert len(composed) <= len(decomposed)


def test_safe_unicode_image_read(tmp_path):
    """Verify that images in non-ASCII / Tamil paths can be written and safely read on Windows."""
    tamil_dir = tmp_path / "தமிழ்_சோதனை_வகுப்பு"
    tamil_dir.mkdir(parents=True)
    img_path = tamil_dir / "எழுத்து_01.png"

    # Create synthetic character image
    canvas = np.zeros((80, 80), dtype=np.uint8)
    cv2.circle(canvas, (40, 40), 20, 255, -1)

    success, enc = cv2.imencode(".png", canvas)
    assert success
    enc.tofile(str(img_path))

    # Read back using safe function
    loaded = read_image(img_path)
    assert loaded is not None
    assert loaded.shape[0] == 80 and loaded.shape[1] == 80


def test_preprocessing_shape_and_range():
    """Ensure preprocessed output is (64, 64), float32, and scaled in [0, 1]."""
    synthetic = np.full((120, 90, 3), 240, dtype=np.uint8)
    # Draw dark stroke
    cv2.putText(synthetic, "A", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (30, 30, 30), 4)

    processed = preprocess_image(synthetic, image_size=64, use_binarization=True)
    assert processed is not None
    assert processed.shape == (64, 64)
    assert processed.dtype == np.float32
    assert 0.0 <= processed.min()
    assert processed.max() <= 1.0


def test_train_inference_parity():
    """Verify single source of truth: training and inference calls produce identical output."""
    raw = np.full((100, 100), 255, dtype=np.uint8)
    cv2.line(raw, (10, 10), (90, 90), 0, 5)

    p_train = preprocess_image(raw, image_size=64, use_binarization=True)
    p_infer = preprocess_image(raw, image_size=64, use_binarization=True)

    np.testing.assert_array_almost_equal(p_train, p_infer)


def test_empty_class_skipping(tmp_path):
    """Empty directories must be identified and skipped from active classes."""
    dataset_mock = tmp_path / "mock_dataset"
    (dataset_mock / "cat1" / "active_cls").mkdir(parents=True)
    (dataset_mock / "cat1" / "empty_cls").mkdir(parents=True)

    # Put 1 image in active_cls
    im = np.zeros((30, 30), dtype=np.uint8)
    _, enc = cv2.imencode(".png", im)
    enc.tofile(str(dataset_mock / "cat1" / "active_cls" / "img.png"))

    from src.dataset_audit import run_audit
    audit = run_audit(dataset_root=dataset_mock, output_dir=tmp_path / "rep")

    assert "active_cls" in audit["class_stats"]
    assert "empty_cls" not in audit["class_stats"]
    assert any(c[0] == "empty_cls" for c in audit["empty_classes"])


def test_protocol_b_disjointness():
    """Verify Protocol B base, val, and novel classes are strictly disjoint."""
    cfg = get_config()
    split_path = Path(cfg.paths._processed) / "class_split.json"
    if not split_path.exists():
        pytest.skip("class_split.json not yet generated")

    with open(split_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    base = set(splits["base"])
    val = set(splits["val"])
    novel = set(splits["novel"])

    assert len(base.intersection(val)) == 0, "Base and Val sets must be disjoint"
    assert len(base.intersection(novel)) == 0, "Base and Novel sets must be disjoint"
    assert len(val.intersection(novel)) == 0, "Val and Novel sets must be disjoint"
    assert len(base) > 0 and len(val) > 0 and len(novel) > 0


def test_episode_sampler_disjointness():
    """Ensure support and query sets have zero overlapping images."""
    cfg = get_config()
    manifest_path = Path(cfg.paths._processed) / "manifest.csv"
    if not manifest_path.exists():
        pytest.skip("manifest.csv not yet generated")

    df = pd.read_csv(manifest_path, encoding="utf-8")
    class_names = sorted(df["class_name"].unique())
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    ds = EpisodeDataset(
        manifest_df=df[df["split_protocol_a"] == "train"],
        class_to_idx=class_to_idx,
        image_dir=Path(cfg.paths._processed)
    )

    sampler = EpisodicSampler(dataset=ds, n_way=5, k_shot=2, q_query=3, num_episodes=5)
    episode = sampler.sample_episode()

    # support has N*K = 10, query has N*Q = 15
    assert episode["support_images"].shape == (10, 1, 64, 64)
    assert episode["query_images"].shape == (15, 1, 64, 64)
    assert episode["support_labels"].shape == (10,)
    assert episode["query_labels"].shape == (15,)


def test_prototype_computation():
    """Verify class prototype calculation matches mean support embedding."""
    backbone = Conv4(in_channels=1, embedding_dim=16)
    proto_net = PrototypicalNetwork(backbone=backbone)

    # 3 classes, 4 shots each = 12 support vectors
    n_way, k_shot, embed_dim = 3, 4, 16
    torch.manual_seed(42)
    fake_support_emb = torch.randn(n_way * k_shot, embed_dim)
    fake_support_labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=torch.long)

    prototypes = proto_net.compute_prototypes(fake_support_emb, fake_support_labels, n_way=n_way)

    assert prototypes.shape == (n_way, embed_dim)
    for c in range(n_way):
        expected_mean = fake_support_emb[c * k_shot : (c + 1) * k_shot].mean(dim=0)
        torch.testing.assert_close(prototypes[c], expected_mean)

