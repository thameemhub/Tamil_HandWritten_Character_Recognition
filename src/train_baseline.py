"""
src/train_baseline.py – Training script for the Supervised CNN baseline.

Trains a standard CNN with cross-entropy loss over all training images (Protocol A).
Saves model checkpoint and training curves to models/supervised_cnn_best.pth.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import torch
import torch.optim as optim

from src.augment import TrainAugmentor
from src.baselines import SupervisedCNN, create_dataloader, evaluate_supervised, train_supervised
from src.config import get_config, seed_everything

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("train_baseline")


def run_baseline_training(
    num_epochs: int = 35,
    batch_size: int = 64,
    lr: float = 0.001,
    backbone_name: str = "conv4",
    embedding_dim: int = 64,
    patience: int = 10,
    device_str: Optional[str] = None
) -> Dict[str, Any]:
    cfg = get_config()
    seed_everything(cfg.seed)

    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info(f"Using device: {device} for Supervised CNN baseline")

    processed_dir = Path(cfg.paths._processed)
    manifest_path = processed_dir / "manifest.csv"
    class_names_path = processed_dir / "class_names.json"
    models_dir = Path(cfg.paths._models)
    models_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}. Run prepare_data.py first.")

    manifest_df = pd.read_csv(manifest_path, encoding="utf-8")
    with open(class_names_path, "r", encoding="utf-8") as f:
        class_meta = json.load(f)
    num_classes = class_meta["num_classes"]

    logger.info(f"Training Supervised CNN on {num_classes} classes...")

    augmentor = TrainAugmentor(cfg)
    train_loader = create_dataloader(manifest_df, split="train", batch_size=batch_size, shuffle=True, augmentor=augmentor)
    val_loader = create_dataloader(manifest_df, split="val", batch_size=batch_size, shuffle=False, augmentor=None)
    test_loader = create_dataloader(manifest_df, split="test", batch_size=batch_size, shuffle=False, augmentor=None)

    model = SupervisedCNN(
        num_classes=num_classes,
        in_channels=1,
        backbone_name=backbone_name,
        embedding_dim=embedding_dim
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    save_path = models_dir / "supervised_cnn_best.pth"

    history = train_supervised(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=num_epochs,
        patience=patience,
        save_path=save_path
    )

    # Evaluate on test set
    if save_path.exists():
        ckpt = torch.load(save_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])

    test_results = evaluate_supervised(model, test_loader, device=device)
    logger.info(f"Supervised CNN Test Accuracy (All {num_classes} classes): {test_results['accuracy'] * 100:.2f}%")

    # Save metrics
    results_file = models_dir / "supervised_cnn_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({
            "test_accuracy": test_results["accuracy"],
            "history": history
        }, f, indent=2)

    return test_results


def main():
    parser = argparse.ArgumentParser(description="Train Supervised CNN Baseline")
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--backbone", type=str, default="conv4", help="Backbone name")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    args = parser.parse_args()

    run_baseline_training(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        backbone_name=args.backbone,
        patience=args.patience
    )


if __name__ == "__main__":
    main()

