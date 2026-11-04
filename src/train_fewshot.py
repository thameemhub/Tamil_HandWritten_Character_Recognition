"""
src/train_fewshot.py – Episodic training script for ProtoNet, Siamese, and MatchingNet.

Supports:
  - Methods: protonet (primary), siamese, matching
  - Protocols:
      Protocol A (known-class few-shot): train on train pool of all classes, validate on val pool
      Protocol B (novel-class few-shot): train ONLY on base classes, validate on val classes
  - N-way K-shot Q-query episodic sampling
  - On-the-fly augmentation during train episodes
  - Cosine / Step learning rate decay and early stopping on validation episode accuracy
  - Checkpoint saving to models/<method>_<protocol>_best.pth
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm

from src.augment import TrainAugmentor
from src.backbones import get_backbone
from src.config import get_config, seed_everything
from src.episodes import create_episode_loader
from src.fewshot.matchingnet import MatchingNetwork, train_matching_epoch, evaluate_matching
from src.fewshot.protonet import PrototypicalNetwork, evaluate_episodes, train_one_epoch
from src.fewshot.siamese import (
    SiameseNetwork,
    evaluate_siamese_fewshot,
    train_siamese_epoch,
    triplet_loss,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("train_fewshot")


def run_training(
    method: str = "protonet",
    protocol: str = "a",
    n_way: int = 10,
    k_shot: int = 1,
    q_query: int = 5,
    num_episodes: int = 100,
    num_epochs: int = 25,
    lr: float = 0.001,
    backbone_name: str = "conv4",
    embedding_dim: int = 64,
    device_str: Optional[str] = None,
    save_path: Optional[Path] = None,
    patience: int = 10
) -> Dict[str, Any]:
    cfg = get_config()
    seed_everything(cfg.seed)

    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info(f"Using device: {device} for {method.upper()} on Protocol {protocol.upper()}")

    # Paths
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
    class_to_idx = class_meta["class_to_idx"]

    # Filter data according to protocol
    if protocol.lower() == "a":
        # Protocol A: All classes seen. Train pool vs Val pool
        train_df = manifest_df[manifest_df["split_protocol_a"] == "train"].reset_index(drop=True)
        val_df = manifest_df[manifest_df["split_protocol_a"] == "val"].reset_index(drop=True)
    else:
        # Protocol B: Class-level split. Train on base classes only, val on val classes
        train_df = manifest_df[manifest_df["split_protocol_b"] == "base"].reset_index(drop=True)
        val_df = manifest_df[manifest_df["split_protocol_b"] == "val"].reset_index(drop=True)

    logger.info(f"Protocol {protocol.upper()} Split: Train set has {len(train_df)} images ({train_df['class_name'].nunique()} classes), "
                f"Val set has {len(val_df)} images ({val_df['class_name'].nunique()} classes)")

    # Adjust n_way if fewer classes exist in train/val sets
    max_train_classes = train_df["class_name"].nunique()
    max_val_classes = val_df["class_name"].nunique()
    train_way = min(n_way, max_train_classes)
    val_way = min(n_way, max_val_classes)

    logger.info(f"Configuring episodes: Train {train_way}-way {k_shot}-shot {q_query}-query | Val {val_way}-way {k_shot}-shot")

    # Augmentor for training
    augmentor = TrainAugmentor(cfg)

    # Episode Loaders
    train_loader = create_episode_loader(
        manifest_df=train_df,
        class_to_idx=class_to_idx,
        processed_dir=processed_dir,
        n_way=train_way,
        k_shot=k_shot,
        q_query=q_query,
        num_episodes=num_episodes,
        augmentor=augmentor
    )

    val_loader = create_episode_loader(
        manifest_df=val_df,
        class_to_idx=class_to_idx,
        processed_dir=processed_dir,
        n_way=val_way,
        k_shot=k_shot,
        q_query=q_query,
        num_episodes=min(100, num_episodes),
        augmentor=None
    )

    # Initialize Model Backbone
    backbone = get_backbone(backbone_name, in_channels=1, embedding_dim=embedding_dim)

    if method.lower() == "protonet":
        model = PrototypicalNetwork(backbone=backbone, learnable_temp=True).to(device)
    elif method.lower() == "siamese":
        model = SiameseNetwork(backbone=backbone).to(device)
    elif method.lower() == "matching":
        model = MatchingNetwork(backbone=backbone).to(device)
    else:
        raise ValueError(f"Unknown fewshot method: {method}")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    if save_path is None:
        save_path = models_dir / f"{method}_{protocol}_best.pth"

    best_val_acc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_acc": []}

    logger.info(f"Starting training for {num_epochs} epochs...")

    for epoch in range(1, num_epochs + 1):
        if method.lower() == "protonet":
            train_metrics = train_one_epoch(
                model=model,
                episode_loader=train_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                n_way=train_way,
                k_shot=k_shot
            )
            val_metrics = evaluate_episodes(
                model=model,
                episode_loader=val_loader,
                device=device,
                n_way=val_way,
                k_shot=k_shot,
                num_episodes=min(100, num_episodes)
            )
        elif method.lower() == "matching":
            train_metrics = train_matching_epoch(
                model=model,
                episode_loader=train_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                n_way=train_way,
                k_shot=k_shot
            )
            val_metrics = evaluate_matching(
                model=model,
                episode_loader=val_loader,
                device=device,
                n_way=val_way,
                k_shot=k_shot,
                num_episodes=min(100, num_episodes)
            )
        elif method.lower() == "siamese":
            train_metrics = train_siamese_epoch(
                model=model,
                episode_loader=train_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                margin=1.0
            )
            val_metrics = evaluate_siamese_fewshot(
                model=model,
                episode_loader=val_loader,
                device=device,
                n_way=val_way,
                k_shot=k_shot,
                num_episodes=min(100, num_episodes)
            )

        scheduler.step()

        t_acc = train_metrics.get("accuracy", 0.0)
        t_loss = train_metrics.get("loss", 0.0)
        v_acc = val_metrics["accuracy"]

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)

        logger.info(f"Epoch {epoch:02d}/{num_epochs:02d} | Train Loss: {t_loss:.4f} Acc: {t_acc*100:.1f}% | Val Acc: {v_acc*100:.2f}% (CI95: ±{val_metrics.get('ci95', 0)*100:.2f}%)")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "backbone_state": backbone.state_dict(),
                "method": method,
                "protocol": protocol,
                "val_acc": v_acc,
                "epoch": epoch,
                "config": {
                    "backbone": backbone_name,
                    "embedding_dim": embedding_dim,
                    "way": n_way,
                    "shot": k_shot
                }
            }, save_path)
            logger.info(f"--> Saved best model checkpoint to {save_path} (Val Acc: {best_val_acc*100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

    # Save training history
    history_file = models_dir / f"{method}_{protocol}_history.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    logger.info(f"Training completed. Best Val Acc: {best_val_acc*100:.2f}%")
    return {"best_val_acc": best_val_acc, "history": history, "save_path": save_path}


def main():
    parser = argparse.ArgumentParser(description="Episodic Training for Few-Shot Tamil HTR")
    parser.add_argument("--method", type=str, default="protonet", choices=["protonet", "siamese", "matching"], help="Few-shot learning method")
    parser.add_argument("--protocol", type=str, default="a", choices=["a", "b"], help="Evaluation protocol: 'a' (known-class) or 'b' (novel-class)")
    parser.add_argument("--way", type=int, default=10, help="Number of ways per episode")
    parser.add_argument("--shot", type=int, default=1, help="Number of support shots per class")
    parser.add_argument("--query", type=int, default=5, help="Number of query samples per class")
    parser.add_argument("--episodes", type=int, default=100, help="Episodes per epoch")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--backbone", type=str, default="conv4", choices=["conv4", "resnet12"], help="Backbone architecture")
    parser.add_argument("--embedding_dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu' or 'cuda')")
    parser.add_argument("--save_path", type=str, default=None, help="Custom checkpoint save path")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")

    args = parser.parse_args()

    run_training(
        method=args.method,
        protocol=args.protocol,
        n_way=args.way,
        k_shot=args.shot,
        q_query=args.query,
        num_episodes=args.episodes,
        num_epochs=args.epochs,
        lr=args.lr,
        backbone_name=args.backbone,
        embedding_dim=args.embedding_dim,
        device_str=args.device,
        save_path=Path(args.save_path) if args.save_path else None,
        patience=args.patience
    )


if __name__ == "__main__":
    main()

