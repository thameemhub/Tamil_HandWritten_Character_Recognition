"""
src/baselines.py – Baseline models for comparison with few-shot methods.

Provides:
  1. SupervisedCNN: Standard CNN with classification head trained on all images (Protocol A).
  2. PrototypeBaseline: Nearest prototype classification using raw/untrained or frozen backbone embeddings.
  3. FineTuneBaseline: Fine-tuning last layer on support set of novel classes (Protocol B).
  4. SupervisedDataset & DataLoader helpers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.backbones import get_backbone
from src.config import get_config

logger = logging.getLogger("baselines")


class SupervisedCNN(nn.Module):
    """
    Supervised CNN baseline that shares the exact same backbone architecture
    as few-shot models, followed by a linear classification head.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        backbone_name: str = "conv4",
        embedding_dim: int = 64
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.backbone = get_backbone(backbone_name, in_channels=in_channels, embedding_dim=embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits (B, num_classes)."""
        emb = self.backbone(x)
        logits = self.classifier(emb)
        return logits

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings from backbone."""
        return self.backbone(x)


class PrototypeBaseline:
    """
    Non-episodic nearest-prototype baseline.
    Computes class prototypes from support images and classifies query images
    by minimum Euclidean distance.
    """

    def __init__(self, backbone: nn.Module, device: str = "cpu"):
        self.backbone = backbone.to(device)
        self.device = device
        self.backbone.eval()

    def predict(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        n_way: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Classify queries by nearest mean support embedding."""
        with torch.no_grad():
            s_emb = self.backbone(support_images.to(self.device))  # (N*K, D)
            q_emb = self.backbone(query_images.to(self.device))    # (Q, D)

            # Compute prototypes
            prototypes = torch.zeros(n_way, s_emb.shape[-1], device=self.device)
            for c in range(n_way):
                mask = (support_labels.to(self.device) == c)
                if mask.sum() > 0:
                    prototypes[c] = s_emb[mask].mean(dim=0)

            # Distance from each query to each prototype: (Q, n_way)
            dists = torch.cdist(q_emb, prototypes)  # Euclidean distance
            confidences, preds = torch.min(dists, dim=-1)
            # Convert distances to softmax confidence
            probs = F.softmax(-dists, dim=-1)
            confidences, preds = torch.max(probs, dim=-1)

        return preds, confidences


class FineTuneBaseline:
    """
    Protocol B baseline: Fine-tunes a linear classification head on top
    of the frozen backbone using support images for novel classes.
    """

    def __init__(self, backbone: nn.Module, n_way: int, embedding_dim: int = 64, device: str = "cpu"):
        self.backbone = backbone.to(device)
        self.device = device
        self.n_way = n_way
        self.embedding_dim = embedding_dim
        self.head = nn.Linear(embedding_dim, n_way).to(device)

    def fine_tune_on_support(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        lr: float = 0.01,
        num_steps: int = 40
    ):
        """Fine-tune the linear head on support examples."""
        self.backbone.eval()
        self.head.train()

        with torch.no_grad():
            s_emb = self.backbone(support_images.to(self.device))

        optimizer = torch.optim.Adam(self.head.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        labels = support_labels.to(self.device)

        for _ in range(num_steps):
            optimizer.zero_grad()
            logits = self.head(s_emb)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

    def predict(self, query_images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.backbone.eval()
        self.head.eval()
        with torch.no_grad():
            q_emb = self.backbone(query_images.to(self.device))
            logits = self.head(q_emb)
            probs = F.softmax(logits, dim=-1)
            confidences, preds = torch.max(probs, dim=-1)
        return preds, confidences


class SupervisedDataset(Dataset):
    """PyTorch Dataset for standard supervised training using preprocessed images."""

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        split: Optional[str] = None,
        split_col: str = "split_protocol_a",
        augmentor: Any = None
    ):
        if split is not None:
            self.df = manifest_df[manifest_df[split_col] == split].reset_index(drop=True)
        else:
            self.df = manifest_df.reset_index(drop=True)
        self.augmentor = augmentor

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_path = row["processed_path"]

        # Read preprocessed 8-bit png
        # Use np.fromfile for safe non-ASCII path handling
        raw = np.fromfile(str(img_path), dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback black canvas
            img = np.zeros((64, 64), dtype=np.uint8)

        img_float = img.astype(np.float32) / 255.0

        if self.augmentor is not None:
            img_float = self.augmentor(img_float)

        tensor = torch.from_numpy(img_float).unsqueeze(0)  # (1, H, W)
        label = int(row["class_idx"])
        return tensor, label


def create_dataloader(
    manifest_df: pd.DataFrame,
    split: str,
    split_col: str = "split_protocol_a",
    batch_size: int = 64,
    shuffle: bool = True,
    augmentor: Any = None
) -> DataLoader:
    dataset = SupervisedDataset(manifest_df, split=split, split_col=split_col, augmentor=augmentor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_supervised(
    model: SupervisedCNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    num_epochs: int = 40,
    patience: int = 10,
    save_path: Optional[Path] = None
) -> Dict[str, Any]:
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, num_epochs + 1):
        # Train
        model.train()
        t_losses, t_correct, t_total = [], 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            t_losses.append(loss.item())
            preds = torch.argmax(logits, dim=-1)
            t_correct += (preds == y).sum().item()
            t_total += y.size(0)

        train_loss = float(np.mean(t_losses))
        train_acc = t_correct / t_total if t_total > 0 else 0.0

        # Validate
        model.eval()
        v_losses, v_correct, v_total = [], 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                v_losses.append(loss.item())
                preds = torch.argmax(logits, dim=-1)
                v_correct += (preds == y).sum().item()
                v_total += y.size(0)

        val_loss = float(np.mean(v_losses)) if v_losses else 0.0
        val_acc = v_correct / v_total if v_total > 0 else 0.0

        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        logger.info(f"Epoch {epoch:02d}/{num_epochs:02d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            if save_path:
                torch.save({
                    "model_state": model.state_dict(),
                    "backbone_state": model.backbone.state_dict(),
                    "val_acc": val_acc,
                    "epoch": epoch
                }, save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

    return history


def evaluate_supervised(
    model: SupervisedCNN,
    test_loader: DataLoader,
    device: torch.device
) -> Dict[str, Any]:
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(y.cpu().numpy().tolist())

    all_preds_arr = np.array(all_preds)
    all_targets_arr = np.array(all_targets)
    acc = float((all_preds_arr == all_targets_arr).mean()) if len(all_preds_arr) > 0 else 0.0

    return {
        "accuracy": acc,
        "predictions": all_preds,
        "targets": all_targets
    }

