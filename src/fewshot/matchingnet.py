"""
src/fewshot/matchingnet.py – Matching Network with Attention for Few-Shot Learning.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger("matchingnet")


class MatchingNetwork(nn.Module):
    """
    Matching Network with Cosine-Attention over Support Set.
    Computes an attention-weighted sum of one-hot support labels to form query probabilities.
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        n_way: int,
        k_shot: int
    ) -> torch.Tensor:
        """
        Args:
            support_images: (n_support, 1, H, W)
            support_labels: (n_support,) with integer values in [0, n_way - 1]
            query_images: (n_query, 1, H, W)
            n_way: number of classes
            k_shot: number of samples per class
        Returns:
            log_probabilities: (n_query, n_way)
        """
        # Embed representations
        support_emb = self.backbone(support_images)  # (n_support, D), already L2-normalized
        query_emb = self.backbone(query_images)      # (n_query, D), already L2-normalized

        # Cosine similarity between each query and all support embeddings: (n_query, n_support)
        # Since embeddings are L2 normalized, dot product = cosine similarity
        similarities = torch.mm(query_emb, support_emb.t())  # (n_query, n_support)
        attention_weights = F.softmax(similarities, dim=-1)   # (n_query, n_support)

        # One-hot encode support labels: (n_support, n_way)
        support_one_hot = F.one_hot(support_labels, num_classes=n_way).float()

        # Class probabilities for query: (n_query, n_way)
        probs = torch.mm(attention_weights, support_one_hot)
        # Add epsilon for numerical stability before log
        log_probs = torch.log(probs + 1e-8)
        return log_probs

    def predict(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        n_way: int,
        k_shot: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (predicted_labels, confidences)."""
        log_probs = self.forward(support_images, support_labels, query_images, n_way, k_shot)
        probs = torch.exp(log_probs)
        confidences, preds = torch.max(probs, dim=-1)
        return preds, confidences


def matching_loss(log_probs: torch.Tensor, query_labels: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Negative log-likelihood loss and accuracy."""
    loss = F.nll_loss(log_probs, query_labels)
    with torch.no_grad():
        preds = torch.argmax(log_probs, dim=-1)
        acc = (preds == query_labels).float().mean().item()
    return loss, acc


def train_matching_epoch(
    model: MatchingNetwork,
    episode_loader: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    n_way: int,
    k_shot: int
) -> Dict[str, float]:
    model.train()
    losses = []
    accuracies = []

    for ep in episode_loader:
        sup_x = ep["support_images"].to(device)
        sup_y = ep["support_labels"].to(device)
        qry_x = ep["query_images"].to(device)
        qry_y = ep["query_labels"].to(device)

        optimizer.zero_grad()
        log_probs = model(sup_x, sup_y, qry_x, n_way, k_shot)
        loss, acc = matching_loss(log_probs, qry_y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        accuracies.append(acc)

    return {"loss": float(np.mean(losses)), "accuracy": float(np.mean(accuracies))}


def evaluate_matching(
    model: MatchingNetwork,
    episode_loader: Any,
    device: torch.device,
    n_way: int,
    k_shot: int,
    num_episodes: int = 600
) -> Dict[str, Any]:
    model.eval()
    accuracies = []

    with torch.no_grad():
        for count, ep in enumerate(episode_loader):
            if count >= num_episodes:
                break
            sup_x = ep["support_images"].to(device)
            sup_y = ep["support_labels"].to(device)
            qry_x = ep["query_images"].to(device)
            qry_y = ep["query_labels"].to(device)

            log_probs = model(sup_x, sup_y, qry_x, n_way, k_shot)
            _, acc = matching_loss(log_probs, qry_y)
            accuracies.append(acc)

    mean_acc = float(np.mean(accuracies))
    std_acc = float(np.std(accuracies))
    ci95 = 1.96 * (std_acc / np.sqrt(len(accuracies))) if accuracies else 0.0

    return {"accuracy": mean_acc, "ci95": ci95, "accuracies": accuracies}

