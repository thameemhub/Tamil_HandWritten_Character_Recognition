"""Prototypical Network for Tamil few-shot handwritten character recognition.

Implements the Prototypical Network (Snell et al., 2017) that learns a metric
space where classification is performed by computing distances to class
prototypes – the mean embeddings of support examples for each class.

Key components
--------------
- :class:`PrototypicalNetwork` – main model that wraps a backbone encoder and
  performs prototype-based classification with temperature-scaled Euclidean
  distance.
- :func:`prototypical_loss` – cross-entropy loss over N-way episode logits with
  accuracy tracking.
- :func:`train_one_epoch` – single training epoch over episodic batches.
- :func:`evaluate_episodes` – evaluation with mean accuracy and 95% CI.

Usage
-----
>>> from src.backbones import get_backbone
>>> from src.fewshot.protonet import PrototypicalNetwork
>>> backbone = get_backbone("conv4", in_channels=1, embedding_dim=64)
>>> model = PrototypicalNetwork(backbone, temperature=1.0, learnable_temp=True)
>>> logits = model(support_imgs, support_labels, query_imgs, n_way=5, k_shot=5)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)

__all__ = [
    "PrototypicalNetwork",
    "prototypical_loss",
    "train_one_epoch",
    "evaluate_episodes",
]


# ---------------------------------------------------------------------------
# Prototypical Network
# ---------------------------------------------------------------------------


class PrototypicalNetwork(nn.Module):
    """Prototypical Network for few-shot classification.

    The model embeds support and query images using a shared backbone,
    computes class prototypes as the mean support embedding per class, and
    classifies queries by (negative) squared Euclidean distance to each
    prototype, scaled by a temperature parameter.

    Parameters
    ----------
    backbone : nn.Module
        Embedding network that maps ``(B, C, H, W)`` images to
        ``(B, embed_dim)`` vectors.
    temperature : float, optional
        Initial temperature scalar for distance scaling (default ``1.0``).
    learnable_temp : bool, optional
        If ``True``, temperature is registered as a learnable
        :class:`nn.Parameter` (default ``True``).
    """

    def __init__(
        self,
        backbone: nn.Module,
        temperature: float = 1.0,
        learnable_temp: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone

        if learnable_temp:
            self.temperature = nn.Parameter(torch.tensor(temperature))
            logger.info(
                "PrototypicalNetwork created with learnable temperature=%.4f",
                temperature,
            )
        else:
            # Register as a buffer so it moves with .to(device) but is not
            # updated by the optimiser.
            self.register_buffer(
                "temperature", torch.tensor(temperature)
            )
            logger.info(
                "PrototypicalNetwork created with fixed temperature=%.4f",
                temperature,
            )

    # ------------------------------------------------------------------
    # Prototype computation
    # ------------------------------------------------------------------

    def compute_prototypes(
        self,
        support_embeddings: torch.Tensor,
        support_labels: torch.Tensor,
        n_way: int,
    ) -> torch.Tensor:
        """Compute the class prototype for each of the *n_way* classes.

        A prototype is the mean of the support embeddings belonging to that
        class.

        Parameters
        ----------
        support_embeddings : torch.Tensor
            Shape ``(n_way * k_shot, embed_dim)``.
        support_labels : torch.Tensor
            Shape ``(n_way * k_shot,)`` with integer values in ``[0, n_way)``.
        n_way : int
            Number of classes in the episode.

        Returns
        -------
        torch.Tensor
            Prototypes of shape ``(n_way, embed_dim)``.
        """
        embed_dim = support_embeddings.size(-1)
        prototypes = torch.zeros(
            n_way, embed_dim,
            device=support_embeddings.device,
            dtype=support_embeddings.dtype,
        )
        for c in range(n_way):
            mask = support_labels == c  # (n_way * k_shot,)
            prototypes[c] = support_embeddings[mask].mean(dim=0)
        return prototypes

    # ------------------------------------------------------------------
    # Distance helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _squared_euclidean_distance(
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        """Compute pairwise squared Euclidean distance.

        Parameters
        ----------
        a : torch.Tensor
            Shape ``(N, D)``.
        b : torch.Tensor
            Shape ``(M, D)``.

        Returns
        -------
        torch.Tensor
            Distance matrix of shape ``(N, M)`` where entry ``(i, j)`` is
            ``||a_i - b_j||^2``.
        """
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
        aa = (a * a).sum(dim=1, keepdim=True)  # (N, 1)
        bb = (b * b).sum(dim=1, keepdim=True)  # (M, 1)
        ab = torch.mm(a, b.t())                 # (N, M)
        return aa + bb.t() - 2.0 * ab

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        n_way: int,
        k_shot: int,
    ) -> torch.Tensor:
        """Run a full prototypical-network episode.

        Parameters
        ----------
        support_images : torch.Tensor
            Shape ``(n_way * k_shot, C, H, W)``.
        support_labels : torch.Tensor
            Shape ``(n_way * k_shot,)`` with values in ``[0, n_way)``.
        query_images : torch.Tensor
            Shape ``(n_queries, C, H, W)``.
        n_way : int
            Number of classes in the episode.
        k_shot : int
            Number of support examples per class.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(n_queries, n_way)`` — higher means closer
            to the prototype (negative distance / temperature).
        """
        # Embed support and query images through the shared backbone.
        support_embeddings = self.backbone(support_images)   # (N*K, D)
        query_embeddings = self.backbone(query_images)       # (Q, D)

        # Compute prototypes.
        prototypes = self.compute_prototypes(
            support_embeddings, support_labels, n_way,
        )  # (N, D)

        # Negative squared Euclidean distance → logits.
        distances = self._squared_euclidean_distance(
            query_embeddings, prototypes,
        )  # (Q, N)
        logits = -distances / self.temperature

        return logits

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        n_way: int,
        k_shot: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict class indices and confidence scores for query images.

        Parameters
        ----------
        support_images : torch.Tensor
            Shape ``(n_way * k_shot, C, H, W)``.
        support_labels : torch.Tensor
            Shape ``(n_way * k_shot,)``.
        query_images : torch.Tensor
            Shape ``(n_queries, C, H, W)``.
        n_way : int
            Number of classes.
        k_shot : int
            Support shots per class.

        Returns
        -------
        predicted_indices : torch.Tensor
            Shape ``(n_queries,)`` — predicted class index per query.
        confidence_scores : torch.Tensor
            Shape ``(n_queries,)`` — softmax probability of the predicted
            class.
        """
        self.eval()
        logits = self.forward(
            support_images, support_labels, query_images, n_way, k_shot,
        )
        probabilities = F.softmax(logits, dim=1)           # (Q, N)
        confidence_scores, predicted_indices = probabilities.max(dim=1)
        return predicted_indices, confidence_scores

    @torch.no_grad()
    def predict_with_prototypes(
        self,
        prototypes: torch.Tensor,
        query_images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Classify queries using pre-computed prototypes.

        This is useful at inference time when a prototype registry has
        already been built from support sets.

        Parameters
        ----------
        prototypes : torch.Tensor
            Shape ``(n_way, embed_dim)`` — pre-computed class prototypes.
        query_images : torch.Tensor
            Shape ``(n_queries, C, H, W)``.

        Returns
        -------
        predicted_indices : torch.Tensor
            Shape ``(n_queries,)`` — predicted class index per query.
        confidence_scores : torch.Tensor
            Shape ``(n_queries,)`` — softmax probability of the predicted
            class.
        """
        self.eval()
        query_embeddings = self.backbone(query_images)  # (Q, D)
        distances = self._squared_euclidean_distance(
            query_embeddings, prototypes,
        )  # (Q, N)
        logits = -distances / self.temperature
        probabilities = F.softmax(logits, dim=1)
        confidence_scores, predicted_indices = probabilities.max(dim=1)
        return predicted_indices, confidence_scores


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------


def prototypical_loss(
    logits: torch.Tensor,
    query_labels: torch.Tensor,
) -> Tuple[torch.Tensor, float]:
    """Cross-entropy loss for prototypical network logits.

    Parameters
    ----------
    logits : torch.Tensor
        Shape ``(n_queries, n_way)`` — raw (unscaled) logits from the
        prototypical network forward pass.
    query_labels : torch.Tensor
        Shape ``(n_queries,)`` — ground-truth class indices in ``[0, n_way)``.

    Returns
    -------
    loss : torch.Tensor
        Scalar cross-entropy loss (differentiable).
    accuracy : float
        Classification accuracy for this episode (non-differentiable).
    """
    loss = F.cross_entropy(logits, query_labels)

    # Accuracy (detached, no grad needed).
    with torch.no_grad():
        predictions = logits.argmax(dim=1)
        correct = (predictions == query_labels).sum().item()
        accuracy = correct / query_labels.size(0)

    return loss, accuracy


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: PrototypicalNetwork,
    episode_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    n_way: int,
    k_shot: int,
) -> Dict[str, float]:
    """Train the prototypical network for a single epoch over episodes.

    Each item yielded by *episode_loader* must be a tuple of::

        (support_images, support_labels, query_images, query_labels)

    where labels are in ``[0, n_way)`` for the episode.

    Parameters
    ----------
    model : PrototypicalNetwork
        The prototypical network model.
    episode_loader : iterable
        Yields ``(support_images, support_labels, query_images, query_labels)``
        tuples.
    optimizer : torch.optim.Optimizer
        Optimiser instance (already wrapping ``model.parameters()``).
    device : torch.device
        Target device for tensors.
    epoch : int
        Current epoch number (1-indexed, for display).
    n_way : int
        Number of classes per episode.
    k_shot : int
        Number of support examples per class.

    Returns
    -------
    dict
        ``{'loss': float, 'accuracy': float}`` — averages over all
        episodes in the epoch.
    """
    model.train()

    running_loss: float = 0.0
    running_acc: float = 0.0
    n_episodes: int = 0

    pbar = tqdm(
        episode_loader,
        desc=f"Epoch {epoch} [train]",
        leave=True,
        dynamic_ncols=True,
    )

    for item in pbar:
        if isinstance(item, dict):
            support_images = item["support_images"]
            support_labels = item["support_labels"]
            query_images = item["query_images"]
            query_labels = item["query_labels"]
        else:
            support_images, support_labels, query_images, query_labels = item

        # Move to device.
        support_images = support_images.to(device)
        support_labels = support_labels.to(device)
        query_images = query_images.to(device)
        query_labels = query_labels.to(device)

        # Forward pass.
        logits = model(support_images, support_labels, query_images, n_way, k_shot)
        loss, acc = prototypical_loss(logits, query_labels)

        # Backward pass.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track statistics.
        running_loss += loss.item()
        running_acc += acc
        n_episodes += 1

        # Update progress bar.
        pbar.set_postfix(
            loss=f"{running_loss / n_episodes:.4f}",
            acc=f"{running_acc / n_episodes:.4f}",
        )

    avg_loss = running_loss / max(n_episodes, 1)
    avg_acc = running_acc / max(n_episodes, 1)

    logger.info(
        "Epoch %d train — loss: %.4f, accuracy: %.4f (%d episodes)",
        epoch, avg_loss, avg_acc, n_episodes,
    )

    return {"loss": avg_loss, "accuracy": avg_acc}


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_episodes(
    model: PrototypicalNetwork,
    episode_loader,
    device: torch.device,
    n_way: int,
    k_shot: int,
    num_episodes: int = 600,
) -> Dict[str, object]:
    """Evaluate the model over many episodes and compute statistics.

    Parameters
    ----------
    model : PrototypicalNetwork
        The prototypical network model.
    episode_loader : iterable
        Yields ``(support_images, support_labels, query_images, query_labels)``
        tuples.
    device : torch.device
        Target device for tensors.
    n_way : int
        Number of classes per episode.
    k_shot : int
        Number of support examples per class.
    num_episodes : int, optional
        Maximum number of episodes to evaluate (default ``600``).

    Returns
    -------
    dict
        ``{'accuracy': float, 'ci95': float, 'accuracies': list[float]}``
        where *accuracy* is the mean, *ci95* is the 95% confidence interval
        half-width, and *accuracies* is the per-episode accuracy list.
    """
    model.eval()

    accuracies: List[float] = []

    pbar = tqdm(
        episode_loader,
        desc="Evaluating",
        total=num_episodes,
        leave=True,
        dynamic_ncols=True,
    )

    for idx, item in enumerate(pbar):
        if idx >= num_episodes:
            break

        if isinstance(item, dict):
            support_images = item["support_images"]
            support_labels = item["support_labels"]
            query_images = item["query_images"]
            query_labels = item["query_labels"]
        else:
            support_images, support_labels, query_images, query_labels = item

        support_images = support_images.to(device)
        support_labels = support_labels.to(device)
        query_images = query_images.to(device)
        query_labels = query_labels.to(device)

        logits = model(support_images, support_labels, query_images, n_way, k_shot)
        _, acc = prototypical_loss(logits, query_labels)
        accuracies.append(acc)

        pbar.set_postfix(acc=f"{np.mean(accuracies):.4f}")

    acc_array = np.array(accuracies)
    mean_acc = float(acc_array.mean())
    # 95% confidence interval: mean ± 1.96 * (std / sqrt(n))
    ci95 = float(1.96 * acc_array.std() / np.sqrt(len(acc_array)))

    logger.info(
        "Evaluation — accuracy: %.4f ± %.4f (95%% CI, %d episodes)",
        mean_acc, ci95, len(accuracies),
    )

    return {
        "accuracy": mean_acc,
        "ci95": ci95,
        "accuracies": accuracies,
    }
