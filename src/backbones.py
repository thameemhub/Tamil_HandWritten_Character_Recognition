"""Embedding network backbones for Tamil few-shot handwritten character recognition.

Provides lightweight CNN backbones that map single-channel 64×64 grayscale
images to L2-normalised embedding vectors on the unit hypersphere.

Architectures
-------------
- **Conv4**: Classic 4-block convolutional encoder (Vinyals et al., 2016).
- **ResNet12Lite**: Simplified ResNet-12 with two-conv residual blocks.

Usage
-----
>>> from src.backbones import get_backbone
>>> model = get_backbone("conv4", in_channels=1, embedding_dim=64)
>>> z = model(torch.randn(4, 1, 64, 64))  # (4, 64), unit-norm
"""

from __future__ import annotations

import logging
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

__all__ = ["Conv4", "ResNet12Lite", "get_backbone"]


# ---------------------------------------------------------------------------
# Helper: convolutional block used by Conv4
# ---------------------------------------------------------------------------

class _ConvBlock(nn.Module):
    """Conv2d → BatchNorm2d → ReLU → MaxPool2d(2)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Conv4
# ---------------------------------------------------------------------------

class Conv4(nn.Module):
    """Four-block convolutional embedding network.

    Architecture
    ------------
    Four successive ``_ConvBlock`` layers with channel progression
    ``in_channels → 64 → 64 → 64 → 64``.  A 64×64 input is spatially
    reduced to 4×4 after four 2× max-pool operations, yielding a
    1024-dimensional flat vector that is linearly projected to
    *embedding_dim* and L2-normalised.

    Parameters
    ----------
    in_channels : int, optional
        Number of input image channels (default ``1`` for grayscale).
    embedding_dim : int, optional
        Dimensionality of the output embedding (default ``64``).
    """

    def __init__(self, in_channels: int = 1, embedding_dim: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            _ConvBlock(in_channels, 64),
            _ConvBlock(64, 64),
            _ConvBlock(64, 64),
            _ConvBlock(64, 64),
        )
        # 64 channels × 4 × 4 spatial = 1024
        self.fc = nn.Linear(64 * 4 * 4, embedding_dim)

        self._init_weights()
        logger.info(
            "Conv4 backbone created: in_channels=%d, embedding_dim=%d",
            in_channels,
            embedding_dim,
        )

    # ---- weight initialisation -------------------------------------------

    def _init_weights(self) -> None:
        """Apply Kaiming Normal init for convolutions and constant init for BN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    # ---- forward ---------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute L2-normalised embedding.

        Parameters
        ----------
        x : torch.Tensor
            Input images of shape ``(batch, in_channels, 64, 64)``.

        Returns
        -------
        torch.Tensor
            Embedding of shape ``(batch, embedding_dim)`` on the unit
            hypersphere.
        """
        h = self.encoder(x)                       # (B, 64, 4, 4)
        h = h.view(h.size(0), -1)                 # (B, 1024)
        h = self.fc(h)                            # (B, embedding_dim)
        return F.normalize(h, p=2, dim=1)         # L2-normalise


# ---------------------------------------------------------------------------
# Residual block used by ResNet12Lite
# ---------------------------------------------------------------------------

class _ResBlock(nn.Module):
    """Two-conv residual block with optional channel-changing skip connection.

    Structure::

        x ──► Conv3×3 → BN → ReLU → Conv3×3 → BN ──(+)──► ReLU → MaxPool2d(2)
        │                                              ▲
        └─────────── [1×1 Conv → BN] (if channels differ)

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection with 1×1 projection when channel dims differ.
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = self.relu(out + identity)
        return self.pool(out)


# ---------------------------------------------------------------------------
# ResNet12Lite
# ---------------------------------------------------------------------------

class ResNet12Lite(nn.Module):
    """Simplified ResNet-12 embedding network for small images.

    Architecture
    ------------
    Four ``_ResBlock`` layers with channel progression
    ``in_channels → 64 → 128 → 256 → 512``, each halving spatial size via
    MaxPool2d(2).  An adaptive average pool collapses the spatial dimensions
    to 1×1 before a linear projection to *embedding_dim*.  The output is
    L2-normalised.

    Parameters
    ----------
    in_channels : int, optional
        Number of input image channels (default ``1`` for grayscale).
    embedding_dim : int, optional
        Dimensionality of the output embedding (default ``64``).
    """

    CHANNELS: List[int] = [64, 128, 256, 512]

    def __init__(self, in_channels: int = 1, embedding_dim: int = 64) -> None:
        super().__init__()

        blocks: List[nn.Module] = []
        prev_channels = in_channels
        for ch in self.CHANNELS:
            blocks.append(_ResBlock(prev_channels, ch))
            prev_channels = ch
        self.encoder = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.CHANNELS[-1], embedding_dim)

        self._init_weights()
        logger.info(
            "ResNet12Lite backbone created: in_channels=%d, embedding_dim=%d",
            in_channels,
            embedding_dim,
        )

    # ---- weight initialisation -------------------------------------------

    def _init_weights(self) -> None:
        """Apply Kaiming Normal init for convolutions and constant init for BN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    # ---- forward ---------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute L2-normalised embedding.

        Parameters
        ----------
        x : torch.Tensor
            Input images of shape ``(batch, in_channels, 64, 64)``.

        Returns
        -------
        torch.Tensor
            Embedding of shape ``(batch, embedding_dim)`` on the unit
            hypersphere.
        """
        h = self.encoder(x)                       # (B, 512, 4, 4)
        h = self.pool(h)                          # (B, 512, 1, 1)
        h = h.view(h.size(0), -1)                 # (B, 512)
        h = self.fc(h)                            # (B, embedding_dim)
        return F.normalize(h, p=2, dim=1)         # L2-normalise


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_BACKBONE_REGISTRY = {
    "conv4": Conv4,
    "resnet12": ResNet12Lite,
}


def get_backbone(
    name: str,
    in_channels: int = 1,
    embedding_dim: int = 64,
) -> nn.Module:
    """Instantiate a backbone by name.

    Parameters
    ----------
    name : str
        Backbone identifier.  One of ``"conv4"`` or ``"resnet12"``
        (case-insensitive).
    in_channels : int, optional
        Number of input image channels (default ``1``).
    embedding_dim : int, optional
        Dimensionality of the output embedding (default ``64``).

    Returns
    -------
    nn.Module
        The instantiated backbone network.

    Raises
    ------
    ValueError
        If *name* is not a recognised backbone identifier.
    """
    key = name.strip().lower()
    if key not in _BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone '{name}'. "
            f"Available: {sorted(_BACKBONE_REGISTRY.keys())}"
        )
    backbone = _BACKBONE_REGISTRY[key](
        in_channels=in_channels,
        embedding_dim=embedding_dim,
    )
    return backbone
