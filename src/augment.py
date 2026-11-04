"""
augment.py – Data augmentation for Tamil Few-Shot HTR training episodes.

Provides per-image stochastic augmentations that are safe for Tamil script:
NO flips or 90°/180°/270° rotations (Tamil letters are NOT symmetric).

All augmentations operate on single-channel float32 images in [0, 1] range
(H×W format). Each transform is applied with ≈50 % probability so that
every call to the augmentor yields a different random subset of transforms.

Dependencies: numpy, cv2 (no PyTorch).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.config import get_config, _CfgNode

logger = logging.getLogger(__name__)

# ── Default augmentation parameters ──────────────────────────────
_DEFAULTS: Dict[str, Any] = {
    "rotation_range": 10,            # degrees ±
    "shift_range": 0.1,              # fraction of image size
    "zoom_range": 0.1,               # fraction
    "shear_range": 5,                # degrees
    "elastic_alpha": 15.0,
    "elastic_sigma": 3.0,
    "noise_std": 0.02,
    "blur_kernel_range": [3, 5],
    "brightness_range": [0.8, 1.2],
    "contrast_range": [0.8, 1.2],
    "stroke_thickness_range": [-1, 1],
}


class TrainAugmentor:
    """Stochastic augmentor applied **per-image** during TRAIN episodes only.

    Each augmentation is applied independently with ~50 % probability.
    The augmentor is stateless apart from its parameter configuration.

    Parameters
    ----------
    cfg : _CfgNode | None
        Full project config.  If *None*, :func:`get_config` is called and
        ``cfg.augmentation`` is read.  Missing keys fall back to
        ``_DEFAULTS``.
    """

    def __init__(self, cfg: Optional[_CfgNode] = None) -> None:
        if cfg is None:
            try:
                cfg = get_config()
            except FileNotFoundError:
                logger.warning(
                    "config.yaml not found – using built-in augmentation defaults."
                )
                cfg = None

        aug_cfg: Dict[str, Any] = {}
        if cfg is not None:
            aug_section = getattr(cfg, "augmentation", {})
            if isinstance(aug_section, dict):
                aug_cfg = dict(aug_section)

        # Merge: config values override defaults
        self.rotation_range: float = float(
            aug_cfg.get("rotation_range", _DEFAULTS["rotation_range"])
        )
        self.shift_range: float = float(
            aug_cfg.get("shift_range", _DEFAULTS["shift_range"])
        )
        self.zoom_range: float = float(
            aug_cfg.get("zoom_range", _DEFAULTS["zoom_range"])
        )
        self.shear_range: float = float(
            aug_cfg.get("shear_range", _DEFAULTS["shear_range"])
        )
        self.elastic_alpha: float = float(
            aug_cfg.get("elastic_alpha", _DEFAULTS["elastic_alpha"])
        )
        self.elastic_sigma: float = float(
            aug_cfg.get("elastic_sigma", _DEFAULTS["elastic_sigma"])
        )
        self.noise_std: float = float(
            aug_cfg.get("noise_std", _DEFAULTS["noise_std"])
        )
        self.blur_kernel_range: List[int] = list(
            aug_cfg.get("blur_kernel_range", _DEFAULTS["blur_kernel_range"])
        )
        self.brightness_range: List[float] = list(
            aug_cfg.get("brightness_range", _DEFAULTS["brightness_range"])
        )
        self.contrast_range: List[float] = list(
            aug_cfg.get("contrast_range", _DEFAULTS["contrast_range"])
        )
        self.stroke_thickness_range: List[int] = list(
            aug_cfg.get("stroke_thickness_range", _DEFAULTS["stroke_thickness_range"])
        )

        logger.debug(
            "TrainAugmentor initialised: rot=±%.1f°, shift=%.2f, zoom=%.2f, "
            "shear=±%.1f°, elastic=(%.1f, %.1f), noise_std=%.3f",
            self.rotation_range,
            self.shift_range,
            self.zoom_range,
            self.shear_range,
            self.elastic_alpha,
            self.elastic_sigma,
            self.noise_std,
        )

    # ── Public entry point ───────────────────────────────────────

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply a random subset of augmentations to *image*.

        Parameters
        ----------
        image : np.ndarray
            Single-channel float32 image of shape ``(H, W)`` with values
            in ``[0, 1]``.

        Returns
        -------
        np.ndarray
            Augmented image (same shape / dtype), clipped to ``[0, 1]``.
        """
        if image.ndim != 2:
            raise ValueError(
                f"Expected 2-D image (H, W), got shape {image.shape}"
            )
        img = image.astype(np.float32, copy=True)

        # Each transform applied independently with ~50 % probability
        if np.random.random() < 0.5:
            img = self.random_affine(img)

        if np.random.random() < 0.5:
            img = self.elastic_deformation(img)

        if np.random.random() < 0.5:
            img = self.add_noise(img)

        if np.random.random() < 0.5:
            img = self.random_blur(img)

        if np.random.random() < 0.5:
            img = self.brightness_contrast(img)

        if np.random.random() < 0.5:
            img = self.stroke_thickness_jitter(img)

        return np.clip(img, 0.0, 1.0)

    # ── Individual augmentations ─────────────────────────────────

    def random_affine(self, image: np.ndarray) -> np.ndarray:
        """Apply random rotation + translation + zoom + shear.

        Uses ``cv2.getRotationMatrix2D`` for rotation/zoom combined with
        manual shear and translation offsets, then applies a single
        ``cv2.warpAffine`` call.

        Parameters
        ----------
        image : np.ndarray
            ``(H, W)`` float32 image in ``[0, 1]``.

        Returns
        -------
        np.ndarray
            Affine-transformed image, same shape.
        """
        h, w = image.shape[:2]
        cx, cy = w / 2.0, h / 2.0

        # Random parameters
        angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        scale = np.random.uniform(1.0 - self.zoom_range, 1.0 + self.zoom_range)
        tx = np.random.uniform(-self.shift_range, self.shift_range) * w
        ty = np.random.uniform(-self.shift_range, self.shift_range) * h
        shear_rad = np.deg2rad(
            np.random.uniform(-self.shear_range, self.shear_range)
        )

        # Rotation + scale matrix (2×3)
        M = cv2.getRotationMatrix2D((cx, cy), angle, scale)  # type: ignore[call-overload]

        # Add translation
        M[0, 2] += tx
        M[1, 2] += ty

        # Compose shear into the existing 2×3 matrix
        # Shear matrix: [[1, shear_x, 0], [0, 1, 0]]
        shear_mat = np.array(
            [[1.0, np.tan(shear_rad), 0.0],
             [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        # Convert M to 3×3 for composition, then take top 2 rows
        M_3x3 = np.vstack([M, [0.0, 0.0, 1.0]])
        S_3x3 = np.vstack([shear_mat, [0.0, 0.0, 1.0]])

        # Translate to origin → shear → translate back → then apply M
        T_to_origin = np.array(
            [[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64
        )
        T_back = np.array(
            [[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=np.float64
        )
        combined = M_3x3 @ T_back @ S_3x3 @ T_to_origin
        M_final = combined[:2, :]

        return cv2.warpAffine(
            image,
            M_final,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    def elastic_deformation(self, image: np.ndarray) -> np.ndarray:
        """Apply elastic deformation using a smoothed random displacement field.

        Generates per-pixel random displacements, smooths them with a
        Gaussian kernel (``elastic_sigma``), scales by ``elastic_alpha``,
        and remaps with ``cv2.remap``.

        Parameters
        ----------
        image : np.ndarray
            ``(H, W)`` float32 image in ``[0, 1]``.

        Returns
        -------
        np.ndarray
            Elastically deformed image, same shape.
        """
        h, w = image.shape[:2]
        alpha = self.elastic_alpha
        sigma = self.elastic_sigma

        # Gaussian-smoothed random displacement fields
        # Kernel size must be odd and large enough relative to sigma
        ksize = int(6 * sigma + 1) | 1  # ensure odd
        dx = cv2.GaussianBlur(
            np.random.uniform(-1.0, 1.0, (h, w)).astype(np.float32),
            (ksize, ksize),
            sigma,
        ) * alpha
        dy = cv2.GaussianBlur(
            np.random.uniform(-1.0, 1.0, (h, w)).astype(np.float32),
            (ksize, ksize),
            sigma,
        ) * alpha

        # Build remapping grids
        x, y = np.meshgrid(np.arange(w, dtype=np.float32),
                           np.arange(h, dtype=np.float32))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)

        return cv2.remap(
            image,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    def add_noise(self, image: np.ndarray) -> np.ndarray:
        """Add zero-mean Gaussian noise.

        Parameters
        ----------
        image : np.ndarray
            ``(H, W)`` float32 image in ``[0, 1]``.

        Returns
        -------
        np.ndarray
            Noisy image (not clipped here; final clip in ``__call__``).
        """
        noise = np.random.normal(0.0, self.noise_std, image.shape).astype(
            np.float32
        )
        return image + noise

    def random_blur(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur with a randomly chosen kernel size.

        The kernel size is drawn uniformly from ``blur_kernel_range`` and
        forced to be odd.

        Parameters
        ----------
        image : np.ndarray
            ``(H, W)`` float32 image in ``[0, 1]``.

        Returns
        -------
        np.ndarray
            Blurred image.
        """
        lo, hi = self.blur_kernel_range
        k = np.random.randint(lo, hi + 1)
        k = k | 1  # ensure odd
        return cv2.GaussianBlur(image, (k, k), 0)

    def brightness_contrast(self, image: np.ndarray) -> np.ndarray:
        """Apply random brightness and contrast jitter.

        Contrast is applied multiplicatively, brightness as an additive
        offset derived from the contrast-scaled range.

        Parameters
        ----------
        image : np.ndarray
            ``(H, W)`` float32 image in ``[0, 1]``.

        Returns
        -------
        np.ndarray
            Adjusted image (not clipped here; final clip in ``__call__``).
        """
        contrast = np.random.uniform(
            self.contrast_range[0], self.contrast_range[1]
        )
        brightness = np.random.uniform(
            self.brightness_range[0] - 1.0, self.brightness_range[1] - 1.0
        )
        return image * contrast + brightness

    def stroke_thickness_jitter(self, image: np.ndarray) -> np.ndarray:
        """Randomly thicken or thin stroke via morphological operations.

        A positive value from ``stroke_thickness_range`` triggers dilation
        (thicker strokes); a negative value triggers erosion (thinner
        strokes); zero is a no-op.  A small circular structuring element
        is used.

        Parameters
        ----------
        image : np.ndarray
            ``(H, W)`` float32 image in ``[0, 1]``.

        Returns
        -------
        np.ndarray
            Morphologically adjusted image.
        """
        lo, hi = self.stroke_thickness_range
        delta = np.random.randint(lo, hi + 1)
        if delta == 0:
            return image

        radius = abs(delta)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )

        if delta > 0:
            return cv2.dilate(image, kernel, iterations=1)
        else:
            return cv2.erode(image, kernel, iterations=1)


# ── Batch helper ─────────────────────────────────────────────────


def augment_batch(
    images: np.ndarray,
    augmentor: TrainAugmentor,
) -> np.ndarray:
    """Apply augmentation independently to every image in a batch.

    Parameters
    ----------
    images : np.ndarray
        Batch of single-channel images, shape ``(B, H, W)``, float32
        in ``[0, 1]``.
    augmentor : TrainAugmentor
        Configured augmentor instance.

    Returns
    -------
    np.ndarray
        Augmented batch with the same shape and dtype, values clipped
        to ``[0, 1]``.
    """
    if images.ndim != 3:
        raise ValueError(
            f"Expected 3-D batch (B, H, W), got shape {images.shape}"
        )
    batch_size = images.shape[0]
    out = np.empty_like(images)
    for i in range(batch_size):
        out[i] = augmentor(images[i])
    return out


# ── CLI quick-test ───────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Visualise augmentations on a sample image."
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a single grayscale image (PNG/JPG).  "
             "If omitted, a synthetic 64×64 test pattern is used.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of augmented copies to generate (default: 5).",
    )
    args = parser.parse_args()

    from src.config import seed_everything

    seed_everything(42)

    # Load or create a test image
    if args.image is not None:
        buf = np.fromfile(args.image, dtype=np.uint8)
        raw = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            print(f"ERROR: Cannot read image '{args.image}'")
            raise SystemExit(1)
        img = raw.astype(np.float32) / 255.0
    else:
        # Synthetic: centred circle on black background
        img = np.zeros((64, 64), dtype=np.float32)
        cv2.circle(img, (32, 32), 20, 1.0, 2)

    augmentor = TrainAugmentor()
    print(f"Original shape: {img.shape}, dtype: {img.dtype}")

    for i in range(args.n):
        aug_img = augmentor(img)
        print(
            f"  Augmented #{i + 1}: "
            f"min={aug_img.min():.4f}, max={aug_img.max():.4f}, "
            f"mean={aug_img.mean():.4f}"
        )

    # Batch test
    batch = np.stack([img] * args.n, axis=0)
    aug_batch = augment_batch(batch, augmentor)
    print(
        f"Batch augment: input {batch.shape} -> output {aug_batch.shape}, "
        f"range [{aug_batch.min():.4f}, {aug_batch.max():.4f}]"
    )
