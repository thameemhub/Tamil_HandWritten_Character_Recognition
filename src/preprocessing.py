"""
preprocessing.py – Single source of truth for image preprocessing.

Every image (training AND inference) passes through the same pipeline
defined here, ensuring identical transforms at all stages.

Pipeline (preprocess_image):
    read → composite alpha → grayscale → denoise → normalize polarity
    → binarize → crop to ink → pad to square → resize → scale [0, 1]
"""

from __future__ import annotations

import logging
import unicodedata
from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# 1. Image I/O
# ─────────────────────────────────────────────────────────────────────

def read_image(path: Union[str, Path]) -> np.ndarray | None:
    """Read an image from *path*, handling Windows Unicode filenames.

    Uses ``np.fromfile`` + ``cv2.imdecode`` to avoid OpenCV's ASCII-only
    ``imread``.  Falls back to PIL if the primary method fails.

    Parameters
    ----------
    path : str | Path
        Absolute or relative path to the image file.

    Returns
    -------
    np.ndarray | None
        BGR (or BGRA) uint8 array, or ``None`` if the file is unreadable.
    """
    path = Path(path)
    if not path.exists():
        logger.warning("Image path does not exist: %s", path)
        return None

    # Primary: numpy + cv2.imdecode (handles Unicode paths on Windows)
    try:
        raw = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img
    except Exception as exc:
        logger.debug("cv2.imdecode failed for %s: %s", path.name, exc)

    # Fallback: PIL → numpy
    try:
        pil_img = Image.open(str(path))
        pil_img.load()
        arr = np.array(pil_img)
        # PIL returns RGB(A); convert to BGR(A) for consistency
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
        elif arr.ndim == 2:
            # Grayscale → expand to BGR so downstream code is uniform
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        return arr
    except Exception as exc:
        logger.warning("Cannot read image %s (PIL fallback also failed): %s",
                       path.name, exc)
        return None


# ─────────────────────────────────────────────────────────────────────
# 2. Alpha compositing
# ─────────────────────────────────────────────────────────────────────

def composite_alpha(
    img: np.ndarray,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Composite a BGRA image onto a solid background, returning BGR.

    If the image already has ≤ 3 channels it is returned as-is (or
    converted to BGR if single-channel).

    Parameters
    ----------
    img : np.ndarray
        Input image (BGR or BGRA, uint8).
    bg_color : tuple[int, int, int]
        Background colour in BGR order.

    Returns
    -------
    np.ndarray
        3-channel BGR uint8 image.
    """
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    channels = img.shape[2]
    if channels == 3:
        return img

    if channels == 4:
        alpha = img[:, :, 3].astype(np.float32) / 255.0
        bgr = img[:, :, :3].astype(np.float32)
        bg = np.full_like(bgr, bg_color, dtype=np.float32)
        alpha3 = np.stack([alpha] * 3, axis=-1)
        blended = bgr * alpha3 + bg * (1.0 - alpha3)
        return np.clip(blended, 0, 255).astype(np.uint8)

    # Unexpected channel count – just take first 3
    logger.debug("Unexpected channel count %d; taking first 3.", channels)
    return img[:, :, :3]


# ─────────────────────────────────────────────────────────────────────
# 3. Grayscale conversion
# ─────────────────────────────────────────────────────────────────────

def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert an image to single-channel grayscale.

    Handles alpha compositing automatically before conversion.

    Parameters
    ----------
    img : np.ndarray
        Input image (grayscale, BGR, or BGRA).

    Returns
    -------
    np.ndarray
        Single-channel uint8 grayscale image.
    """
    if img.ndim == 2:
        return img

    # Composite alpha first (produces BGR)
    bgr = composite_alpha(img)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


# ─────────────────────────────────────────────────────────────────────
# 4. Polarity detection
# ─────────────────────────────────────────────────────────────────────

def detect_polarity(gray: np.ndarray) -> str:
    """Detect whether ink is darker or lighter than the background.

    Examines a 5-pixel border strip and compares its median intensity
    to the median of the central region.

    Parameters
    ----------
    gray : np.ndarray
        Single-channel uint8 image.

    Returns
    -------
    str
        ``"dark_on_light"`` (standard paper) or ``"light_on_dark"``.
    """
    h, w = gray.shape[:2]
    strip = 5

    # Collect border pixels (outer 5-pixel strip)
    border_pixels: list[np.ndarray] = []
    if h > 2 * strip and w > 2 * strip:
        border_pixels.append(gray[:strip, :].ravel())       # top
        border_pixels.append(gray[-strip:, :].ravel())      # bottom
        border_pixels.append(gray[strip:-strip, :strip].ravel())   # left
        border_pixels.append(gray[strip:-strip, -strip:].ravel())  # right
    else:
        # Image is very small – use entire image as border estimate
        border_pixels.append(gray.ravel())

    border_median = float(np.median(np.concatenate(border_pixels)))

    # Center region
    if h > 2 * strip and w > 2 * strip:
        center = gray[strip:-strip, strip:-strip]
    else:
        center = gray
    center_median = float(np.median(center.ravel()))

    # Border brighter than center → dark ink on light paper
    if border_median >= center_median:
        return "dark_on_light"
    return "light_on_dark"


# ─────────────────────────────────────────────────────────────────────
# 5. Polarity normalisation
# ─────────────────────────────────────────────────────────────────────

def normalize_polarity(
    gray: np.ndarray,
    target_ink_white: bool = True,
) -> np.ndarray:
    """Normalise so ink is white on a black background (or vice-versa).

    Parameters
    ----------
    gray : np.ndarray
        Single-channel uint8 image.
    target_ink_white : bool
        If ``True`` (default), ink will be white (255) on black (0).
        If ``False``, ink will be black on white.

    Returns
    -------
    np.ndarray
        Polarity-normalised single-channel uint8 image.
    """
    polarity = detect_polarity(gray)

    if target_ink_white:
        # We want ink=white, bg=black.
        # "dark_on_light" means ink is dark → invert.
        if polarity == "dark_on_light":
            return cv2.bitwise_not(gray)
        return gray.copy()
    else:
        # We want ink=black, bg=white.
        if polarity == "light_on_dark":
            return cv2.bitwise_not(gray)
        return gray.copy()


# ─────────────────────────────────────────────────────────────────────
# 6. Binarisation
# ─────────────────────────────────────────────────────────────────────

def _foreground_ratio(binary: np.ndarray) -> float:
    """Return the fraction of white (255) pixels."""
    total = binary.size
    if total == 0:
        return 0.0
    return float(np.count_nonzero(binary)) / total


def binarize(gray: np.ndarray, use_binarization: bool = True) -> np.ndarray:
    """Binarise a grayscale image using the best of Otsu / adaptive.

    The method that produces a foreground-pixel ratio in [5 %, 50 %] is
    preferred.  If both qualify, Otsu is chosen; if neither qualifies
    the one closest to the valid range is returned.

    Parameters
    ----------
    gray : np.ndarray
        Single-channel uint8 image (ink should already be white on black).
    use_binarization : bool
        If ``False``, the image is returned unchanged.

    Returns
    -------
    np.ndarray
        Binary uint8 image (values 0 or 255).
    """
    if not use_binarization:
        return gray

    # Otsu
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    otsu_ratio = _foreground_ratio(otsu)

    # Adaptive Gaussian
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=11, C=2,
    )
    adaptive_ratio = _foreground_ratio(adaptive)

    lo, hi = 0.05, 0.50

    otsu_ok = lo <= otsu_ratio <= hi
    adaptive_ok = lo <= adaptive_ratio <= hi

    if otsu_ok and adaptive_ok:
        return otsu
    if otsu_ok:
        return otsu
    if adaptive_ok:
        return adaptive

    # Neither is ideal – pick the one whose ratio is closest to [lo, hi]
    def _dist(r: float) -> float:
        if r < lo:
            return lo - r
        if r > hi:
            return r - hi
        return 0.0

    if _dist(otsu_ratio) <= _dist(adaptive_ratio):
        return otsu
    return adaptive


# ─────────────────────────────────────────────────────────────────────
# 7. Ink cropping
# ─────────────────────────────────────────────────────────────────────

def crop_to_ink(
    gray: np.ndarray,
    margin_fraction: float = 0.1,
) -> np.ndarray:
    """Crop to the bounding box of non-zero (ink) pixels with a margin.

    Parameters
    ----------
    gray : np.ndarray
        Single-channel image where ink is non-zero.
    margin_fraction : float
        Fraction of bbox width/height to add as margin on each side.

    Returns
    -------
    np.ndarray
        Cropped image, or the original if no ink is found.
    """
    coords = cv2.findNonZero(gray)
    if coords is None:
        logger.debug("crop_to_ink: no non-zero pixels found; returning as-is.")
        return gray

    x, y, bw, bh = cv2.boundingRect(coords)
    h, w = gray.shape[:2]

    margin_x = int(bw * margin_fraction)
    margin_y = int(bh * margin_fraction)

    x0 = max(x - margin_x, 0)
    y0 = max(y - margin_y, 0)
    x1 = min(x + bw + margin_x, w)
    y1 = min(y + bh + margin_y, h)

    cropped = gray[y0:y1, x0:x1]
    if cropped.size == 0:
        return gray
    return cropped


# ─────────────────────────────────────────────────────────────────────
# 8. Square padding
# ─────────────────────────────────────────────────────────────────────

def pad_to_square(gray: np.ndarray) -> np.ndarray:
    """Pad the shorter side with zeros so the image is square, centred.

    Parameters
    ----------
    gray : np.ndarray
        Single-channel image.

    Returns
    -------
    np.ndarray
        Square image with the original content centred.
    """
    h, w = gray.shape[:2]
    if h == w:
        return gray

    side = max(h, w)
    result = np.zeros((side, side), dtype=gray.dtype)
    y_off = (side - h) // 2
    x_off = (side - w) // 2
    result[y_off:y_off + h, x_off:x_off + w] = gray
    return result


# ─────────────────────────────────────────────────────────────────────
# 9. Full preprocessing pipeline
# ─────────────────────────────────────────────────────────────────────

def preprocess_image(
    img_or_path: Union[np.ndarray, str, Path],
    image_size: int = 64,
    use_binarization: bool = True,
    denoise_strength: int = 5,
    crop_margin_fraction: float = 0.1,
    composite_bg: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray | None:
    """Run the complete preprocessing pipeline on a single image.

    This function is the **single source of truth** – it is called
    identically during dataset preparation and at inference time.

    Parameters
    ----------
    img_or_path : np.ndarray | str | Path
        Either an already-loaded image array or a file path.
    image_size : int
        Target spatial resolution (square).
    use_binarization : bool
        Whether to apply adaptive / Otsu binarisation.
    denoise_strength : int
        ``h`` parameter for ``cv2.fastNlMeansDenoising``.
    crop_margin_fraction : float
        Margin fraction passed to :func:`crop_to_ink`.
    composite_bg : tuple[int, int, int]
        BGR background colour for alpha compositing.

    Returns
    -------
    np.ndarray | None
        Float32 array of shape ``(image_size, image_size)`` in [0, 1],
        or ``None`` if the image could not be read/processed.
    """
    # --- Step 1: Read (if path) -----------------------------------------
    if isinstance(img_or_path, (str, Path)):
        img = read_image(img_or_path)
        if img is None:
            return None
    else:
        img = img_or_path.copy()

    # --- Step 2: Composite alpha ----------------------------------------
    img_bgr = composite_alpha(img, bg_color=composite_bg)

    # --- Step 3: Grayscale ----------------------------------------------
    gray = to_grayscale(img_bgr)

    # --- Step 4: Denoise ------------------------------------------------
    if denoise_strength > 0:
        gray = cv2.fastNlMeansDenoising(
            gray, None, h=denoise_strength, templateWindowSize=7,
            searchWindowSize=21,
        )

    # --- Step 5: Normalize polarity (ink → white, bg → black) -----------
    gray = normalize_polarity(gray, target_ink_white=True)

    # --- Step 6: Binarize -----------------------------------------------
    gray = binarize(gray, use_binarization=use_binarization)

    # --- Step 7: Crop to ink --------------------------------------------
    gray = crop_to_ink(gray, margin_fraction=crop_margin_fraction)

    # --- Step 8: Pad to square ------------------------------------------
    gray = pad_to_square(gray)

    # --- Step 9: Resize -------------------------------------------------
    gray = cv2.resize(
        gray, (image_size, image_size), interpolation=cv2.INTER_AREA,
    )

    # --- Step 10: Scale to [0, 1] float32 --------------------------------
    result = gray.astype(np.float32) / 255.0
    return result


# ─────────────────────────────────────────────────────────────────────
# 10. Model-ready tensor
# ─────────────────────────────────────────────────────────────────────

def preprocess_for_model(
    img_or_path: Union[np.ndarray, str, Path],
    cfg: object | None = None,
) -> torch.Tensor | None:
    """Preprocess an image and return a model-ready ``torch.Tensor``.

    Parameters
    ----------
    img_or_path : np.ndarray | str | Path
        Raw image or file path.
    cfg : _CfgNode | None
        Configuration object.  If ``None``, loaded via
        :func:`src.config.get_config`.

    Returns
    -------
    torch.Tensor | None
        Tensor of shape ``(1, image_size, image_size)`` (float32), or
        ``None`` if the image could not be processed.
    """
    if cfg is None:
        from src.config import get_config
        cfg = get_config()

    pp = cfg.preprocessing
    image_size: int = int(pp.get("image_size", 64))
    use_bin: bool = bool(pp.get("use_binarization", True))
    denoise: int = int(pp.get("denoise_strength", 5))
    margin: float = float(pp.get("crop_margin_fraction", 0.1))
    bg_color: Tuple[int, int, int] = tuple(
        pp.get("composite_bg_color", [255, 255, 255]),
    )

    arr = preprocess_image(
        img_or_path,
        image_size=image_size,
        use_binarization=use_bin,
        denoise_strength=denoise,
        crop_margin_fraction=margin,
        composite_bg=bg_color,
    )
    if arr is None:
        return None

    # Shape: (1, H, W) – single-channel
    tensor = torch.from_numpy(arr).unsqueeze(0)
    return tensor
