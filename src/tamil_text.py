"""
tamil_text.py – Tamil text rendering helpers for OpenCV images and matplotlib.

Provides utilities to render Tamil Unicode text onto NumPy images (via Pillow)
and configure matplotlib for Tamil-capable font families.  Standard OpenCV
``cv2.putText`` cannot render complex scripts like Tamil, so all on-image text
drawing goes through PIL's ``ImageDraw``.

Functions
---------
find_tamil_font      – Locate a Tamil-capable TrueType font and return an
                        ``ImageFont`` instance (cached).
put_tamil_text       – Draw Tamil text onto an OpenCV BGR image in-place.
configure_matplotlib_tamil – Set matplotlib's default font to a Tamil-capable
                        family so that Tamil labels render correctly in plots.
render_tamil_label   – Render a short Tamil string to a grayscale NumPy array
                        (useful for grid/report thumbnails).
"""

from __future__ import annotations

import functools
import logging
import unicodedata
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Font search configuration
# ---------------------------------------------------------------------------

# Ordered preference list – first match wins.
_PREFERRED_FONT_FILENAMES: list[str] = [
    "Nirmala.ttc",
    "NirmalaB.ttc",       # Bold variant
    "NotoSansTamil-Regular.ttf",
    "NotoSansTamil-Bold.ttf",
    "Latha.ttf",          # Legacy Windows Tamil font
]

# Directories to search (Windows-centric, but portable enough).
_FONT_DIRS: list[Path] = [
    Path(r"C:\Windows\Fonts"),
    Path.home() / "AppData" / "Local" / "Microsoft" / "Windows" / "Fonts",
    Path("/usr/share/fonts"),
    Path("/usr/local/share/fonts"),
    Path.home() / ".fonts",
]


# ---------------------------------------------------------------------------
# 1. find_tamil_font
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=16)
def find_tamil_font(size: int = 18) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Locate a Tamil-capable TrueType font and return a sized ``ImageFont``.

    Search order:
        1. Nirmala UI  (ships with Windows 8+)
        2. Noto Sans Tamil
        3. Any other known Tamil font (Latha, etc.)

    The result is cached per *size* so repeated calls are cheap.

    Parameters
    ----------
    size : int, optional
        Point size for the returned font (default ``18``).

    Returns
    -------
    ImageFont.FreeTypeFont | ImageFont.ImageFont
        A Pillow font object.  Falls back to the built-in default bitmap
        font (with a warning) if no Tamil-capable font is found.
    """
    for font_dir in _FONT_DIRS:
        if not font_dir.is_dir():
            continue
        for fname in _PREFERRED_FONT_FILENAMES:
            candidate = font_dir / fname
            if candidate.is_file():
                try:
                    font = ImageFont.truetype(str(candidate), size=size)
                    logger.info("Tamil font loaded: %s (size=%d)", candidate, size)
                    return font
                except Exception:
                    logger.debug("Failed to load font %s, trying next.", candidate)

    # Fallback – scan font dirs for *any* .ttf / .ttc containing 'tamil'
    for font_dir in _FONT_DIRS:
        if not font_dir.is_dir():
            continue
        try:
            for p in font_dir.iterdir():
                if p.suffix.lower() in {".ttf", ".ttc", ".otf"}:
                    if "tamil" in p.stem.lower():
                        try:
                            font = ImageFont.truetype(str(p), size=size)
                            logger.info(
                                "Tamil font (wildcard) loaded: %s (size=%d)",
                                p,
                                size,
                            )
                            return font
                        except Exception:
                            continue
        except PermissionError:
            continue

    logger.warning(
        "No Tamil-capable font found on this system. "
        "Text rendering will use Pillow's default bitmap font and may not "
        "display Tamil characters correctly.  Install 'Nirmala UI' or "
        "'Noto Sans Tamil' to fix this."
    )
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# 2. put_tamil_text
# ---------------------------------------------------------------------------

def put_tamil_text(
    img: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_size: int = 18,
    color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Draw Tamil text onto an OpenCV BGR image using Pillow.

    ``cv2.putText`` cannot render complex scripts (Tamil, Devanagari, …).
    This helper converts the image to a PIL ``Image``, draws text with
    ``ImageDraw``, and converts back.

    Parameters
    ----------
    img : np.ndarray
        OpenCV image in BGR uint8 format (H×W×3).  Modified **in-place** and
        also returned for convenience.
    text : str
        Tamil (or any Unicode) string to render.
    position : tuple[int, int]
        ``(x, y)`` pixel coordinates for the top-left corner of the text.
    font_size : int, optional
        Font point size (default ``18``).
    color : tuple[int, int, int], optional
        BGR colour tuple (default white ``(255, 255, 255)``).

    Returns
    -------
    np.ndarray
        The same image array with text drawn on it.
    """
    text = unicodedata.normalize("NFC", text)

    # BGR → RGB for Pillow
    rgb = img[:, :, ::-1].copy()
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)

    font = find_tamil_font(size=font_size)

    # Colour must also be RGB for Pillow
    color_rgb = (color[2], color[1], color[0])
    draw.text(position, text, font=font, fill=color_rgb)

    # Write back into the original array (in-place)
    rendered = np.asarray(pil_img)[:, :, ::-1]  # RGB → BGR
    np.copyto(img, rendered)
    return img


# ---------------------------------------------------------------------------
# 3. configure_matplotlib_tamil
# ---------------------------------------------------------------------------

def configure_matplotlib_tamil() -> None:
    """Configure matplotlib to use a Tamil-capable font family.

    Searches for Nirmala UI or Noto Sans Tamil in the system font directories
    and registers the first match as the default ``font.family``.  Call this
    **before** creating any figures or axes.

    If no suitable font is found a warning is logged and matplotlib's default
    sans-serif family is left unchanged.
    """
    import matplotlib
    import matplotlib.font_manager as fm

    # Candidate (family-name, file-glob-hint) pairs
    _CANDIDATES: list[Tuple[str, list[str]]] = [
        ("Nirmala UI", ["Nirmala.ttc", "NirmalaB.ttc"]),
        ("Noto Sans Tamil", ["NotoSansTamil-Regular.ttf", "NotoSansTamil-Bold.ttf"]),
        ("Latha", ["Latha.ttf"]),
    ]

    for family_name, filenames in _CANDIDATES:
        for font_dir in _FONT_DIRS:
            if not font_dir.is_dir():
                continue
            for fname in filenames:
                fpath = font_dir / fname
                if fpath.is_file():
                    try:
                        # Register with matplotlib's font manager
                        fm.fontManager.addfont(str(fpath))
                        matplotlib.rcParams["font.family"] = family_name
                        # Ensure the font is in the sans-serif fallback list too
                        sans_list: list[str] = matplotlib.rcParams.get(
                            "font.sans-serif", []
                        )
                        if family_name not in sans_list:
                            matplotlib.rcParams["font.sans-serif"] = [
                                family_name,
                                *sans_list,
                            ]
                        logger.info(
                            "matplotlib Tamil font set to '%s' from %s",
                            family_name,
                            fpath,
                        )
                        return
                    except Exception as exc:
                        logger.debug(
                            "Could not register '%s' with matplotlib: %s",
                            fpath,
                            exc,
                        )

    logger.warning(
        "No Tamil-capable font found for matplotlib.  Tamil labels in plots "
        "may render as empty boxes.  Install 'Nirmala UI' or 'Noto Sans Tamil'."
    )


# ---------------------------------------------------------------------------
# 4. render_tamil_label
# ---------------------------------------------------------------------------

def render_tamil_label(
    text: str,
    size: int = 64,
    font_size: int = 24,
    padding: int = 4,
) -> np.ndarray:
    """Render a Tamil string to a square grayscale NumPy image.

    Useful for embedding class-name thumbnails in image grids or evaluation
    reports.

    Parameters
    ----------
    text : str
        Tamil (or any Unicode) string to render.
    size : int, optional
        Output image dimensions ``(size, size)`` in pixels (default ``64``).
    font_size : int, optional
        Font point size used for rendering (default ``24``).
    padding : int, optional
        Minimum padding in pixels around the text (default ``4``).

    Returns
    -------
    np.ndarray
        Grayscale ``uint8`` image of shape ``(size, size)`` with white text on
        a black background.
    """
    text = unicodedata.normalize("NFC", text)
    font = find_tamil_font(size=font_size)

    # Render onto a temporary canvas to measure bounding box
    tmp = Image.new("L", (size * 4, size * 4), color=0)
    draw = ImageDraw.Draw(tmp)
    bbox = draw.textbbox((0, 0), text, font=font)  # (left, top, right, bottom)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Scale font down if text doesn't fit
    effective_font_size = font_size
    effective_font = font
    max_dim = size - 2 * padding
    while (text_w > max_dim or text_h > max_dim) and effective_font_size > 8:
        effective_font_size -= 2
        effective_font = find_tamil_font(size=effective_font_size)
        bbox = draw.textbbox((0, 0), text, font=effective_font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

    # Create final canvas and center the text
    canvas = Image.new("L", (size, size), color=0)
    draw_final = ImageDraw.Draw(canvas)
    x = max(padding, (size - text_w) // 2 - bbox[0])
    y = max(padding, (size - text_h) // 2 - bbox[1])
    draw_final.text((x, y), text, font=effective_font, fill=255)

    return np.asarray(canvas, dtype=np.uint8).copy()


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(description="Tamil text rendering smoke-test")
    parser.add_argument(
        "--text",
        default="\u0b85",  # Tamil letter 'அ'
        help="Tamil text to render (default: அ)",
    )
    parser.add_argument("--size", type=int, default=128, help="Output image size")
    args = parser.parse_args()

    # Test find_tamil_font
    font = find_tamil_font(size=24)
    print(f"Font found: {font}")

    # Test render_tamil_label
    label_img = render_tamil_label(args.text, size=args.size, font_size=32)
    print(f"Label image shape: {label_img.shape}, dtype: {label_img.dtype}")

    # Test put_tamil_text
    test_img = np.zeros((200, 400, 3), dtype=np.uint8)
    put_tamil_text(test_img, args.text, (50, 80), font_size=36, color=(0, 255, 0))
    print(f"put_tamil_text completed. Non-zero pixels: {np.count_nonzero(test_img)}")

    # Test configure_matplotlib_tamil
    configure_matplotlib_tamil()
    print("matplotlib Tamil font configuration complete.")
