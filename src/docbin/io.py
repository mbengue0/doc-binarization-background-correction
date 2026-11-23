"""Image loading/saving with explicit polarity handling."""
from pathlib import Path

import cv2
import numpy as np

TEXT, BACKGROUND = 0, 255


def load_gray(path) -> np.ndarray:
    """Load any image as uint8 grayscale [0, 255]."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def load_binary_gt(path) -> np.ndarray:
    """Load a DIBCO ground truth as uint8 with text=0, background=255.

    DIBCO GT files are white background / black text, sometimes 1-bit,
    sometimes 24-bit. Any non-zero pixel is treated as background, which is
    what the official MATLAB code does (logical(z)).
    """
    g = load_gray(path)
    return np.where(g > 0, BACKGROUND, TEXT).astype(np.uint8)


def text_mask_to_image(is_text: np.ndarray) -> np.ndarray:
    """Boolean text mask -> uint8 binary image in DIBCO polarity."""
    return np.where(is_text, TEXT, BACKGROUND).astype(np.uint8)


def save(path, img: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
