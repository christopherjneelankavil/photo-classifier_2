"""
Multi-metric blur detection module.

Combines three complementary frequency/gradient measures:
  1. Laplacian Variance  – second-derivative sharpness
  2. FFT High-Frequency Ratio – frequency-domain energy
  3. Brenner Gradient – first-derivative contrast

Each metric is normalised to [0, 1] range, then combined with tunable
weights.  The weighted score is compared against a single threshold to
decide blurred / sharp.
"""

import cv2
import numpy as np


# --------------- individual metrics ---------------

def _laplacian_variance(gray: np.ndarray) -> float:
    """Higher value → sharper image."""
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def _fft_high_freq_ratio(gray: np.ndarray, radius_fraction: float = 0.1) -> float:
    """
    Ratio of high-frequency energy to total energy in the FFT.
    Higher value → sharper image.
    """
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    rows, cols = gray.shape
    cy, cx = rows // 2, cols // 2
    radius = int(min(rows, cols) * radius_fraction)

    # Create low-frequency mask
    y, x = np.ogrid[:rows, :cols]
    low_freq_mask = ((x - cx) ** 2 + (y - cy) ** 2) <= radius ** 2

    total_energy = np.sum(magnitude ** 2) + 1e-10
    low_energy = np.sum((magnitude * low_freq_mask) ** 2)
    high_energy = total_energy - low_energy

    return float(high_energy / total_energy)


def _brenner_gradient(gray: np.ndarray) -> float:
    """
    Sum of squared differences between pixels two apart.
    Higher value → sharper image.
    """
    diff = gray[:, 2:].astype(np.float64) - gray[:, :-2].astype(np.float64)
    return float(np.mean(diff ** 2))


# --------------- combined scorer ---------------

# Normalisation ceilings (values above these are clamped to 1.0)
_LAP_CEIL = 500.0
_FFT_CEIL = 1.0       # already 0-1
_BRE_CEIL = 800.0

# Weights for the three metrics
_W_LAP = 0.40
_W_FFT = 0.25
_W_BRE = 0.35

# Final threshold – below this the image is considered blurry
BLUR_THRESHOLD = 0.30


def compute_blur_score(image_path: str) -> float:
    """
    Return a blur score in [0, 1].  Lower → blurrier.

    Parameters
    ----------
    image_path : str
        Path to the image file.

    Returns
    -------
    float
        Combined sharpness score (0 = very blurry, 1 = very sharp).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize large images for consistent scoring
    max_dim = 1024
    h, w = gray.shape
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    lap = min(_laplacian_variance(gray) / _LAP_CEIL, 1.0)
    fft = min(_fft_high_freq_ratio(gray) / _FFT_CEIL, 1.0)
    bre = min(_brenner_gradient(gray) / _BRE_CEIL, 1.0)

    score = _W_LAP * lap + _W_FFT * fft + _W_BRE * bre
    return round(score, 4)


def is_blurred(image_path: str, threshold: float = BLUR_THRESHOLD) -> bool:
    """
    Return True if the image is considered blurry.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    threshold : float
        Score below this is blurry.  Default 0.30.
    """
    return compute_blur_score(image_path) < threshold
