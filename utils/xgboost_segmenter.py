from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


# Per-pixel feature stack: raw intensity + multi-scale Gaussian blur + gradient magnitude.
_SIGMAS: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
N_FEATURES: int = 1 + len(_SIGMAS) + 1  # raw + blurs + gradient magnitude


def extract_pixel_features(image: np.ndarray) -> np.ndarray:
    """Return an ``[H*W, N_FEATURES]`` float32 feature matrix for a ``[H, W]`` image in ``[0, 1]``."""

    image = image.astype(np.float32, copy=False)
    channels: list[np.ndarray] = [image]
    for sigma in _SIGMAS:
        channels.append(gaussian_filter(image, sigma=sigma))
    grad_y, grad_x = np.gradient(image)
    channels.append(np.sqrt(grad_x * grad_x + grad_y * grad_y))
    return np.stack(channels, axis=-1).reshape(-1, N_FEATURES).astype(np.float32)


__all__ = ["extract_pixel_features", "N_FEATURES"]
