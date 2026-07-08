from __future__ import annotations

import pickle
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter


# Per-pixel feature stack: raw intensity + multi-scale Gaussian blur + gradient magnitude.
# Keep _SIGMAS and N_FEATURES in sync; fit() and forward() must extract identical features.
_SIGMAS: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
N_FEATURES: int = 1 + len(_SIGMAS) + 1  # raw + blurs + gradient magnitude

# Safety cap on the number of training pixels handed to XGBoost after per-image
# background subsampling. Bounds fit() memory/time even on dense masks.
_MAX_TRAIN_PIXELS: int = 20_000_000


def extract_pixel_features(image: np.ndarray) -> np.ndarray:
    """Return an ``[H*W, N_FEATURES]`` float32 feature matrix for a ``[H, W]`` image in ``[0, 1]``."""

    image = image.astype(np.float32, copy=False)
    channels: list[np.ndarray] = [image]
    for sigma in _SIGMAS:
        channels.append(gaussian_filter(image, sigma=sigma))
    grad_y, grad_x = np.gradient(image)
    channels.append(np.sqrt(grad_x * grad_x + grad_y * grad_y))
    return np.stack(channels, axis=-1).reshape(-1, N_FEATURES).astype(np.float32)


class XGBoostSegmenter(nn.Module):
    """Pixel-wise binary segmentation backed by an XGBoost classifier.

    Acts as a drop-in ``nn.Module`` for the rest of the pipeline: ``forward`` returns
    ``{"out": logits}`` exactly like ``WrappedModel`` in ``model_factory.py``, so
    ``visualize_nearby.py`` and ``specify_visualize.py`` work without modification.

    Training is a single ``fit`` call rather than a gradient-descent epoch loop, so
    ``main.py`` routes this model to ``run_fold_xgboost`` instead of ``run_fold``.

    The classifier predicts ``P(foreground)`` per pixel; ``forward`` converts that to
    log-odds so the downstream ``sigmoid`` -> threshold -> hole-fill path is unchanged.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        max_bg_ratio: int = 5,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.subsample = float(subsample)
        self.colsample_bytree = float(colsample_bytree)
        # Background pixels kept per foreground pixel during training subsampling.
        self.max_bg_ratio = int(max_bg_ratio)
        self.seed = int(seed)
        self._clf: Any = None
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, images: np.ndarray, masks: np.ndarray) -> None:
        """Train the pixel classifier.

        Args:
            images: float32 array ``[N, H, W]`` normalized to ``[0, 1]``.
            masks:  float32/uint8 array ``[N, H, W]`` with values in ``{0, 1}``.
        """

        try:
            import xgboost as xgb
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "xgboost is required for the 'xgboost' model. Install it with: pip install xgboost"
            ) from exc

        rng = np.random.default_rng(self.seed)
        feature_blocks: list[np.ndarray] = []
        label_blocks: list[np.ndarray] = []

        print(f"  Extracting pixel features from {len(images)} training images...")
        for image, mask in zip(images, masks):
            features = extract_pixel_features(image)
            labels = (mask.ravel() >= 0.5).astype(np.int8)

            fg_idx = np.flatnonzero(labels == 1)
            bg_idx = np.flatnonzero(labels == 0)
            n_bg_keep = min(len(bg_idx), self.max_bg_ratio * max(len(fg_idx), 1))
            if n_bg_keep < len(bg_idx):
                bg_idx = rng.choice(bg_idx, size=n_bg_keep, replace=False)
            keep = np.concatenate([fg_idx, bg_idx])

            feature_blocks.append(features[keep])
            label_blocks.append(labels[keep])

        x_train = np.concatenate(feature_blocks, axis=0)
        y_train = np.concatenate(label_blocks, axis=0)

        if len(x_train) > _MAX_TRAIN_PIXELS:
            sel = rng.choice(len(x_train), size=_MAX_TRAIN_PIXELS, replace=False)
            x_train = x_train[sel]
            y_train = y_train[sel]

        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        scale_pos_weight = n_neg / max(n_pos, 1)

        print(
            f"  Training XGBoost on {len(x_train):,} pixels "
            f"({n_pos:,} foreground / {n_neg:,} background, "
            f"scale_pos_weight={scale_pos_weight:.2f})"
        )

        self._clf = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
            n_jobs=-1,
            random_state=self.seed,
            eval_metric="logloss",
        )
        self._clf.fit(x_train, y_train)
        self._fitted = True
        print("  XGBoost fit complete.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return ``{"out": logits}`` with logits shaped ``[B, 1, H, W]`` on ``images.device``."""

        if not self._fitted or self._clf is None:
            raise RuntimeError("XGBoostSegmenter must be fitted (or loaded) before inference.")

        device = images.device
        images_np = images.detach().to("cpu", dtype=torch.float32).numpy()[:, 0]  # [B, H, W]
        batch, height, width = images_np.shape

        logits_np = np.empty((batch, height, width), dtype=np.float32)
        for index in range(batch):
            features = extract_pixel_features(images_np[index])
            proba = self._clf.predict_proba(features)
            if proba.shape[1] == 1:
                # Degenerate single-class model: map to the one seen class.
                fg_prob = np.full(features.shape[0], float(self._clf.classes_[0]), dtype=np.float32)
            else:
                fg_prob = proba[:, 1]
            fg_prob = np.clip(fg_prob, 1e-7, 1.0 - 1e-7)
            logits_np[index] = np.log(fg_prob / (1.0 - fg_prob)).reshape(height, width)

        logits = torch.from_numpy(logits_np).unsqueeze(1).to(device)
        return {"out": logits}

    # ------------------------------------------------------------------
    # Persistence
    #
    # The booster is pickled and stored as a uint8 tensor so torch.save/torch.load
    # round-trip it as plain tensor data. This keeps loading compatible with
    # torch.load(weights_only=True) (the default on torch >= 2.6), which the rest
    # of the pipeline relies on. The booster is unpickled here, not by torch.
    # ------------------------------------------------------------------

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        payload: dict[str, Any] = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "max_bg_ratio": self.max_bg_ratio,
            "seed": self.seed,
            "fitted": self._fitted,
        }
        if self._fitted and self._clf is not None:
            clf_bytes = pickle.dumps(self._clf, protocol=pickle.HIGHEST_PROTOCOL)
            payload["clf_bytes"] = torch.frombuffer(bytearray(clf_bytes), dtype=torch.uint8).clone()
        return payload

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> None:  # type: ignore[override]
        self.n_estimators = int(state_dict["n_estimators"])
        self.max_depth = int(state_dict["max_depth"])
        self.learning_rate = float(state_dict["learning_rate"])
        self.subsample = float(state_dict["subsample"])
        self.colsample_bytree = float(state_dict["colsample_bytree"])
        self.max_bg_ratio = int(state_dict["max_bg_ratio"])
        self.seed = int(state_dict.get("seed", 42))
        self._fitted = bool(state_dict.get("fitted", False))
        clf_tensor = state_dict.get("clf_bytes")
        if self._fitted and clf_tensor is not None:
            self._clf = pickle.loads(bytes(clf_tensor.cpu().numpy().tobytes()))
        else:
            self._clf = None


__all__ = ["XGBoostSegmenter", "extract_pixel_features", "N_FEATURES"]
