from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy.ndimage import binary_fill_holes
from torch import nn
from torch.nn import functional as F


class BCEDiceloss(nn.Module):
    """Combined BCE-with-logits and Dice loss for binary segmentation."""

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1e-6,
        pos_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.register_buffer("pos_weight", pos_weight)

        if bce_weight < 0 or dice_weight < 0:
            raise ValueError("bce_weight and dice_weight must be non-negative")
        if bce_weight == 0 and dice_weight == 0:
            raise ValueError("At least one loss weight must be greater than zero")

    def forward(self, pred_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(pred_logits, targets, pos_weight=self.pos_weight)
        dice = dice_loss_from_logits(pred_logits, targets, smooth=self.smooth)
        return self.bce_weight * bce + self.dice_weight * dice


BCEDiceLoss = BCEDiceloss


def dice_loss_from_logits(
    pred_logits: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """Compute soft Dice loss from raw logits."""

    probs = torch.sigmoid(pred_logits)
    targets = targets.float()
    dims = tuple(range(1, probs.ndim))

    intersection = torch.sum(probs * targets, dim=dims)
    denominator = torch.sum(probs, dim=dims) + torch.sum(targets, dim=dims)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()


def get_postprocessed_mask(pred_logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Convert logits to a binary mask and apply hole filling."""

    preds = torch.sigmoid(pred_logits)
    preds = (preds > threshold).float()

    preds_np = preds.detach().cpu().numpy()
    filled_np = np.zeros_like(preds_np, dtype=np.float32)

    for b in range(preds_np.shape[0]):
        for c in range(preds_np.shape[1]):
            filled_np[b, c] = binary_fill_holes(preds_np[b, c]).astype(np.float32)

    return torch.from_numpy(filled_np).to(pred_logits.device)


def get_binary_mask(pred_logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Convert logits to a binary float mask without postprocessing."""

    return (torch.sigmoid(pred_logits) > threshold).float()


def dice_coefficient(
    preds: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """Compute Dice coefficient from binary prediction and target masks."""

    preds = preds.float()
    targets = targets.float()
    dims = tuple(range(1, preds.ndim))

    intersection = torch.sum(preds * targets, dim=dims)
    denominator = torch.sum(preds, dim=dims) + torch.sum(targets, dim=dims)
    return ((2.0 * intersection + smooth) / (denominator + smooth)).mean()


def dice_score_from_logits(
    pred_logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """Compute binary Dice score from logits without extra metric bookkeeping."""

    preds = get_binary_mask(pred_logits, threshold=threshold)
    return dice_coefficient(preds, targets, smooth=smooth)


def iou_score(
    preds: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """Compute intersection-over-union from binary prediction and target masks."""

    preds = preds.float()
    targets = targets.float()
    dims = tuple(range(1, preds.ndim))

    intersection = torch.sum(preds * targets, dim=dims)
    union = torch.sum(preds, dim=dims) + torch.sum(targets, dim=dims) - intersection
    return ((intersection + smooth) / (union + smooth)).mean()


def compute_binary_metrics(
    pred_logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    postprocess: bool = True,
    smooth: float = 1e-6,
) -> dict[str, float]:
    """Compute common binary segmentation metrics from logits."""

    with torch.no_grad():
        preds = (
            get_postprocessed_mask(pred_logits, threshold=threshold)
            if postprocess
            else get_binary_mask(pred_logits, threshold=threshold)
        )
        return compute_binary_mask_metrics(preds, targets, smooth=smooth)


def compute_binary_mask_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-6,
) -> dict[str, float]:
    """Compute common binary segmentation metrics from binary masks."""

    with torch.no_grad():
        preds = preds.float()
        targets = targets.float()

        tp = torch.sum(preds * targets)
        fp = torch.sum(preds * (1.0 - targets))
        fn = torch.sum((1.0 - preds) * targets)
        tn = torch.sum((1.0 - preds) * (1.0 - targets))

        metrics: dict[str, Any] = {
            "dice": (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth),
            "iou": (tp + smooth) / (tp + fp + fn + smooth),
            "precision": (tp + smooth) / (tp + fp + smooth),
            "recall": (tp + smooth) / (tp + fn + smooth),
            "specificity": (tn + smooth) / (tn + fp + smooth),
            "accuracy": (tp + tn + smooth) / (tp + tn + fp + fn + smooth),
        }

    return {name: float(value.detach().cpu().item()) for name, value in metrics.items()}


__all__ = [
    "BCEDiceLoss",
    "BCEDiceloss",
    "compute_binary_metrics",
    "compute_binary_mask_metrics",
    "dice_coefficient",
    "dice_loss_from_logits",
    "dice_score_from_logits",
    "get_binary_mask",
    "get_postprocessed_mask",
    "iou_score",
]
