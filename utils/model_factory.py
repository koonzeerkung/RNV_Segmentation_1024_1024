from __future__ import annotations

import re
from typing import Final

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

# XGBoost is a pixel classifier with no encoder backbone. It is handled separately
# from the segmentation_models_pytorch architectures throughout this module.
XGBOOST_MODEL: Final[str] = "xgboost"

# FRNet is a custom full-resolution CNN (Ning et al., ICASSP 2024) with no encoder
# backbone. It trains with gradient descent like the smp models, so it uses the normal
# training loop; only model construction and backbone handling are special-cased.
FRNET_MODEL: Final[str] = "frnet"

SUPPORTED_MODELS: Final[tuple[str, ...]] = (
    "unet",
    "unetplusplus",
    "deeplabv3plus",
    "deeplabv3",
    FRNET_MODEL,
    XGBOOST_MODEL,
)
SUPPORTED_BACKBONES: Final[tuple[str, ...]] = ("efficientnet-b3", "inceptionv4", "densenet169")
_MODEL_ALIASES: Final[dict[str, str]] = {
    "unet": "unet",
    "u_net": "unet",
    "unetplusplus": "unetplusplus",
    "unet_plus_plus": "unetplusplus",
    "unetpp": "unetplusplus",
    "unet_pp": "unetplusplus",
    "unet_plusplus": "unetplusplus",
    "deeplab": "deeplabv3plus",
    "deep_lab": "deeplabv3plus",
    "deeplabv3plus": "deeplabv3plus",
    "deeplab_v3_plus": "deeplabv3plus",
    "deeplabv3": "deeplabv3",
    "deeplab_v3": "deeplabv3",
    "frnet": FRNET_MODEL,
    "fr_net": FRNET_MODEL,
    "full_resolution_network": FRNET_MODEL,
    "xgboost": XGBOOST_MODEL,
    "xgb": XGBOOST_MODEL,
    "xg_boost": XGBOOST_MODEL,
    "xgboost_segmenter": XGBOOST_MODEL,
}
_BACKBONE_ALIASES: Final[dict[str, str]] = {
    "efficientnet_b3": "efficientnet-b3",
    "efficientnetb3": "efficientnet-b3",
    "inceptionv4": "inceptionv4",
    "inception_v4": "inceptionv4",
    "densenet169": "densenet169",
    "densenet_169": "densenet169",
    "dense_net_169": "densenet169",
}
_ARCHITECTURE_CLASSES: Final[dict[str, type[nn.Module]]] = {
    "unet": smp.Unet,
    "unetplusplus": smp.UnetPlusPlus,
    "deeplabv3plus": smp.DeepLabV3Plus,
    "deeplabv3": smp.DeepLabV3,
}


def _lookup_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def normalize_model_name(model_name: str) -> str:
    raw = model_name.strip().lower().replace("-", "_")
    if raw in {"unet++", "u_net++", "u-net++"}:
        return "unetplusplus"
    if raw in {"deeplabv3+", "deeplab_v3+", "deep_lab_v3+"}:
        return "deeplabv3plus"

    key = _lookup_key(model_name)
    if key in _MODEL_ALIASES:
        return _MODEL_ALIASES[key]

    choices = ", ".join(SUPPORTED_MODELS)
    raise ValueError(f"Unsupported model_name '{model_name}'. Expected one of: {choices}")


def normalize_backbone_name(backbone: str) -> str:
    key = _lookup_key(backbone)
    if key in _BACKBONE_ALIASES:
        return _BACKBONE_ALIASES[key]

    choices = ", ".join(SUPPORTED_BACKBONES)
    raise ValueError(f"Unsupported backbone '{backbone}'. Expected one of: {choices}")


def is_xgboost_model(model_name: str) -> bool:
    """Return True when ``model_name`` resolves to the XGBoost pixel classifier."""

    return normalize_model_name(model_name) == XGBOOST_MODEL


def validate_model_backbone(model_name: str, backbone: str) -> None:
    model_name = normalize_model_name(model_name)

    # XGBoost and FRNet have no encoder backbone, so backbone compatibility does not apply.
    if model_name in (XGBOOST_MODEL, FRNET_MODEL):
        return

    backbone = normalize_backbone_name(backbone)

    if model_name.startswith("deeplab") and backbone in {"densenet169", "inceptionv4"}:
        raise ValueError(
            f"{model_name} cannot use backbone '{backbone}' in segmentation_models_pytorch because that encoder "
            "does not support the dilated mode required by DeepLab. Use backbone='efficientnet-b3' for DeepLab, "
            "or use model_name='unet'/'unetplusplus' with this backbone."
        )


def model_experiment_token(model_name: str, backbone: str = "efficientnet-b3") -> str:
    model_name = normalize_model_name(model_name)
    # XGBoost and FRNet have no backbone, so the token is just the model name.
    if model_name in (XGBOOST_MODEL, FRNET_MODEL):
        return model_name
    backbone = normalize_backbone_name(backbone)
    backbone_token = backbone.replace("-", "")
    return f"{model_name}_{backbone_token}"


def get_model(
    device: torch.device | str,
    model_name: str = "unet",
    backbone: str = "efficientnet-b3",
    in_channels: int = 1,
    out_channels: int = 1,
) -> nn.Module:
    normalized_name = normalize_model_name(model_name)

    if normalized_name == XGBOOST_MODEL:
        from utils.xgboost_segmenter import XGBoostSegmenter

        print("Loading model: xgboost (pixel classifier, no encoder backbone)")
        return XGBoostSegmenter().to(device)

    if normalized_name == FRNET_MODEL:
        from utils.frnet import build_frnet

        print("Loading model: frnet (full-resolution network, no encoder backbone)")
        model: nn.Module = build_frnet(in_channels=in_channels, out_channels=out_channels)
    else:
        normalized_backbone = normalize_backbone_name(backbone)
        validate_model_backbone(normalized_name, normalized_backbone)

        model_class = _ARCHITECTURE_CLASSES[normalized_name]
        print(f"Loading model: {normalized_name} (encoder: {normalized_backbone})")

        model = model_class(
            encoder_name=normalized_backbone,
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=out_channels,
            activation=None,
        )

    class WrappedModel(nn.Module):
        def __init__(self, base_model: nn.Module) -> None:
            super().__init__()
            self.base = base_model

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            return {"out": self.base(x)}

    return WrappedModel(model).to(device)
