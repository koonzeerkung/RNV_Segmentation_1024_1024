from __future__ import annotations

import re
from typing import Final

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

SUPPORTED_MODELS: Final[tuple[str, ...]] = ("unet", "unetplusplus", "deeplabv3plus", "deeplabv3")
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


def normalize_encoder_weights(encoder_weights: str | None) -> str | None:
    if encoder_weights is None:
        return None

    normalized = encoder_weights.strip()
    if normalized.lower() in {"", "none", "null", "false", "no"}:
        return None
    return normalized


def validate_model_backbone(model_name: str, backbone: str) -> None:
    model_name = normalize_model_name(model_name)
    backbone = normalize_backbone_name(backbone)

    if model_name.startswith("deeplab") and backbone in {"densenet169", "inceptionv4"}:
        raise ValueError(
            f"{model_name} cannot use backbone '{backbone}' in segmentation_models_pytorch because that encoder "
            "does not support the dilated mode required by DeepLab. Use backbone='efficientnet-b3' for DeepLab, "
            "or use model_name='unet'/'unetplusplus' with this backbone."
        )


def model_experiment_token(model_name: str, backbone: str = "efficientnet-b3") -> str:
    model_name = normalize_model_name(model_name)
    backbone = normalize_backbone_name(backbone)
    backbone_token = backbone.replace("-", "")
    return f"{model_name}_{backbone_token}"


def get_model(
    device: torch.device | str,
    model_name: str = "unet",
    backbone: str = "efficientnet-b3",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 1,
    out_channels: int = 1,
) -> nn.Module:
    normalized_name = normalize_model_name(model_name)
    normalized_backbone = normalize_backbone_name(backbone)
    normalized_encoder_weights = normalize_encoder_weights(encoder_weights)
    validate_model_backbone(normalized_name, normalized_backbone)

    model_class = _ARCHITECTURE_CLASSES[normalized_name]
    print(
        f"Loading model: {normalized_name} "
        f"(encoder: {normalized_backbone}, encoder_weights: {normalized_encoder_weights})"
    )

    model = model_class(
        encoder_name=normalized_backbone,
        encoder_weights=normalized_encoder_weights,
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
