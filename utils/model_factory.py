from __future__ import annotations

import re
from typing import Final

import torch
import torch.nn as nn

from utils.pretrained_unet import get_model as get_unet_model


SUPPORTED_MODELS: Final[tuple[str, ...]] = ("unet",)
_MODEL_ALIASES: Final[dict[str, str]] = {
    "unet": "unet",
    "u_net": "unet",
    "efficientnet_b3_unet": "unet",
    "unet_efficientnet_b3": "unet",
    "unet_efficientnetb3": "unet",
}
_EXPERIMENT_TOKENS: Final[dict[str, str]] = {
    "unet": "unet_efficientnetb3",
}


def _model_lookup_key(model_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model_name.strip().lower()).strip("_")


def normalize_model_name(model_name: str) -> str:
    key = _model_lookup_key(model_name)
    if key in _MODEL_ALIASES:
        return _MODEL_ALIASES[key]

    choices = ", ".join(SUPPORTED_MODELS)
    raise ValueError(f"Unsupported model_name '{model_name}'. Expected one of: {choices}")


def model_experiment_token(model_name: str) -> str:
    return _EXPERIMENT_TOKENS[normalize_model_name(model_name)]


def get_model(
    device: torch.device | str,
    model_name: str = "unet",
    in_channels: int = 1,
    out_channels: int = 1,
) -> nn.Module:
    normalized_name = normalize_model_name(model_name)

    if normalized_name == "unet":
        if in_channels != 1 or out_channels != 1:
            raise ValueError("The configured U-Net factory currently supports only in_channels=1 and out_channels=1")
        return get_unet_model(device)

    raise AssertionError(f"Unhandled model name: {normalized_name}")
