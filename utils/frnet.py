"""FRNet: a full-resolution network for OCTA vessel segmentation.

Faithful reimplementation of the official FRNet architecture from:

    H. Ning, C. Wang, X. Chen, S. Li, "An Accurate and Efficient Neural Network for
    OCTA Vessel Segmentation and a New Dataset", ICASSP 2024.
    Official code: https://github.com/nhjydywd/OCTA-FRNet

FRNet keeps feature maps at the input resolution throughout (no down/upsampling), which
preserves thin vessels that encoder-decoder models lose. The canonical model is a stack of
full-resolution blocks (a Recurrent-Residual init block followed by Recurrent-ConvNeXt
blocks), ending in an 11x11 convolution to ``out_channels * 4`` heads that are collapsed to
one channel by a per-pixel max (maxout). It has only ~100k parameters.

One deliberate adaptation for this pipeline: the official network applies ``Sigmoid`` before
the maxout, so it emits probabilities. This pipeline's loss/metrics expect raw logits. Because
sigmoid is monotonic, ``max`` over logits followed by ``sigmoid`` is identical to ``sigmoid``
followed by ``max``. So we drop the inner ``Sigmoid`` and take the maxout over logits; the
returned logit's sigmoid equals the original FRNet probability exactly, and BCEDiceLoss /
get_postprocessed_mask / dice_score_from_logits work unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # timm is already a pipeline dependency; tolerate either import path / absence.
    from timm.layers import DropPath
except Exception:  # pragma: no cover - depends on timm version
    try:
        from timm.models.layers import DropPath
    except Exception:  # pragma: no cover
        class DropPath(nn.Identity):  # type: ignore[no-redef]
            def __init__(self, drop_prob: float = 0.0) -> None:
                super().__init__()


class LayerNorm(nn.Module):
    """LayerNorm supporting channels_last (default) and channels_first layouts."""

    def __init__(self, normalized_shape: int, eps: float = 1e-6, data_format: str = "channels_last") -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ("channels_last", "channels_first"):
            raise NotImplementedError(self.data_format)
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class ConvBlock(nn.Module):
    """Base class so subclasses share the same constructor signature via init()."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, k_size: int = 3, dilation: int = 1) -> None:
        super().__init__()
        self.init(in_channels, out_channels, stride, k_size, dilation)

    def init(self, in_channels: int, out_channels: int, stride: int, k_size: int, dilation: int) -> None:
        raise NotImplementedError


class ResidualBlock(ConvBlock):
    expansion = 1

    def init(self, in_channels: int, out_channels: int, stride: int, k_size: int, dilation: int) -> None:
        p = k_size // 2 * dilation
        self.conv1 = nn.Conv2d(in_channels, out_channels, k_size, stride=stride, padding=p, bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, k_size, stride=1, padding=p, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class RecurrentBlock(nn.Module):
    """Recurrent convolution: the same conv weights are applied t times over (x + x1)."""

    def __init__(self, out_ch: int, k_size: int, t: int = 2, groups: int = 1) -> None:
        super().__init__()
        self.t = t
        self.out_ch = out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=k_size, stride=1, padding=k_size // 2, bias=False, groups=groups),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = None
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            x1 = self.conv(x + x1)
        return x1


class RRCNNBlock(ConvBlock):
    """Recurrent Residual CNN block used as the input (channel-changing) stage."""

    def init(self, in_channels: int, out_channels: int, stride: int, k_size: int, dilation: int) -> None:
        assert dilation == 1
        t = 2
        self.RCNN = nn.Sequential(
            RecurrentBlock(out_channels, k_size=k_size, t=t),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            RecurrentBlock(out_channels, k_size=k_size, t=t),
            nn.BatchNorm2d(out_channels),
        )
        self.Conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.Conv(x)
        x2 = self.RCNN(x1)
        return F.relu(x1 + x2)


class RecurrentConvNeXtBlock(nn.Module):
    """ConvNeXt block whose depthwise and pointwise convs are made recurrent."""

    def __init__(self, dim: int, drop_path: float = 0.0, layer_scale_init_value: float = 1e-6) -> None:
        super().__init__()
        self.dwconv = RecurrentBlock(dim, k_size=7, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = RecurrentBlock(dim, k_size=3)
        self.act = nn.GELU()
        self.pwconv2 = RecurrentBlock(dim, k_size=3)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # back to (N, C, H, W)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = x.permute(0, 2, 3, 1)
            x = self.gamma * x
            x = x.permute(0, 3, 1, 2)
        return inp + self.drop_path(x)


class FRNet(nn.Module):
    """Full-resolution network. Emits per-pixel logits (see module docstring for the
    sigmoid/maxout adaptation that keeps it compatible with this pipeline's logit-based
    loss and metrics)."""

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        ls_mid_ch: tuple[int, ...] = tuple([32] * 6),
        out_k_size: int = 11,
        k_size: int = 3,
        cls_init_block: type[nn.Module] = RRCNNBlock,
        cls_conv_block: type[nn.Module] = RecurrentConvNeXtBlock,
    ) -> None:
        super().__init__()
        self.dict_module = nn.ModuleDict()
        self.ls_mid_ch = ls_mid_ch

        ch1 = ch_in
        for i, ch2 in enumerate(ls_mid_ch):
            if ch1 != ch2:
                module = cls_init_block(ch1, ch2, k_size=k_size)
            elif cls_conv_block is RecurrentConvNeXtBlock:
                module = RecurrentConvNeXtBlock(dim=ch1, layer_scale_init_value=1)
            else:
                module = cls_conv_block(ch1, ch2, k_size=k_size)
            self.dict_module.add_module(f"conv{i}", module)
            ch1 = ch2

        # Final 11x11 conv to ch_out*4 maxout heads. No Sigmoid here: we keep logits and
        # take the max over logits, which equals the original (sigmoid then max) after a
        # downstream sigmoid.
        self.dict_module.add_module(
            "final", nn.Conv2d(ch1, ch_out * 4, out_k_size, padding=out_k_size // 2, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.ls_mid_ch)):
            x = self.dict_module[f"conv{i}"](x)
        x = self.dict_module["final"](x)
        return torch.max(x, dim=1, keepdim=True)[0]


def build_frnet(in_channels: int = 1, out_channels: int = 1) -> FRNet:
    """Build the canonical FRNet (RRCNN init block + Recurrent-ConvNeXt blocks)."""

    return FRNet(
        ch_in=in_channels,
        ch_out=out_channels,
        ls_mid_ch=tuple([32] * 6),
        out_k_size=11,
        k_size=3,
        cls_init_block=RRCNNBlock,
        cls_conv_block=RecurrentConvNeXtBlock,
    )


__all__ = ["FRNet", "build_frnet"]
