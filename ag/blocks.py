from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SiLU(nn.Module):

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x)


class GroupNorm32(nn.GroupNorm):

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


def normalization(channels: int) -> GroupNorm32:
    return GroupNorm32(32, channels)


def conv3d(in_channels: int, out_channels: int, *, stride: int = 1) -> nn.Conv3d:
    return nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)


def conv1x1x1(in_channels: int, out_channels: int) -> nn.Conv3d:
    return nn.Conv3d(in_channels, out_channels, kernel_size=1)


def zero_module(module: nn.Module) -> nn.Module:
    for parameter in module.parameters():
        parameter.detach().zero_()
    return module


def zero_linear(in_features: int, out_features: int) -> nn.Linear:
    with torch.random.fork_rng(devices=[]):
        return zero_module(nn.Linear(in_features, out_features))


class TimeFourierEmbedding(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.t_max = 64.0
        self.num_bands = 4
        self.mlp = nn.Sequential(nn.Linear(10, 128), SiLU(), nn.Linear(128, 128))

    def forward(self, delta_t: Tensor) -> Tensor:
        dt = delta_t.float().reshape(-1, 1)
        normalized = (dt / self.t_max).clamp(0.0, 1.0)
        parts = [normalized, torch.log1p(dt)]
        for band in range(self.num_bands):
            arg = (2**band) * math.pi * normalized
            parts.extend([torch.sin(arg), torch.cos(arg)])
        return self.mlp(torch.cat(parts, dim=-1))


class TimeCondition(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.time_norm = nn.LayerNorm(128)

    def forward(self, embedding: Tensor) -> Tensor:
        return self.time_norm(embedding.float())


class CMSRBlock3D(nn.Module):

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.branch_channels = (channels // 2, channels // 2)
        self.norm1 = normalization(channels)
        self.activation1 = SiLU()
        self.local_branch = conv3d(channels, channels // 2)
        self.dilated_branch = nn.Conv3d(
            channels, channels // 2, kernel_size=5, padding=2
        )
        self.norm2 = normalization(channels)
        self.activation2 = SiLU()
        self.output_conv = zero_module(conv3d(channels, channels))
        self.condition_projection = zero_linear(128, channels)

    def forward(self, features: Tensor, condition: Tensor) -> Tensor:
        logits = self.condition_projection(condition).to(dtype=features.dtype)
        scales = 1.0 + 0.1 * torch.tanh(logits)
        hidden = self.activation1(self.norm1(features))
        branches = [self.local_branch(hidden), self.dilated_branch(hidden)]
        normalized_branches = torch.split(
            self.norm2(torch.cat(branches, dim=1)), self.branch_channels, dim=1
        )
        branch_scales = [
            scale.view(-1, channels, 1, 1, 1)
            for scale, channels in zip(
                torch.split(scales, self.branch_channels, dim=1), self.branch_channels
            )
        ]
        conditioned = [
            branch * scale for branch, scale in zip(normalized_branches, branch_scales)
        ]
        response = self.output_conv(self.activation2(torch.cat(conditioned, dim=1)))
        return features + response


class CMSRBackboneBlock3D(nn.Module):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.input_projection = (
            conv1x1x1(in_channels, out_channels)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.cmsr = CMSRBlock3D(out_channels)

    def forward(self, features: Tensor, condition: Tensor) -> Tensor:
        return self.cmsr(self.input_projection(features), condition)


class CMSREncoderBlock3D(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, *, stride: int) -> None:
        super().__init__()
        self.cmsr1 = CMSRBackboneBlock3D(in_channels, out_channels)
        self.down = conv3d(out_channels, out_channels, stride=stride)

    def forward(self, features: Tensor, condition: Tensor) -> tuple[Tensor, Tensor]:
        skip = self.cmsr1(features, condition)
        return self.down(skip), skip


class CTEncoder3D(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.input_conv = nn.Sequential(conv3d(1, 32), normalization(32), SiLU())
        self.blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        channels = 32
        for level, out_channels in enumerate((32, 64, 64, 64)):
            self.blocks.append(
                nn.Sequential(
                    conv3d(channels, out_channels),
                    normalization(out_channels),
                    SiLU(),
                    conv3d(out_channels, out_channels),
                    normalization(out_channels),
                    SiLU(),
                )
            )
            channels = out_channels
            if level < 3:
                self.downs.append(conv3d(channels, channels, stride=2))

    def forward(self, x: Tensor) -> list[Tensor]:
        x = self.input_conv(x)
        features = []
        for index, block in enumerate(self.blocks):
            x = block(x)
            features.append(x)
            if index < len(self.blocks) - 1:
                x = self.downs[index](x)
        return features


class PlainSpatialBlock3D(nn.Sequential):

    def __init__(self, channels: int) -> None:
        super().__init__(conv3d(channels, channels), normalization(channels), SiLU())


class ThreeWayConcatFusion3D(nn.Module):

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.fusion = nn.Sequential(
            normalization(3 * channels), SiLU(), conv3d(3 * channels, channels)
        )

    def forward(self, ct: Tensor, mask: Tensor, morphology: Tensor) -> Tensor:
        return self.fusion(torch.cat([ct, mask, morphology], dim=1))


class CMSRDecoderBlock3D(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, *, upsample: bool) -> None:
        super().__init__()
        self.up = (
            nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2, stride=2)
            if upsample
            else conv1x1x1(in_channels, in_channels)
        )
        self.projection = conv1x1x1(2 * in_channels, out_channels)
        self.spatial_mix = CMSRBlock3D(out_channels)

    def forward(self, x: Tensor, skip: Tensor, condition: Tensor) -> Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = self.projection(torch.cat([x, skip], dim=1))
        return self.spatial_mix(x, condition)
