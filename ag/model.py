from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .blocks import (
    CMSRBackboneBlock3D,
    CMSRDecoderBlock3D,
    CMSREncoderBlock3D,
    CTEncoder3D,
    PlainSpatialBlock3D,
    SiLU,
    ThreeWayConcatFusion3D,
    TimeCondition,
    TimeFourierEmbedding,
    conv1x1x1,
    conv3d,
    normalization,
    zero_module,
)
from .spatial import SpatialTransformer3D


class InferenceModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.ct_encoder = CTEncoder3D()
        self.time_embed = TimeFourierEmbedding()
        self.cmsr_conditioner = TimeCondition()
        self.input_conv = nn.Sequential(conv3d(2, 32), normalization(32), SiLU())
        widths = (32, 64, 64, 64)
        self.encoders = self._conditioned_encoder()
        self.fusion_blocks = nn.ModuleList(ThreeWayConcatFusion3D(w) for w in widths)
        self.decoders = nn.ModuleList(
            [
                CMSRDecoderBlock3D(64, 64, upsample=False),
                CMSRDecoderBlock3D(64, 64, upsample=True),
                CMSRDecoderBlock3D(64, 32, upsample=True),
                CMSRDecoderBlock3D(32, 32, upsample=True),
            ]
        )
        self.flow_head = nn.Sequential(
            normalization(32), SiLU(), zero_module(conv1x1x1(32, 3))
        )
        self.disp_head = self.flow_head
        self.flow_scale_epsilon = 1e-6
        initial_positive_scale = 8.0 - self.flow_scale_epsilon
        alpha_init = initial_positive_scale + math.log(-math.expm1(-initial_positive_scale))
        self.flow_scale_alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        self.warp_ct = SpatialTransformer3D()
        self.warp_mask = SpatialTransformer3D()
        self.mask_input_conv = nn.Sequential(conv3d(1, 32), normalization(32), SiLU())
        self.mask_encoders = self._conditioned_encoder()
        self.ct_terminal = conv3d(64, 64)
        self.private_bottlenecks = nn.ModuleDict(
            {
                "ct": PlainSpatialBlock3D(64),
                "morphology": CMSRBackboneBlock3D(64, 64),
                "mask": CMSRBackboneBlock3D(64, 64),
            }
        )
        self.deep_fusion = ThreeWayConcatFusion3D(64)

    @staticmethod
    def _conditioned_encoder() -> nn.ModuleList:
        return nn.ModuleList(
            [
                CMSREncoderBlock3D(32, 32, stride=2),
                CMSREncoderBlock3D(32, 64, stride=2),
                CMSREncoderBlock3D(64, 64, stride=2),
                CMSREncoderBlock3D(64, 64, stride=1),
            ]
        )

    def current_displacement_scale(self) -> Tensor:
        return F.softplus(self.flow_scale_alpha) + self.flow_scale_epsilon

    def forward(self, ct_t0: Tensor, mask_t0: Tensor, delta_t: Tensor) -> dict[str, Tensor]:
        if ct_t0.ndim != 5 or ct_t0.shape[1] != 1 or ct_t0.shape != mask_t0.shape:
            raise ValueError("CT and mask must have identical [B,1,D,H,W] shapes")
        if ct_t0.shape[0] < 1:
            raise ValueError("the input batch must contain at least one sample")
        if not ct_t0.is_floating_point() or not mask_t0.is_floating_point():
            raise ValueError("CT and mask must be floating-point tensors")
        if ct_t0.dtype != mask_t0.dtype:
            raise ValueError("CT and mask must have the same dtype")
        if ct_t0.device != mask_t0.device or ct_t0.device != delta_t.device:
            raise ValueError("CT, mask, and delta_t must be on the same device")
        dt = delta_t.float().reshape(-1)
        if dt.numel() != ct_t0.shape[0] or not torch.isfinite(dt).all() or torch.any(dt < 1):
            raise ValueError("delta_t must contain one finite interval >= 1 per sample")
        if min(ct_t0.shape[2:]) < 32 or any(size % 16 for size in ct_t0.shape[2:]):
            raise ValueError("spatial dimensions must be >= 32 and divisible by 16")

        condition = self.cmsr_conditioner(self.time_embed(dt))
        ct_skips = self.ct_encoder(ct_t0)
        ct_deep = self.ct_terminal(ct_skips[-1])
        joint = self.input_conv(torch.cat([ct_t0, mask_t0], dim=1))
        mask = self.mask_input_conv(mask_t0)
        skips = []
        for level, (joint_encoder, mask_encoder) in enumerate(zip(self.encoders, self.mask_encoders)):
            joint, joint_skip = joint_encoder(joint, condition)
            mask, mask_skip = mask_encoder(mask, condition)
            skips.append(self.fusion_blocks[level](ct_skips[level], mask_skip, joint_skip))

        ct_deep = self.private_bottlenecks["ct"](ct_deep)
        joint = self.private_bottlenecks["morphology"](joint, condition)
        mask = self.private_bottlenecks["mask"](mask, condition)
        features = self.deep_fusion(ct_deep, mask, joint)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            features = decoder(features, skip, condition)
        scale = self.current_displacement_scale()
        flow = torch.tanh(self.disp_head(features)) * scale
        return {
            "pred_ct": self.warp_ct(ct_t0, flow),
            "pred_mask": self.warp_mask(mask_t0, flow).clamp(0, 1),
            "disp_bwd": flow,
            "displacement_scale": scale,
        }
