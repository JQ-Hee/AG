from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SpatialTransformer3D(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_grid", torch.empty(0), persistent=False)
        self._grid_shape = None

    def _base_grid(self, shape: tuple[int, ...], device: torch.device, dtype: torch.dtype) -> Tensor:
        if self._grid_shape == shape and self._grid.device == device and self._grid.dtype == dtype:
            return self._grid
        depth, height, width = shape
        z = torch.linspace(-1.0, 1.0, depth, device=device, dtype=dtype)
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        grid = torch.stack([xx, yy, zz], dim=-1).unsqueeze(0)
        self._grid = grid
        self._grid_shape = shape
        return grid

    def forward(self, src: Tensor, flow: Tensor) -> Tensor:
        if src.ndim != 5 or flow.ndim != 5:
            raise ValueError("src and flow must be 5D tensors")
        if flow.shape[1] != 3:
            raise ValueError("flow must have three channels ordered as (dz, dy, dx)")
        if src.shape[0] != flow.shape[0] or src.shape[2:] != flow.shape[2:]:
            raise ValueError("src and flow must have matching batch and spatial shapes")
        if src.device != flow.device:
            raise ValueError("src and flow must be on the same device")
        depth, height, width = src.shape[2:]
        scale_z = 2.0 / max(depth - 1, 1)
        scale_y = 2.0 / max(height - 1, 1)
        scale_x = 2.0 / max(width - 1, 1)
        flow_norm = torch.stack(
            [flow[:, 2] * scale_x, flow[:, 1] * scale_y, flow[:, 0] * scale_z], dim=-1
        )
        grid = self._base_grid((depth, height, width), src.device, src.dtype)
        return F.grid_sample(
            src, grid + flow_norm, mode="bilinear", padding_mode="border", align_corners=True
        )
