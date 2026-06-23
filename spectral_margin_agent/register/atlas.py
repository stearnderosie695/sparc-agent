from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from spectral_margin_agent.register.indices import haemoglobin_index, nerve_mask, vessel_mask
from spectral_margin_agent.register.proximity import proximity_field

ATLAS_LABELS: tuple[str, ...] = ("organ_boundary", "vessel", "nerve")


@dataclass
class AtlasOut:
    mask: torch.Tensor
    proximity: torch.Tensor
    label: torch.Tensor
    confidence: torch.Tensor


def _ring_template(size: int) -> torch.Tensor:
    ys = torch.linspace(-1, 1, size).view(-1, 1)
    xs = torch.linspace(-1, 1, size).view(1, -1)
    radius = torch.sqrt(ys**2 + xs**2)
    return ((radius > 0.55) & (radius < 0.8)).float()


class AtlasReasoner(nn.Module):
    template: torch.Tensor

    def __init__(self, template_size: int = 48, k_residual: float = 4.0) -> None:
        super().__init__()
        self.k_residual = k_residual
        template = _ring_template(template_size).view(1, 1, template_size, template_size)
        self.register_buffer("template", template)
        self.affine = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 4),
        )

    def _theta(self, params: torch.Tensor) -> torch.Tensor:
        tx, ty, log_s, rot = params.unbind(dim=-1)
        scale = torch.exp(0.3 * torch.tanh(log_s))
        angle = 0.3 * torch.tanh(rot)
        cos, sin = torch.cos(angle), torch.sin(angle)
        row0 = torch.stack([scale * cos, -scale * sin, torch.tanh(tx)], dim=-1)
        row1 = torch.stack([scale * sin, scale * cos, torch.tanh(ty)], dim=-1)
        return torch.stack([row0, row1], dim=1)

    def forward(self, cube: torch.Tensor, wl: torch.Tensor) -> AtlasOut:
        n, _, height, width = cube.shape
        structure = haemoglobin_index(cube, wl)
        lo = structure.amin(dim=(-2, -1), keepdim=True)
        hi = structure.amax(dim=(-2, -1), keepdim=True)
        structure_norm = (structure - lo) / (hi - lo + 1e-4)

        params = self.affine(structure_norm.unsqueeze(1))
        theta = self._theta(params)
        grid = F.affine_grid(theta, [n, 1, height, width], align_corners=False)
        warped = F.grid_sample(
            self.template.expand(n, 1, -1, -1), grid, align_corners=False
        ).squeeze(1)

        vmask = vessel_mask(cube, wl).float()
        nmask = nerve_mask(cube, wl).float()
        mask = torch.clamp(warped + vmask + nmask, 0.0, 1.0)

        residual = ((warped - structure_norm) ** 2).mean(dim=(-2, -1))
        confidence = torch.exp(-self.k_residual * residual)

        coverage = torch.stack(
            [warped.mean(dim=(-2, -1)), vmask.mean(dim=(-2, -1)), nmask.mean(dim=(-2, -1))], dim=-1
        )
        label = coverage.argmax(dim=-1)
        proximity = proximity_field(mask > 0.5)
        return AtlasOut(mask=mask, proximity=proximity, label=label, confidence=confidence)
