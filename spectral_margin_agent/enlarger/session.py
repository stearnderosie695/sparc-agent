from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from spectral_margin_agent.emulsion.encoder import EncoderOut
from spectral_margin_agent.register.atlas import AtlasOut

TOOL_NAMES: tuple[str, ...] = (
    "spec_cls",
    "atlas_lookup",
    "vasc_det",
    "pareto_sample",
    "plan_verify",
)


def dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    kernel = 2 * radius + 1
    return F.max_pool2d(mask.float().unsqueeze(0), kernel, stride=1, padding=radius).squeeze(0)


def contour_from(tumour_map: torch.Tensor, radius: int, threshold: float) -> torch.Tensor:
    core = (tumour_map >= threshold).float()
    return (dilate(core, radius) > 0.5).float()


@dataclass
class Session:
    cube: torch.Tensor
    wl: torch.Tensor
    encoder_out: EncoderOut
    pref: torch.Tensor
    pixel_mm: float = 0.4
    tumour_map: torch.Tensor | None = None
    c_spec: float = 0.0
    atlas: AtlasOut | None = None
    vessels: torch.Tensor | None = None
    contour: torch.Tensor | None = None
    margin_radius: int = 2
    history: list[int] = field(default_factory=list)
    confidences: dict[str, float] = field(default_factory=dict)

    def critical_mask(self) -> torch.Tensor:
        spatial = self.cube.shape[-2:]
        mask = torch.zeros(spatial)
        if self.atlas is not None:
            mask = torch.clamp(mask + self.atlas.mask.squeeze(0), 0.0, 1.0)
        if self.vessels is not None:
            mask = torch.clamp(mask + self.vessels.squeeze(0).float(), 0.0, 1.0)
        return mask

    def achieved_margin(self) -> float:
        return self.margin_radius * self.pixel_mm
