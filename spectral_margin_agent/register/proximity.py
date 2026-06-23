from __future__ import annotations

import torch
import torch.nn.functional as F


def proximity_field(mask: torch.Tensor, max_radius: int = 16) -> torch.Tensor:
    reached = mask.float().unsqueeze(1)
    distance = torch.zeros_like(reached)
    for _ in range(max_radius):
        distance = distance + (reached < 0.5).float()
        reached = F.max_pool2d(reached, kernel_size=3, stride=1, padding=1)
    return distance.squeeze(1)


def overlap_fraction(contour: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    inside = contour.float()
    hit = (inside * mask.float()).sum(dim=(-2, -1))
    total = inside.sum(dim=(-2, -1)).clamp_min(1.0)
    return hit / total
