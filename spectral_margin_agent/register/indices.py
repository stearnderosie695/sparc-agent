from __future__ import annotations

import torch
import torch.nn.functional as F

from spectral_margin_agent.reels.grid import window_mask


def _band_mean(
    cube: torch.Tensor, wl: torch.Tensor, center: float, halfwidth: float
) -> torch.Tensor:
    mask = window_mask(wl, center, halfwidth)
    if int(mask.sum()) == 0:
        nearest = int(torch.argmin((wl - center).abs()))
        return cube[:, nearest]
    return cube[:, mask].mean(dim=1)


def haemoglobin_index(cube: torch.Tensor, wl: torch.Tensor) -> torch.Tensor:
    absorb = 0.5 * (_band_mean(cube, wl, 542.0, 12.0) + _band_mean(cube, wl, 576.0, 12.0))
    baseline = _band_mean(cube, wl, 660.0, 20.0)
    return (baseline - absorb) / (baseline + absorb + 1e-4)


def nerve_index(cube: torch.Tensor, wl: torch.Tensor) -> torch.Tensor:
    lipid = _band_mean(cube, wl, 930.0, 18.0)
    reference = _band_mean(cube, wl, 860.0, 18.0)
    return (reference - lipid) / (reference + lipid + 1e-4)


def _quantile_threshold(field: torch.Tensor, q: float) -> torch.Tensor:
    flat = field.reshape(field.shape[0], -1)
    return torch.quantile(flat, q, dim=1).view(-1, 1, 1)


def vessel_mask(cube: torch.Tensor, wl: torch.Tensor, q: float = 0.9) -> torch.Tensor:
    field = haemoglobin_index(cube, wl)
    return field >= _quantile_threshold(field, q)


def nerve_mask(cube: torch.Tensor, wl: torch.Tensor, q: float = 0.92) -> torch.Tensor:
    field = nerve_index(cube, wl)
    return field >= _quantile_threshold(field, q)


def thin(mask: torch.Tensor) -> torch.Tensor:
    f = mask.float().unsqueeze(1)
    eroded = -F.max_pool2d(-f, kernel_size=3, stride=1, padding=1)
    border = F.max_pool2d(f, kernel_size=3, stride=1, padding=1) - eroded
    skeleton = (f > 0) & (border > 0)
    return skeleton.squeeze(1)


def skeleton_edges(skeleton: torch.Tensor) -> torch.Tensor:
    coords = torch.nonzero(skeleton, as_tuple=False)
    if coords.shape[0] < 2:
        return coords.new_zeros((2, 0))
    pos = coords[:, 1:].float()
    dist = torch.cdist(pos, pos)
    adjacency = (dist > 0) & (dist <= 1.5)
    edges = torch.nonzero(adjacency, as_tuple=False).t()
    return edges
