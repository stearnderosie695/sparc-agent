from __future__ import annotations

import torch


def dominates(a: torch.Tensor, b: torch.Tensor) -> bool:
    return bool((a >= b).all() and (a > b).any())


def nondominated(points: torch.Tensor) -> torch.Tensor:
    geq = (points.unsqueeze(0) >= points.unsqueeze(1)).all(dim=-1)
    gt = (points.unsqueeze(0) > points.unsqueeze(1)).any(dim=-1)
    dominated_by = geq & gt
    dominated_by.fill_diagonal_(False)
    return ~dominated_by.any(dim=1)


def hv2d(points: torch.Tensor, ref: torch.Tensor) -> float:
    viable = points[(points > ref).all(dim=-1)]
    if viable.shape[0] == 0:
        return 0.0
    front = viable[nondominated(viable)]
    order = torch.argsort(front[:, 0], descending=True)
    front = front[order]
    prev_y = ref[1]
    volume = 0.0
    for x, y in front:
        volume += float((x - ref[0]) * (y - prev_y))
        prev_y = y
    return volume


def hypervolume_mc(
    points: torch.Tensor, ref: torch.Tensor, upper: torch.Tensor, samples: int, seed: int
) -> float:
    generator = torch.Generator().manual_seed(seed)
    unit = torch.rand(samples, ref.shape[0], generator=generator)
    probes = ref + unit * (upper - ref)
    box = float(torch.prod(upper - ref))
    dominated = (points.unsqueeze(0) >= probes.unsqueeze(1)).all(dim=-1).any(dim=-1)
    return box * float(dominated.float().mean())


def hypervolume(points: torch.Tensor, ref: torch.Tensor) -> float:
    if points.shape[-1] == 2:
        return hv2d(points, ref)
    viable = points[(points > ref).all(dim=-1)]
    if viable.shape[0] == 0:
        return 0.0
    levels = torch.unique(viable[:, 2], sorted=True).flip(0)
    total = 0.0
    for k in range(levels.shape[0]):
        upper = levels[k]
        lower = levels[k + 1] if k + 1 < levels.shape[0] else ref[2]
        active = viable[viable[:, 2] >= upper]
        area = hv2d(active[:, :2], ref[:2])
        total += area * float(upper - lower)
    return total
