from __future__ import annotations

import torch

from spectral_margin_agent.dodge_burn.hypernet import sample_preferences
from spectral_margin_agent.dodge_burn.pareto import hypervolume_mc

_MC_SAMPLES = 40000
_MC_SEED = 17


def octant_front(preferences: torch.Tensor, radius: float) -> torch.Tensor:
    direction = preferences / preferences.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return radius * direction


def reference_hypervolume(radius: float, ref: torch.Tensor) -> float:
    generator = torch.Generator().manual_seed(0)
    dense = octant_front(sample_preferences([1.0, 1.0, 1.0], 8192, generator), radius)
    upper = ref.new_full((3,), radius)
    return hypervolume_mc(dense, ref, upper, _MC_SAMPLES, _MC_SEED)


def regret_sequence(
    radius: float, ref: torch.Tensor, counts: list[int], seed: int
) -> list[tuple[int, float]]:
    true_hv = reference_hypervolume(radius, ref)
    upper = ref.new_full((3,), radius)
    out = []
    for n in counts:
        generator = torch.Generator().manual_seed(seed + n)
        prefs = sample_preferences([1.0, 1.0, 1.0], n, generator)
        points = octant_front(prefs, radius)
        approx = hypervolume_mc(points, ref, upper, _MC_SAMPLES, _MC_SEED)
        out.append((n, max(true_hv - approx, 0.0)))
    return out
