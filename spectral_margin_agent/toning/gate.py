from __future__ import annotations

import torch

from spectral_margin_agent.toning.reparam import reparameterize


def confidence_gated_augment(
    cube: torch.Tensor,
    wl: torch.Tensor,
    confidence: torch.Tensor,
    threshold: float,
    shift: dict[str, float],
) -> torch.Tensor:
    augmented = reparameterize(cube, wl, shift)
    low = (confidence < threshold).unsqueeze(1).to(cube.dtype)
    return low * augmented + (1.0 - low) * cube


def gap_closure(source_only: float, source_plus: float, target: float) -> float:
    gap = target - source_only
    if abs(gap) < 1e-8:
        return 0.0
    return (source_plus - source_only) / gap
