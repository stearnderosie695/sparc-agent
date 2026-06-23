from __future__ import annotations

import torch

from spectral_margin_agent.register.proximity import overlap_fraction


def critical_preservation(contour: torch.Tensor, critical: torch.Tensor) -> float:
    return float(1.0 - overlap_fraction(contour, critical))


def violation_rate(
    contours: list[torch.Tensor], criticals: list[torch.Tensor], threshold: float = 0.05
) -> float:
    flags = [float(overlap_fraction(c, m)) >= threshold for c, m in zip(contours, criticals)]
    if not flags:
        return 0.0
    return float(sum(flags) / len(flags))


def constraint_satisfaction(objectives: torch.Tensor, eps_c: float) -> float:
    satisfied = objectives[:, 1] >= (1.0 - eps_c)
    return float(satisfied.float().mean())


def mean_pareto_margin(margins_mm: list[float]) -> float:
    if not margins_mm:
        return 0.0
    return float(sum(margins_mm) / len(margins_mm))
