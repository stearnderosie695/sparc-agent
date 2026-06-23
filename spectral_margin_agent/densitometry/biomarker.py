from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from scipy.stats import spearmanr

from spectral_margin_agent.densitometry.classification import auroc
from spectral_margin_agent.reels.grid import window_mask

ScoreFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _occlude(cube: torch.Tensor, band_mask: torch.Tensor) -> torch.Tensor:
    occluded = cube.clone()
    fill = cube.mean(dim=(0, 2, 3), keepdim=True)
    occluded[:, band_mask] = fill[:, band_mask]
    return occluded


def occlusion_importance(
    score_fn: ScoreFn,
    cube: torch.Tensor,
    wl: torch.Tensor,
    labels: np.ndarray,
    n_groups: int,
) -> tuple[np.ndarray, np.ndarray]:
    base = score_fn(cube, wl).reshape(-1).cpu().numpy()
    base_auc = auroc(labels, base)
    edges = torch.linspace(0, wl.shape[0], n_groups + 1).long()
    centers = np.empty(n_groups)
    drops = np.empty(n_groups)
    for g in range(n_groups):
        band_mask = torch.zeros(wl.shape[0], dtype=torch.bool)
        start, stop = int(edges[g]), max(int(edges[g + 1]), int(edges[g]) + 1)
        band_mask[start:stop] = True
        centers[g] = float(wl[start:stop].mean())
        scores = score_fn(_occlude(cube, band_mask), wl).reshape(-1).cpu().numpy()
        drops[g] = base_auc - auroc(labels, scores)
    return centers, drops


def window_concentration(
    centers: np.ndarray, importance: np.ndarray, center: float, halfwidth: float
) -> float:
    inside = (centers >= center - halfwidth) & (centers <= center + halfwidth)
    if inside.sum() == 0:
        return 0.0
    uniform = importance.mean()
    if abs(uniform) < 1e-9:
        return 0.0
    return float(importance[inside].mean() / uniform)


def masked_auroc_drop(
    score_fn: ScoreFn,
    cube: torch.Tensor,
    wl: torch.Tensor,
    labels: np.ndarray,
    center: float,
    halfwidth: float,
) -> float:
    base = score_fn(cube, wl).reshape(-1).cpu().numpy()
    band_mask = window_mask(wl, center, halfwidth)
    scores = score_fn(_occlude(cube, band_mask), wl).reshape(-1).cpu().numpy()
    return auroc(labels, base) - auroc(labels, scores)


def profile_rank_correlation(profile_a: np.ndarray, profile_b: np.ndarray) -> float:
    return float(spearmanr(profile_a, profile_b).statistic)
