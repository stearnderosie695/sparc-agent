from __future__ import annotations

import numpy as np
import torch

from spectral_margin_agent.contact_sheet.evaluate import MarginReport, evaluate_margin
from spectral_margin_agent.densitometry.biomarker import (
    masked_auroc_drop,
    occlusion_importance,
    window_concentration,
)
from spectral_margin_agent.densitometry.surgical import (
    constraint_satisfaction,
    critical_preservation,
    mean_pareto_margin,
    violation_rate,
)
from spectral_margin_agent.dodge_burn.hypernet import sample_preferences
from spectral_margin_agent.dodge_burn.pareto import hypervolume
from spectral_margin_agent.enlarger.orchestrator import SparcAgent
from spectral_margin_agent.reels.corpora import CORPORA, build_split
from spectral_margin_agent.reels.spool import build_loader
from spectral_margin_agent.safelight.dials import Recipe


def cross_tissue(
    agent: SparcAgent, recipe: Recipe, corpora: list[str], device: torch.device
) -> dict[str, MarginReport]:
    out = {}
    for corpus in corpora:
        stack = build_split(
            corpus, "test", 8, recipe.run.seed, recipe.stock.root, recipe.stock.tile
        )
        loader = build_loader(stack, recipe.stock.micro_batch, 0, recipe.run.seed, False)
        out[corpus] = evaluate_margin(
            agent, loader, recipe.enlarger.tumour_index, recipe.enlarger.t_calib, device
        )
    return out


def pareto_sweep(
    agent: SparcAgent, recipe: Recipe, n_pref: int, device: torch.device
) -> dict[str, float]:
    stack = build_split(
        recipe.stock.corpus, "test", 6, recipe.run.seed, recipe.stock.root, recipe.stock.tile
    )
    generator = torch.Generator().manual_seed(recipe.run.seed)
    prefs = sample_preferences(recipe.enlarger.pareto.dirichlet_alpha, n_pref, generator)
    wl = stack[0].wavelengths.to(device)
    objectives, contours, criticals, margins = [], [], [], []
    for index in range(min(n_pref, len(stack))):
        cube = stack[index].cube.unsqueeze(0).to(device)
        result = agent.plan(cube, wl, prefs[index].view(1, 3).to(device), explore=False)
        objectives.append(result.objective.detach())
        contours.append(result.contour)
        criticals.append(result.critical)
        margins.append(result.margin_mm)
    stacked = torch.stack(objectives)
    return {
        "hypervolume": hypervolume(stacked, stacked.new_zeros(3)),
        "preservation": float(
            np.mean([critical_preservation(c, m) for c, m in zip(contours, criticals)])
        ),
        "pareto_margin_mm": mean_pareto_margin(margins),
        "constraint_satisfaction": constraint_satisfaction(stacked, recipe.enlarger.pareto.eps_c),
        "violation_rate": violation_rate(contours, criticals),
    }


def biomarker(
    agent: SparcAgent, recipe: Recipe, corpus: str, device: torch.device
) -> dict[str, float]:
    spec = CORPORA[corpus]
    stack = build_split(corpus, "test", 4, recipe.run.seed, recipe.stock.root, recipe.stock.tile)
    tumour_index = recipe.enlarger.tumour_index

    def score_fn(cube: torch.Tensor, wl: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.softmax(agent.encoder(cube, wl).dense_logits, dim=1)[0, tumour_index]

    frame = stack[0]
    cube = frame.cube.unsqueeze(0).to(device)
    wl = frame.wavelengths.to(device)
    labels = (frame.label == tumour_index).reshape(-1).cpu().numpy().astype(int)
    centers, drops = occlusion_importance(score_fn, cube, wl, labels, min(16, spec.n_bands))
    importance = np.abs(drops)
    return {
        "conc_975": window_concentration(centers, importance, 975.0, 25.0),
        "mask_auroc_drop": masked_auroc_drop(score_fn, cube, wl, labels, 975.0, 25.0),
        "peak_nm": float(centers[int(np.argmax(importance))]),
    }
