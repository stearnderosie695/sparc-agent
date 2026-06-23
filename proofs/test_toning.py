from __future__ import annotations

import torch

from spectral_margin_agent.reels.chromo import CHROMOPHORES, mix_reflectance
from spectral_margin_agent.reels.grid import make_grid
from spectral_margin_agent.toning.gate import confidence_gated_augment, gap_closure
from spectral_margin_agent.toning.reparam import estimate_concentrations, reparameterize


def _recovered_water(value: float) -> float:
    wl = make_grid(400.0, 1000.0, 120)
    concentrations = {name: torch.full((1, 8, 8), 0.1) for name in CHROMOPHORES}
    concentrations["water"] = torch.full((1, 8, 8), value)
    reflectance = mix_reflectance(concentrations, wl).movedim(-1, 1)
    return float(estimate_concentrations(reflectance, wl)["water"].mean())


def test_concentration_estimate_responds_to_planted_water():
    assert _recovered_water(0.9) > _recovered_water(0.2) + 0.2


def test_gate_only_changes_low_confidence_pixels():
    wl = make_grid(400.0, 1000.0, 60)
    cube = torch.rand(1, 60, 16, 16) * 0.5 + 0.25
    confidence = torch.ones(1, 16, 16)
    confidence[:, :8] = 0.1
    shift = {"water": 0.5}
    augmented = confidence_gated_augment(cube, wl, confidence, 0.5, shift)
    changed = (augmented - cube).abs().sum(dim=1).squeeze(0)
    assert changed[:8].sum() > 0
    assert torch.allclose(changed[8:], torch.zeros_like(changed[8:]), atol=1e-5)


def test_reparameterize_shifts_spectrum():
    wl = make_grid(400.0, 1000.0, 60)
    cube = torch.rand(1, 60, 12, 12) * 0.4 + 0.3
    augmented = reparameterize(cube, wl, {"water": 0.4, "lipid": -0.2})
    assert (augmented - cube).abs().mean() > 1e-3


def test_gap_closure_fraction():
    assert abs(gap_closure(0.5, 0.7, 1.0) - 0.4) < 1e-6
