from __future__ import annotations

import torch

from spectral_margin_agent.enlarger.ccatr import (
    ABSTAIN,
    ATLAS,
    CCATR,
    FUSE,
    SPECTRAL,
    quadrant_route,
)
from spectral_margin_agent.enlarger.reward import composite_reward


def test_plan_returns_valid_structures(agent, cube):
    tile, wl = cube
    pref = torch.tensor([[0.5, 0.3, 0.2]])
    result = agent.plan(tile, wl, pref, explore=False)
    assert result.contour.shape == tile.shape[-2:]
    assert 0.0 <= float(result.c_final) <= 1.0
    assert result.objective.shape == (3,)
    assert result.steps >= 1
    assert result.provenance.trace().shape[0] == 5


def test_quadrant_route_matches_proposition_one():
    assert quadrant_route(0.9, 0.9, 0.5, 0.5) == FUSE
    assert quadrant_route(0.9, 0.1, 0.5, 0.5) == SPECTRAL
    assert quadrant_route(0.1, 0.9, 0.5, 0.5) == ATLAS
    assert quadrant_route(0.1, 0.1, 0.5, 0.5) == ABSTAIN


def test_bayesian_average_is_convex_combination():
    weights = torch.tensor([[0.2, 0.3, 0.5]])
    confidences = torch.tensor([[0.4, 0.6, 0.8]])
    combined = float(CCATR.bayesian_average(weights, confidences))
    assert min(0.4, 0.6, 0.8) <= combined <= max(0.4, 0.6, 0.8)


def test_reward_rewards_clean_verified_plan():
    alpha = [1.0, 1.0, 0.05, 2.0]
    good = composite_reward(2.0, 1.0, 0.0, 3, True, alpha)
    bad = composite_reward(0.5, 1.0, 0.4, 12, False, alpha)
    assert good > bad
