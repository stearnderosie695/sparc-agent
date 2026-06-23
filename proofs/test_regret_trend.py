from __future__ import annotations

import math

import torch

from spectral_margin_agent.dodge_burn.regret import regret_sequence


def test_regret_decreases_with_samples():
    sequence = regret_sequence(1.0, torch.zeros(3), [16, 64, 256, 1024], seed=3)
    regrets = [r for _, r in sequence]
    for earlier, later in zip(regrets, regrets[1:]):
        assert later <= earlier + 0.03


def test_regret_respects_root_n_envelope():
    sequence = regret_sequence(1.0, torch.zeros(3), [64, 256, 1024], seed=5)
    scaled = [r * math.sqrt(n) for n, r in sequence]
    assert max(scaled) < 3.0
    assert min(scaled) > 0.0
