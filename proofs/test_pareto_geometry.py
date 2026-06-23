from __future__ import annotations

import torch

from spectral_margin_agent.dodge_burn.hypernet import sample_preferences
from spectral_margin_agent.dodge_burn.pareto import dominates, hypervolume, nondominated


def test_nondominated_are_truly_nondominated():
    generator = torch.Generator().manual_seed(0)
    for _ in range(200):
        size = int(torch.randint(1, 12, (1,), generator=generator))
        points = torch.rand(size, 2, generator=generator)
        keep = nondominated(points)
        for i in range(size):
            if keep[i]:
                assert not any(dominates(points[j], points[i]) for j in range(size) if j != i)


def test_hypervolume_monotone_under_addition():
    generator = torch.Generator().manual_seed(1)
    ref = torch.zeros(2)
    for _ in range(200):
        size = int(torch.randint(1, 8, (1,), generator=generator))
        base = torch.rand(size, 2, generator=generator)
        extra = torch.tensor([[1.5, 1.5]])
        augmented = torch.cat([base, extra], dim=0)
        assert hypervolume(augmented, ref) >= hypervolume(base, ref) - 1e-6


def test_known_hypervolume_values():
    ref = torch.zeros(2)
    assert abs(hypervolume(torch.tensor([[2.0, 1.0], [1.0, 2.0]]), ref) - 3.0) < 1e-5
    ref3 = torch.zeros(3)
    assert abs(hypervolume(torch.tensor([[1.0, 1.0, 1.0]]), ref3) - 1.0) < 1e-5


def test_preferences_live_on_simplex():
    generator = torch.Generator().manual_seed(0)
    prefs = sample_preferences([1.0, 1.0, 1.0], 256, generator)
    assert torch.all(prefs >= 0)
    assert torch.allclose(prefs.sum(dim=-1), torch.ones(256), atol=1e-5)
