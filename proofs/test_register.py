from __future__ import annotations

import torch

from spectral_margin_agent.reels.synth import plant_cube
from spectral_margin_agent.register.atlas import AtlasReasoner
from spectral_margin_agent.register.indices import skeleton_edges, thin, vessel_mask
from spectral_margin_agent.register.proximity import overlap_fraction, proximity_field


def test_atlas_reasoner_outputs():
    torch.manual_seed(0)
    reasoner = AtlasReasoner()
    planted = plant_cube(4, 60, 400.0, 1000.0, 40, centre=1)
    out = reasoner(planted.cube.unsqueeze(0), planted.wavelengths)
    assert out.mask.shape == (1, 40, 40)
    assert 0.0 <= float(out.confidence) <= 1.0
    assert float(out.proximity.min()) >= 0.0
    assert int(out.label) in (0, 1, 2)


def test_vessel_detection_and_skeleton():
    planted = plant_cube(2, 60, 400.0, 1000.0, 40, centre=1)
    mask = vessel_mask(planted.cube.unsqueeze(0), planted.wavelengths)
    assert 0.0 < float(mask.float().mean()) < 0.5
    edges = skeleton_edges(thin(mask))
    assert edges.shape[0] == 2


def test_proximity_zero_on_structure():
    mask = torch.zeros(1, 20, 20, dtype=torch.bool)
    mask[:, 10, 10] = True
    field = proximity_field(mask)
    assert float(field[0, 10, 10]) == 0.0
    assert float(field[0, 0, 0]) > 0.0


def test_overlap_fraction_known():
    contour = torch.zeros(1, 10, 10)
    contour[:, :5, :] = 1.0
    mask = torch.zeros(1, 10, 10)
    mask[:, :5, :5] = 1.0
    assert abs(float(overlap_fraction(contour, mask)) - 0.5) < 1e-5
