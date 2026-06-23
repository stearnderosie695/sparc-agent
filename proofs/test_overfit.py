from __future__ import annotations

import torch
from conftest import tiny_emulsion

from spectral_margin_agent.developing.objectives import segmentation_loss
from spectral_margin_agent.dodge_burn.hypernet import (
    ParetoHyperNet,
    psl_loss,
    sample_preferences,
)
from spectral_margin_agent.emulsion.encoder import Encoder
from spectral_margin_agent.reels.synth import plant_cube
from spectral_margin_agent.safelight.dials import ParetoDial


def test_hypernet_overfits_preference_direction():
    torch.manual_seed(0)
    net = ParetoHyperNet(ParetoDial(width=64, pref_embed=32))
    generator = torch.Generator().manual_seed(1)
    prefs = sample_preferences([1.0, 1.0, 1.0], 16, generator)
    context = torch.zeros(16, 8)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    first = float(psl_loss(net(prefs, context), prefs))
    for _ in range(250):
        optimizer.zero_grad()
        loss = psl_loss(net(prefs, context), prefs)
        loss.backward()
        optimizer.step()
    assert float(loss) < 0.02
    assert float(loss) < first


def test_encoder_overfits_single_tile():
    torch.manual_seed(0)
    encoder = Encoder(tiny_emulsion(), n_classes=4)
    planted = plant_cube(3, 40, 400.0, 1000.0, 32, centre=1)
    cube = planted.cube.unsqueeze(0)
    wl = planted.wavelengths
    labels = planted.label.clamp_max(3).unsqueeze(0)
    optimizer = torch.optim.Adam([p for p in encoder.parameters() if p.requires_grad], lr=5e-3)
    first = float(segmentation_loss(encoder(cube, wl).dense_logits, labels))
    for _ in range(40):
        optimizer.zero_grad()
        loss = segmentation_loss(encoder(cube, wl).dense_logits, labels)
        loss.backward()
        optimizer.step()
    assert float(loss) < first * 0.6
