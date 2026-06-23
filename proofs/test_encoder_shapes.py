from __future__ import annotations

import pytest
import torch
from conftest import tiny_emulsion

from spectral_margin_agent.emulsion.encoder import Encoder
from spectral_margin_agent.enlarger.orchestrator import build_agent
from spectral_margin_agent.safelight.bench import count_total, count_trainable
from spectral_margin_agent.safelight.dials import EmulsionDial, EnlargerDial


@pytest.mark.parametrize("bands", [30, 60, 128, 400, 826])
def test_encoder_handles_variable_band_count(bands):
    encoder = Encoder(tiny_emulsion(), n_classes=4)
    cube = torch.rand(2, bands, 32, 32)
    wl = torch.linspace(400.0, 1000.0, bands)
    out = encoder(cube, wl)
    assert out.dense_logits.shape == (2, 4, 32, 32)
    assert out.embedding.shape[-1] == 96


def test_mamba_blocks_zero_is_valid():
    dial = tiny_emulsion()
    dial.mamba_blocks = 0
    encoder = Encoder(dial, n_classes=4)
    out = encoder(torch.rand(1, 40, 32, 32), torch.linspace(400.0, 1000.0, 40))
    assert out.dense_logits.shape == (1, 4, 32, 32)


def test_paper_parameter_budget():
    agent = build_agent(EmulsionDial(), EnlargerDial())
    trainable = count_trainable(agent)
    frozen = count_total(agent) - trainable
    assert 20e6 <= trainable <= 28e6
    assert 70e6 <= frozen <= 85e6
    assert 8e6 <= count_trainable(agent.encoder.fusion) <= 9e6
    assert 0.8e6 <= count_trainable(agent.ccatr) <= 1.0e6
    assert 3.0e6 <= count_trainable(agent.hypernet) <= 3.4e6
