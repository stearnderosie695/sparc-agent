from __future__ import annotations

import pytest
import torch

from spectral_margin_agent.enlarger.orchestrator import build_agent
from spectral_margin_agent.safelight.dials import (
    BathDial,
    EmulsionDial,
    EnlargerDial,
    ParetoDial,
)


def tiny_emulsion() -> EmulsionDial:
    return EmulsionDial(
        backbone_depth=2,
        d_emb=96,
        backbone_heads=4,
        spectral_tokens=8,
        adapter_depth=1,
        fuse_width=64,
        mamba_blocks=2,
    )


def tiny_enlarger() -> EnlargerDial:
    return EnlargerDial(
        planning_steps=4,
        router_hidden=48,
        pareto=ParetoDial(width=48, pref_embed=24, n_pref_samples=8),
    )


def tiny_bath() -> BathDial:
    return BathDial(lr=1e-3, warmup_steps=2, a3c_workers=2, grad_accum=1)


@pytest.fixture
def agent():
    torch.manual_seed(0)
    return build_agent(tiny_emulsion(), tiny_enlarger())


@pytest.fixture
def cube():
    torch.manual_seed(1)
    return torch.rand(1, 48, 32, 32), torch.linspace(400.0, 1000.0, 48)
