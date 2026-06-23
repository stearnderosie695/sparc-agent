from __future__ import annotations

import torch
from conftest import tiny_bath, tiny_emulsion, tiny_enlarger

from spectral_margin_agent.developing.trainer import Trainer
from spectral_margin_agent.enlarger.orchestrator import build_agent
from spectral_margin_agent.reels.corpora import build_split
from spectral_margin_agent.reels.spool import build_loader
from spectral_margin_agent.safelight.bench import set_seed


def _trainer(tmp_root):
    set_seed(0)
    agent = build_agent(tiny_emulsion(), tiny_enlarger())
    return Trainer(agent, tiny_bath(), tiny_enlarger(), torch.device("cpu"), seed=0, total_steps=10)


def test_training_reduces_loss_over_several_steps(tmp_path):
    trainer = _trainer(tmp_path)
    stack = build_split("sarcoma_synth", "train", 12, 0, str(tmp_path), 32)
    loader = build_loader(stack, 2, 0, 0, True)
    history = trainer.fit(loader, max_steps=4)
    assert len(history) == 4
    assert all(r.total == r.total for r in history)
    assert history[-1].total < history[0].total + 1e-3


def test_checkpoint_round_trip(tmp_path):
    trainer = _trainer(tmp_path)
    stack = build_split("sarcoma_synth", "train", 6, 0, str(tmp_path), 32)
    loader = build_loader(stack, 2, 0, 0, True)
    trainer.fit(loader, max_steps=2)
    path = str(tmp_path / "ck.pt")
    trainer.save(path)
    trainer.load(path)
    assert trainer.step_index == 2
