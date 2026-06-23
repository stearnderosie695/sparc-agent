from __future__ import annotations

import torch

from spectral_margin_agent.contact_sheet.evaluate import evaluate_margin
from spectral_margin_agent.contact_sheet.experiments import biomarker, cross_tissue, pareto_sweep
from spectral_margin_agent.enlarger.orchestrator import build_agent
from spectral_margin_agent.reels.corpora import build_split
from spectral_margin_agent.reels.spool import build_loader
from spectral_margin_agent.safelight.mix import load_recipe


def _probe_recipe():
    return load_recipe(
        None,
        [
            "stock.tile=32",
            "emulsion.d_emb=96",
            "emulsion.backbone_depth=2",
            "emulsion.spectral_tokens=8",
            "emulsion.fuse_width=64",
            "emulsion.mamba_blocks=2",
            "enlarger.planning_steps=3",
            "enlarger.router_hidden=48",
            "enlarger.pareto.width=48",
            "enlarger.pareto.pref_embed=24",
        ],
    )


def _agent(recipe):
    torch.manual_seed(0)
    return build_agent(recipe.emulsion, recipe.enlarger)


def test_evaluate_margin_reports_finite_metrics():
    recipe = _probe_recipe()
    agent = _agent(recipe)
    stack = build_split("sarcoma_synth", "test", 8, 0, "plates", 32)
    loader = build_loader(stack, 2, 0, 0, False)
    report = evaluate_margin(agent, loader, 1, 1.7, torch.device("cpu"))
    assert 0.0 <= report.auroc <= 1.0
    assert 0.0 <= report.ece <= 1.0


def test_cross_tissue_covers_all_corpora():
    recipe = _probe_recipe()
    agent = _agent(recipe)
    out = cross_tissue(agent, recipe, ["helicoid", "hsi_dermoscopy"], torch.device("cpu"))
    assert set(out) == {"helicoid", "hsi_dermoscopy"}


def test_pareto_sweep_keys_and_bounds():
    recipe = _probe_recipe()
    agent = _agent(recipe)
    out = pareto_sweep(agent, recipe, 4, torch.device("cpu"))
    assert {"hypervolume", "preservation", "pareto_margin_mm", "violation_rate"} <= set(out)
    assert 0.0 <= out["preservation"] <= 1.0
    assert 0.0 <= out["violation_rate"] <= 1.0


def test_biomarker_runner_reports_peak():
    recipe = _probe_recipe()
    agent = _agent(recipe)
    out = biomarker(agent, recipe, "sarcoma_synth", torch.device("cpu"))
    assert "conc_975" in out and "peak_nm" in out
