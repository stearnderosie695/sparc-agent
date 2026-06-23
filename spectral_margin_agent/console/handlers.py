from __future__ import annotations

import torch

from spectral_margin_agent.contact_sheet.evaluate import evaluate_margin
from spectral_margin_agent.contact_sheet.experiments import biomarker, cross_tissue, pareto_sweep
from spectral_margin_agent.developing.trainer import Trainer
from spectral_margin_agent.dodge_burn.hypernet import sample_preferences
from spectral_margin_agent.enlarger.orchestrator import build_agent
from spectral_margin_agent.reels.corpora import build_split
from spectral_margin_agent.reels.spool import build_loader
from spectral_margin_agent.safelight.bench import atomic_save, get_logger, pick_device, set_seed
from spectral_margin_agent.safelight.dials import Recipe
from spectral_margin_agent.safelight.mix import load_recipe

_LOG = get_logger("console")
_SYNTH_ITEMS = 64


def _prepare(recipe_path, overrides):
    recipe = load_recipe(recipe_path, overrides)
    set_seed(recipe.run.seed)
    device = pick_device(recipe.run.device)
    agent = build_agent(recipe.emulsion, recipe.enlarger).to(device)
    return recipe, device, agent


def _train_loader(recipe: Recipe):
    stack = build_split(
        recipe.stock.corpus,
        "train",
        _SYNTH_ITEMS,
        recipe.run.seed,
        recipe.stock.root,
        recipe.stock.tile,
    )
    return build_loader(
        stack, recipe.stock.micro_batch, recipe.stock.num_workers, recipe.run.seed, True
    )


def develop(recipe_path, overrides, steps, out_path):
    recipe, device, agent = _prepare(recipe_path, overrides)
    loader = _train_loader(recipe)
    total = steps if steps is not None else recipe.bath.epochs
    trainer = Trainer(agent, recipe.bath, recipe.enlarger, device, recipe.run.seed, total)
    history = trainer.fit(loader, total)
    target = out_path or f"{recipe.run.out_dir}/negative.pt"
    trainer.save(target)
    _LOG.info(
        "developed %d steps; final loss %.4f; saved %s", len(history), history[-1].total, target
    )


def _load_ckpt(agent, ckpt_path, device):
    if ckpt_path is None:
        return
    state = torch.load(ckpt_path, map_location=device)
    agent.load_state_dict(state["model"])
    _LOG.info("loaded checkpoint %s", ckpt_path)


def read(recipe_path, overrides, ckpt_path, table):
    recipe, device, agent = _prepare(recipe_path, overrides)
    _load_ckpt(agent, ckpt_path, device)
    if table in (None, "main", "clinical"):
        stack = build_split(
            recipe.stock.corpus, "test", 16, recipe.run.seed, recipe.stock.root, recipe.stock.tile
        )
        loader = build_loader(stack, recipe.stock.micro_batch, 0, recipe.run.seed, False)
        report = evaluate_margin(
            agent, loader, recipe.enlarger.tumour_index, recipe.enlarger.t_calib, device
        )
        _LOG.info(
            "margin AUROC %.3f sensitivity %.3f dice %.3f macroF1 %.3f ECE %.3f",
            report.auroc,
            report.tumour_sensitivity,
            report.dice,
            report.macro_f1,
            report.ece,
        )
    elif table == "cross_tissue":
        for corpus, report in cross_tissue(
            agent, recipe, ["helicoid", "heiporspectral", "odsi_db", "hsi_dermoscopy"], device
        ).items():
            _LOG.info(
                "%s AUROC %.3f macroF1 %.3f sens %.3f",
                corpus,
                report.auroc,
                report.macro_f1,
                report.tumour_sensitivity,
            )
    elif table == "pareto":
        for key, value in pareto_sweep(
            agent, recipe, recipe.enlarger.pareto.n_pref_samples, device
        ).items():
            _LOG.info("%s %.4f", key, value)
    elif table == "biomarker":
        for corpus in ["helicoid", "heiporspectral", "odsi_db"]:
            stats = biomarker(agent, recipe, corpus, device)
            _LOG.info(
                "%s 975-conc %.2f mask-dAUROC %.3f peak %.0f nm",
                corpus,
                stats["conc_975"],
                stats["mask_auroc_drop"],
                stats["peak_nm"],
            )
    else:
        raise ValueError(f"unknown table {table}")


def expose(recipe_path, overrides, ckpt_path):
    recipe, device, agent = _prepare(recipe_path, overrides)
    _load_ckpt(agent, ckpt_path, device)
    stack = build_split(
        recipe.stock.corpus, "test", 4, recipe.run.seed, recipe.stock.root, recipe.stock.tile
    )
    frame = stack[0]
    pref = sample_preferences(
        recipe.enlarger.pareto.dirichlet_alpha, 1, torch.Generator().manual_seed(recipe.run.seed)
    )
    result = agent.plan(
        frame.cube.unsqueeze(0).to(device),
        frame.wavelengths.to(device),
        pref.to(device),
        explore=False,
    )
    _LOG.info(
        "plan margin %.2f mm decision %d verified %s steps %d c_final %.3f",
        result.margin_mm,
        result.decision,
        result.verified,
        result.steps,
        float(result.c_final),
    )
    _LOG.info(
        "provenance %s",
        {t: round(float(v), 3) for t, v in zip(result.provenance.tools, result.provenance.trace())},
    )


def transfer(recipe_path, overrides, ckpt_path, onnx_path):
    recipe, device, agent = _prepare(recipe_path, overrides)
    _load_ckpt(agent, ckpt_path, device)
    target = onnx_path or f"{recipe.run.out_dir}/encoder.onnx"
    bands = recipe.emulsion.in_bands
    example = torch.rand(1, bands, recipe.stock.tile, recipe.stock.tile, device=device)
    wl = torch.linspace(recipe.stock.band_lo, recipe.stock.band_hi, bands, device=device)

    class Wrap(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, cube):
            return self.encoder(cube, wl).dense_logits

    try:
        torch.onnx.export(
            Wrap(agent.encoder),
            (example,),
            target,
            input_names=["cube"],
            output_names=["dense_logits"],
            opset_version=17,
            dynamo=False,
        )
        atomic_save({"onnx": target}, f"{target}.meta")
        _LOG.info("exported encoder to %s", target)
    except (RuntimeError, ValueError) as error:
        _LOG.error("onnx export failed: %s", error)
