from __future__ import annotations

import torch

from spectral_margin_agent.reels.corpora import CORPORA, build_split
from spectral_margin_agent.reels.grid import make_grid, resample_to_grid, slope_feature
from spectral_margin_agent.reels.spool import build_loader


def test_corpora_band_counts():
    expected = {
        "helicoid": 826,
        "heiporspectral": 100,
        "odsi_db": 204,
        "hsi_dermoscopy": 16,
        "sarcoma_synth": 826,
    }
    for key, bands in expected.items():
        stack = build_split(key, "test", 4, 7, "plates", 24)
        assert stack[0].cube.shape[0] == bands
        assert CORPORA[key].n_bands == bands


def test_resample_preserves_linear_signal():
    src = make_grid(400.0, 1000.0, 60)
    dst = make_grid(420.0, 980.0, 33)
    signal = (src * 2.0 + 1.0).view(1, 60)
    out = resample_to_grid(signal, src, dst)
    assert torch.allclose(out.view(-1), dst * 2.0 + 1.0, atol=1e-3)


def test_slope_feature_recovers_known_gradient():
    wl = make_grid(950.0, 1000.0, 20)
    spectra = (wl * -0.01 + 5.0).view(1, 20)
    slope = slope_feature(spectra, wl, 975.0, 25.0)
    assert abs(float(slope) + 0.01) < 1e-3


def test_loader_collates_batch():
    stack = build_split("sarcoma_synth", "train", 8, 0, "plates", 32)
    loader = build_loader(stack, 4, 0, 0, True)
    frame = next(iter(loader))
    assert frame.cube.shape[0] <= 4
    assert frame.label.shape[-2:] == frame.cube.shape[-2:]
