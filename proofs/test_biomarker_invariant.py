from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import recall_score, roc_auc_score

from spectral_margin_agent.densitometry.biomarker import (
    masked_auroc_drop,
    occlusion_importance,
    window_concentration,
)
from spectral_margin_agent.reels.grid import slope_feature
from spectral_margin_agent.reels.synth import NORMAL, TUMOUR, plant_cube


def _planted_margin_scores():
    scores, labels = [], []
    for seed in range(16):
        planted = plant_cube(1000 + seed, 120, 400.0, 1000.0, 48, centre=1 + seed % 4)
        spectra = planted.cube.movedim(0, -1)
        slope = slope_feature(spectra, planted.wavelengths, 975.0, 25.0)
        keep = (planted.label == TUMOUR) | (planted.label == NORMAL)
        scores.append(slope[keep])
        labels.append((planted.label[keep] == TUMOUR).long())
    return torch.cat(scores).numpy(), torch.cat(labels).numpy()


def test_planted_cohort_reproduces_table_one_headline():
    scores, labels = _planted_margin_scores()
    auc = roc_auc_score(labels, scores)
    oriented = max(auc, 1.0 - auc)
    flip = auc < 0.5
    thresholds = np.quantile(scores, np.linspace(0.02, 0.98, 120))
    best = 0.0
    for t in thresholds:
        pred = (scores < t).astype(int) if flip else (scores >= t).astype(int)
        youden = recall_score(labels, pred) + recall_score(1 - labels, 1 - pred) - 1
        if youden > best:
            best = youden
            sensitivity = recall_score(labels, pred)
    assert 0.88 <= oriented <= 0.97
    assert sensitivity >= 0.83


def test_margin_signal_concentrates_at_975():
    planted = plant_cube(7, 120, 400.0, 1000.0, 48, centre=1)
    cube = planted.cube.unsqueeze(0)
    labels = (planted.label == TUMOUR).reshape(-1).numpy().astype(int)

    def score_fn(c: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return slope_feature(c.squeeze(0).movedim(0, -1), w, 975.0, 25.0)

    centers, drops = occlusion_importance(score_fn, cube, planted.wavelengths, labels, 16)
    importance = np.abs(drops)
    peak = centers[int(np.argmax(importance))]
    assert abs(peak - 975.0) <= 35.0
    assert window_concentration(centers, importance, 975.0, 25.0) > 1.5
    assert abs(masked_auroc_drop(score_fn, cube, planted.wavelengths, labels, 975.0, 25.0)) > 0.2
