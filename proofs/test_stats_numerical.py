from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from spectral_margin_agent.densitometry.classification import dice, expected_calibration_error
from spectral_margin_agent.densitometry.stats import (
    benjamini_hochberg,
    cohens_d,
    delong_test,
    paired_wilcoxon,
)


def test_delong_matches_sklearn_auc():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 400)
    a = labels * 0.6 + rng.normal(0, 0.5, 400)
    b = labels * 0.3 + rng.normal(0, 0.5, 400)
    auc_a, auc_b, p = delong_test(labels.astype(float), a, b)
    assert abs(auc_a - roc_auc_score(labels, a)) < 1e-6
    assert abs(auc_b - roc_auc_score(labels, b)) < 1e-6
    assert p < 0.05


def test_benjamini_hochberg_threshold():
    rejected = benjamini_hochberg(np.array([0.001, 0.04, 0.2, 0.5]), 0.05)
    assert rejected.tolist() == [True, False, False, False]


def test_cohens_d_and_wilcoxon():
    a = np.array([0.9, 0.8, 0.85, 0.92, 0.88])
    b = np.array([0.7, 0.72, 0.69, 0.75, 0.71])
    assert cohens_d(a, b) > 1.0
    assert paired_wilcoxon(a, b) < 0.1


def test_dice_and_ece_bounds():
    assert dice(np.ones(10), np.ones(10)) == 1.0
    assert dice(np.zeros(10), np.zeros(10)) == 1.0
    labels = np.array([1, 0, 1, 0])
    conf = np.array([0.9, 0.9, 0.9, 0.9])
    pred = np.array([1, 0, 1, 0])
    assert 0.0 <= expected_calibration_error(labels, conf, pred) <= 1.0
