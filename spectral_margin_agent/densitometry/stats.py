from __future__ import annotations

import numpy as np
from scipy.stats import norm, wilcoxon


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    n = x.shape[0]
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def _fast_delong(predictions: np.ndarray, positive_count: int) -> tuple[np.ndarray, np.ndarray]:
    m = positive_count
    n = predictions.shape[1] - m
    pos = predictions[:, :m]
    neg = predictions[:, m:]
    k = predictions.shape[0]
    tx = np.empty([k, m])
    ty = np.empty([k, n])
    tz = np.empty([k, m + n])
    for r in range(k):
        tx[r] = _midrank(pos[r])
        ty[r] = _midrank(neg[r])
        tz[r] = _midrank(predictions[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    return aucs, np.atleast_2d(cov)


def delong_test(
    labels: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray
) -> tuple[float, float, float]:
    order = np.argsort(-labels)
    positive_count = int(labels.sum())
    stacked = np.vstack([scores_a, scores_b])[:, order]
    aucs, cov = _fast_delong(stacked, positive_count)
    contrast = np.array([[1.0, -1.0]])
    variance = float(contrast @ cov @ contrast.T)
    z = (aucs[0] - aucs[1]) / np.sqrt(variance + 1e-12)
    p = 2.0 * (1.0 - norm.cdf(abs(z)))
    return float(aucs[0]), float(aucs[1]), float(p)


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    if np.allclose(diff, 0.0):
        return 1.0
    return float(wilcoxon(a, b).pvalue)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    spread = diff.std(ddof=1)
    if spread < 1e-12:
        return 0.0
    return float(diff.mean() / spread)


def benjamini_hochberg(pvalues: np.ndarray, q: float) -> np.ndarray:
    n = len(pvalues)
    order = np.argsort(pvalues)
    thresholds = q * (np.arange(1, n + 1) / n)
    passed = pvalues[order] <= thresholds
    rejected = np.zeros(n, dtype=bool)
    if passed.any():
        cutoff = np.max(np.where(passed)[0])
        rejected[order[: cutoff + 1]] = True
    return rejected


def cluster_bootstrap_ci(
    values: np.ndarray, clusters: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    unique = np.unique(clusters)
    estimates = np.empty(n_boot)
    for b in range(n_boot):
        drawn = rng.choice(unique, size=len(unique), replace=True)
        pooled = np.concatenate([values[clusters == c] for c in drawn])
        estimates[b] = pooled.mean()
    lo = float(np.quantile(estimates, alpha / 2))
    hi = float(np.quantile(estimates, 1 - alpha / 2))
    return float(values.mean()), lo, hi
