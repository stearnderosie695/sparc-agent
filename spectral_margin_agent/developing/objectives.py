from __future__ import annotations

import torch
import torch.nn.functional as F

from spectral_margin_agent.dodge_burn.hypernet import ParetoHyperNet, psl_loss, sample_preferences
from spectral_margin_agent.emulsion.encoder import Encoder
from spectral_margin_agent.safelight.dials import ParetoDial
from spectral_margin_agent.toning.gate import confidence_gated_augment
from spectral_margin_agent.toning.reparam import sample_shift


def segmentation_loss(dense_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(dense_logits, labels)


def pareto_front_loss(
    hypernet: ParetoHyperNet, dial: ParetoDial, generator: torch.Generator
) -> torch.Tensor:
    prefs = sample_preferences(dial.dirichlet_alpha, dial.n_pref_samples, generator)
    context = prefs.new_zeros(dial.n_pref_samples, dial.context_dim)
    objectives = hypernet(prefs, context)
    return psl_loss(objectives, prefs)


def transfer_consistency(
    encoder: Encoder,
    cube: torch.Tensor,
    wl: torch.Tensor,
    tumour_index: int,
    threshold: float,
    generator: torch.Generator,
) -> torch.Tensor:
    clean = encoder(cube, wl)
    posterior = torch.softmax(clean.dense_logits, dim=1)
    confidence = posterior.amax(dim=1)
    shift = sample_shift(generator)
    augmented = confidence_gated_augment(cube, wl, confidence, threshold, shift)
    toned = encoder(augmented, wl)
    return F.mse_loss(torch.softmax(toned.dense_logits, dim=1), posterior.detach())
