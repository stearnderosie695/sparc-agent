from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from spectral_margin_agent.dodge_burn.hypernet import ParetoHyperNet
from spectral_margin_agent.enlarger.session import TOOL_NAMES, Session, contour_from
from spectral_margin_agent.reels.grid import slope_feature
from spectral_margin_agent.register.atlas import AtlasReasoner
from spectral_margin_agent.register.indices import vessel_mask
from spectral_margin_agent.register.proximity import overlap_fraction
from spectral_margin_agent.safelight.glossary import ToolReport


@dataclass
class ToolDeps:
    atlas_reasoner: AtlasReasoner
    hypernet: ParetoHyperNet
    tumour_index: int = 1
    t_calib: float = 1.7
    eps_overlap: float = 0.05
    homogeneity_thresh: float = 0.06
    budget_steps: int = 15
    max_radius: int = 6


def context_vector(session: Session, deps: ToolDeps) -> torch.Tensor:
    zero = session.cube.new_zeros(())
    tumour = session.tumour_map.mean() if session.tumour_map is not None else zero
    c_atlas = session.atlas.confidence.mean() if session.atlas is not None else zero
    vessel_frac = session.vessels.float().mean() if session.vessels is not None else zero
    margin_norm = torch.tensor(session.margin_radius / deps.max_radius)
    latency_norm = torch.tensor(len(session.history) / deps.budget_steps)
    feats = torch.stack(
        [
            tumour.detach(),
            torch.tensor(float(session.c_spec)),
            c_atlas.detach(),
            vessel_frac.detach(),
            margin_norm,
            latency_norm,
            session.pref.squeeze(0)[0].detach(),
            session.pref.squeeze(0)[1].detach(),
        ]
    )
    return feats.view(1, -1)


def run_spec_cls(session: Session, deps: ToolDeps) -> ToolReport:
    posterior = torch.softmax(session.encoder_out.dense_logits / deps.t_calib, dim=1)
    session.tumour_map = posterior[:, deps.tumour_index].squeeze(0)
    confidence = float(posterior.amax(dim=1).mean().detach())
    session.c_spec = confidence
    session.confidences["spec_cls"] = confidence
    return ToolReport("spec_cls", session.tumour_map.detach(), torch.tensor(confidence))


def run_atlas_lookup(session: Session, deps: ToolDeps) -> ToolReport:
    out = deps.atlas_reasoner(session.cube, session.wl)
    session.atlas = out
    confidence = float(out.confidence.mean().detach())
    session.confidences["atlas_lookup"] = confidence
    return ToolReport(
        "atlas_lookup",
        out.label.float(),
        torch.tensor(confidence),
        {"mask": out.mask.detach(), "proximity": out.proximity.detach()},
    )


def run_vasc_det(session: Session, deps: ToolDeps) -> ToolReport:
    mask = vessel_mask(session.cube, session.wl).squeeze(0)
    session.vessels = mask
    confidence = float((mask.float().mean() * 4.0).clamp(0.0, 1.0))
    session.confidences["vasc_det"] = confidence
    return ToolReport("vasc_det", mask.float(), torch.tensor(confidence))


def run_pareto_sample(session: Session, deps: ToolDeps) -> ToolReport:
    if session.tumour_map is None:
        run_spec_cls(session, deps)
    tumour = session.tumour_map
    assert tumour is not None
    objective = deps.hypernet(session.pref, context_vector(session, deps))
    radius = int(1 + round(float(objective[0, 0].detach()) * (deps.max_radius - 1)))
    session.margin_radius = radius
    contour = contour_from(tumour.detach(), radius, 0.5)
    session.contour = contour
    preservation = 1.0 - float(overlap_fraction(contour, session.critical_mask()).detach())
    session.confidences["pareto_sample"] = preservation
    return ToolReport(
        "pareto_sample",
        objective.squeeze(0).detach(),
        torch.tensor(preservation),
        {"contour": contour.detach()},
    )


def run_plan_verify(session: Session, deps: ToolDeps) -> ToolReport:
    if session.contour is None:
        run_pareto_sample(session, deps)
    contour = session.contour
    assert contour is not None
    overlap = float(overlap_fraction(contour, session.critical_mask()).detach())
    boundary = contour - torch.clamp(contour - contour.roll(1, dims=-1), 0.0, 1.0)
    edge = (contour > 0.5) & (boundary < 0.5)
    spectra = session.cube.squeeze(0).movedim(0, -1)
    slope = slope_feature(spectra, session.wl, 975.0, 25.0)
    homogeneity = float(slope[edge].std()) if int(edge.sum()) > 1 else 0.0
    latency_ok = len(session.history) <= deps.budget_steps
    checks = [overlap < deps.eps_overlap, homogeneity < deps.homogeneity_thresh, latency_ok]
    confidence = float(sum(checks) / len(checks))
    session.confidences["plan_verify"] = confidence
    return ToolReport(
        "plan_verify",
        torch.tensor(float(all(checks))),
        torch.tensor(confidence),
        {"overlap": torch.tensor(overlap), "homogeneity": torch.tensor(homogeneity)},
    )


REGISTRY: dict[str, Callable[[Session, ToolDeps], ToolReport]] = {
    "spec_cls": run_spec_cls,
    "atlas_lookup": run_atlas_lookup,
    "vasc_det": run_vasc_det,
    "pareto_sample": run_pareto_sample,
    "plan_verify": run_plan_verify,
}


def dispatch(index: int, session: Session, deps: ToolDeps) -> ToolReport:
    return REGISTRY[TOOL_NAMES[index]](session, deps)
