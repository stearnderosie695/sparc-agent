from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical

from spectral_margin_agent.dodge_burn.hypernet import ParetoHyperNet
from spectral_margin_agent.emulsion.encoder import Encoder
from spectral_margin_agent.enlarger.ccatr import CCATR, quadrant_route, tool_confidence_vector
from spectral_margin_agent.enlarger.reward import composite_reward
from spectral_margin_agent.enlarger.session import TOOL_NAMES, Session
from spectral_margin_agent.enlarger.tools import (
    ToolDeps,
    dispatch,
    run_pareto_sample,
    run_plan_verify,
)
from spectral_margin_agent.register.atlas import AtlasReasoner
from spectral_margin_agent.register.proximity import overlap_fraction
from spectral_margin_agent.safelight.dials import EmulsionDial, EnlargerDial
from spectral_margin_agent.safelight.glossary import Provenance


@dataclass
class Rollout:
    log_probs: torch.Tensor
    values: torch.Tensor
    entropies: torch.Tensor
    rewards: torch.Tensor


@dataclass
class PlanResult:
    contour: torch.Tensor
    objective: torch.Tensor
    provenance: Provenance
    c_final: torch.Tensor
    decision: int
    verified: bool
    steps: int
    margin_mm: float
    critical: torch.Tensor
    rollout: Rollout


class SparcAgent(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        atlas_reasoner: AtlasReasoner,
        hypernet: ParetoHyperNet,
        ccatr: CCATR,
        dial: EnlargerDial,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.atlas_reasoner = atlas_reasoner
        self.hypernet = hypernet
        self.ccatr = ccatr
        self.dial = dial

    def _deps(self) -> ToolDeps:
        return ToolDeps(
            atlas_reasoner=self.atlas_reasoner,
            hypernet=self.hypernet,
            tumour_index=self.dial.tumour_index,
            t_calib=self.dial.t_calib,
            eps_overlap=self.dial.eps_overlap,
            homogeneity_thresh=self.dial.homogeneity_thresh,
            budget_steps=self.dial.planning_steps,
            max_radius=self.dial.max_radius,
        )

    def plan(
        self, cube: torch.Tensor, wl: torch.Tensor, pref: torch.Tensor, explore: bool = True
    ) -> PlanResult:
        encoder_out = self.encoder(cube, wl)
        session = Session(cube, wl, encoder_out, pref, pixel_mm=self.dial.pixel_mm)
        deps = self._deps()

        log_probs, values, entropies = [], [], []
        verified = False
        objective = pref.new_zeros(3)
        weights = pref.new_full((1, len(TOOL_NAMES)), 1.0 / len(TOOL_NAMES))

        for _ in range(self.dial.planning_steps):
            features = self.ccatr.features(session)
            weights, value, _ = self.ccatr(features)
            distribution = Categorical(weights)
            action = distribution.sample() if explore else weights.argmax(dim=-1)
            log_probs.append(distribution.log_prob(action))
            entropies.append(distribution.entropy())
            values.append(value)
            report = dispatch(int(action), session, deps)
            session.history.append(int(action))
            if report.name == "pareto_sample":
                objective = report.value
            if report.name == "plan_verify" and float(report.value) > 0.5:
                verified = True
                break

        if session.contour is None:
            objective = run_pareto_sample(session, deps).value
        final_check = run_plan_verify(session, deps)
        verified = verified or float(final_check.value) > 0.5

        contour = session.contour
        assert contour is not None
        overlap = float(overlap_fraction(contour, session.critical_mask()).detach())
        reward = composite_reward(
            session.achieved_margin(),
            self.dial.margin_threshold_mm,
            overlap,
            len(session.history),
            verified,
            self.dial.reward_alpha,
        )
        rewards = [0.0] * (len(values) - 1) + [reward] if values else [reward]

        confidences = tool_confidence_vector(session).view(1, -1)
        c_final = self.ccatr.bayesian_average(weights, confidences).detach()
        provenance = Provenance(weights.squeeze(0).detach(), confidences.squeeze(0), TOOL_NAMES)
        c_atlas = (
            float(session.atlas.confidence.mean().detach()) if session.atlas is not None else 0.0
        )
        decision = quadrant_route(session.c_spec, c_atlas, self.dial.tau_spec, self.dial.tau_atlas)
        rollout = Rollout(
            log_probs=torch.stack(log_probs) if log_probs else pref.new_zeros(1),
            values=torch.stack(values).squeeze(-1) if values else pref.new_zeros(1),
            entropies=torch.stack(entropies) if entropies else pref.new_zeros(1),
            rewards=torch.tensor(rewards),
        )
        return PlanResult(
            contour=contour,
            objective=objective,
            provenance=provenance,
            c_final=c_final,
            decision=decision,
            verified=verified,
            steps=len(session.history),
            margin_mm=session.achieved_margin(),
            critical=session.critical_mask(),
            rollout=rollout,
        )


def build_agent(emulsion: EmulsionDial, enlarger: EnlargerDial) -> SparcAgent:
    encoder = Encoder(emulsion, enlarger.n_classes)
    atlas_reasoner = AtlasReasoner()
    hypernet = ParetoHyperNet(enlarger.pareto)
    ccatr = CCATR(emulsion.d_emb, len(TOOL_NAMES), enlarger.router_hidden, enlarger.router_depth)
    return SparcAgent(encoder, atlas_reasoner, hypernet, ccatr, enlarger)
