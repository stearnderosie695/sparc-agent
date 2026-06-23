from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from spectral_margin_agent.enlarger.session import TOOL_NAMES, Session

SPECTRAL, ATLAS, FUSE, ABSTAIN = 0, 1, 2, 3


class CCATR(nn.Module):
    def __init__(self, d_emb: int, n_tools: int, hidden: int, depth: int) -> None:
        super().__init__()
        self.n_tools = n_tools
        in_dim = d_emb + 4 + 3 + n_tools
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        self.trunk = nn.Sequential(*layers)
        self.policy = nn.Linear(hidden, n_tools)
        self.value = nn.Linear(hidden, 1)

    def features(self, session: Session) -> torch.Tensor:
        pooled = session.encoder_out.embedding.mean(dim=1).squeeze(0)
        if session.atlas is not None:
            label_oh = F.one_hot(session.atlas.label.squeeze(0), num_classes=3).float()
            atlas_summary = torch.cat([session.atlas.confidence.mean().view(1), label_oh])
        else:
            atlas_summary = pooled.new_zeros(4)
        pref = session.pref.squeeze(0)
        counts = pooled.new_zeros(self.n_tools)
        for index in session.history:
            counts[index] += 1.0
        history_pool = counts / max(len(session.history), 1)
        return torch.cat([pooled, atlas_summary, pref, history_pool]).view(1, -1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.trunk(features)
        logits = self.policy(hidden)
        value = self.value(hidden).squeeze(-1)
        return torch.softmax(logits, dim=-1), value, logits

    @staticmethod
    def bayesian_average(weights: torch.Tensor, confidences: torch.Tensor) -> torch.Tensor:
        return (weights * confidences).sum(dim=-1)

    @staticmethod
    def provenance(weights: torch.Tensor, confidences: torch.Tensor) -> torch.Tensor:
        return weights * confidences


def quadrant_route(c_spec: float, c_atlas: float, tau_spec: float, tau_atlas: float) -> int:
    spec_high = c_spec >= tau_spec
    atlas_high = c_atlas >= tau_atlas
    if spec_high and atlas_high:
        return FUSE
    if spec_high and not atlas_high:
        return SPECTRAL
    if atlas_high and not spec_high:
        return ATLAS
    return ABSTAIN


def tool_confidence_vector(session: Session) -> torch.Tensor:
    return torch.tensor([session.confidences.get(name, 0.0) for name in TOOL_NAMES])
