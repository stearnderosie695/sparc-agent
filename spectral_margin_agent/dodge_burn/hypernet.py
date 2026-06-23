from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from spectral_margin_agent.safelight.dials import ParetoDial


def sample_preferences(alpha: list[float], count: int, generator: torch.Generator) -> torch.Tensor:
    concentration = torch.tensor(alpha)
    gamma = torch._standard_gamma(concentration.expand(count, -1), generator=generator)
    return gamma / gamma.sum(dim=-1, keepdim=True)


def psl_loss(objective: torch.Tensor, preference: torch.Tensor) -> torch.Tensor:
    cosine = F.cosine_similarity(objective, preference, dim=-1)
    return (1.0 - cosine).mean()


class ParetoHyperNet(nn.Module):
    def __init__(self, dial: ParetoDial) -> None:
        super().__init__()
        width = dial.width
        self.context = nn.Linear(dial.context_dim, width)
        self.pref = nn.Sequential(
            nn.Linear(3, dial.pref_embed),
            nn.GELU(),
            nn.Linear(dial.pref_embed, dial.pref_embed),
            nn.GELU(),
        )
        self.layers = nn.ModuleList([nn.Linear(width, width) for _ in range(dial.film_layers)])
        self.films = nn.ModuleList(
            [nn.Linear(dial.pref_embed, 2 * width) for _ in range(dial.film_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(dial.film_layers)])
        self.head = nn.Linear(width, 3)

    def forward(self, preference: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.context(context))
        embedded = self.pref(preference)
        for layer, film, norm in zip(self.layers, self.films, self.norms):
            gamma, beta = film(embedded).chunk(2, dim=-1)
            hidden = hidden + gamma * layer(norm(hidden)) + beta
            hidden = F.gelu(hidden)
        return torch.sigmoid(self.head(hidden))
