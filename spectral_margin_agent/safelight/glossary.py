from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class Frame:
    cube: torch.Tensor
    wavelengths: torch.Tensor
    label: torch.Tensor
    centre: torch.Tensor
    corpus: str


@dataclass(frozen=True)
class ToolReport:
    name: str
    value: torch.Tensor
    confidence: torch.Tensor
    payload: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass(frozen=True)
class PlanScore:
    margin: torch.Tensor
    function: torch.Tensor
    latency: torch.Tensor

    def vector(self) -> torch.Tensor:
        return torch.stack([self.margin, self.function, self.latency], dim=-1)


@dataclass(frozen=True)
class Provenance:
    weights: torch.Tensor
    confidences: torch.Tensor
    tools: tuple[str, ...]

    def trace(self) -> torch.Tensor:
        return self.weights * self.confidences
