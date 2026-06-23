from __future__ import annotations

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, scale: float = 1.0) -> None:
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.down = nn.Parameter(torch.zeros(rank, base.in_features))
        self.up = nn.Parameter(torch.zeros(base.out_features, rank))
        self.scale = scale / rank
        nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = (x @ self.down.t()) @ self.up.t()
        return self.base(x) + self.scale * residual


def graft_lora(module: nn.Module, names: tuple[str, ...], rank: int) -> int:
    grafted = 0
    for child_name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and child_name in names:
            setattr(module, child_name, LoRALinear(child, rank))
            grafted += 1
        else:
            grafted += graft_lora(child, names, rank)
    return grafted
