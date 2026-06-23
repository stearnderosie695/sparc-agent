from __future__ import annotations

from typing import cast

import torch
from torch import nn

from spectral_margin_agent.emulsion.ssm import MambaBlock, init_dt_bias


class MambaFusion(nn.Module):
    def __init__(self, d_emb: int, fuse_width: int, d_state: int, blocks: int, expand: int) -> None:
        super().__init__()
        self.down_s = nn.Linear(d_emb, fuse_width)
        self.down_e = nn.Linear(d_emb, fuse_width)
        self.blocks = nn.ModuleList(
            [MambaBlock(fuse_width, d_state, expand) for _ in range(blocks)]
        )
        for block in self.blocks:
            init_dt_bias(cast(MambaBlock, block))
        self.gates = nn.ModuleList([nn.Linear(fuse_width, fuse_width) for _ in range(blocks)])
        self.fuse = nn.Linear(2 * fuse_width, d_emb)

    def forward(self, spatial: torch.Tensor, spectral: torch.Tensor) -> torch.Tensor:
        s = self.down_s(spatial)
        e = self.down_e(spectral)
        for index, block in enumerate(self.blocks):
            if index % 2 == 0:
                context = s.mean(dim=1, keepdim=True)
                e = block(e + torch.sigmoid(self.gates[index](context)) * e)
            else:
                context = e.mean(dim=1, keepdim=True)
                s = block(s + torch.sigmoid(self.gates[index](context)) * s)
        pooled = e.mean(dim=1, keepdim=True).expand_as(s)
        return self.fuse(torch.cat([s, pooled], dim=-1))
