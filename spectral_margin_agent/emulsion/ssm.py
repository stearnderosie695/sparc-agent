from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    a_log: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
) -> torch.Tensor:
    n, length, d_inner = u.shape
    state_dim = a_log.shape[-1]
    a = -torch.exp(a_log)
    delta = F.softplus(delta)
    da = torch.exp(delta.unsqueeze(-1) * a.view(1, 1, d_inner, state_dim))
    db = delta.unsqueeze(-1) * b.unsqueeze(2)
    state = u.new_zeros((n, d_inner, state_dim))
    outputs = []
    for t in range(length):
        state = da[:, t] * state + db[:, t] * u[:, t].unsqueeze(-1)
        outputs.append(torch.einsum("nds,ns->nd", state, c[:, t]))
    y = torch.stack(outputs, dim=1)
    return y + u * d.view(1, 1, d_inner)


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int, expand: int, conv_kernel: int = 4) -> None:
        super().__init__()
        self.d_inner = expand * d_model
        self.dt_rank = max(1, d_model // 16)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            conv_kernel,
            groups=self.d_inner,
            padding=conv_kernel - 1,
            bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        a = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.a_log = nn.Parameter(torch.log(a))
        self.d_skip = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.d_state = d_state

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        residual = hidden
        hidden = self.norm(hidden)
        length = hidden.shape[1]
        xz = self.in_proj(hidden)
        x, z = xz.chunk(2, dim=-1)
        x = x.transpose(1, 2)
        x = self.conv(x)[..., :length]
        x = x.transpose(1, 2)
        x = F.silu(x)
        proj = self.x_proj(x)
        delta, b, c = torch.split(proj, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = self.dt_proj(delta)
        y = selective_scan(x, delta, self.a_log, b, c, self.d_skip)
        y = y * F.silu(z)
        return residual + self.out_proj(y)


def init_dt_bias(block: MambaBlock, floor: float = 1e-3, ceil: float = 0.1) -> None:
    with torch.no_grad():
        span = math.log(ceil) - math.log(floor)
        rand = torch.rand(block.dt_proj.bias.shape) * span + math.log(floor)
        block.dt_proj.bias.copy_(rand + torch.log(-torch.expm1(-torch.exp(rand))))
