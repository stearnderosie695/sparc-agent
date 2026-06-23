from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def sinusoidal(values: torch.Tensor, dim: int, scale: float) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -torch.arange(half, dtype=torch.float32, device=values.device)
        * (torch.log(torch.tensor(10000.0)) / max(half - 1, 1))
    )
    angles = (values.unsqueeze(-1) / scale) * freqs
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb


def grid_encoding(rows: int, cols: int, dim: int, device: torch.device) -> torch.Tensor:
    ys = torch.arange(rows, device=device, dtype=torch.float32)
    xs = torch.arange(cols, device=device, dtype=torch.float32)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    half = dim // 2
    enc = torch.cat(
        [sinusoidal(gy.reshape(-1), half, 1.0), sinusoidal(gx.reshape(-1), half, 1.0)], dim=-1
    )
    if enc.shape[-1] < dim:
        enc = F.pad(enc, (0, dim - enc.shape[-1]))
    return enc


class SpectralTokenizer(nn.Module):
    def __init__(
        self, d_emb: int, patch: int, heads: int, spectral_tokens: int, adapter_depth: int = 1
    ) -> None:
        super().__init__()
        self.patch = patch
        self.d_emb = d_emb
        self.heads = heads
        self.spectral_tokens = spectral_tokens
        self.value_proj = nn.Linear(1, d_emb)
        self.query = nn.Parameter(torch.randn(1, 1, d_emb) * 0.02)
        self.k_proj = nn.Linear(d_emb, d_emb)
        self.v_proj = nn.Linear(d_emb, d_emb)
        self.o_proj = nn.Linear(d_emb, d_emb)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_emb), nn.Linear(d_emb, 4 * d_emb), nn.GELU(), nn.Linear(4 * d_emb, d_emb)
        )
        self.refine = nn.ModuleList(
            [ViTBlock(d_emb, heads, mlp_ratio=2) for _ in range(adapter_depth)]
        )

    def _band_tokens(self, spectra: torch.Tensor, wl: torch.Tensor) -> torch.Tensor:
        value = self.value_proj(spectra.unsqueeze(-1))
        wl_pe = sinusoidal(wl, self.d_emb, 200.0).view(1, 1, wl.shape[0], self.d_emb)
        return value + wl_pe

    def forward(
        self, cube: torch.Tensor, wl: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        n, bands, height, width = cube.shape
        rows, cols = height // self.patch, width // self.patch
        folded = cube.reshape(n, bands, rows, self.patch, cols, self.patch)
        per_patch = folded.mean(dim=(3, 5)).permute(0, 2, 3, 1).reshape(n, rows * cols, bands)

        tokens = self._band_tokens(per_patch, wl)
        flat = tokens.reshape(n * rows * cols, bands, self.d_emb)
        q = self.query.expand(flat.shape[0], 1, self.d_emb)
        k = self.k_proj(flat)
        v = self.v_proj(flat)
        attn = torch.softmax(q @ k.transpose(1, 2) / (self.d_emb**0.5), dim=-1)
        pooled = self.o_proj((attn @ v).squeeze(1)).reshape(n, rows * cols, self.d_emb)
        patch_tokens = pooled + self.ffn(pooled)
        for block in self.refine:
            patch_tokens = block(patch_tokens)

        band_mean = tokens.mean(dim=1)
        edges = torch.linspace(0, bands, self.spectral_tokens + 1).long()
        chunks = []
        for i in range(self.spectral_tokens):
            start = int(edges[i])
            stop = max(int(edges[i + 1]), start + 1)
            chunks.append(band_mean[:, start:stop].mean(dim=1))
        spectral = torch.stack(chunks, dim=1)
        return patch_tokens, spectral, (rows, cols)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, length, dim = x.shape
        qkv = self.qkv(x).reshape(n, length, 3, self.heads, dim // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = torch.softmax(q @ k.transpose(-2, -1) / ((dim // self.heads) ** 0.5), dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(n, length, dim)
        return self.proj(out)


class Mlp(nn.Module):
    def __init__(self, dim: int, ratio: int = 4) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, ratio * dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(ratio * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class ViTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class FrozenBackbone(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([ViTBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, tokens: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
        pos = grid_encoding(rows, cols, tokens.shape[-1], tokens.device).unsqueeze(0)
        x = tokens + pos
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


def load_pretrained(backbone: FrozenBackbone, path: str) -> None:
    state = torch.load(path, map_location="cpu")
    backbone.load_state_dict(state, strict=False)
