from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from spectral_margin_agent.emulsion.backbone import FrozenBackbone, SpectralTokenizer
from spectral_margin_agent.emulsion.fusion import MambaFusion
from spectral_margin_agent.emulsion.lora import graft_lora
from spectral_margin_agent.safelight.dials import EmulsionDial

_LORA_TARGETS = ("qkv", "proj", "fc1", "fc2")


@dataclass
class EncoderOut:
    embedding: torch.Tensor
    patch_logits: torch.Tensor
    dense_logits: torch.Tensor
    grid: tuple[int, int]


class Encoder(nn.Module):
    def __init__(self, dial: EmulsionDial, n_classes: int) -> None:
        super().__init__()
        self.dial = dial
        self.n_classes = n_classes
        self.tokenizer = SpectralTokenizer(
            dial.d_emb, dial.patch, dial.backbone_heads, dial.spectral_tokens, dial.adapter_depth
        )
        self.backbone = FrozenBackbone(dial.d_emb, dial.backbone_depth, dial.backbone_heads)
        self.lora_blocks = 0
        for index, block in enumerate(self.backbone.blocks):
            if index % dial.lora_every == 0:
                self.lora_blocks += graft_lora(block, _LORA_TARGETS, dial.lora_rank)
        self.fusion = MambaFusion(
            dial.d_emb, dial.fuse_width, dial.d_state, dial.mamba_blocks, dial.mamba_expand
        )
        self.head = nn.Sequential(nn.LayerNorm(dial.d_emb), nn.Linear(dial.d_emb, n_classes))

    def forward(self, cube: torch.Tensor, wl: torch.Tensor) -> EncoderOut:
        patch_tokens, spectral_tokens, (rows, cols) = self.tokenizer(cube, wl)
        contextual = self.backbone(patch_tokens, rows, cols)
        fused = self.fusion(contextual, spectral_tokens)
        patch_logits = self.head(fused)
        dense = patch_logits.transpose(1, 2).reshape(cube.shape[0], self.n_classes, rows, cols)
        dense = F.interpolate(dense, size=cube.shape[-2:], mode="bilinear", align_corners=False)
        return EncoderOut(
            embedding=fused, patch_logits=patch_logits, dense_logits=dense, grid=(rows, cols)
        )

    def tumour_map(self, out: EncoderOut, tumour_index: int) -> torch.Tensor:
        return torch.softmax(out.dense_logits, dim=1)[:, tumour_index]
