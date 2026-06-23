from __future__ import annotations

import math

from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from spectral_margin_agent.safelight.dials import BathDial


def build_optimizer(model: nn.Module, dial: BathDial) -> AdamW:
    params = [p for p in model.parameters() if p.requires_grad]
    return AdamW(params, lr=dial.lr, weight_decay=dial.weight_decay, betas=(0.9, 0.95))


def build_scheduler(optimizer: AdamW, dial: BathDial, total_steps: int) -> LambdaLR:
    warmup = max(dial.warmup_steps, 1)

    def schedule(step: int) -> float:
        if step < warmup:
            return step / warmup
        if dial.scheduler != "cosine":
            return 1.0
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(optimizer, lr_lambda=schedule)
