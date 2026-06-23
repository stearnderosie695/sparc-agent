from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from spectral_margin_agent.reels.corpora import CubeStack
from spectral_margin_agent.safelight.bench import seeded_generator
from spectral_margin_agent.safelight.glossary import Frame


def collate_frames(items: list[Frame]) -> Frame:
    return Frame(
        cube=torch.stack([it.cube for it in items], dim=0),
        wavelengths=items[0].wavelengths,
        label=torch.stack([it.label for it in items], dim=0),
        centre=torch.stack([it.centre for it in items], dim=0),
        corpus=items[0].corpus,
    )


def build_loader(
    stack: CubeStack,
    micro_batch: int,
    num_workers: int,
    seed: int,
    shuffle: bool,
) -> DataLoader[Frame]:
    return DataLoader(
        stack,
        batch_size=micro_batch,
        shuffle=shuffle,
        num_workers=0 if num_workers < 1 else num_workers,
        generator=seeded_generator(seed),
        collate_fn=collate_frames,
        drop_last=False,
    )
