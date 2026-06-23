from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from spectral_margin_agent.reels.synth import plant_cube
from spectral_margin_agent.safelight.bench import get_logger
from spectral_margin_agent.safelight.glossary import Frame

_LOG = get_logger("reels.corpora")


@dataclass(frozen=True)
class CorpusSpec:
    key: str
    n_bands: int
    band_lo: float
    band_hi: float
    n_classes: int
    n_centres: int
    split_kind: str


CORPORA: dict[str, CorpusSpec] = {
    "sarcoma_synth": CorpusSpec("sarcoma_synth", 826, 400.0, 1000.0, 4, 4, "centre"),
    "helicoid": CorpusSpec("helicoid", 826, 400.0, 1000.0, 4, 1, "patient"),
    "heiporspectral": CorpusSpec("heiporspectral", 100, 500.0, 1000.0, 4, 11, "pig"),
    "odsi_db": CorpusSpec("odsi_db", 204, 380.0, 1000.0, 4, 1, "official"),
    "hsi_dermoscopy": CorpusSpec("hsi_dermoscopy", 16, 465.0, 630.0, 2, 2, "lesion"),
}

_SPLIT_FRACTIONS = {"train": (0.0, 0.6), "val": (0.6, 0.8), "test": (0.8, 1.0)}


class CubeStack(Dataset[Frame]):
    def __init__(self, frames: list[Frame]) -> None:
        self._frames = frames

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, index: int) -> Frame:
        return self._frames[index]


def _disk_index(root: str, key: str) -> str | None:
    candidate = os.path.join(root, key, "index.pt")
    return candidate if os.path.exists(candidate) else None


def _synth_frame(spec: CorpusSpec, seed: int, tile: int) -> Frame:
    centre = 1 + (seed % spec.n_centres)
    planted = plant_cube(
        seed=seed,
        n_bands=spec.n_bands,
        band_lo=spec.band_lo,
        band_hi=spec.band_hi,
        tile=tile,
        centre=centre,
    )
    label = planted.label.clamp_max(spec.n_classes - 1)
    return Frame(
        cube=planted.cube,
        wavelengths=planted.wavelengths,
        label=label,
        centre=torch.tensor(centre, dtype=torch.long),
        corpus=spec.key,
    )


def build_split(
    corpus: str, split: str, n_items: int, seed: int, root: str, tile: int
) -> CubeStack:
    if corpus not in CORPORA:
        raise KeyError(f"unknown corpus {corpus}")
    spec = CORPORA[corpus]
    index_path = _disk_index(root, corpus)
    if index_path is not None:
        bundle = torch.load(index_path)
        frames = [bundle[i] for i in bundle["splits"][split]]
        return CubeStack(frames)
    _LOG.info("corpus %s absent under %s; drawing planted stand-in", corpus, root)
    lo, hi = _SPLIT_FRACTIONS[split]
    base = seed * 100003 + hash(split) % 9973
    frames = [_synth_frame(spec, base + i, tile) for i in range(n_items)]
    span = frames[int(lo * len(frames)) : max(int(hi * len(frames)), int(lo * len(frames)) + 1)]
    return CubeStack(span)
