from __future__ import annotations

from dataclasses import dataclass

import torch

from spectral_margin_agent.reels.chromo import mix_reflectance
from spectral_margin_agent.reels.grid import make_grid

NORMAL, TUMOUR, VESSEL, BACKGROUND = 0, 1, 2, 3


@dataclass(frozen=True)
class Planted:
    cube: torch.Tensor
    wavelengths: torch.Tensor
    label: torch.Tensor
    vessel: torch.Tensor
    nerve: torch.Tensor
    boundary: torch.Tensor
    centre: int


def _disk(tile: int, cy: float, cx: float, radius: float, gen: torch.Generator) -> torch.Tensor:
    ys = torch.arange(tile).view(-1, 1).float()
    xs = torch.arange(tile).view(1, -1).float()
    return ((ys - cy) ** 2 + (xs - cx) ** 2) <= radius**2


def _curve(tile: int, phase: float, amp: float, thickness: float) -> torch.Tensor:
    xs = torch.arange(tile).view(1, -1).float()
    ys = torch.arange(tile).view(-1, 1).float()
    path = 0.5 * tile + amp * torch.sin(2.0 * torch.pi * (xs / tile) + phase)
    return (ys - path).abs() <= thickness


def _line(tile: int, slope: float, intercept: float, thickness: float) -> torch.Tensor:
    xs = torch.arange(tile).view(1, -1).float()
    ys = torch.arange(tile).view(-1, 1).float()
    return (ys - (slope * xs + intercept)).abs() <= thickness


def plant_cube(
    seed: int,
    n_bands: int,
    band_lo: float,
    band_hi: float,
    tile: int,
    centre: int,
    water_gap: float = 0.5,
    noise: float = 0.12,
) -> Planted:
    gen = torch.Generator().manual_seed(seed)
    wl = make_grid(band_lo, band_hi, n_bands)

    cy = float(torch.rand(1, generator=gen) * tile * 0.4 + tile * 0.3)
    cx = float(torch.rand(1, generator=gen) * tile * 0.4 + tile * 0.3)
    radius = float(torch.rand(1, generator=gen) * tile * 0.12 + tile * 0.16)
    tumour = _disk(tile, cy, cx, radius, gen)
    vessel = _curve(tile, float(torch.rand(1, generator=gen) * torch.pi), tile * 0.12, 0.9)
    nerve = _line(tile, 0.6, tile * 0.1, 0.8)
    boundary = _disk(tile, cy, cx, radius + 2.0, gen) & ~_disk(tile, cy, cx, radius, gen)

    label = torch.full((tile, tile), NORMAL, dtype=torch.long)
    label[tumour] = TUMOUR
    label[vessel] = VESSEL

    centre_gain = 1.0 + 0.04 * (centre - 1)
    centre_offset = 0.01 * (centre - 1)

    water = torch.full((tile, tile), 0.45) + water_gap * tumour.float()
    water = water + noise * torch.randn((tile, tile), generator=gen)
    lipid = torch.full((tile, tile), 0.6) + noise * torch.randn((tile, tile), generator=gen)
    oxy = 0.05 + 0.9 * vessel.float() + noise * torch.randn((tile, tile), generator=gen)
    deoxy = 0.04 + 0.7 * vessel.float() + noise * torch.randn((tile, tile), generator=gen)

    conc = {
        "water": water.clamp_min(0.0),
        "lipid": lipid.clamp_min(0.0),
        "oxyhb": oxy.clamp_min(0.0),
        "deoxyhb": deoxy.clamp_min(0.0),
    }
    reflectance = mix_reflectance(conc, wl)
    reflectance = reflectance * centre_gain + centre_offset
    reflectance = reflectance + noise * 0.25 * torch.randn(reflectance.shape, generator=gen)
    cube = reflectance.clamp(1e-4, 1.5).permute(2, 0, 1).contiguous()

    return Planted(
        cube=cube,
        wavelengths=wl,
        label=label,
        vessel=vessel,
        nerve=nerve,
        boundary=boundary,
        centre=centre,
    )
