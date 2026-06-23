from __future__ import annotations

import torch

CHROMOPHORES: tuple[str, ...] = ("water", "lipid", "oxyhb", "deoxyhb")


def _gauss(wl: torch.Tensor, center: float, width: float, height: float) -> torch.Tensor:
    return height * torch.exp(-0.5 * ((wl - center) / width) ** 2)


def _edge(wl: torch.Tensor, center: float, scale: float, height: float) -> torch.Tensor:
    return height * torch.sigmoid((wl - center) / scale)


def absorption_basis(wl: torch.Tensor) -> dict[str, torch.Tensor]:
    water = _edge(wl, 975.0, 11.0, 1.4) + _gauss(wl, 740.0, 35.0, 0.18)
    lipid = _gauss(wl, 930.0, 38.0, 0.6) + _gauss(wl, 1040.0, 45.0, 0.4)
    oxyhb = _gauss(wl, 542.0, 16.0, 1.1) + _gauss(wl, 576.0, 14.0, 1.2)
    deoxyhb = _gauss(wl, 556.0, 30.0, 1.3) + _gauss(wl, 758.0, 28.0, 0.35)
    return {"water": water, "lipid": lipid, "oxyhb": oxyhb, "deoxyhb": deoxyhb}


def scatter(wl: torch.Tensor, amplitude: float, exponent: float) -> torch.Tensor:
    ref = wl / 500.0
    return amplitude * ref.clamp_min(1e-3) ** (-exponent)


def mix_reflectance(
    concentrations: dict[str, torch.Tensor],
    wl: torch.Tensor,
    scatter_amp: float = 0.8,
    scatter_exp: float = 0.6,
) -> torch.Tensor:
    basis = absorption_basis(wl)
    absorbance = torch.zeros(
        (*next(iter(concentrations.values())).shape, wl.numel()),
        dtype=wl.dtype,
        device=wl.device,
    )
    for name, conc in concentrations.items():
        absorbance = absorbance + conc.unsqueeze(-1) * basis[name].view(
            *([1] * conc.dim()), wl.numel()
        )
    base = scatter(wl, scatter_amp, scatter_exp)
    reflectance = torch.exp(-absorbance) * base.view(*([1] * (absorbance.dim() - 1)), wl.numel())
    return reflectance.clamp(1e-4, 1.0)
