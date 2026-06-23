from __future__ import annotations

import torch

from spectral_margin_agent.reels.chromo import CHROMOPHORES, absorption_basis, mix_reflectance


def basis_matrix(wl: torch.Tensor) -> torch.Tensor:
    basis = absorption_basis(wl)
    return torch.stack([basis[name] for name in CHROMOPHORES], dim=-1)


def estimate_concentrations(cube: torch.Tensor, wl: torch.Tensor) -> dict[str, torch.Tensor]:
    spectra = cube.movedim(1, -1)
    absorbance = -torch.log(spectra.clamp(1e-4, 1.0))
    matrix = basis_matrix(wl).to(absorbance.dtype)
    pinv = torch.linalg.pinv(matrix)
    coeffs = absorbance @ pinv.t()
    coeffs = coeffs.clamp_min(0.0)
    return {name: coeffs[..., i] for i, name in enumerate(CHROMOPHORES)}


def resynthesize(concentrations: dict[str, torch.Tensor], wl: torch.Tensor) -> torch.Tensor:
    reflectance = mix_reflectance(concentrations, wl)
    return reflectance.movedim(-1, 1)


def reparameterize(cube: torch.Tensor, wl: torch.Tensor, shift: dict[str, float]) -> torch.Tensor:
    concentrations = estimate_concentrations(cube, wl)
    perturbed = {name: conc * (1.0 + shift.get(name, 0.0)) for name, conc in concentrations.items()}
    return resynthesize(perturbed, wl)


def sample_shift(generator: torch.Generator, scale: float = 0.3) -> dict[str, float]:
    draw = torch.randn(len(CHROMOPHORES), generator=generator) * scale
    return {name: float(draw[i]) for i, name in enumerate(CHROMOPHORES)}
