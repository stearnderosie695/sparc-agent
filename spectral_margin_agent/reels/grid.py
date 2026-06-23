from __future__ import annotations

import torch


def make_grid(lo: float, hi: float, n_bands: int) -> torch.Tensor:
    return torch.linspace(lo, hi, n_bands)


def resample_to_grid(
    spectra: torch.Tensor, src_wl: torch.Tensor, dst_wl: torch.Tensor
) -> torch.Tensor:
    src = src_wl.to(spectra.dtype)
    dst = dst_wl.to(spectra.dtype)
    flat = spectra.reshape(-1, spectra.shape[-1])
    idx = torch.searchsorted(src, dst).clamp(1, src.numel() - 1)
    lo = idx - 1
    w_lo = src[lo]
    w_hi = src[idx]
    span = (w_hi - w_lo).clamp_min(1e-6)
    frac = ((dst - w_lo) / span).clamp(0.0, 1.0)
    left = flat.index_select(-1, lo)
    right = flat.index_select(-1, idx)
    out = left + (right - left) * frac
    return out.reshape(*spectra.shape[:-1], dst.numel())


def snv(spectra: torch.Tensor) -> torch.Tensor:
    mean = spectra.mean(dim=-1, keepdim=True)
    std = spectra.std(dim=-1, keepdim=True).clamp_min(1e-6)
    return (spectra - mean) / std


def window_mask(wl: torch.Tensor, center: float, halfwidth: float) -> torch.Tensor:
    return (wl >= center - halfwidth) & (wl <= center + halfwidth)


def slope_feature(
    spectra: torch.Tensor, wl: torch.Tensor, center: float, halfwidth: float
) -> torch.Tensor:
    mask = window_mask(wl, center, halfwidth)
    if int(mask.sum()) < 2:
        return torch.zeros(spectra.shape[:-1], dtype=spectra.dtype, device=spectra.device)
    w = wl[mask].to(spectra.dtype)
    r = spectra[..., mask]
    w_bar = w.mean()
    r_bar = r.mean(dim=-1, keepdim=True)
    num = ((w - w_bar) * (r - r_bar)).sum(dim=-1)
    den = ((w - w_bar) ** 2).sum().clamp_min(1e-6)
    return num / den
