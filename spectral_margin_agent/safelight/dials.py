from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RunDial:
    seed: int = 20
    device: str = "cuda"
    out_dir: str = "darkroom_runs/main"
    world_size: int = 8
    precision: str = "bf16"
    log_every: int = 50


@dataclass
class StockDial:
    corpus: str = "sarcoma_synth"
    root: str = "plates"
    n_bands: int = 826
    band_lo: float = 400.0
    band_hi: float = 1000.0
    tile: int = 256
    micro_batch: int = 2
    num_workers: int = 8
    slope_center: float = 975.0
    slope_halfwidth: float = 25.0
    synth_patients: int = 386
    synth_centres: int = 4
    synth_regions: int = 2


@dataclass
class EmulsionDial:
    in_bands: int = 826
    patch: int = 16
    d_emb: int = 768
    backbone_depth: int = 11
    backbone_heads: int = 8
    spectral_tokens: int = 64
    adapter_depth: int = 1
    lora_rank: int = 16
    lora_every: int = 2
    mamba_blocks: int = 6
    fuse_width: int = 512
    d_state: int = 16
    mamba_expand: int = 1
    freeze_backbone: bool = True


@dataclass
class ParetoDial:
    film_layers: int = 6
    width: int = 512
    pref_embed: int = 256
    context_dim: int = 8
    dirichlet_alpha: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    n_pref_samples: int = 64
    eps_c: float = 0.1


@dataclass
class EnlargerDial:
    n_classes: int = 4
    tumour_index: int = 1
    tau_spec: float = 0.55
    tau_atlas: float = 0.5
    t_calib: float = 1.7
    latency_budget_s: float = 5.2
    planning_steps: int = 15
    reward_alpha: list[float] = field(default_factory=lambda: [1.0, 1.0, 0.05, 2.0])
    margin_threshold_mm: float = 1.0
    pixel_mm: float = 0.4
    max_radius: int = 6
    eps_overlap: float = 0.05
    homogeneity_thresh: float = 0.06
    gamma: float = 0.99
    router_hidden: int = 512
    router_depth: int = 3
    pareto: ParetoDial = field(default_factory=ParetoDial)


@dataclass
class BathDial:
    epochs: int = 800
    lr: float = 3.0e-6
    warmup_steps: int = 12000
    weight_decay: float = 0.05
    grad_accum: int = 64
    grad_clip: float = 1.0
    ema_decay: float = 0.9999
    scheduler: str = "cosine"
    a3c_workers: int = 8
    a3c_sync: bool = False
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    pareto_coef: float = 0.5
    transfer_coef: float = 0.1


@dataclass
class NegativeDial:
    seeds: int = 20
    bootstrap: int = 2000
    fdr_q: float = 0.05
    pareto_ref: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class Recipe:
    run: RunDial = field(default_factory=RunDial)
    stock: StockDial = field(default_factory=StockDial)
    emulsion: EmulsionDial = field(default_factory=EmulsionDial)
    enlarger: EnlargerDial = field(default_factory=EnlargerDial)
    bath: BathDial = field(default_factory=BathDial)
    negative: NegativeDial = field(default_factory=NegativeDial)


def effective_batch(recipe: Recipe) -> int:
    return recipe.stock.micro_batch * recipe.bath.grad_accum * recipe.run.world_size
