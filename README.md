# spectral_margin_agent

A darkroom for surgical hyperspectral cubes. The raw negative is an intra-operative
HSI cube; the print on the easel is a structure-preserving resection contour with a
calibrated, auditable trail of how it was reached. The pipeline reads the tumour/normal
boundary from the 975 nm water–lipid slope, routes five callable tools from a frozen
spectral backbone through a confidence-calibrated controller, and resolves the
margin-versus-preservation trade-off as a learned Pareto front rather than a single
recommendation.

This repository is the working bench for the SPARC-Agent study: a frozen HyperSIGMA
backbone with a LoRA surgical adapter and a Mamba spectral/spatial fusion head, an
anatomical reasoner over atlas and vessel/nerve evidence, the CCATR tool router with a
sub-1M-parameter policy, the PSL-SURG margin–function hypernetwork, and the
physiology-aware cross-tissue transfer step.

## Loading the bath

The reporting stack is pinned to an older CUDA/PyTorch line on purpose; the numbers in
this repository were exposed on that emulsion and nothing else is supported.

    # pip, from the pinned index
    python -m venv .bench && . .bench/bin/activate
    pip install -r requirements.txt
    pip install -e .

    # conda
    conda env create -f environment.yml
    conda activate smarg
    pip install -e .

    # container (CUDA 11.8 / Torch 2.3.0)
    docker build -t smarg .
    docker run --gpus all --rm smarg read --table main

The `accel` extra pulls the real `mamba-ssm`, `causal-conv1d`, and `transformers`
backends. Without it the selective-state-space scan and adapter run in pure Torch, which
is what the test bench uses.

## Plates

Every corpus consumed by the auxiliary layer is listed in `plate_sources.txt` with its
band range, split convention, licence, and access route. Four public corpora are used
for backbone pre-training, the cross-tissue invariance analysis, and external
comparison: HELICoiD (brain), HeiPorSPECTRAL (porcine), ODSI-DB (oral), and HSI
Dermoscopy (skin). The primary-validation sarcoma layer is a private de-identified
clinical cohort and is not redistributable; when its files are absent under `plates/`
the loader draws a deterministic planted-signal stand-in that carries the 975 nm slope
invariant, so the bench runs end to end without the restricted data.

To stage a corpus, place its prepared `index.pt` under `plates/<corpus>/`. Absent that,
the planted stand-in is drawn and a line is logged so the substitution is never silent.

## Developing

One command per reported run. Defaults live in `recipes/session/main.yaml`; override any
field on the command line with `--set group.key=value`.

    # primary sarcoma run (heavy reference recipe)
    darkroom develop --recipe recipes/session/main.yaml --out darkroom_runs/main/negative.pt

    # cross-tissue, ablations, and the five hypothesis recipes
    darkroom develop --recipe recipes/session/h1_helicoid.yaml
    darkroom develop --recipe recipes/session/ablation_no_mamba.yaml
    darkroom develop --recipe recipes/session/ablation_no_transfer.yaml

The objective stacks supervised margin segmentation, the PSL-SURG cosine
direction-alignment loss over Dirichlet preferences, the asynchronous advantage
actor–critic update for the tool-routing policy, and the confidence-gated transfer
consistency term.

## Reading the negative

    darkroom read --recipe recipes/session/main.yaml --ckpt darkroom_runs/main/negative.pt --table main
    darkroom read --recipe recipes/session/cross_tissue.yaml --table cross_tissue
    darkroom read --recipe recipes/session/biomarker.yaml --table biomarker
    darkroom read --recipe recipes/session/pareto.yaml --table pareto
    darkroom expose --recipe recipes/session/main.yaml --ckpt darkroom_runs/main/negative.pt
    darkroom transfer --recipe recipes/session/main.yaml --ckpt darkroom_runs/main/negative.pt --onnx encoder.onnx

Reference values on the reporting recipe, with tolerances:

| Readout | Expected |
| --- | --- |
| Intra-operative margin AUROC | 0.93 (0.91–0.95) |
| Tumour-class sensitivity | 0.88 (0.85–0.91) |
| HELICoiD inter-patient macro-F1 / tumour-sens | 0.78 / 0.73 |
| HeiPorSPECTRAL margin Dice | 0.925 ± 0.015 |
| ODSI-DB macro-F1 | 0.80 ± 0.02 |
| HSI Dermoscopy melanoma AUROC | 0.91 ± 0.02 |
| 975 nm spectral concentration (×uniform) | 1.79 |
| Masked 950–1000 nm AUROC change | −8.0 pp |
| Band-importance rank correlation vs pooled | 0.78 |
| Pareto hypervolume / preservation / margin | 0.65 / 87% / 3.4 mm |
| Critical-structure violation rate | 7% |

The planted stand-in reproduces the margin headline (AUROC ≈ 0.93, sensitivity ≈ 0.88)
deterministically under the recorded seed, and the band-importance peak sits inside the
950–1000 nm window.

## Bench timing

The reporting recipe is not a quick-start. It is eight A100-class GPUs in DDP, an
effective batch of 1024 cubes (micro-batch 2 × grad-accum 64 × world-size 8), bf16 with
tf32 matmul, an extremely low learning rate (3e-6) under a 12 000-step warmup and cosine
decay, and 800 epochs with 20 seeds per dataset per condition. The frozen backbone holds
≈ 78 M parameters; the trainable surgical adapter, Mamba fusion head, CCATR router, and
PSL-SURG hypernetwork together hold ≈ 24 M. Expect tens of GPU-days for a full seed
sweep and roughly 100 GB of staged corpora on disk. `recipes/session/_probe.yaml` is a
CPU fixture for the test bench only and is not a reporting configuration.

## Contact sheet

    spectral_margin_agent/
      reels/         corpora, band resampling, 975 nm slope feature, planted cohort
      emulsion/      frozen backbone, LoRA adapter, selective-SSM Mamba fusion head
      register/      atlas alignment, vessel/nerve detection, proximity fields
      enlarger/      five tools, CCATR router, A3C policy, the planning orchestrator
      dodge_burn/    PSL-SURG hypernetwork, Pareto geometry, hypervolume regret
      toning/        Beer–Lambert chromophore re-parameterization, confidence gate
      developing/    trainer, optimiser, schedule, EMA, A3C objective
      contact_sheet/ evaluation loops and per-table experiment runners
      densitometry/  AUROC, DeLong, Wilcoxon, Cohen d, FDR, ECE, hypervolume, biomarker
      safelight/     dials (typed config), seeding, atomic IO, recipe loading
      console/       develop / read / expose / transfer entry points

## Proof prints

    pytest                         # property, shape, overfit, numerical, integration, end-to-end
    ruff check spectral_margin_agent proofs
    mypy

The suite checks Pareto dominance and hypervolume against hand values, the O(1/√N)
hypervolume-regret envelope, encoder behaviour across 30–826 bands, single-batch overfit
of the hypernetwork and encoder, DeLong against the closed-form AUC, the planted 975 nm
invariant, the tool-routing loop with its provenance accounting, and a two-step develop
on the CPU fixture.
