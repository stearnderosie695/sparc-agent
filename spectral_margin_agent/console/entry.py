from __future__ import annotations

from docopt import docopt

from spectral_margin_agent.console import handlers

USAGE = """darkroom

Usage:
  darkroom develop [--recipe=PATH] [--steps=N] [--out=PATH] [--set=KV]...
  darkroom read [--recipe=PATH] [--ckpt=PATH] [--table=NAME] [--set=KV]...
  darkroom expose [--recipe=PATH] [--ckpt=PATH] [--set=KV]...
  darkroom transfer [--recipe=PATH] [--ckpt=PATH] [--onnx=PATH] [--set=KV]...
  darkroom (-h | --help)

Options:
  -h --help      Show this screen.
  --recipe=PATH  OmegaConf recipe file.
  --steps=N      Override the developed step count.
  --out=PATH     Checkpoint output path.
  --ckpt=PATH    Checkpoint to load.
  --table=NAME   One of main, cross_tissue, pareto, biomarker.
  --onnx=PATH    ONNX export target.
  --set=KV       Dotted recipe override, e.g. --set run.seed=7.
"""


def main() -> None:
    args = docopt(USAGE)
    overrides = args["--set"]
    recipe = args["--recipe"]
    if args["develop"]:
        steps = int(args["--steps"]) if args["--steps"] else None
        handlers.develop(recipe, overrides, steps, args["--out"])
    elif args["read"]:
        handlers.read(recipe, overrides, args["--ckpt"], args["--table"])
    elif args["expose"]:
        handlers.expose(recipe, overrides, args["--ckpt"])
    elif args["transfer"]:
        handlers.transfer(recipe, overrides, args["--ckpt"], args["--onnx"])


if __name__ == "__main__":
    main()
