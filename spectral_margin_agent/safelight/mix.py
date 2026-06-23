from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from omegaconf import OmegaConf

from spectral_margin_agent.safelight.dials import Recipe


def blank_recipe() -> Recipe:
    schema = OmegaConf.structured(Recipe)
    return cast(Recipe, OmegaConf.to_object(schema))


def load_recipe(path: str | None, overrides: Sequence[str] | None = None) -> Recipe:
    schema = OmegaConf.structured(Recipe)
    merged = schema
    if path is not None:
        merged = OmegaConf.merge(merged, OmegaConf.load(path))
    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(list(overrides)))
    return cast(Recipe, OmegaConf.to_object(merged))


def dump_recipe(recipe: Recipe, path: str) -> None:
    conf = OmegaConf.structured(recipe)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(OmegaConf.to_yaml(conf))
