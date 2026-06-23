from __future__ import annotations

import logging
import os
import random
import tempfile
from typing import Any

import numpy as np
import torch
from torch import nn

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seeded_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def pick_device(preference: str) -> torch.device:
    if preference.startswith("cuda") and torch.cuda.is_available():
        return torch.device(preference)
    return torch.device("cpu")


def atomic_save(payload: dict[str, Any], path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".part")
    os.close(fd)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def count_trainable(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def count_total(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
