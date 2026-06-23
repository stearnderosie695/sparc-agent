from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from spectral_margin_agent.densitometry.classification import (
    auroc,
    class_sensitivity,
    dice,
    expected_calibration_error,
    macro_f1,
)
from spectral_margin_agent.enlarger.orchestrator import SparcAgent
from spectral_margin_agent.reels.spool import Frame

NORMAL = 0


@dataclass
class MarginReport:
    auroc: float
    tumour_sensitivity: float
    dice: float
    macro_f1: float
    ece: float


@torch.no_grad()
def evaluate_margin(
    agent: SparcAgent,
    loader: DataLoader[Frame],
    tumour_index: int,
    t_calib: float,
    device: torch.device,
) -> MarginReport:
    agent.eval()
    scores, binary, preds, golds, confs = [], [], [], [], []
    dice_scores = []
    for frame in loader:
        cube = frame.cube.to(device)
        labels = frame.label.to(device)
        wl = frame.wavelengths.to(device)
        posterior = torch.softmax(agent.encoder(cube, wl).dense_logits / t_calib, dim=1)
        tumour = posterior[:, tumour_index]
        prediction = posterior.argmax(dim=1)
        confidence = posterior.amax(dim=1)
        keep = (labels == NORMAL) | (labels == tumour_index)
        scores.append(tumour[keep].cpu().numpy())
        binary.append((labels[keep] == tumour_index).cpu().numpy().astype(int))
        preds.append(prediction.reshape(-1).cpu().numpy())
        golds.append(labels.reshape(-1).cpu().numpy())
        confs.append(confidence.reshape(-1).cpu().numpy())
        dice_scores.append(
            dice((prediction == tumour_index).cpu().numpy(), (labels == tumour_index).cpu().numpy())
        )
    score = np.concatenate(scores)
    label = np.concatenate(binary)
    prediction = np.concatenate(preds)
    gold = np.concatenate(golds)
    confidence = np.concatenate(confs)
    return MarginReport(
        auroc=auroc(label, score),
        tumour_sensitivity=class_sensitivity(gold, prediction, tumour_index),
        dice=float(np.mean(dice_scores)),
        macro_f1=macro_f1(gold, prediction),
        ece=expected_calibration_error(gold, confidence, prediction),
    )


@torch.no_grad()
def margin_scores(
    agent: SparcAgent, loader: DataLoader[Frame], tumour_index: int, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    agent.eval()
    scores, labels = [], []
    for frame in loader:
        cube = frame.cube.to(device)
        gold = frame.label.to(device)
        wl = frame.wavelengths.to(device)
        tumour = torch.softmax(agent.encoder(cube, wl).dense_logits, dim=1)[:, tumour_index]
        keep = (gold == NORMAL) | (gold == tumour_index)
        scores.append(tumour[keep].cpu().numpy())
        labels.append((gold[keep] == tumour_index).cpu().numpy().astype(int))
    return np.concatenate(scores), np.concatenate(labels)
