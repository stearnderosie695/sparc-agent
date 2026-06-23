from __future__ import annotations

import torch


def composite_reward(
    achieved_margin: float,
    margin_threshold: float,
    atlas_overlap: float,
    history_len: int,
    verified: bool,
    alpha: list[float],
) -> float:
    margin_term = alpha[0] * float(achieved_margin >= margin_threshold)
    atlas_term = alpha[1] * float(atlas_overlap == 0.0)
    latency_term = alpha[2] * float(history_len)
    verify_term = alpha[3] * float(not verified)
    return margin_term + atlas_term - latency_term - verify_term


def discounted_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    out = torch.zeros(len(rewards))
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * running
        out[t] = running
    return out
