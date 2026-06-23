from __future__ import annotations

import torch
import torch.nn.functional as F

from spectral_margin_agent.enlarger.orchestrator import Rollout
from spectral_margin_agent.enlarger.reward import discounted_returns


def advantage_objective(
    rollout: Rollout, gamma: float, value_coef: float, entropy_coef: float
) -> tuple[torch.Tensor, dict[str, float]]:
    returns = discounted_returns(rollout.rewards.tolist(), gamma).to(rollout.values.device)
    values = rollout.values
    advantage = returns - values.detach()
    log_probs = rollout.log_probs.reshape(-1)
    policy_loss = -(log_probs * advantage).mean()
    value_loss = F.mse_loss(values, returns)
    entropy = rollout.entropies.mean()
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    metrics = {
        "policy": float(policy_loss.detach()),
        "value": float(value_loss.detach()),
        "entropy": float(entropy.detach()),
        "return": float(returns[-1]),
    }
    return loss, metrics


def gather_workers(
    rollouts: list[Rollout], gamma: float, value_coef: float, entropy_coef: float
) -> tuple[torch.Tensor, dict[str, float]]:
    losses = []
    aggregate: dict[str, float] = {}
    for rollout in rollouts:
        loss, metrics = advantage_objective(rollout, gamma, value_coef, entropy_coef)
        losses.append(loss)
        for key, value in metrics.items():
            aggregate[key] = aggregate.get(key, 0.0) + value / len(rollouts)
    return torch.stack(losses).mean(), aggregate
