from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from spectral_margin_agent.developing.a3c import gather_workers
from spectral_margin_agent.developing.ema import Ema
from spectral_margin_agent.developing.objectives import (
    pareto_front_loss,
    segmentation_loss,
    transfer_consistency,
)
from spectral_margin_agent.developing.optim import build_optimizer, build_scheduler
from spectral_margin_agent.dodge_burn.hypernet import sample_preferences
from spectral_margin_agent.enlarger.orchestrator import Rollout, SparcAgent
from spectral_margin_agent.reels.spool import Frame
from spectral_margin_agent.safelight.bench import atomic_save, get_logger, seeded_generator
from spectral_margin_agent.safelight.dials import BathDial, EnlargerDial

_LOG = get_logger("developing.trainer")


@dataclass
class StepReport:
    total: float
    segmentation: float
    pareto: float
    transfer: float
    policy: float
    value: float


class Trainer:
    def __init__(
        self,
        agent: SparcAgent,
        bath: BathDial,
        enlarger: EnlargerDial,
        device: torch.device,
        seed: int,
        total_steps: int,
    ) -> None:
        self.agent = agent.to(device)
        self.bath = bath
        self.enlarger = enlarger
        self.device = device
        self.seed = seed
        self.optimizer = build_optimizer(self.agent, bath)
        self.scheduler = build_scheduler(self.optimizer, bath, total_steps)
        self.ema = Ema(self.agent, bath.ema_decay)
        self.generator = seeded_generator(seed)
        self.step_index = 0

    def _rollouts(self, cube: torch.Tensor, wl: torch.Tensor) -> list[Rollout]:
        workers = self.bath.a3c_workers
        prefs = sample_preferences(self.enlarger.pareto.dirichlet_alpha, workers, self.generator)
        rollouts = []
        for w in range(workers):
            sample = cube[w % cube.shape[0]].unsqueeze(0)
            result = self.agent.plan(sample, wl, prefs[w].view(1, 3).to(self.device), explore=True)
            rollouts.append(result.rollout)
        return rollouts

    def train_step(self, frame: Frame) -> StepReport:
        self.agent.train()
        cube = frame.cube.to(self.device)
        labels = frame.label.to(self.device)
        wl = frame.wavelengths.to(self.device)

        encoded = self.agent.encoder(cube, wl)
        seg = segmentation_loss(encoded.dense_logits, labels)
        psl = pareto_front_loss(self.agent.hypernet, self.enlarger.pareto, self.generator)
        transfer = transfer_consistency(
            self.agent.encoder, cube, wl, self.enlarger.tumour_index, 0.5, self.generator
        )
        policy_loss, policy_metrics = gather_workers(
            self._rollouts(cube, wl),
            self.enlarger.gamma,
            self.bath.value_coef,
            self.bath.entropy_coef,
        )
        total = seg + self.bath.pareto_coef * psl + self.bath.transfer_coef * transfer + policy_loss

        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.bath.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        self.ema.update(self.agent)
        self.step_index += 1

        return StepReport(
            total=float(total.detach()),
            segmentation=float(seg.detach()),
            pareto=float(psl.detach()),
            transfer=float(transfer.detach()),
            policy=policy_metrics["policy"],
            value=policy_metrics["value"],
        )

    def fit(self, loader: DataLoader[Frame], max_steps: int) -> list[StepReport]:
        history = []
        for frame in loader:
            if self.step_index >= max_steps:
                break
            report = self.train_step(frame)
            history.append(report)
            if self.step_index % max(self.bath.a3c_workers, 1) == 0:
                _LOG.info("step %d loss %.4f", self.step_index, report.total)
        return history

    def save(self, path: str) -> None:
        atomic_save(
            {
                "model": self.agent.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "ema": self.ema.shadow,
                "seed": self.seed,
                "step": self.step_index,
            },
            path,
        )

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.ema.shadow = state["ema"]
        self.seed = state["seed"]
        self.step_index = state["step"]
