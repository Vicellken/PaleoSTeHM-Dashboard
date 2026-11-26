"""Streamlit progress helpers for GP training."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional


@dataclass
class RunProgress:
    stage: str = "idle"
    stage_message: str = ""
    iteration: int = 0
    total_iterations: int = 0
    loss_history: List[float] = field(default_factory=list)
    log_lines: List[str] = field(default_factory=list)
    start_time: float = 0.0

    def set_stage(self, stage_id: str, message: str) -> None:
        self.stage = stage_id
        self.stage_message = message
        self.log_lines.append(f"[{stage_id}] {message}")

    def on_iteration(
        self, iteration: int, total: int, loss: float, _track_row: list
    ) -> None:
        if self.start_time == 0.0:
            self.start_time = time.time()
        self.iteration = iteration
        self.total_iterations = total
        self.loss_history.append(float(loss))

    @property
    def elapsed_s(self) -> float:
        if self.start_time == 0.0:
            return 0.0
        return max(0.0, time.time() - self.start_time)

    @property
    def iterations_per_second(self) -> float:
        if self.iteration <= 0 or self.elapsed_s <= 0:
            return 0.0
        return self.iteration / self.elapsed_s

    @property
    def eta_s(self) -> Optional[float]:
        if self.iteration <= 0 or self.total_iterations <= 0:
            return None
        rate = self.iterations_per_second
        if rate <= 0:
            return None
        remaining = self.total_iterations - self.iteration
        return remaining / rate

    def format_training_status(self) -> str:
        if self.total_iterations <= 0:
            return "Training…"
        parts = [f"Iteration {self.iteration}/{self.total_iterations}"]
        rate = self.iterations_per_second
        if rate > 0:
            parts.append(f"{rate:.2f} iter/s")
        eta = self.eta_s
        if eta is not None:
            if eta >= 3600:
                parts.append(f"ETA {eta / 3600:.1f} h")
            elif eta >= 60:
                parts.append(f"ETA {eta / 60:.1f} min")
            else:
                parts.append(f"ETA {eta:.0f} s")
        if self.loss_history:
            parts.append(f"loss {self.loss_history[-1]:.4f}")
        return " · ".join(parts)

    _POST_TRAIN_STAGES = frozenset(
        {
            "save_model",
            "nll",
            "boundary",
            "temporal",
            "budget",
            "spatial",
            "spatial_rates",
            "load_model",
            "complete",
        }
    )

    def format_status(self) -> str:
        training_done = (
            self.total_iterations > 0 and self.iteration >= self.total_iterations
        )
        if (training_done or self.stage in self._POST_TRAIN_STAGES) and self.stage_message:
            return self.stage_message
        if self.total_iterations > 0 and self.iteration > 0:
            return self.format_training_status()
        if self.stage_message:
            return self.stage_message
        return "Starting…"

    def status_label(self) -> str:
        if self.stage == "train" or (
            self.total_iterations > 0 and self.iteration < self.total_iterations
        ):
            return "Training"
        if self.stage in self._POST_TRAIN_STAGES or (
            self.total_iterations > 0 and self.iteration >= self.total_iterations
        ):
            return "Post-processing"
        return "Stage"


def make_progress_callbacks(
    progress: RunProgress,
) -> tuple[Callable[[int, int, float, list], None], Callable[[str, str], None]]:
    def on_iteration(iteration: int, total: int, loss: float, track_row: list) -> None:
        progress.on_iteration(iteration, total, loss, track_row)

    def on_stage(stage_id: str, message: str) -> None:
        progress.set_stage(stage_id, message)

    return on_iteration, on_stage
