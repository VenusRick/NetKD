"""Lightweight live training monitor for reporting metrics as models run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .status_tracker import TrainingStatusLogger


@dataclass
class StageHistory:
    """Container tracking progress for a specific training stage."""

    total_epochs: int
    records: List[Dict[str, float]] = field(default_factory=list)
    last_val_acc: Optional[float] = None


class LiveTrainingMonitor:
    """Console-friendly monitor that prints live metrics per epoch.

    The monitor keeps a small history per stage and highlights the
    change of validation accuracy to make convergence trends easy to see.
    """

    def __init__(self, verbose: bool = True, status_logger: TrainingStatusLogger | None = None):
        self.verbose = verbose
        self._stages: Dict[str, StageHistory] = {}
        self.status_logger = status_logger
        if self.status_logger:
            self.status_logger.reset()

    # ------------------------------------------------------------------
    # Stage lifecycle helpers
    # ------------------------------------------------------------------
    def stage_start(self, stage_name: str, total_epochs: int) -> None:
        self._stages[stage_name] = StageHistory(total_epochs=total_epochs)
        if self.verbose:
            print(f"[Monitor] >>> Stage '{stage_name}' started ({total_epochs} epochs)")

    def stage_end(self, stage_name: str) -> None:
        stage = self._stages.get(stage_name)
        if not stage:
            return
        if self.verbose:
            final_msg = "[Monitor] <<< Stage '{}' finished after {} epochs".format(
                stage_name, len(stage.records)
            )
            if stage.last_val_acc is not None:
                final_msg += f" | final acc={stage.last_val_acc:.4f}"
            print(final_msg)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def log(self, stage_name: str, epoch: int, metrics: Dict[str, float]) -> None:
        stage = self._stages.setdefault(stage_name, StageHistory(total_epochs=epoch))
        record = {"epoch": epoch}
        record.update(metrics)
        stage.records.append(record)

        delta_acc: Optional[float] = None
        val_acc = metrics.get("val_acc")
        if val_acc is not None:
            prev = stage.last_val_acc
            if prev is not None:
                delta_acc = val_acc - prev
            stage.last_val_acc = val_acc

        if self.verbose:
            parts = [f"epoch {epoch}/{stage.total_epochs}"]
            for key in ("train_loss", "val_loss", "val_acc"):
                if key in metrics and metrics[key] is not None:
                    parts.append(f"{key}={metrics[key]:.4f}")
            if delta_acc is not None:
                parts.append(f"Δacc={delta_acc:+.4f}")
            print(f"[Monitor][{stage_name}] " + " | ".join(parts))

        if self.status_logger:
            self.status_logger.log(stage_name, epoch, **metrics)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def history(self, stage_name: str) -> List[Dict[str, float]]:
        stage = self._stages.get(stage_name)
        return list(stage.records) if stage else []

    def as_dict(self) -> Dict[str, List[Dict[str, float]]]:
        return {name: list(stage.records) for name, stage in self._stages.items()}


__all__ = ["LiveTrainingMonitor"]
