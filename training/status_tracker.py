"""Lightweight training status logger used for live monitoring scripts."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Optional


@dataclass
class StatusRecord:
    """Single training status entry."""

    stage: str
    epoch: int
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    val_acc: Optional[float] = None
    message: Optional[str] = None
    timestamp: float = field(default_factory=lambda: time.time())


class TrainingStatusLogger:
    """Append-only JSONL logger consumed by monitor_training_status.py."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, stage: str, epoch: int, **kwargs: Optional[float | str]):
        record = StatusRecord(stage=stage, epoch=epoch, **kwargs)
        with self.path.open("a", encoding="utf-8") as f:
            json.dump(asdict(record), f, ensure_ascii=False)
            f.write("\n")

    def reset(self):
        if self.path.exists():
            self.path.unlink()

    def latest(self) -> Dict[str, StatusRecord]:
        """Return the last record per stage (best-effort, used in tests)."""

        entries: Dict[str, StatusRecord] = {}
        if not self.path.exists():
            return entries
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                record = StatusRecord(**data)
                entries[record.stage] = record
        return entries


__all__ = ["TrainingStatusLogger", "StatusRecord"]
