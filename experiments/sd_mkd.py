"""High-level SD-MKD experiment runner backed by the real-data pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

from train_with_real_data import run_pipeline


@dataclass
class ExperimentConfig:
    dataset: str = "ISCXVPN2016"
    data_root: str = "/walnut_data/yqm/Dataset"
    batch_size: int = 32
    num_workers: int = 0
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    epochs_teacher: int = 1
    epochs_stacking: int = 1
    epochs_student: int = 1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    temperature: float = 4.0
    lamb_ce: float = 1.0
    lamb_f: float = 0.5
    lamb_r: float = 0.5
    lamb_s: float = 0.1
    output_dir: str = "./experiments/checkpoints"
    mode: str = "full_pipeline"
    student_ckpt: str = "student_sd_mkd.pth"
    max_samples_per_class: int | None = None
    disable_monitor: bool = False
    status_file: str | None = None


def _prepare_output_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_experiment(cfg: ExperimentConfig) -> Dict:
    output_dir = _prepare_output_dir(cfg.output_dir)
    status_file = cfg.status_file or str(output_dir / "training_status.jsonl")

    args = SimpleNamespace(
        dataset=cfg.dataset,
        data_root=cfg.data_root,
        batch_size=cfg.batch_size,
        max_samples_per_class=cfg.max_samples_per_class or 0,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        mode=cfg.mode,
        epochs_teacher=cfg.epochs_teacher,
        epochs_stacking=cfg.epochs_stacking,
        epochs_student=cfg.epochs_student,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        lamb_ce=cfg.lamb_ce,
        lamb_f=cfg.lamb_f,
        lamb_r=cfg.lamb_r,
        lamb_s=cfg.lamb_s,
        temperature=cfg.temperature,
        num_workers=cfg.num_workers,
        student_ckpt=cfg.student_ckpt,
        output_dir=str(output_dir),
        disable_monitor=cfg.disable_monitor,
        status_file=status_file,
    )

    return run_pipeline(args)


__all__ = ["ExperimentConfig", "run_experiment"]
