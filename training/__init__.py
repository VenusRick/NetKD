"""Training package for NetKD.

This package contains training logic, loss functions, and evaluation utilities.
"""

from .loss_functions import (
    ce_loss,
    forward_kl,
    reverse_kl,
    sinkhorn_distance,
    distillation_loss,
    class_cost_matrix,
)
from .evaluation import evaluate_model, summarize_metrics
from .monitor import LiveTrainingMonitor
from .status_tracker import TrainingStatusLogger
from .train import (
    train_teachers,
    train_stacking_model_stage,
    train_student_stage,
    run_demo_pipeline,
    run_real_pipeline,
)

__all__ = [
    # Loss functions
    "ce_loss",
    "forward_kl",
    "reverse_kl",
    "sinkhorn_distance",
    "distillation_loss",
    "class_cost_matrix",
    # Evaluation
    "evaluate_model",
    "summarize_metrics",
    "LiveTrainingMonitor",
    "TrainingStatusLogger",
    # Training pipelines
    "train_teachers",
    "train_stacking_model_stage",
    "train_student_stage",
    "run_demo_pipeline",
    "run_real_pipeline",
]
