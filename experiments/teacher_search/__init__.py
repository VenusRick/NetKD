"""Teacher Search Experiment Package

This package provides tools for searching optimal teacher model combinations:
- metrics: Evaluation metrics for teachers and ensembles
- result_schema: Data structures for storing experiment results
- run_train_teacher: Training script for individual teachers
- run_train_stacking: Training script for stacking ensembles
"""

from .metrics import (
    TeacherMetrics,
    EnsembleMetrics,
    compute_classification_metrics,
    compute_disagreement_rate,
    compute_oracle_accuracy,
    compute_diversity_score,
    evaluate_teacher,
    evaluate_ensemble,
)

from .result_schema import (
    TrainingConfig,
    TeacherResult,
    StackingResult,
    ExperimentResult,
    NumpyEncoder,
    create_comparison_table,
)

__all__ = [
    # Metrics
    "TeacherMetrics",
    "EnsembleMetrics", 
    "compute_classification_metrics",
    "compute_disagreement_rate",
    "compute_oracle_accuracy",
    "compute_diversity_score",
    "evaluate_teacher",
    "evaluate_ensemble",
    # Result Schema
    "TrainingConfig",
    "TeacherResult",
    "StackingResult",
    "ExperimentResult",
    "NumpyEncoder",
    "create_comparison_table",
]
