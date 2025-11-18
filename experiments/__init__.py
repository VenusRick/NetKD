"""Experiments package for NetKD.

This package contains experimental scripts for SD-MKD and ablation studies.
"""

from .sd_mkd import train_student as sd_mkd_train_student
from .ablation import (
    AblationConfig,
    AblationResult,
    default_ablation_configs,
    run_ablation_suite,
)

__all__ = [
    "sd_mkd_train_student",
    "AblationConfig",
    "AblationResult",
    "default_ablation_configs",
    "run_ablation_suite",
]
