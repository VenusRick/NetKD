"""
Student KD Experiments - 学生模型知识蒸馏实验

包含:
- train_student_kd.py: 完整 TrafficKD 蒸馏训练
- train_student_baseline.py: CE baseline 训练
- train_student_kd_subsampled.py: 数据效率实验
"""

from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).parent
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]

__all__ = [
    "EXPERIMENT_ROOT",
    "PROJECT_ROOT",
]
