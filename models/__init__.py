"""Models package for NetKD.

This package contains all neural network model definitions:
- Student model: Lightweight ShuffleNetV2 with AgentAttention
- Teacher models: ResNet50, MobileNetV3, DenseNet121
- Ensemble models: Stacking and MLP fusion
"""

from .student_model import StudentNet, AgentAttention2D, ShuffleNetBackbone, softmax_with_temperature
from .teacher_models import (
    ResNet50Teacher,
    MobileNetV3LargeTeacher,
    DenseNet121Teacher,
    TeacherEnsemble,
    StackingModel,
    TrainResult,
    train_single_teacher,
    train_stacking_model,
)

__all__ = [
    # Student model
    "StudentNet",
    "AgentAttention2D",
    "ShuffleNetBackbone",
    "softmax_with_temperature",
    # Teacher models
    "ResNet50Teacher",
    "MobileNetV3LargeTeacher",
    "DenseNet121Teacher",
    "TeacherEnsemble",
    "StackingModel",
    "TrainResult",
    "train_single_teacher",
    "train_stacking_model",
]
