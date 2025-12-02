"""Teacher Search Experiment - Result Schema Module"""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        return super().default(obj)

@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    optimizer: str
    scheduler: str
    label_smoothing: float
    warmup_epochs: int = 0
    def to_dict(self) -> Dict[str, Any]: return asdict(self)
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig": return cls(**data)

@dataclass
class TeacherResult:
    name: str
    class_name: str
    pretrained: bool
    use_eca: bool
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    val_accs: List[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_acc: float = 0.0
    test_accuracy: float = 0.0
    test_f1_macro: float = 0.0
    test_f1_weighted: float = 0.0
    params_millions: float = 0.0
    checkpoint_path: str = ""
    training_time_seconds: float = 0.0
    def to_dict(self) -> Dict[str, Any]: return asdict(self)
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TeacherResult": return cls(**data)

@dataclass
class StackingResult:
    teacher_set_name: str
    teachers: List[str]
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    val_accs: List[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_acc: float = 0.0
    test_accuracy: float = 0.0
    test_f1_macro: float = 0.0
    test_f1_weighted: float = 0.0
    disagreement_rate: float = 0.0
    oracle_accuracy: float = 0.0
    diversity_score: float = 0.0
    hidden_dim: int = 384
    checkpoint_path: str = ""
    training_time_seconds: float = 0.0
    def to_dict(self) -> Dict[str, Any]: return asdict(self)
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StackingResult": return cls(**data)

@dataclass
class ExperimentResult:
    experiment_name: str
    dataset_name: str
    num_classes: int
    training_config: TrainingConfig = field(default_factory=lambda: TrainingConfig(
        epochs=50, batch_size=256, learning_rate=0.001, weight_decay=0.0001,
        optimizer="adamw", scheduler="cosine", label_smoothing=0.05))
    teacher_results: Dict[str, TeacherResult] = field(default_factory=dict)
    stacking_results: Dict[str, StackingResult] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    status: str = "in_progress"
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {"experiment_name": self.experiment_name, "dataset_name": self.dataset_name,
                "num_classes": self.num_classes, "training_config": self.training_config.to_dict(),
                "teacher_results": {k: v.to_dict() for k, v in self.teacher_results.items()},
                "stacking_results": {k: v.to_dict() for k, v in self.stacking_results.items()},
                "created_at": self.created_at, "completed_at": self.completed_at,
                "status": self.status, "notes": self.notes}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentResult":
        teacher_results = {k: TeacherResult.from_dict(v) for k, v in data.get("teacher_results", {}).items()}
        stacking_results = {k: StackingResult.from_dict(v) for k, v in data.get("stacking_results", {}).items()}
        return cls(experiment_name=data["experiment_name"], dataset_name=data["dataset_name"],
                   num_classes=data["num_classes"],
                   training_config=TrainingConfig.from_dict(data.get("training_config", {})),
                   teacher_results=teacher_results, stacking_results=stacking_results,
                   created_at=data.get("created_at", ""), completed_at=data.get("completed_at"),
                   status=data.get("status", "in_progress"), notes=data.get("notes", ""))
    
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, cls=NumpyEncoder, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: Path) -> "ExperimentResult":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
    
    def add_teacher_result(self, result: TeacherResult) -> None:
        self.teacher_results[result.name] = result
    
    def add_stacking_result(self, result: StackingResult) -> None:
        self.stacking_results[result.teacher_set_name] = result
    
    def mark_completed(self) -> None:
        self.completed_at = datetime.now().isoformat()
        self.status = "completed"

def create_comparison_table(results: List[ExperimentResult]) -> Dict[str, Any]:
    comparison = {"experiments": [], "teacher_comparison": {}, "stacking_comparison": {}}
    for result in results:
        comparison["experiments"].append({"name": result.experiment_name, "dataset": result.dataset_name, "status": result.status})
        for name, teacher in result.teacher_results.items():
            if name not in comparison["teacher_comparison"]: comparison["teacher_comparison"][name] = []
            comparison["teacher_comparison"][name].append({"experiment": result.experiment_name, "test_accuracy": teacher.test_accuracy})
        for name, stacking in result.stacking_results.items():
            if name not in comparison["stacking_comparison"]: comparison["stacking_comparison"][name] = []
            comparison["stacking_comparison"][name].append({"experiment": result.experiment_name, "test_accuracy": stacking.test_accuracy})
    return comparison

__all__ = ["TrainingConfig", "TeacherResult", "StackingResult", "ExperimentResult", "NumpyEncoder", "create_comparison_table"]
