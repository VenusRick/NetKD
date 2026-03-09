"""Teacher Search Experiment - Metrics Module"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

@dataclass
class TeacherMetrics:
    """单个教师模型的评估指标"""
    name: str
    accuracy: float
    f1_macro: float
    f1_weighted: float
    precision_macro: float
    recall_macro: float
    confusion_matrix: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "accuracy": self.accuracy,
            "f1_macro": self.f1_macro, "f1_weighted": self.f1_weighted,
            "precision_macro": self.precision_macro, "recall_macro": self.recall_macro,
            "confusion_matrix": self.confusion_matrix.tolist() if self.confusion_matrix is not None else None,
        }

@dataclass
class EnsembleMetrics:
    """集成模型的评估指标"""
    teacher_set_name: str
    teachers: List[str]
    individual_metrics: Dict[str, TeacherMetrics] = field(default_factory=dict)
    stacking_accuracy: float = 0.0
    stacking_f1_macro: float = 0.0
    stacking_f1_weighted: float = 0.0
    disagreement_rate: float = 0.0
    oracle_accuracy: float = 0.0
    diversity_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "teacher_set_name": self.teacher_set_name, "teachers": self.teachers,
            "individual_metrics": {k: v.to_dict() for k, v in self.individual_metrics.items()},
            "stacking_accuracy": self.stacking_accuracy, "stacking_f1_macro": self.stacking_f1_macro,
            "stacking_f1_weighted": self.stacking_f1_weighted,
            "disagreement_rate": self.disagreement_rate, "oracle_accuracy": self.oracle_accuracy,
            "diversity_score": self.diversity_score,
        }

def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, name: str = "model") -> TeacherMetrics:
    return TeacherMetrics(
        name=name, accuracy=accuracy_score(y_true, y_pred),
        f1_macro=f1_score(y_true, y_pred, average='macro', zero_division=0),
        f1_weighted=f1_score(y_true, y_pred, average='weighted', zero_division=0),
        precision_macro=precision_score(y_true, y_pred, average='macro', zero_division=0),
        recall_macro=recall_score(y_true, y_pred, average='macro', zero_division=0),
        confusion_matrix=confusion_matrix(y_true, y_pred),
    )

def compute_disagreement_rate(predictions: List[np.ndarray]) -> float:
    if len(predictions) < 2: return 0.0
    n_samples = len(predictions[0])
    disagreements = sum(1 for i in range(n_samples) if len(set(p[i] for p in predictions)) > 1)
    return disagreements / n_samples

def compute_oracle_accuracy(predictions: List[np.ndarray], y_true: np.ndarray) -> float:
    n_samples = len(y_true)
    correct = sum(1 for i in range(n_samples) if any(p[i] == y_true[i] for p in predictions))
    return correct / n_samples

def compute_diversity_score(predictions: List[np.ndarray]) -> float:
    if len(predictions) < 2: return 0.0
    n_samples, n_teachers = len(predictions[0]), len(predictions)
    entropies = []
    for i in range(n_samples):
        preds_i = [p[i] for p in predictions]
        unique, counts = np.unique(preds_i, return_counts=True)
        probs = counts / n_teachers
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        entropies.append(entropy)
    max_entropy = np.log2(n_teachers)
    return np.mean(entropies) / max_entropy if max_entropy > 0 else 0.0

@torch.no_grad()
def evaluate_teacher(model: nn.Module, dataloader, device: torch.device, name: str = "model"):
    model.eval()
    model.to(device)
    all_preds, all_logits, all_labels = [], [], []
    for x, y in dataloader:
        x = x.to(device)
        logits = model(x)
        all_preds.append(logits.argmax(dim=-1).cpu().numpy())
        all_logits.append(logits.cpu().numpy())
        all_labels.append(y.numpy())
    y_pred = np.concatenate(all_preds)
    y_logits = np.concatenate(all_logits)
    y_true = np.concatenate(all_labels)
    metrics = compute_classification_metrics(y_true, y_pred, name)
    return metrics, y_pred, y_logits

@torch.no_grad()
def evaluate_ensemble(teachers: Dict[str, nn.Module], stacking_model: nn.Module, dataloader, device: torch.device, teacher_set_name: str = "ensemble"):
    teacher_predictions, individual_metrics = {}, {}
    for name, model in teachers.items():
        metrics, preds, _ = evaluate_teacher(model, dataloader, device, name)
        individual_metrics[name] = metrics
        teacher_predictions[name] = preds
    
    stacking_model.eval()
    stacking_model.to(device)
    all_stacking_preds, all_labels = [], []
    for x, y in dataloader:
        x = x.to(device)
        teacher_logits = [teachers[name](x) for name in teachers.keys()]
        stacking_logits = stacking_model(*teacher_logits)
        all_stacking_preds.append(stacking_logits.argmax(dim=-1).cpu().numpy())
        all_labels.append(y.numpy())
    
    stacking_pred = np.concatenate(all_stacking_preds)
    y_true = np.concatenate(all_labels)
    stacking_metrics = compute_classification_metrics(y_true, stacking_pred, "stacking")
    predictions_list = list(teacher_predictions.values())
    
    return EnsembleMetrics(
        teacher_set_name=teacher_set_name, teachers=list(teachers.keys()),
        individual_metrics=individual_metrics,
        stacking_accuracy=stacking_metrics.accuracy, stacking_f1_macro=stacking_metrics.f1_macro,
        stacking_f1_weighted=stacking_metrics.f1_weighted,
        disagreement_rate=compute_disagreement_rate(predictions_list),
        oracle_accuracy=compute_oracle_accuracy(predictions_list, y_true),
        diversity_score=compute_diversity_score(predictions_list),
    )

__all__ = ["TeacherMetrics", "EnsembleMetrics", "compute_classification_metrics",
           "compute_disagreement_rate", "compute_oracle_accuracy", "compute_diversity_score",
           "evaluate_teacher", "evaluate_ensemble"]
