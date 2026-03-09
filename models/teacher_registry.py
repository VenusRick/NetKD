"""
Teacher Model Registry - 教师模型注册表

提供工厂模式用于动态创建和管理教师模型。
支持从配置文件加载模型定义，便于实验和消融研究。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

import torch
import torch.nn as nn
import yaml

logger = logging.getLogger(__name__)

# 全局注册表
_TEACHER_REGISTRY: Dict[str, Type[nn.Module]] = {}


@dataclass
class TeacherConfig:
    """教师模型配置数据类"""
    name: str
    class_name: str
    pretrained: bool = True
    use_eca: bool = True
    description: str = ""
    expected_accuracy: Optional[float] = None
    params_m: Optional[float] = None
    notes: str = ""
    
    @classmethod
    def from_dict(cls, name: str, config: Dict[str, Any]) -> "TeacherConfig":
        """从字典创建配置"""
        return cls(
            name=name,
            class_name=config.get("class", ""),
            pretrained=config.get("pretrained", True),
            use_eca=config.get("use_eca", True),
            description=config.get("description", ""),
            expected_accuracy=config.get("expected_accuracy"),
            params_m=config.get("params_m"),
            notes=config.get("notes", ""),
        )


@dataclass
class TeacherSetConfig:
    """教师组合配置数据类"""
    name: str
    display_name: str
    teachers: List[str]
    description: str = ""
    
    @classmethod
    def from_dict(cls, name: str, config: Dict[str, Any]) -> "TeacherSetConfig":
        """从字典创建配置"""
        return cls(
            name=name,
            display_name=config.get("name", name),
            teachers=config.get("teachers", []),
            description=config.get("description", ""),
        )


def register_teacher(name: str) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    """装饰器：注册教师模型类到全局注册表"""
    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        if name in _TEACHER_REGISTRY:
            logger.warning(f"Teacher '{name}' already registered, overwriting...")
        _TEACHER_REGISTRY[name] = cls
        logger.debug(f"Registered teacher: {name} -> {cls.__name__}")
        return cls
    return decorator


def get_teacher_class(name: str) -> Type[nn.Module]:
    """获取注册的教师模型类"""
    if name not in _TEACHER_REGISTRY:
        available = list(_TEACHER_REGISTRY.keys())
        raise KeyError(f"Teacher '{name}' not found. Available: {available}")
    return _TEACHER_REGISTRY[name]


def list_registered_teachers() -> List[str]:
    """列出所有已注册的教师模型"""
    return list(_TEACHER_REGISTRY.keys())


def build_teacher(
    name: str,
    num_classes: int,
    pretrained: bool = True,
    use_eca: bool = True,
    **kwargs
) -> nn.Module:
    """工厂函数：根据名称构建教师模型"""
    cls = get_teacher_class(name)
    logger.info(f"Building teacher: {name} (pretrained={pretrained}, use_eca={use_eca})")
    return cls(num_classes=num_classes, pretrained=pretrained, use_eca=use_eca, **kwargs)


class TeacherRegistry:
    """教师模型注册表管理器"""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_path = Path(config_path) if config_path else None
        self._config: Dict[str, Any] = {}
        self._teacher_configs: Dict[str, TeacherConfig] = {}
        self._teacher_sets: Dict[str, TeacherSetConfig] = {}
        
        if self.config_path and self.config_path.exists():
            self.load_config(self.config_path)
    
    def load_config(self, config_path: Union[str, Path]) -> None:
        """加载YAML配置文件"""
        config_path = Path(config_path)
        logger.info(f"Loading teacher config from: {config_path}")
        
        with open(config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)
        
        # 解析教师候选
        candidates = self._config.get("teacher_candidates", {})
        for name, cfg in candidates.items():
            self._teacher_configs[name] = TeacherConfig.from_dict(name, cfg)
        
        # 解析教师组合
        sets = self._config.get("teacher_sets", {})
        for name, cfg in sets.items():
            self._teacher_sets[name] = TeacherSetConfig.from_dict(name, cfg)
    
    def get_teacher_config(self, name: str) -> TeacherConfig:
        """获取教师配置"""
        if name not in self._teacher_configs:
            available = list(self._teacher_configs.keys())
            raise KeyError(f"Teacher config '{name}' not found. Available: {available}")
        return self._teacher_configs[name]
    
    def get_teacher_set(self, name: str) -> TeacherSetConfig:
        """获取教师组合配置"""
        if name not in self._teacher_sets:
            available = list(self._teacher_sets.keys())
            raise KeyError(f"Teacher set '{name}' not found. Available: {available}")
        return self._teacher_sets[name]
    
    def list_teachers(self) -> List[str]:
        """列出所有配置的教师"""
        return list(self._teacher_configs.keys())
    
    def list_teacher_sets(self) -> List[str]:
        """列出所有教师组合"""
        return list(self._teacher_sets.keys())
    
    def get_defaults(self, section: str = "training") -> Dict[str, Any]:
        """获取默认训练参数"""
        defaults = self._config.get("defaults", {})
        return defaults.get(section, {})
    
    def build_teacher(self, name: str, num_classes: int, **kwargs) -> nn.Module:
        """根据配置名称构建教师模型"""
        config = self.get_teacher_config(name)
        return build_teacher(
            name=name,
            num_classes=num_classes,
            pretrained=kwargs.get("pretrained", config.pretrained),
            use_eca=kwargs.get("use_eca", config.use_eca),
        )
    
    def build_teacher_set(
        self,
        set_name: str,
        num_classes: int,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> Dict[str, nn.Module]:
        """构建教师组合"""
        teacher_set = self.get_teacher_set(set_name)
        teachers = {}
        
        for teacher_name in teacher_set.teachers:
            model = self.build_teacher(teacher_name, num_classes, **kwargs)
            if device is not None:
                model = model.to(device)
            teachers[teacher_name] = model
        
        return teachers
    
    def get_output_config(self) -> Dict[str, str]:
        """获取输出路径配置"""
        return self._config.get("output", {
            "checkpoint_dir": "checkpoints/teacher_search",
            "results_dir": "results/teacher_search",
            "log_dir": "logs/teacher_search",
        })


class DynamicStackingModel(nn.Module):
    """动态Stacking模型，支持任意数量的教师"""
    
    def __init__(
        self,
        num_classes: int,
        num_teachers: int,
        hidden_dim: int = 384,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_teachers = num_teachers
        
        in_dim = num_teachers * num_classes
        
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    
    def forward(self, *teacher_logits: torch.Tensor) -> torch.Tensor:
        if len(teacher_logits) != self.num_teachers:
            raise ValueError(
                f"Expected {self.num_teachers} teacher logits, got {len(teacher_logits)}"
            )
        x = torch.cat(teacher_logits, dim=-1)
        return self.mlp(x)


def _auto_register_teachers():
    """自动注册 teacher_models.py 中定义的教师模型"""
    try:
        from . import teacher_models
        
        if hasattr(teacher_models, "ResNet50Teacher"):
            _TEACHER_REGISTRY["resnet50_eca"] = teacher_models.ResNet50Teacher
        if hasattr(teacher_models, "DenseNet121Teacher"):
            _TEACHER_REGISTRY["densenet121_eca"] = teacher_models.DenseNet121Teacher
        if hasattr(teacher_models, "MobileNetV3LargeTeacher"):
            _TEACHER_REGISTRY["mobilenetv3_large_eca"] = teacher_models.MobileNetV3LargeTeacher
        if hasattr(teacher_models, "ConvNeXtV2TinyTeacher"):
            _TEACHER_REGISTRY["convnextv2_tiny_eca"] = teacher_models.ConvNeXtV2TinyTeacher
        if hasattr(teacher_models, "EfficientNetV2STeacher"):
            _TEACHER_REGISTRY["efficientnetv2_s_eca"] = teacher_models.EfficientNetV2STeacher
            
        logger.debug(f"Auto-registered teachers: {list(_TEACHER_REGISTRY.keys())}")
        
    except ImportError as e:
        logger.warning(f"Could not auto-register teachers: {e}")


_auto_register_teachers()


__all__ = [
    "TeacherConfig",
    "TeacherSetConfig",
    "TeacherRegistry",
    "DynamicStackingModel",
    "register_teacher",
    "get_teacher_class",
    "list_registered_teachers",
    "build_teacher",
]
