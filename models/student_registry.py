"""
Student Model Registry - 学生模型注册表

提供工厂模式用于动态创建和管理学生模型。
支持从配置文件加载模型定义，便于实验和消融研究。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

import torch
import torch.nn as nn
import yaml

logger = logging.getLogger(__name__)

# 全局注册表
_STUDENT_REGISTRY: Dict[str, Type[nn.Module]] = {}


@dataclass
class StudentConfig:
    """学生模型配置数据类"""
    name: str
    class_name: str
    backbone: str
    width_mult: float = 1.0
    description: str = ""
    expected_params_m: Optional[float] = None
    notes: str = ""

    @classmethod
    def from_dict(cls, name: str, config: Dict[str, Any]) -> "StudentConfig":
        """从字典创建配置"""
        return cls(
            name=name,
            class_name=config.get("class", ""),
            backbone=config.get("backbone", ""),
            width_mult=config.get("width_mult", 1.0),
            description=config.get("description", ""),
            expected_params_m=config.get("expected_params_m"),
            notes=config.get("notes", ""),
        )


@dataclass
class KDConfig:
    """知识蒸馏配置数据类"""
    name: str
    display_name: str
    use_ce: bool = True
    use_fkl: bool = False
    use_rkl: bool = False
    use_sinkhorn: bool = False
    alpha_ce: float = 1.0
    alpha_fkl: float = 0.0
    alpha_rkl: float = 0.0
    alpha_sinkhorn: float = 0.0
    temperature: float = 3.0
    description: str = ""

    @classmethod
    def from_dict(cls, name: str, config: Dict[str, Any]) -> "KDConfig":
        """从字典创建配置"""
        return cls(
            name=name,
            display_name=config.get("name", name),
            use_ce=config.get("use_ce", True),
            use_fkl=config.get("use_fkl", False),
            use_rkl=config.get("use_rkl", False),
            use_sinkhorn=config.get("use_sinkhorn", False),
            alpha_ce=config.get("alpha_ce", 1.0),
            alpha_fkl=config.get("alpha_fkl", 0.0),
            alpha_rkl=config.get("alpha_rkl", 0.0),
            alpha_sinkhorn=config.get("alpha_sinkhorn", 0.0),
            temperature=config.get("temperature", 3.0),
            description=config.get("description", ""),
        )


def register_student(name: str) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    """装饰器：注册学生模型类到全局注册表"""
    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        if name in _STUDENT_REGISTRY:
            logger.warning(f"Student '{name}' already registered, overwriting...")
        _STUDENT_REGISTRY[name] = cls
        logger.debug(f"Registered student: {name} -> {cls.__name__}")
        return cls
    return decorator


def get_student_class(name: str) -> Type[nn.Module]:
    """获取注册的学生模型类"""
    if name not in _STUDENT_REGISTRY:
        available = list(_STUDENT_REGISTRY.keys())
        raise KeyError(f"Student '{name}' not found. Available: {available}")
    return _STUDENT_REGISTRY[name]


def list_registered_students() -> List[str]:
    """列出所有已注册的学生模型"""
    return list(_STUDENT_REGISTRY.keys())


def build_student(
    name: str,
    num_classes: int,
    width_mult: float = 1.0,
    in_channels: int = 1,
    **kwargs
) -> nn.Module:
    """工厂函数：根据名称构建学生模型"""
    cls = get_student_class(name)
    logger.info(f"Building student: {name} (width_mult={width_mult})")
    return cls(num_classes=num_classes, width_mult=width_mult, in_channels=in_channels, **kwargs)


class StudentRegistry:
    """学生模型注册表管理器"""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_path = Path(config_path) if config_path else None
        self._config: Dict[str, Any] = {}
        self._student_configs: Dict[str, StudentConfig] = {}
        self._kd_configs: Dict[str, KDConfig] = {}

        if self.config_path and self.config_path.exists():
            self.load_config(self.config_path)

    def load_config(self, config_path: Union[str, Path]) -> None:
        """加载YAML配置文件"""
        config_path = Path(config_path)
        logger.info(f"Loading student config from: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

        # 解析学生模型候选
        students = self._config.get("students", {})
        for name, cfg in students.items():
            self._student_configs[name] = StudentConfig.from_dict(name, cfg)

        # 解析KD配置
        kd_configs = self._config.get("kd_configs", {})
        for name, cfg in kd_configs.items():
            self._kd_configs[name] = KDConfig.from_dict(name, cfg)

    def get_student_config(self, name: str) -> StudentConfig:
        """获取学生配置"""
        if name not in self._student_configs:
            available = list(self._student_configs.keys())
            raise KeyError(f"Student config '{name}' not found. Available: {available}")
        return self._student_configs[name]

    def get_kd_config(self, name: str) -> KDConfig:
        """获取KD配置"""
        if name not in self._kd_configs:
            available = list(self._kd_configs.keys())
            raise KeyError(f"KD config '{name}' not found. Available: {available}")
        return self._kd_configs[name]

    def list_students(self) -> List[str]:
        """列出所有配置的学生"""
        return list(self._student_configs.keys())

    def list_kd_configs(self) -> List[str]:
        """列出所有KD配置"""
        return list(self._kd_configs.keys())

    def get_defaults(self, section: str = "training") -> Dict[str, Any]:
        """获取默认训练参数"""
        defaults = self._config.get("defaults", {})
        return defaults.get(section, {})

    def get_dataset_config(self) -> Dict[str, Any]:
        """获取数据集配置"""
        return self._config.get("dataset", {})

    def get_experiment_matrix(self) -> Dict[str, Any]:
        """获取实验矩阵"""
        return self._config.get("experiment_matrix", {})

    def build_student(self, name: str, num_classes: int, **kwargs) -> nn.Module:
        """根据配置名称构建学生模型"""
        config = self.get_student_config(name)
        return build_student(
            name=name,
            num_classes=num_classes,
            width_mult=kwargs.get("width_mult", config.width_mult),
            **kwargs
        )

    def get_output_config(self) -> Dict[str, str]:
        """获取输出路径配置"""
        return self._config.get("output", {
            "checkpoint_dir": "checkpoints/student_kd",
            "results_dir": "results/student_kd",
            "log_dir": "logs/student_kd",
        })


def _auto_register_students():
    """自动注册 student_models_v2.py 中定义的学生模型"""
    try:
        from . import student_models_v2 as sm

        # 注册已有模型
        if hasattr(sm, "MobileNetV2Student"):
            _STUDENT_REGISTRY["mobilenetv2"] = sm.MobileNetV2Student
        if hasattr(sm, "MobileNetV3SmallStudent"):
            _STUDENT_REGISTRY["mobilenetv3_small"] = sm.MobileNetV3SmallStudent
        if hasattr(sm, "GhostNetStudent"):
            _STUDENT_REGISTRY["ghostnet_v1_1_0x"] = sm.GhostNetStudent
        if hasattr(sm, "GhostNetV3Student"):
            _STUDENT_REGISTRY["ghostnetv3_0_75x"] = sm.GhostNetV3Student
        if hasattr(sm, "RepViTStudent"):
            _STUDENT_REGISTRY["repvit_tiny_m0_9"] = sm.RepViTStudent

        logger.debug(f"Auto-registered students: {list(_STUDENT_REGISTRY.keys())}")

    except ImportError as e:
        logger.warning(f"Could not auto-register students: {e}")


# 模块加载时自动注册
_auto_register_students()


__all__ = [
    "StudentConfig",
    "KDConfig",
    "StudentRegistry",
    "register_student",
    "get_student_class",
    "list_registered_students",
    "build_student",
]
