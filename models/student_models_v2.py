"""
多种学生模型实现，用于知识蒸馏实验

支持的模型:
1. StudentNet (原始): ShuffleNet v2 0.5x + AgentAttention2D (~9.8M params)
2. GhostNet: GhostNet 1.0x (~3.9M params) - 轻量级高效
3. EfficientNet-Lite0: (~4.6M params) - 移动端优化
4. MobileNetV2: MobileNetV2 (~2.2M params) - 经典轻量级
"""

from __future__ import annotations

import torch
import torch.nn as nn
import timm


class GhostNetStudent(nn.Module):
    """GhostNet 作为学生模型"""
    
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        model_name: str = "ghostnet_100",
        pretrained: bool = False,
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_channels,
        )
        self.feature_dim = self.model.num_features
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    @property
    def fc(self) -> nn.Module:
        return self.model.classifier


class EfficientNetLiteStudent(nn.Module):
    """EfficientNet-Lite 作为学生模型"""
    
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        model_name: str = "efficientnet_lite0",
        pretrained: bool = False,
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_channels,
        )
        self.feature_dim = self.model.num_features
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    @property
    def fc(self) -> nn.Module:
        return self.model.classifier


class MobileNetV2Student(nn.Module):
    """MobileNetV2 作为学生模型"""
    
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        model_name: str = "mobilenetv2_100",
        pretrained: bool = False,
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_channels,
        )
        self.feature_dim = self.model.num_features
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    @property
    def fc(self) -> nn.Module:
        return self.model.classifier


class MobileNetV3SmallStudent(nn.Module):
    """MobileNetV3-Small 作为学生模型"""
    
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        model_name: str = "mobilenetv3_small_100",
        pretrained: bool = False,
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_channels,
        )
        self.feature_dim = self.model.num_features
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    @property
    def fc(self) -> nn.Module:
        return self.model.classifier


def create_student_model(
    model_type: str,
    num_classes: int,
    in_channels: int = 1,
    pretrained: bool = False,
    **kwargs,
) -> nn.Module:
    """创建学生模型的工厂函数"""
    model_type = model_type.lower()
    
    if model_type in ('shufflenet', 'studentnet', 'original'):
        from models.student_model import StudentNet
        return StudentNet(
            num_classes=num_classes,
            in_channels=in_channels,
            pretrained_backbone=pretrained,
            **kwargs,
        )
    
    elif model_type in ('ghostnet', 'ghost'):
        model_name = kwargs.get('model_name', 'ghostnet_100')
        return GhostNetStudent(
            num_classes=num_classes,
            in_channels=in_channels,
            model_name=model_name,
            pretrained=pretrained,
        )
    
    elif model_type in ('efficientnet_lite', 'effnet_lite', 'lite'):
        model_name = kwargs.get('model_name', 'efficientnet_lite0')
        return EfficientNetLiteStudent(
            num_classes=num_classes,
            in_channels=in_channels,
            model_name=model_name,
            pretrained=pretrained,
        )
    
    elif model_type in ('mobilenetv2', 'mnv2'):
        model_name = kwargs.get('model_name', 'mobilenetv2_100')
        return MobileNetV2Student(
            num_classes=num_classes,
            in_channels=in_channels,
            model_name=model_name,
            pretrained=pretrained,
        )
    
    elif model_type in ('mobilenetv3_small', 'mnv3_small', 'mnv3s'):
        model_name = kwargs.get('model_name', 'mobilenetv3_small_100')
        return MobileNetV3SmallStudent(
            num_classes=num_classes,
            in_channels=in_channels,
            model_name=model_name,
            pretrained=pretrained,
        )
    
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Supported: shufflenet, ghostnet, efficientnet_lite, mobilenetv2, mobilenetv3_small"
        )


def get_model_info(model: nn.Module) -> dict:
    """获取模型信息"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_params_M": total_params / 1e6,
        "trainable_params_M": trainable_params / 1e6,
    }
