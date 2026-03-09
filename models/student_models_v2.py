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


# =============================================================================
# Student 2.0 新增模型
# =============================================================================

class GhostNetV3Student(nn.Module):
    """GhostNetV3 作为学生模型 (Student 2.0)
    
    GhostNetV3 引入了 DFC attention 和改进的 ghost module，
    在保持轻量级的同时获得更好的性能。
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        width_mult: float = 0.75,
        pretrained: bool = False,
        **kwargs,
    ):
        super().__init__()
        # 尝试使用 timm 的 ghostnetv2 作为替代
        # 注意：timm 可能没有 ghostnetv3，使用 ghostnetv2 或 ghostnet
        model_name = f"ghostnetv2_{int(width_mult * 100):03d}"
        try:
            self.model = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=num_classes,
                in_chans=in_channels,
            )
        except RuntimeError:
            # 如果 ghostnetv2 不可用，回退到 ghostnet
            width_str = "100" if width_mult >= 1.0 else f"{int(width_mult * 100):03d}"
            self.model = timm.create_model(
                f"ghostnet_{width_str}",
                pretrained=pretrained,
                num_classes=num_classes,
                in_chans=in_channels,
            )
        self.feature_dim = self.model.num_features
        self.width_mult = width_mult

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @property
    def fc(self) -> nn.Module:
        return self.model.classifier


class RepViTStudent(nn.Module):
    """RepViT 作为学生模型 (Student 2.0)
    
    RepViT 是一种轻量级视觉Transformer，结合了
    重参数化技术和移动端友好的设计。
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        width_mult: float = 1.0,
        model_variant: str = "m0_9",
        pretrained: bool = False,
        **kwargs,
    ):
        super().__init__()
        # timm 中 repvit 的命名格式
        model_name = f"repvit_{model_variant}"
        try:
            self.model = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=num_classes,
                in_chans=in_channels,
            )
        except RuntimeError:
            # 如果特定变体不可用，尝试其他变体
            available_variants = ["m0_9", "m1_0", "m1_1", "m1_5", "m2_3"]
            for var in available_variants:
                try:
                    self.model = timm.create_model(
                        f"repvit_{var}",
                        pretrained=pretrained,
                        num_classes=num_classes,
                        in_chans=in_channels,
                    )
                    break
                except RuntimeError:
                    continue
            else:
                # 如果所有变体都不可用，使用 MobileViT 作为替代
                self.model = timm.create_model(
                    "mobilevit_xxs",
                    pretrained=pretrained,
                    num_classes=num_classes,
                    in_chans=in_channels,
                )
        
        self.feature_dim = self.model.num_features
        self.width_mult = width_mult
        self.model_variant = model_variant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @property
    def fc(self) -> nn.Module:
        if hasattr(self.model, 'classifier'):
            return self.model.classifier
        elif hasattr(self.model, 'head'):
            return self.model.head
        else:
            return self.model.fc


# 更新工厂函数以支持新模型
def create_student_model_v2(
    model_type: str,
    num_classes: int,
    in_channels: int = 1,
    pretrained: bool = False,
    width_mult: float = 1.0,
    **kwargs,
) -> nn.Module:
    """创建学生模型的工厂函数 v2 - 支持 Student 2.0 模型"""
    model_type = model_type.lower()

    # 原有模型
    if model_type in ('shufflenet', 'studentnet', 'original'):
        from models.student_model import StudentNet
        return StudentNet(
            num_classes=num_classes,
            in_channels=in_channels,
            pretrained_backbone=pretrained,
            **kwargs,
        )

    elif model_type in ('ghostnet', 'ghost', 'ghostnet_v1'):
        return GhostNetStudent(
            num_classes=num_classes,
            in_channels=in_channels,
            pretrained=pretrained,
        )

    elif model_type in ('efficientnet_lite', 'effnet_lite', 'lite'):
        return EfficientNetLiteStudent(
            num_classes=num_classes,
            in_channels=in_channels,
            pretrained=pretrained,
        )

    elif model_type in ('mobilenetv2', 'mnv2'):
        return MobileNetV2Student(
            num_classes=num_classes,
            in_channels=in_channels,
            pretrained=pretrained,
        )

    elif model_type in ('mobilenetv3_small', 'mnv3_small', 'mnv3s'):
        return MobileNetV3SmallStudent(
            num_classes=num_classes,
            in_channels=in_channels,
            pretrained=pretrained,
        )

    # Student 2.0 新增模型
    elif model_type in ('ghostnetv3', 'ghostnet_v3', 'ghost_v3'):
        return GhostNetV3Student(
            num_classes=num_classes,
            in_channels=in_channels,
            width_mult=width_mult,
            pretrained=pretrained,
        )

    elif model_type in ('repvit', 'rep_vit'):
        model_variant = kwargs.get('model_variant', 'm0_9')
        return RepViTStudent(
            num_classes=num_classes,
            in_channels=in_channels,
            width_mult=width_mult,
            model_variant=model_variant,
            pretrained=pretrained,
        )

    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Supported: shufflenet, ghostnet, efficientnet_lite, mobilenetv2, "
            f"mobilenetv3_small, ghostnetv3, repvit"
        )


# 导出列表
__all__ = [
    "GhostNetStudent",
    "EfficientNetLiteStudent", 
    "MobileNetV2Student",
    "MobileNetV3SmallStudent",
    "GhostNetV3Student",
    "RepViTStudent",
    "create_student_model",
    "create_student_model_v2",
    "get_model_info",
]


# ============== 新增超轻量学生模型 ==============

class GhostNet050Student(nn.Module):
    """GhostNet 0.5x - 超轻量版本 (1.32M params)"""
    def __init__(self, num_classes: int, in_channels: int = 1, pretrained: bool = False):
        super().__init__()
        self.model = timm.create_model("ghostnet_050", pretrained=pretrained, 
                                       num_classes=num_classes, in_chans=in_channels)
    def forward(self, x): return self.model(x)


class GhostNetV3050Student(nn.Module):
    """GhostNetV3 0.5x - 最新轻量版本 (2.12M params)"""
    def __init__(self, num_classes: int, in_channels: int = 1, pretrained: bool = False):
        super().__init__()
        self.model = timm.create_model("ghostnetv3_050", pretrained=pretrained,
                                       num_classes=num_classes, in_chans=in_channels)
    def forward(self, x): return self.model(x)


class RepViTM09Student(nn.Module):
    """RepViT M0.9 - 轻量视觉Transformer (4.72M params)"""
    def __init__(self, num_classes: int, in_channels: int = 1, pretrained: bool = False):
        super().__init__()
        self.model = timm.create_model("repvit_m0_9", pretrained=pretrained,
                                       num_classes=num_classes, in_chans=in_channels)
    def forward(self, x): return self.model(x)


class MobileNetV2050Student(nn.Module):
    """MobileNetV2 0.5x - 超轻量版本 (0.70M params)"""
    def __init__(self, num_classes: int, in_channels: int = 1, pretrained: bool = False):
        super().__init__()
        self.model = timm.create_model("mobilenetv2_050", pretrained=pretrained,
                                       num_classes=num_classes, in_chans=in_channels)
    def forward(self, x): return self.model(x)


class MobileNetV3Small050Student(nn.Module):
    """MobileNetV3-Small 0.5x - 最轻量版本 (0.58M params)"""
    def __init__(self, num_classes: int, in_channels: int = 1, pretrained: bool = False):
        super().__init__()
        self.model = timm.create_model("mobilenetv3_small_050", pretrained=pretrained,
                                       num_classes=num_classes, in_chans=in_channels)
    def forward(self, x): return self.model(x)


# 扩展的工厂函数
def create_lightweight_student(model_type: str, num_classes: int, in_channels: int = 1, 
                               pretrained: bool = False) -> nn.Module:
    """创建轻量学生模型"""
    model_map = {
        "ghostnet_050": GhostNet050Student,
        "ghostnetv3_050": GhostNetV3050Student,
        "repvit_m0_9": RepViTM09Student,
        "mobilenetv2_050": MobileNetV2050Student,
        "mobilenetv3_small_050": MobileNetV3Small050Student,
    }
    
    if model_type not in model_map:
        raise ValueError(f"Unknown lightweight model: {model_type}. Available: {list(model_map.keys())}")
    
    return model_map[model_type](num_classes=num_classes, in_channels=in_channels, pretrained=pretrained)


# 更新原有的create_student_model支持新模型
_original_create = create_student_model

def create_student_model_extended(model_type: str, num_classes: int, in_channels: int = 1,
                                  pretrained: bool = False, **kwargs) -> nn.Module:
    """扩展的学生模型工厂函数"""
    # 先尝试轻量模型
    lightweight_models = ["ghostnet_050", "ghostnetv3_050", "repvit_m0_9", 
                         "mobilenetv2_050", "mobilenetv3_small_050"]
    if model_type.lower() in lightweight_models:
        return create_lightweight_student(model_type.lower(), num_classes, in_channels, pretrained)
    
    # 否则使用原有函数
    return _original_create(model_type, num_classes, in_channels, pretrained, **kwargs)
