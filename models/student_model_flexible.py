"""
Flexible student model with configurable attention mechanism.
用于消融实验的可配置学生模型
"""

import torch
import torch.nn as nn
from torchvision.models import shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights
from models.attention_modules import get_attention_module


class ShuffleNetBackbone(nn.Module):
    """ShuffleNet v2 0.5x feature extractor for single-channel inputs."""

    def __init__(self, in_channels: int = 1, pretrained: bool = False):
        super().__init__()
        weights = ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1 if pretrained else None
        base = shufflenet_v2_x0_5(weights=weights)

        # Adapt the stem to match desired number of input channels
        if in_channels != 3:
            conv1 = base.conv1[0]
            new_conv = nn.Conv2d(
                in_channels,
                conv1.out_channels,
                kernel_size=conv1.kernel_size,
                stride=conv1.stride,
                padding=conv1.padding,
                bias=False,
            )
            if conv1.weight.shape[1] == 3:
                new_conv.weight.data = conv1.weight.data.mean(dim=1, keepdim=True)
            base.conv1[0] = new_conv

        # Recompose features into a single sequential block
        self.features = nn.Sequential(
            base.conv1,
            base.maxpool,
            base.stage2,
            base.stage3,
            base.stage4,
            base.conv5,
        )
        self.out_channels = base.fc.in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class FlexibleStudentNet(nn.Module):
    """ShuffleNet v2 student with configurable attention mechanism.
    
    Args:
        num_classes: Number of output classes
        attention_type: Type of attention ('agent', 'cbam', 'eca', 'simam', 'none')
        in_channels: Number of input channels (default: 1 for grayscale traffic images)
        pretrained_backbone: Whether to use pretrained backbone
        **attention_kwargs: Additional arguments for attention module
    """

    def __init__(
        self,
        num_classes: int,
        attention_type: str = 'agent',
        in_channels: int = 1,
        pretrained_backbone: bool = False,
        **attention_kwargs
    ):
        super().__init__()
        self.attention_type = attention_type
        
        # Backbone
        self.backbone = ShuffleNetBackbone(
            in_channels=in_channels, 
            pretrained=pretrained_backbone
        )
        C = self.backbone.out_channels
        
        # Attention module
        self.attention = get_attention_module(
            name=attention_type,
            channels=C,
            **attention_kwargs
        )
        
        # Classifier
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(C, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)  # [B, C, H', W']
        feat = self.attention(feat)  # [B, C, H', W']
        pooled = self.pool(feat).flatten(1)  # [B, C]
        logits = self.fc(pooled)  # [B, num_classes]
        return logits
    
    def get_model_info(self):
        """返回模型信息用于日志"""
        return {
            'architecture': 'ShuffleNetV2-0.5x',
            'attention': self.attention_type,
            'backbone_channels': self.backbone.out_channels
        }
