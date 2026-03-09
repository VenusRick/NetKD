"""Efficient Channel Attention (ECA) Module

Paper: ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks
Link: https://arxiv.org/abs/1910.03151

ECA uses 1D convolution for efficient channel attention without dimensionality reduction.
"""

import torch
import torch.nn as nn
import math


class ECALayer(nn.Module):
    """Efficient Channel Attention layer.
    
    Args:
        channels: Number of channels in the input feature map
        gamma: Parameter for adaptive kernel size calculation
        b: Bias parameter for adaptive kernel size calculation
    """
    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        # Adaptive kernel size
        t = int(abs(math.log2(channels) + b) / gamma)
        k = t if t % 2 else t + 1  # Ensure odd kernel size
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, H, W]
        Returns:
            Attention-weighted tensor [B, C, H, W]
        """
        # Global average pooling: [B, C, H, W] -> [B, C, 1, 1]
        y = self.avg_pool(x)
        
        # Squeeze and 1D conv: [B, C, 1, 1] -> [B, 1, C] -> [B, 1, C] -> [B, C, 1]
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        
        # Sigmoid activation and scale: [B, C, 1, 1]
        y = self.sigmoid(y)
        
        # Apply attention weights
        return x * y.expand_as(x)


def add_eca_to_resnet(model: nn.Module) -> nn.Module:
    """Add ECA layers after each residual block in ResNet."""
    def add_eca_recursive(module):
        for name, child in module.named_children():
            if hasattr(child, 'conv3') and hasattr(child, 'bn3'):  # Bottleneck block
                # Get output channels from bn3
                out_channels = child.bn3.num_features
                eca = ECALayer(out_channels)
                # Insert ECA after bn3
                child.add_module('eca', eca)
                # Override forward to include ECA
                original_forward = child.forward
                def new_forward(x, orig_fwd=original_forward, eca_layer=eca):
                    out = orig_fwd(x)
                    out = eca_layer(out)
                    return out
                child.forward = new_forward
            else:
                add_eca_recursive(child)
    
    add_eca_recursive(model)
    return model



import torch
import torch.nn as nn
from .eca_module import ECALayer

class DenseNetECAWrapper(nn.Module):
    """Wrapper for DenseNet layers to handle list outputs before ECA"""
    def __init__(self, original_layer, eca_layer):
        super().__init__()
        self.original_layer = original_layer
        self.eca = eca_layer
    
    def forward(self, *prev_features):
        # DenseNet layers output new features (not list)
        new_features = self.original_layer(*prev_features)
        # Apply ECA only on tensor outputs
        if isinstance(new_features, torch.Tensor):
            return self.eca(new_features)
        return new_features

def add_eca_to_densenet_fixed(model, eca_kwargs=None):
    """Add ECA modules to DenseNet dense blocks (fixed for list handling)"""
    eca_kwargs = eca_kwargs or {}
    
    # Find all dense blocks in features
    for name, module in model.features.named_children():
        if 'denseblock' in name:
            # Each dense block contains _DenseLayer modules
            for layer_name, layer in module.named_children():
                if hasattr(layer, 'conv2'):  # _DenseLayer has conv1/bn1/conv2/bn2
                    channels = layer.conv2.out_channels
                    eca = ECALayer(channels, **eca_kwargs)
                    # Wrap the entire layer with ECA
                    wrapped = DenseNetECAWrapper(layer, eca)
                    setattr(module, layer_name, wrapped)
    
    return model


def add_eca_to_densenet(model: nn.Module) -> nn.Module:
    """Add ECA layers after each dense block in DenseNet."""
    if hasattr(model, 'features'):
        features = model.features
        for name, module in features.named_children():
            if 'denseblock' in name:
                # Get the last layer's output channels
                last_layer = list(module.children())[-1]
                if hasattr(last_layer, 'conv2'):
                    out_channels = last_layer.conv2.out_channels
                    eca = ECALayer(out_channels * len(list(module.children())))
                    module.add_module('eca', eca)
    return model


def add_eca_to_mobilenetv3(model: nn.Module) -> nn.Module:
    """Add ECA layers to MobileNetV3's inverted residual blocks."""
    if hasattr(model, 'features'):
        for module in model.features:
            if hasattr(module, 'block') and len(module.block) > 0:
                # Find the last conv layer
                for i in range(len(module.block) - 1, -1, -1):
                    layer = module.block[i]
                    if isinstance(layer, nn.Conv2d):
                        out_channels = layer.out_channels
                        eca = ECALayer(out_channels)
                        module.block.add_module(f'eca_{i}', eca)
                        break
    return model


__all__ = ['ECALayer', 'add_eca_to_resnet', 'add_eca_to_densenet', 'add_eca_to_mobilenetv3']



def add_eca_to_densenet_v2(model, eca_kwargs=None):
    """
    Add ECA modules to DenseNet after each transition layer and final norm.
    This avoids the list concatenation issue in dense blocks.
    """
    eca_kwargs = eca_kwargs or {}
    
    # Add ECA after each transition layer (between dense blocks)
    for name, module in model.features.named_children():
        if 'transition' in name:
            # Transition layers reduce feature maps
            # Get output channels from the last conv in transition
            if hasattr(module, 'conv'):
                channels = module.conv.out_channels
            elif hasattr(module, '1'):  # Sequential: norm, relu, conv, pool
                # Find the conv layer
                channels = None
                for sub_module in module.children():
                    if isinstance(sub_module, nn.Conv2d):
                        channels = sub_module.out_channels
                        break
                if channels is None:
                    continue
            # Create ECA and add it to the transition (register module)
            eca = ECALayer(channels, **eca_kwargs)
            module.add_module('eca', eca)
            # Wrap the transition module's forward to use the registered ECA
            original_forward = module.forward
            def make_forward_with_eca(orig_forward, mod):
                def forward_with_eca(x):
                    x = orig_forward(x)
                    x = mod.eca(x)
                    return x
                return forward_with_eca
            module.forward = make_forward_with_eca(original_forward, module)
    
    # Add ECA after the final norm layer (before classifier)
    # Find the last BatchNorm2d in features
    last_norm_name = None
    last_norm = None
    for name, module in model.features.named_children():
        if isinstance(module, nn.BatchNorm2d):
            last_norm_name = name
            last_norm = module
    if last_norm is not None:
        channels = last_norm.num_features
        eca = ECALayer(channels, **eca_kwargs)
        # Register and wrap
        parent = model.features
        parent.add_module(f'eca_after_{last_norm_name}', eca)
        original_forward = last_norm.forward
        def make_forward_with_eca_final(orig_forward, parent_mod, norm_name):
            def forward_with_eca(x):
                x = orig_forward(x)
                # call the registered eca module
                return getattr(parent_mod, f'eca_after_{norm_name}')(x)
            return forward_with_eca
        last_norm.forward = make_forward_with_eca_final(original_forward, parent, last_norm_name)
    
    return model

