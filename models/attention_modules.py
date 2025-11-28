"""
Attention modules for ablation study.
包含 AgentAttention, CBAM, ECA, SimAM 和 None (无注意力) 的实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AgentAttention2D(nn.Module):
    """Agent attention block with aggregate and broadcast stages.
    
    Original implementation for baseline comparison.
    """

    def __init__(self, dim: int, num_heads: int = 4, num_agents: int = 8):
        super().__init__()
        self.num_agents = num_agents
        self.dim = dim
        self.agent_tokens = nn.Parameter(torch.randn(num_agents, dim))
        self.norm_x = nn.LayerNorm(dim)
        self.norm_a = nn.LayerNorm(dim)
        self.attn_agg = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.attn_broadcast = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.proj_out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        X = x.flatten(2).permute(0, 2, 1)  # [B, N, C]
        X_norm = self.norm_x(X)
        
        A = self.agent_tokens.unsqueeze(0).expand(B, -1, -1)
        A_norm = self.norm_a(A)
        
        # Aggregate
        A_agg, _ = self.attn_agg(query=A_norm, key=X_norm, value=X_norm)
        
        # Broadcast
        X_out, _ = self.attn_broadcast(query=X_norm, key=A_agg, value=A_agg)
        
        X_out = self.proj_out(X_out)
        X_out = X_out + X
        
        X_out = X_out.permute(0, 2, 1).view(B, C, H, W)
        return X_out


class ChannelAttention(nn.Module):
    """Channel attention module for CBAM."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out


class SpatialAttention(nn.Module):
    """Spatial attention module for CBAM."""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))
        return x * out


class CBAM(nn.Module):
    """Convolutional Block Attention Module.
    
    Classic hybrid attention (spatial + channel) mechanism.
    Reference: CBAM: Convolutional Block Attention Module (ECCV 2018)
    """
    
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class ECALayer(nn.Module):
    """Efficient Channel Attention.
    
    Lightweight channel attention without dimensionality reduction.
    Reference: ECA-Net: Efficient Channel Attention for Deep CNNs (CVPR 2020)
    """
    
    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        # Adaptive kernel size calculation
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature descriptor
        y = self.avg_pool(x)  # [B, C, 1, 1]
        
        # Channel attention
        y = self.conv(y.squeeze(-1).transpose(-1, -2))  # [B, 1, C]
        y = y.transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]
        
        # Attention weights
        y = self.sigmoid(y)
        
        return x * y.expand_as(x)


class SimAM(nn.Module):
    """Simple, Parameter-Free Attention Module.
    
    Zero-parameter attention using energy function.
    Reference: SimAM: A Simple, Parameter-Free Attention Module (ICML 2021)
    """
    
    def __init__(self, e_lambda: float = 1e-4):
        super().__init__()
        self.e_lambda = e_lambda
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        n = H * W - 1
        
        # Calculate spatial mean
        x_mean = x.mean(dim=[2, 3], keepdim=True)
        
        # Calculate energy
        x_var = ((x - x_mean) ** 2).sum(dim=[2, 3], keepdim=True) / n
        
        # Energy function: E = 4 * (σ^2 + λ) + 2 * (t - μ)^2
        # Attention: 1 / (1 + E)
        y = (x - x_mean).pow(2) / (4 * (x_var + self.e_lambda)) + 0.5
        
        return x * torch.sigmoid(y)


class NoAttention(nn.Module):
    """Identity module - no attention applied.
    
    Used as baseline to demonstrate the necessity of attention mechanisms.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def get_attention_module(name: str, channels: int, **kwargs):
    """Factory function to get attention module by name.
    
    Args:
        name: One of ['agent', 'cbam', 'eca', 'simam', 'none']
        channels: Number of input channels
        **kwargs: Additional arguments for specific modules
    
    Returns:
        Attention module instance
    """
    name = name.lower()
    
    if name == 'agent':
        return AgentAttention2D(
            dim=channels,
            num_heads=kwargs.get('num_heads', 4),
            num_agents=kwargs.get('num_agents', 8)
        )
    elif name == 'cbam':
        return CBAM(
            channels=channels,
            reduction=kwargs.get('reduction', 16),
            kernel_size=kwargs.get('kernel_size', 7)
        )
    elif name == 'eca':
        return ECALayer(
            channels=channels,
            gamma=kwargs.get('gamma', 2),
            b=kwargs.get('b', 1)
        )
    elif name == 'simam':
        return SimAM(e_lambda=kwargs.get('e_lambda', 1e-4))
    elif name == 'none':
        return NoAttention()
    else:
        raise ValueError(f"Unknown attention module: {name}. "
                        f"Choose from ['agent', 'cbam', 'eca', 'simam', 'none']")
