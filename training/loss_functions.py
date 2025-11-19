"""Distillation losses for SD-MKD with improved numerical stability.

This module implements the composite loss used in SD-MKD training, combining
cross-entropy, forward/reverse KL, and Sinkhorn distances between teacher and
student distributions.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from models.student_model import softmax_with_temperature


def ce_loss(logits_s: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Standard cross-entropy loss for student logits.

    Args:
        logits_s: Student logits of shape ``[B, C]``.
        labels: Ground-truth labels of shape ``[B]``.
    """
    return F.cross_entropy(logits_s, labels)


def forward_kl(P_t: torch.Tensor, P_s: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Forward KL divergence KL(P_t || P_s) with improved numerical stability.

    Both inputs are assumed to be valid probability distributions.
    """
    # Clamp to avoid log(0)
    P_s_safe = torch.clamp(P_s, min=eps)
    P_t_safe = torch.clamp(P_t, min=eps)
    
    # Use log_softmax for stability
    log_P_s = P_s_safe.log()
    kl = F.kl_div(log_P_s, P_t_safe, reduction="batchmean")
    
    # Check for NaN and return safe value
    if torch.isnan(kl) or torch.isinf(kl):
        return torch.tensor(0.0, device=P_t.device, dtype=P_t.dtype)
    return kl


def reverse_kl(P_t: torch.Tensor, P_s: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Reverse KL divergence KL(P_s || P_t) with improved numerical stability."""
    # Clamp to avoid log(0)
    P_s_safe = torch.clamp(P_s, min=eps)
    P_t_safe = torch.clamp(P_t, min=eps)
    
    log_P_t = P_t_safe.log()
    kl = F.kl_div(log_P_t, P_s_safe, reduction="batchmean")
    
    # Check for NaN and return safe value
    if torch.isnan(kl) or torch.isinf(kl):
        return torch.tensor(0.0, device=P_t.device, dtype=P_t.dtype)
    return kl


def class_cost_matrix(num_classes: int, device=None) -> torch.Tensor:
    """Squared-distance cost matrix between class indices."""
    idx = torch.arange(num_classes, device=device, dtype=torch.float32)
    i = idx.unsqueeze(0)
    j = idx.unsqueeze(1)
    return (i - j).pow(2)


def sinkhorn_distance(
    P_t: torch.Tensor,
    P_s: torch.Tensor,
    C: torch.Tensor,
    epsilon: float = 0.1,
    n_iters: int = 50,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute Sinkhorn distance with improved numerical stability.

    Args:
        P_t: Teacher probability distributions ``[B, C]``.
        P_s: Student probability distributions ``[B, C]``.
        C: Cost matrix ``[C, C]``.
        epsilon: Entropic regularization coefficient.
        n_iters: Number of Sinkhorn iterations.
        eps: Small constant for numerical stability.
    Returns:
        Mean Sinkhorn cost over the batch.
    """
    B, C_dim = P_t.shape
    
    # Ensure inputs are valid probability distributions
    P_t_safe = torch.clamp(P_t, min=eps)
    P_s_safe = torch.clamp(P_s, min=eps)
    P_t_safe = P_t_safe / P_t_safe.sum(dim=1, keepdim=True)
    P_s_safe = P_s_safe / P_s_safe.sum(dim=1, keepdim=True)
    
    # Compute kernel matrix with clamping to avoid overflow
    K = torch.exp(-torch.clamp(C / epsilon, max=50.0))

    u = torch.ones(B, C_dim, device=P_t.device, dtype=P_t.dtype) / C_dim
    v = torch.ones(B, C_dim, device=P_t.device, dtype=P_t.dtype) / C_dim

    for _ in range(n_iters):
        Kv = torch.matmul(v, K.T)
        u = P_t_safe / (Kv + eps)
        
        # Clamp u to avoid explosion
        u = torch.clamp(u, max=1e6)
        
        Ku = torch.matmul(u, K)
        v = P_s_safe / (Ku + eps)
        
        # Clamp v to avoid explosion
        v = torch.clamp(v, max=1e6)

    pi = u.unsqueeze(2) * K.unsqueeze(0) * v.unsqueeze(1)
    cost = (pi * C.unsqueeze(0)).sum(dim=(1, 2))
    result = cost.mean()
    
    # Check for NaN and return safe value
    if torch.isnan(result) or torch.isinf(result):
        return torch.tensor(0.0, device=P_t.device, dtype=P_t.dtype)
    
    return result


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    cost_matrix: torch.Tensor,
    T: float = 4.0,
    lamb_ce: float = 1.0,
    lamb_f: float = 0.5,
    lamb_r: float = 0.5,
    lamb_s: float = 0.1,
) -> torch.Tensor:
    """Composite distillation loss for SD-MKD with improved numerical stability.

    Args:
        student_logits: Student logits ``[B, C]``.
        teacher_logits: Teacher logits ``[B, C]``.
        labels: Ground-truth labels ``[B]``.
        T: Softmax temperature for soft targets.
        lamb_ce: Weight for cross-entropy loss.
        lamb_f: Weight for forward KL loss.
        lamb_r: Weight for reverse KL loss.
        lamb_s: Weight for Sinkhorn loss.
        cost_matrix: Precomputed class cost matrix ``[C, C]``.
    """
    # Check for NaN in inputs
    if torch.isnan(student_logits).any() or torch.isnan(teacher_logits).any():
        print("Warning: NaN detected in input logits!")
        return torch.tensor(0.0, device=student_logits.device, requires_grad=True)
    
    L_ce = ce_loss(student_logits, labels)

    P_t_T = softmax_with_temperature(teacher_logits, T).detach()
    P_s_T = softmax_with_temperature(student_logits, T)

    L_f = forward_kl(P_t_T, P_s_T)
    L_r = reverse_kl(P_t_T, P_s_T)
    L_s = sinkhorn_distance(P_t_T, P_s_T, cost_matrix)

    total_loss = lamb_ce * L_ce + lamb_f * L_f + lamb_r * L_r + lamb_s * L_s
    
    # Final NaN check
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        print(f"Warning: NaN/Inf in total loss! L_ce={L_ce:.4f}, L_f={L_f:.4f}, L_r={L_r:.4f}, L_s={L_s:.4f}")
        return torch.tensor(0.0, device=student_logits.device, requires_grad=True)
    
    return total_loss
