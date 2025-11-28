"""
Focal Loss implementation for handling class imbalance
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss: 自动聚焦难分类样本
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    当样本容易分类时(p_t大),损失权重(1-p_t)^γ接近0
    当样本难分类时(p_t小),损失权重接近1
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 类别权重 (num_classes,)
        self.gamma = gamma  # 聚焦参数,通常为2
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) 模型输出logits
            targets: (N,) 真实标签
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        p_t = torch.exp(-ce_loss)
        focal_loss = ((1 - p_t) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ClassBalancedLoss(nn.Module):
    """
    类别平衡损失: 根据样本数反向加权
    """
    def __init__(self, samples_per_class, beta=0.9999):
        super(ClassBalancedLoss, self).__init__()
        effective_num = 1.0 - torch.pow(beta, samples_per_class)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * len(weights)
        self.register_buffer('weights', weights)
    
    def forward(self, inputs, targets):
        return F.cross_entropy(inputs, targets, weight=self.weights)

def get_class_weights(dataset, num_classes=7):
    """计算类别权重 (反比于样本数)"""
    class_counts = torch.zeros(num_classes)
    for _, label in dataset:
        class_counts[label] += 1
    
    # 反比权重
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * num_classes
    return weights

