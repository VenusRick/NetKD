"""
自适应加权知识蒸馏机制
核心思想: 动态调整教师权重,让学生优先学习最不懂的知识
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveWeightedDistillation(nn.Module):
    """
    自适应加权蒸馏:
    1. 计算学生与每个教师的KL散度
    2. KL散度越大,说明学生与该教师差异越大,越需要学习
    3. 动态分配更高权重给差异大的教师
    """
    def __init__(self, temperature=4.0, alpha=0.7, beta=0.3, 
                 adaptation_rate=0.1, min_weight=0.1):
        super(AdaptiveWeightedDistillation, self).__init__()
        self.T = temperature
        self.alpha = alpha  # 蒸馏损失权重
        self.beta = beta   # 硬标签损失权重
        self.adaptation_rate = adaptation_rate  # 权重更新速率
        self.min_weight = min_weight  # 最小权重,避免某个教师完全被忽略
        
    def forward(self, student_logits, teacher_logits_list, labels, 
                teacher_weights=None):
        """
        Args:
            student_logits: (N, C) 学生输出
            teacher_logits_list: List[(N, C)] 多个教师输出
            labels: (N,) 真实标签
            teacher_weights: (num_teachers,) 当前教师权重
        
        Returns:
            loss: 总损失
            new_weights: 更新后的教师权重
            teacher_losses: 每个教师的蒸馏损失(用于分析)
        """
        num_teachers = len(teacher_logits_list)
        
        # 初始化权重 (如果未提供)
        if teacher_weights is None:
            teacher_weights = torch.ones(num_teachers) / num_teachers
        
        teacher_weights = teacher_weights.to(student_logits.device)
        
        # 计算学生与每个教师的KL散度
        student_soft = F.log_softmax(student_logits / self.T, dim=1)
        teacher_losses = []
        
        for teacher_logits in teacher_logits_list:
            teacher_soft = F.softmax(teacher_logits / self.T, dim=1)
            # KL散度: 衡量学生与教师分布的差异
            kl_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
            teacher_losses.append(kl_loss)
        
        teacher_losses_tensor = torch.stack(teacher_losses)
        
        # 自适应权重更新:
        # 1. 计算每个教师损失的相对重要性
        # 2. 损失越大 → 学生越不懂 → 权重越高
        loss_importance = F.softmax(teacher_losses_tensor / self.adaptation_rate, dim=0)
        
        # 3. 指数移动平均更新权重 (避免剧烈变化)
        new_weights = (1 - self.adaptation_rate) * teacher_weights + \
                      self.adaptation_rate * loss_importance
        
        # 4. 确保最小权重 (避免某个教师完全被忽略)
        new_weights = torch.clamp(new_weights, min=self.min_weight)
        new_weights = new_weights / new_weights.sum()  # 归一化
        
        # 加权蒸馏损失
        weighted_distillation_loss = (teacher_losses_tensor * new_weights).sum()
        
        # 硬标签损失
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # 总损失
        total_loss = self.alpha * (self.T ** 2) * weighted_distillation_loss + \
                     self.beta * hard_loss
        
        return total_loss, new_weights.detach(), teacher_losses_tensor.detach()

class FixedWeightDistillation(nn.Module):
    """
    固定权重蒸馏 (Baseline对比)
    所有教师权重相等或预设
    """
    def __init__(self, temperature=4.0, alpha=0.7, beta=0.3, teacher_weights=None):
        super(FixedWeightDistillation, self).__init__()
        self.T = temperature
        self.alpha = alpha
        self.beta = beta
        self.teacher_weights = teacher_weights  # 固定权重
        
    def forward(self, student_logits, teacher_logits_list, labels):
        """
        Args:
            student_logits: (N, C)
            teacher_logits_list: List[(N, C)]
            labels: (N,)
        """
        num_teachers = len(teacher_logits_list)
        
        # 固定权重
        if self.teacher_weights is None:
            weights = torch.ones(num_teachers) / num_teachers
        else:
            weights = self.teacher_weights
        
        weights = weights.to(student_logits.device)
        
        # 计算蒸馏损失
        student_soft = F.log_softmax(student_logits / self.T, dim=1)
        distillation_loss = 0
        
        for i, teacher_logits in enumerate(teacher_logits_list):
            teacher_soft = F.softmax(teacher_logits / self.T, dim=1)
            kl_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
            distillation_loss += weights[i] * kl_loss
        
        # 硬标签损失
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # 总损失
        total_loss = self.alpha * (self.T ** 2) * distillation_loss + \
                     self.beta * hard_loss
        
        return total_loss

class StackingEnsemble(nn.Module):
    """
    Stacking集成教师 (单一强教师)
    可以作为学生的另一种蒸馏方式
    """
    def __init__(self, input_dim=21, hidden_dim=64, num_classes=7):
        super(StackingEnsemble, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

