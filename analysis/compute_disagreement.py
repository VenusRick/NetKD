#!/usr/bin/env python3
"""
计算教师模型之间的多样性（Disagreement）

多样性指标：
1. Disagreement Rate: 教师之间预测不一致的比例
2. Average KL Divergence: 教师输出分布之间的平均KL散度
3. Q-Statistic: 成对教师之间的统计相关性
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict
from sklearn.metrics import cohen_kappa_score


def compute_disagreement_rate(predictions: List[torch.Tensor]) -> float:
    """
    计算教师之间的不一致率
    
    Args:
        predictions: List of prediction tensors, each of shape (N,)
    
    Returns:
        Disagreement rate (0-1)
    """
    n_teachers = len(predictions)
    n_samples = len(predictions[0])
    
    disagreement_count = 0
    for i in range(n_samples):
        preds = [p[i].item() for p in predictions]
        if len(set(preds)) > 1:  # 至少有两个教师预测不同
            disagreement_count += 1
    
    return disagreement_count / n_samples


def compute_kl_divergence_matrix(logits_list: List[torch.Tensor]) -> np.ndarray:
    """
    计算教师输出之间的KL散度矩阵
    
    Args:
        logits_list: List of logit tensors, each of shape (N, C)
    
    Returns:
        KL divergence matrix of shape (n_teachers, n_teachers)
    """
    n_teachers = len(logits_list)
    kl_matrix = np.zeros((n_teachers, n_teachers))
    
    # 转换为概率分布
    probs_list = [F.softmax(logits, dim=-1) for logits in logits_list]
    
    for i in range(n_teachers):
        for j in range(i+1, n_teachers):
            # 计算双向KL散度的平均
            kl_ij = F.kl_div(
                F.log_softmax(logits_list[i], dim=-1),
                probs_list[j],
                reduction='batchmean'
            ).item()
            
            kl_ji = F.kl_div(
                F.log_softmax(logits_list[j], dim=-1),
                probs_list[i],
                reduction='batchmean'
            ).item()
            
            avg_kl = (kl_ij + kl_ji) / 2
            kl_matrix[i, j] = avg_kl
            kl_matrix[j, i] = avg_kl
    
    return kl_matrix


def compute_q_statistic(pred_i: torch.Tensor, pred_j: torch.Tensor, 
                       labels: torch.Tensor) -> float:
    """
    计算两个分类器之间的Q统计量
    
    Q = (N11*N00 - N01*N10) / (N11*N00 + N01*N10)
    其中 Nij 表示分类器i正确/错误 和 分类器j正确/错误的样本数
    
    Q接近1表示模型相似，接近-1表示模型互补，接近0表示独立
    """
    correct_i = (pred_i == labels)
    correct_j = (pred_j == labels)
    
    N11 = torch.sum(correct_i & correct_j).item()
    N00 = torch.sum(~correct_i & ~correct_j).item()
    N01 = torch.sum(correct_i & ~correct_j).item()
    N10 = torch.sum(~correct_i & correct_j).item()
    
    numerator = N11 * N00 - N01 * N10
    denominator = N11 * N00 + N01 * N10
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


def compute_diversity_metrics(models: List[torch.nn.Module], 
                              dataloader, 
                              device: str = 'cuda') -> Dict:
    """
    计算教师集合的多样性指标
    
    Args:
        models: List of teacher models
        dataloader: Data loader
        device: Device to run on
    
    Returns:
        Dictionary of diversity metrics
    """
    n_teachers = len(models)
    
    # 收集所有预测
    all_logits = [[] for _ in range(n_teachers)]
    all_predictions = [[] for _ in range(n_teachers)]
    all_labels = []
    
    for model in models:
        model.eval()
        model.to(device)
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            all_labels.append(y)
            
            for i, model in enumerate(models):
                logits = model(x)
                predictions = logits.argmax(dim=-1)
                
                all_logits[i].append(logits)
                all_predictions[i].append(predictions)
    
    # 合并所有批次
    all_labels = torch.cat(all_labels)
    all_logits = [torch.cat(logits) for logits in all_logits]
    all_predictions = [torch.cat(preds) for preds in all_predictions]
    
    # 计算指标
    disagreement_rate = compute_disagreement_rate(all_predictions)
    kl_matrix = compute_kl_divergence_matrix(all_logits)
    avg_kl = np.mean(kl_matrix[np.triu_indices(n_teachers, k=1)])
    
    # 计算Q统计量矩阵
    q_matrix = np.zeros((n_teachers, n_teachers))
    for i in range(n_teachers):
        for j in range(i+1, n_teachers):
            q = compute_q_statistic(all_predictions[i], all_predictions[j], all_labels)
            q_matrix[i, j] = q
            q_matrix[j, i] = q
    
    avg_q = np.mean(q_matrix[np.triu_indices(n_teachers, k=1)])
    
    # 计算每个教师的准确率
    accuracies = []
    for preds in all_predictions:
        acc = (preds == all_labels).float().mean().item()
        accuracies.append(acc)
    
    return {
        'disagreement_rate': disagreement_rate,
        'avg_kl_divergence': avg_kl,
        'kl_matrix': kl_matrix.tolist(),
        'avg_q_statistic': avg_q,
        'q_matrix': q_matrix.tolist(),
        'teacher_accuracies': accuracies,
        'avg_accuracy': np.mean(accuracies),
    }


def print_diversity_report(metrics: Dict, teacher_names: List[str] = None):
    """打印多样性分析报告"""
    if teacher_names is None:
        teacher_names = [f"Teacher {i+1}" for i in range(len(metrics['teacher_accuracies']))]
    
    print("\n" + "="*80)
    print("教师模型多样性分析报告")
    print("="*80)
    
    print(f"\n📊 总体指标:")
    print(f"  Disagreement Rate: {metrics['disagreement_rate']:.4f}")
    print(f"  Avg KL Divergence: {metrics['avg_kl_divergence']:.4f}")
    print(f"  Avg Q-Statistic: {metrics['avg_q_statistic']:.4f}")
    print(f"  Avg Teacher Accuracy: {metrics['avg_accuracy']:.4f}")
    
    print(f"\n👥 各教师准确率:")
    for name, acc in zip(teacher_names, metrics['teacher_accuracies']):
        print(f"  {name:<20}: {acc:.4f}")
    
    print(f"\n📈 Q-统计量矩阵 (值越小表示越多样):")
    n = len(teacher_names)
    print(f"  {'':20}", end='')
    for name in teacher_names:
        print(f"{name[:15]:<17}", end='')
    print()
    
    q_matrix = np.array(metrics['q_matrix'])
    for i, name in enumerate(teacher_names):
        print(f"  {name[:20]:<20}", end='')
        for j in range(n):
            if i == j:
                print(f"{'---':<17}", end='')
            else:
                print(f"{q_matrix[i][j]:6.3f}{'':<11}", end='')
        print()
    
    print("\n💡 解释:")
    print("  - Disagreement Rate: 教师预测不一致的比例，越高表示越多样")
    print("  - KL Divergence: 输出分布的差异，越大表示越多样")
    print("  - Q-Statistic: [-1, 1]，接近0表示独立，接近1表示相似，接近-1表示互补")
    print("="*80 + "\n")


if __name__ == "__main__":
    print("教师多样性计算模块")
    print("使用示例:")
    print("  from analysis.compute_disagreement import compute_diversity_metrics")
    print("  metrics = compute_diversity_metrics(models, test_loader)")
    print("  print_diversity_report(metrics, teacher_names)")
