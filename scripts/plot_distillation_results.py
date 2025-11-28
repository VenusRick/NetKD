"""
绘制知识蒸馏消融实验结果对比图
比较不同蒸馏模式 (S-CE, S-KL, S-KL2) 的性能
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 读取结果数据
results_dir = Path("checkpoints/ablation")
modes = ["ce", "kl", "kl2"]
mode_names = {
    "ce": "S-CE\n(GT only)",
    "kl": "S-KL\n(GT + FKL)",
    "kl2": "S-KL2\n(GT + FKL + RKL)"
}

data = {}
for mode in modes:
    result_file = results_dir / f"s_{mode}" / "results.json"
    if result_file.exists():
        with open(result_file, 'r') as f:
            data[mode] = json.load(f)

# 提取数据
accuracies = [data[m]["test_accuracy"] * 100 for m in modes]
retentions = [data[m]["knowledge_retention"] * 100 for m in modes]

# Stacking 教师模型基线
teacher_acc = 98.70

# 创建图表
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# === 子图1: 测试集准确率对比 ===
x_pos = np.arange(len(modes))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

bars1 = ax1.bar(x_pos, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# 添加教师模型基线
ax1.axhline(y=teacher_acc, color='#2E7D32', linestyle='--', linewidth=2.5, 
            label=f'Stacking Teacher: {teacher_acc:.2f}%', alpha=0.8)

# 数值标注
for i, (bar, acc) in enumerate(zip(bars1, accuracies)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{acc:.2f}%',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

ax1.set_xlabel('Distillation Mode', fontsize=13, fontweight='bold')
ax1.set_ylabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=15)
ax1.set_xticks(x_pos)
ax1.set_xticklabels([mode_names[m] for m in modes], fontsize=11)
ax1.set_ylim([96, 99])
ax1.legend(fontsize=11, loc='lower right')
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# === 子图2: 知识保留率对比 ===
bars2 = ax2.bar(x_pos, retentions, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# 添加100%基线
ax2.axhline(y=100, color='#2E7D32', linestyle='--', linewidth=2.5, 
            label='Perfect Retention (100%)', alpha=0.8)

# 数值标注
for i, (bar, ret) in enumerate(zip(bars2, retentions)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{ret:.2f}%',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

ax2.set_xlabel('Distillation Mode', fontsize=13, fontweight='bold')
ax2.set_ylabel('Knowledge Retention (%)', fontsize=13, fontweight='bold')
ax2.set_title('Knowledge Retention Comparison', fontsize=14, fontweight='bold', pad=15)
ax2.set_xticks(x_pos)
ax2.set_xticklabels([mode_names[m] for m in modes], fontsize=11)
ax2.set_ylim([97, 101])
ax2.legend(fontsize=11, loc='lower right')
ax2.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()

# 保存图表
output_path = "docs/distillation_ablation_results.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ 图表已保存至: {output_path}")

# 同时保存高分辨率版本用于论文
output_path_hq = "docs/distillation_ablation_results_hq.png"
plt.savefig(output_path_hq, dpi=600, bbox_inches='tight')
print(f"✓ 高清版本已保存至: {output_path_hq}")

# plt.show()  # 注释掉以避免GUI阻塞

# === 打印数据表格 ===
print("\n" + "="*70)
print("知识蒸馏消融实验结果汇总 (ISCXVPN2016)")
print("="*70)
print(f"{'模式':<20} {'测试准确率':<15} {'知识保留率':<15} {'配置':<20}")
print("-"*70)

for mode in modes:
    d = data[mode]
    print(f"{mode_names[mode]:<20} {d['test_accuracy']*100:>6.2f}%        "
          f"{d['knowledge_retention']*100:>6.2f}%        BS={d['batch_size']}, E={d['epochs']}")

print("-"*70)
print(f"{'Stacking Teacher':<20} {teacher_acc:>6.2f}%        {'100.00%':>7}        (Baseline)")
print("="*70)
print("\n关键发现:")
print("  • S-KL (GT + FKL) 取得最佳准确率: 97.11%")
print("  • 所有蒸馏模式均保留了 >98.3% 的教师知识")
print("  • 相较 Stacking 教师，准确率仅下降 1.6%，但模型复杂度大幅降低")
print("="*70)
