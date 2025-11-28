"""
绘制注意力机制消融实验-仅性能对比图(单图版)
移除Agent Attention,使用学术论文配色方案
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 设置样式 - 使用更学术的配置
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'

# 读取结果
results_dir = Path("checkpoints/attention_ablation")
# 移除agent,只保留其他4个机制
attention_types = ['cbam', 'eca', 'simam', 'none']

# 注意力机制的显示名称 - 更简洁的标签
attention_names = {
    'cbam': 'CBAM',
    'eca': 'ECA', 
    'simam': 'SimAM',
    'none': 'Baseline'
}

# 读取所有结果
data = {}
for att_type in attention_types:
    result_file = results_dir / f"attention_{att_type}" / "results.json"
    if result_file.exists():
        with open(result_file, 'r') as f:
            data[att_type] = json.load(f)

if not data:
    print("错误: 未找到实验结果文件")
    exit(1)

# 提取准确率数据
accuracies = [data[att]['best_acc'] * 100 for att in attention_types if att in data]
labels = [attention_names[att] for att in attention_types if att in data]

# 学术论文配色方案 - 更明快、专业
# 使用饱和度较高的颜色,避免过暗或过淡
colors = [
    '#FF6B6B',  # 珊瑚红 - CBAM (最佳性能)
    '#4ECDC4',  # 青绿色 - ECA
    '#95E1D3',  # 薄荷绿 - SimAM  
    '#FFA07A'   # 浅橙色 - Baseline
]

# 创建单图
fig, ax = plt.subplots(figsize=(10, 7), dpi=100)

# 绘制条形图
x_pos = np.arange(len(labels))
bars = ax.bar(x_pos, accuracies, color=colors, alpha=0.85, 
              edgecolor='white', linewidth=2.5, width=0.7)

# 添加数值标注 - 更清晰的字体
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.15,
            f'{acc:.2f}%',
            ha='center', va='bottom', 
            fontsize=13, fontweight='bold',
            color='#2C3E50')  # 深灰蓝色

# 添加最佳准确率参考线
best_acc = max(accuracies)
ax.axhline(y=best_acc, color='#E74C3C', linestyle='--', 
           linewidth=2, alpha=0.7, label=f'Best: {best_acc:.2f}%')

# 设置坐标轴
ax.set_xlabel('Attention Mechanism', fontsize=15, fontweight='bold', color='#2C3E50')
ax.set_ylabel('Test Accuracy (%)', fontsize=15, fontweight='bold', color='#2C3E50')
ax.set_title('Attention Mechanism Performance Comparison', 
             fontsize=17, fontweight='bold', pad=20, color='#2C3E50')

ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=14, fontweight='600')

# Y轴范围优化 - 聚焦在关键区域
y_min = min(accuracies) - 1
y_max = max(accuracies) + 1
ax.set_ylim([y_min, y_max])

# 图例
ax.legend(fontsize=12, loc='lower right', framealpha=0.95, 
          edgecolor='#BDC3C7', fancybox=True, shadow=True)

# 网格优化
ax.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.8, color='#95A5A6')
ax.set_axisbelow(True)  # 网格置于底层

# 边框美化
for spine in ax.spines.values():
    spine.set_edgecolor('#BDC3C7')
    spine.set_linewidth(1.2)

# 背景色微调
ax.set_facecolor('#FAFAFA')
fig.patch.set_facecolor('white')

# 保存高清图
plt.tight_layout()

# 超高清版本 (论文用)
output_path_hq = "docs/attention_performance_comparison_hq.png"
plt.savefig(output_path_hq, dpi=600, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print(f"✓ 高清图表已保存至: {output_path_hq}")

# 标准版本
output_path = "docs/attention_performance_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f"✓ 标准图表已保存至: {output_path}")

# 打印结果摘要
print("\n" + "="*70)
print("注意力机制性能对比 (ISCXVPN2016)")
print("="*70)
print(f"{'机制':<15} {'测试准确率':<15} {'参数量':<15}")
print("-"*70)

for att in attention_types:
    if att in data:
        d = data[att]
        print(f"{attention_names[att]:<15} "
              f"{d['best_acc']*100:>6.2f}%        "
              f"{d['total_params']:>10,}")

print("="*70)
print(f"\n最佳机制: {attention_names[max(data.items(), key=lambda x: x[1]['best_acc'])[0]]} "
      f"({max(accuracies):.2f}%)")
print("\n数据来源:")
print(f"  • 源数据目录: checkpoints/attention_ablation/")
print(f"  • 生成脚本: scripts/plot_attention_performance_only.py")
print(f"  • 高清图片: docs/attention_performance_comparison_hq.png (600 DPI)")
print("="*70)
