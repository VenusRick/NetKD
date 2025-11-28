"""
绘制注意力机制消融实验结果对比图
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-darkgrid')

# 读取结果
results_dir = Path("checkpoints/attention_ablation")
attention_types = ['agent', 'cbam', 'eca', 'simam', 'none']

# 注意力机制的显示名称和描述
attention_names = {
    'agent': 'Agent Attention\n(Baseline)',
    'cbam': 'CBAM\n(Spatial+Channel)',
    'eca': 'ECA\n(Efficient Channel)',
    'simam': 'SimAM\n(Parameter-Free)',
    'none': 'No Attention\n(Baseline)'
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
    print(f"请确保已运行实验并且结果保存在 {results_dir}")
    exit(1)

# 提取数据
accuracies = [data[att]['best_acc'] * 100 for att in attention_types if att in data]
params = [data[att]['total_params'] / 1000 for att in attention_types if att in data]  # 转换为K
times = [data[att]['training_time'] / 60 for att in attention_types if att in data]  # 转换为分钟
labels = [attention_names[att] for att in attention_types if att in data]

# 创建图表 (2行2列布局)
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

# 配色方案
colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#95A5A6']
colors = colors[:len(accuracies)]

# ========== 子图1: 准确率对比 (条形图) ==========
ax1 = fig.add_subplot(gs[0, 0])
x_pos = np.arange(len(labels))
bars1 = ax1.bar(x_pos, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# 添加数值标注
for bar, acc in zip(bars1, accuracies):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.2,
             f'{acc:.2f}%',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# 添加最佳准确率基线
best_acc = max(accuracies)
ax1.axhline(y=best_acc, color='red', linestyle='--', linewidth=2, 
            label=f'Best: {best_acc:.2f}%', alpha=0.6)

ax1.set_xlabel('Attention Mechanism', fontsize=13, fontweight='bold')
ax1.set_ylabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=15)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(labels, fontsize=10)
ax1.set_ylim([min(accuracies) - 2, max(accuracies) + 2])
ax1.legend(fontsize=10, loc='lower right')
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# ========== 子图2: 参数量对比 (条形图) ==========
ax2 = fig.add_subplot(gs[0, 1])
bars2 = ax2.bar(x_pos, params, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# 添加数值标注
for bar, param in zip(bars2, params):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{param:.1f}K',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

ax2.set_xlabel('Attention Mechanism', fontsize=13, fontweight='bold')
ax2.set_ylabel('Parameters (K)', fontsize=13, fontweight='bold')
ax2.set_title('Model Complexity Comparison', fontsize=14, fontweight='bold', pad=15)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(labels, fontsize=10)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# ========== 子图3: 训练时间对比 (条形图) ==========
ax3 = fig.add_subplot(gs[1, 0])
bars3 = ax3.bar(x_pos, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# 添加数值标注
for bar, t in zip(bars3, times):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{t:.1f}min',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

ax3.set_xlabel('Attention Mechanism', fontsize=13, fontweight='bold')
ax3.set_ylabel('Training Time (minutes)', fontsize=13, fontweight='bold')
ax3.set_title('Training Efficiency Comparison', fontsize=14, fontweight='bold', pad=15)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(labels, fontsize=10)
ax3.grid(axis='y', alpha=0.3, linestyle='--')

# ========== 子图4: 综合性能雷达图 ==========
ax4 = fig.add_subplot(gs[1, 1], projection='polar')

# 归一化数据到 0-100 范围
acc_norm = np.array(accuracies)
param_norm = 100 - (np.array(params) - min(params)) / (max(params) - min(params)) * 100  # 参数越少越好
time_norm = 100 - (np.array(times) - min(times)) / (max(times) - min(times)) * 100  # 时间越短越好

# 设置雷达图的角度
categories = ['Accuracy', 'Efficiency\n(Params)', 'Speed\n(Time)']
N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # 闭合

# 绘制每个注意力机制
for i, att in enumerate([att for att in attention_types if att in data]):
    values = [acc_norm[i], param_norm[i], time_norm[i]]
    values += values[:1]  # 闭合
    
    ax4.plot(angles, values, 'o-', linewidth=2, label=attention_names[att].replace('\n', ' '),
             color=colors[i], alpha=0.8)
    ax4.fill(angles, values, alpha=0.15, color=colors[i])

ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(categories, fontsize=11)
ax4.set_ylim(0, 100)
ax4.set_title('Comprehensive Performance Radar', fontsize=14, fontweight='bold', pad=20)
ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
ax4.grid(True, alpha=0.3)

# 保存图表
output_path = "docs/attention_ablation_results.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ 图表已保存至: {output_path}")

# 高分辨率版本
output_path_hq = "docs/attention_ablation_results_hq.png"
plt.savefig(output_path_hq, dpi=600, bbox_inches='tight')
print(f"✓ 高清版本已保存至: {output_path_hq}")

# 打印结果表格
print("\n" + "="*90)
print("注意力机制消融实验结果汇总 (ISCXVPN2016)")
print("="*90)
print(f"{'注意力类型':<20} {'测试准确率':<15} {'参数量':<15} {'训练时间':<15} {'最佳Epoch':<12}")
print("-"*90)

for att in attention_types:
    if att in data:
        d = data[att]
        print(f"{attention_names[att].replace(chr(10), ' '):<20} "
              f"{d['best_acc']*100:>6.2f}%        "
              f"{d['total_params']:>10,}    "
              f"{d['training_time']/60:>6.2f}min      "
              f"{d['best_epoch']:>3}")

print("="*90)

# 关键发现
best_acc_att = max(data.items(), key=lambda x: x[1]['best_acc'])[0]
min_param_att = min(data.items(), key=lambda x: x[1]['total_params'])[0]
min_time_att = min(data.items(), key=lambda x: x[1]['training_time'])[0]

print("\n关键发现:")
print(f"  • 最佳准确率: {attention_names[best_acc_att].replace(chr(10), ' ')} "
      f"({data[best_acc_att]['best_acc']*100:.2f}%)")
print(f"  • 最少参数量: {attention_names[min_param_att].replace(chr(10), ' ')} "
      f"({data[min_param_att]['total_params']:,})")
print(f"  • 最快训练速度: {attention_names[min_time_att].replace(chr(10), ' ')} "
      f"({data[min_time_att]['training_time']/60:.2f}min)")
print(f"  • Agent Attention 相比无注意力提升: "
      f"{(data['agent']['best_acc'] - data['none']['best_acc'])*100:.2f}%")
print("="*90)
