#!/bin/bash
# STEP 3 & STEP 4: 批量运行 KD 消融和数据效率实验

set -e

echo "=========================================="
echo "STEP 3 & STEP 4: KD Ablation + Data Efficiency Experiments"
echo "=========================================="
echo ""

# 学生模型列表
STUDENTS=("repvit_m0_9" "mobilenetv3_small")

# KD 配置
KD_CONFIGS=("ce_only" "ce_kl")

# 数据比例（STEP 4）
TRAIN_FRACTIONS=(1.0 0.5 0.2)

# 输出目录
OUTPUT_DIR="results/kd_ablation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "实验配置:"
echo "  学生模型: ${STUDENTS[@]}"
echo "  KD 配置: ${KD_CONFIGS[@]}"
echo "  数据比例: ${TRAIN_FRACTIONS[@]}"
echo "  输出目录: $OUTPUT_DIR"
echo ""

# 计数器
total_experiments=$((${#STUDENTS[@]} * ${#KD_CONFIGS[@]} * ${#TRAIN_FRACTIONS[@]}))
current=0

# GPU 分配（循环使用3个GPU）
GPUS=(0 1 2)

# 遍历所有实验组合
for student in "${STUDENTS[@]}"; do
    for kd_config in "${KD_CONFIGS[@]}"; do
        for train_frac in "${TRAIN_FRACTIONS[@]}"; do
            current=$((current + 1))
            gpu_id=${GPUS[$((current % 3))]}

            echo "=========================================="
            echo "实验 $current/$total_experiments"
            echo "学生: $student | KD: $kd_config | 数据比例: $train_frac | GPU: $gpu_id"
            echo "=========================================="

            python scripts/run_kd_data_efficiency.py \
                --student "$student" \
                --kd_config "$kd_config" \
                --train_fraction "$train_frac" \
                --gpu "$gpu_id" \
                --output_dir "$OUTPUT_DIR" \
                2>&1 | tee "$OUTPUT_DIR/log_${student}_${kd_config}_frac${train_frac}.txt"

            echo ""
            echo "✅ 完成 $current/$total_experiments"
            echo ""
            sleep 2
        done
    done
done

echo "=========================================="
echo "🎉 所有实验完成！"
echo "结果保存在: $OUTPUT_DIR"
echo "=========================================="

# 生成简要总结
echo ""
echo "=== 生成实验总结 ==="
python << PYTHON_SCRIPT
import json
from pathlib import Path

results_dir = Path("$OUTPUT_DIR")
all_results = []

for json_file in results_dir.rglob("metrics.json"):
    with open(json_file) as f:
        data = json.load(f)
        all_results.append({
            "student": data["student_name"],
            "kd_config": data["kd_config_id"],
            "train_fraction": data["train_fraction"],
            "test_acc": data["test_acc"],
            "macro_f1": data["macro_f1"],
            "params_m": data["params_m"]
        })

# 按学生和配置排序
all_results.sort(key=lambda x: (x["student"], x["kd_config"], x["train_fraction"]))

# 打印表格
print("\n实验结果汇总:")
print("-" * 100)
print(f"{'Student':<25} {'KD Config':<15} {'Train%':<10} {'Test Acc':<12} {'Macro F1':<12} {'Params(M)':<10}")
print("-" * 100)

for r in all_results:
    print(f"{r['student']:<25} {r['kd_config']:<15} {r['train_fraction']:<10.1%} "
          f"{r['test_acc']:<12.4f} {r['macro_f1']:<12.4f} {r['params_m']:<10.2f}")

print("-" * 100)

# 保存到CSV
import csv
with open(results_dir / "summary.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
    writer.writeheader()
    writer.writerows(all_results)

print(f"\n汇总已保存到: {results_dir / 'summary.csv'}")
PYTHON_SCRIPT

