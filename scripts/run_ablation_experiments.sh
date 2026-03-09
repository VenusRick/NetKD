#!/bin/bash

# 消融实验自动化脚本
# 按顺序执行 S-CE, S-KL, S-KL2 三个实验

set -e  # 遇到错误立即退出

# 日志目录
LOG_DIR="logs/ablation"
mkdir -p "$LOG_DIR"

# 时间戳
START_TIME=$(date +%s)
echo "========================================"
echo "开始消融实验"
echo "时间: $(date)"
echo "========================================"
echo

# 实验1: S-CE (仅交叉熵,无教师知识)
echo "[1/3] 运行 S-CE (GT only)..."
EXP_START=$(date +%s)
python run_ablation_student.py ce > "$LOG_DIR/s_ce.log" 2>&1
EXP_END=$(date +%s)
EXP_DURATION=$((EXP_END - EXP_START))
echo "✓ S-CE 完成 (耗时: ${EXP_DURATION}s = $((EXP_DURATION/60))min)"
echo

# 实验2: S-KL (GT + Forward KL)
echo "[2/3] 运行 S-KL (GT + FKL)..."
EXP_START=$(date +%s)
python run_ablation_student.py kl > "$LOG_DIR/s_kl.log" 2>&1
EXP_END=$(date +%s)
EXP_DURATION=$((EXP_END - EXP_START))
echo "✓ S-KL 完成 (耗时: ${EXP_DURATION}s = $((EXP_DURATION/60))min)"
echo

# 实验3: S-KL2 (GT + FKL + RKL)
echo "[3/3] 运行 S-KL2 (GT + FKL + RKL)..."
EXP_START=$(date +%s)
python run_ablation_student.py kl2 > "$LOG_DIR/s_kl2.log" 2>&1
EXP_END=$(date +%s)
EXP_DURATION=$((EXP_END - EXP_START))
echo "✓ S-KL2 完成 (耗时: ${EXP_DURATION}s = $((EXP_DURATION/60))min)"
echo

# 总结
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
echo "========================================"
echo "所有消融实验完成!"
echo "总耗时: ${TOTAL_DURATION}s ($((TOTAL_DURATION / 60))分钟)"
echo "结果保存在: checkpoints/ablation/"
echo "日志保存在: $LOG_DIR/"
echo "========================================"
echo
echo "运行以下命令查看结果对比:"
echo "python scripts/summarize_ablation_results.py"
