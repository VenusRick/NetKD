#!/bin/bash
# 运行两个新实验的脚本
set -e

DATASET="ISCXVPN2016"
DATA_PATH="/workspace/yqm/Dataset/ISCXVPN2016"
GPU=${1:-0}

echo "=============================================="
echo "NetKD 新实验套件"
echo "GPU: $GPU, 数据集: $DATASET"
echo "开始时间: $(date)"
echo "=============================================="

cd /workspace/yqm/NetKD

# 实验1: 学生模型Baseline
echo -e "\n>>> 实验1: 学生模型Baseline"
python scripts/exp1_student_baseline.py --gpu $GPU --epochs 50

# 实验2: 教师模型消融实验
echo -e "\n>>> 实验2: 教师模型消融实验"
python scripts/exp2_teacher_ablation.py --gpu $GPU --epochs 30

echo -e "\n=============================================="
echo "实验完成! $(date)"
echo "结果: results/student_baseline/ 和 results/teacher_ablation/"
echo "=============================================="
