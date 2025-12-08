#!/bin/bash
# TrafficKD 三GPU并行实验启动脚本

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="results/full_pipeline_${TIMESTAMP}"

echo "=========================================="
echo "TrafficKD 全量实验 - 三GPU并行"
echo "时间戳: ${TIMESTAMP}"
echo "输出目录: ${OUTPUT_BASE}"
echo "=========================================="

# 创建主输出目录
mkdir -p ${OUTPUT_BASE}

# GPU 0: 完整实验流程 (包含预训练)
echo "[GPU 0] 启动完整实验 (含预训练)..."
nohup python scripts/full_pipeline_experiment.py \
    --gpu 0 \
    --output_dir ${OUTPUT_BASE}/gpu0_full \
    > ${OUTPUT_BASE}/gpu0.log 2>&1 &
PID0=$!
echo "GPU 0 PID: ${PID0}"

# GPU 1: 直接微调对比实验 (跳过预训练)
echo "[GPU 1] 启动直接微调实验..."
nohup python scripts/full_pipeline_experiment.py \
    --gpu 1 \
    --skip_pretrain \
    --output_dir ${OUTPUT_BASE}/gpu1_scratch \
    > ${OUTPUT_BASE}/gpu1.log 2>&1 &
PID1=$!
echo "GPU 1 PID: ${PID1}"

# GPU 2: 额外学生模型实验
echo "[GPU 2] 启动额外学生实验..."
nohup python scripts/full_pipeline_experiment.py \
    --gpu 2 \
    --skip_pretrain \
    --output_dir ${OUTPUT_BASE}/gpu2_extra \
    > ${OUTPUT_BASE}/gpu2.log 2>&1 &
PID2=$!
echo "GPU 2 PID: ${PID2}"

# 记录PID
echo "${PID0}" > ${OUTPUT_BASE}/pid_gpu0.txt
echo "${PID1}" > ${OUTPUT_BASE}/pid_gpu1.txt
echo "${PID2}" > ${OUTPUT_BASE}/pid_gpu2.txt

echo ""
echo "=========================================="
echo "所有实验已启动!"
echo "查看日志: tail -f ${OUTPUT_BASE}/gpu*.log"
echo "监控GPU: watch -n 5 nvidia-smi"
echo "=========================================="
