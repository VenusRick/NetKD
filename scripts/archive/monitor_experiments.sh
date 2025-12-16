#!/bin/bash
# 实验监控脚本

EXP_DIR="results/full_pipeline_20251208_004257"

while true; do
    clear
    echo "=========================================="
    echo "TrafficKD 实验监控 - $(date)"
    echo "=========================================="
    echo ""
    
    echo "=== GPU 状态 ==="
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv
    echo ""
    
    echo "=== 进程状态 ==="
    ps aux | grep full_pipeline | grep python | grep -v grep | wc -l
    echo "个进程正在运行"
    echo ""
    
    echo "=== GPU 0 (预训练) 最新日志 ==="
    tail -3 ${EXP_DIR}/gpu0_full_fixed/experiment.log 2>/dev/null || echo "等待中..."
    echo ""
    
    echo "=== GPU 1 (直接微调) 最新日志 ==="
    tail -3 ${EXP_DIR}/gpu1_scratch/experiment.log 2>/dev/null || echo "等待中..."
    echo ""
    
    echo "=== GPU 2 (额外实验) 最新日志 ==="
    tail -3 ${EXP_DIR}/gpu2_extra/experiment.log 2>/dev/null || echo "等待中..."
    echo ""
    
    echo "按 Ctrl+C 停止监控"
    sleep 60
done
