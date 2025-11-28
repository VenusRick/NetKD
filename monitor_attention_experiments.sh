#!/bin/bash
# 实时监控注意力机制实验进度

clear
echo "========================================"
echo "注意力机制消融实验 - 实时监控"
echo "========================================"
echo ""

while true; do
    echo -e "\n[$(date '+%H:%M:%S')] 最新进度:"
    echo "----------------------------------------"
    
    # 检查进程是否还在运行
    if pgrep -f "run_attention_experiments.sh" > /dev/null; then
        echo "✓ 实验进程运行中..."
    else
        echo "✗ 实验进程已结束"
    fi
    
    # 显示当前正在训练的模型
    CURRENT_ATTN=$(tail -50 logs/attention_experiments_full.log 2>/dev/null | grep -oP "实验: \K\w+" | tail -1)
    if [ ! -z "$CURRENT_ATTN" ]; then
        echo "📍 当前实验: $CURRENT_ATTN"
    fi
    
    # 显示最新的训练进度
    LATEST_EPOCH=$(tail -20 logs/attention_experiments_full.log 2>/dev/null | grep -oP "Epoch \K\d+/\d+" | tail -1)
    if [ ! -z "$LATEST_EPOCH" ]; then
        echo "🔄 训练进度: Epoch $LATEST_EPOCH"
    fi
    
    LATEST_ACC=$(tail -20 logs/attention_experiments_full.log 2>/dev/null | grep -oP "Val Acc: \K[\d.]+%" | tail -1)
    if [ ! -z "$LATEST_ACC" ]; then
        echo "📊 最新验证准确率: $LATEST_ACC"
    fi
    
    echo ""
    echo "已完成的实验:"
    for att in agent cbam eca simam none; do
        if [ -f "checkpoints/attention_ablation/attention_$att/results.json" ]; then
            ACC=$(python3 -c "import json; print(f\"{json.load(open('checkpoints/attention_ablation/attention_$att/results.json'))['best_acc']*100:.2f}%\")" 2>/dev/null)
            echo "  ✓ $att: $ACC"
        fi
    done
    
    echo ""
    echo "按 Ctrl+C 退出监控"
    
    sleep 10
    clear
    echo "========================================"
    echo "注意力机制消融实验 - 实时监控"
    echo "========================================"
    echo ""
done
