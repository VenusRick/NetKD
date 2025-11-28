#!/bin/bash
# 一次性检查Agent实验进度

echo "========================================"
echo "Agent Attention 修复版进度快照"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo

# 检查进程
if pgrep -f "run_attention_ablation_fixed.py agent" > /dev/null; then
    echo "✓ 实验正在运行中"
    echo
    
    # 提取关键信息
    echo "训练进度:"
    echo "----------------------------------------"
    tail -80 logs/attention_agent_fixed.log | grep -E "(Epoch [0-9]+/50|Val Loss:.*Val Acc:|✓ 保存最佳)" | tail -20
    
    echo
    echo "最佳结果:"
    tail -100 logs/attention_agent_fixed.log | grep "✓ 保存最佳" | tail -1
else
    echo "✗ 实验已完成!"
    echo
    
    # 显示最终结果
    if [ -f checkpoints/attention_ablation_fixed/attention_agent/results.json ]; then
        echo "最终结果:"
        echo "----------------------------------------"
        cat checkpoints/attention_ablation_fixed/attention_agent/results.json | grep -E "(best_acc|best_epoch|training_time)" | head -3
    fi
fi

echo "========================================"
