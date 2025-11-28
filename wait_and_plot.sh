#!/bin/bash
# 等待实验完成后自动绘图

echo "等待Agent Attention修复版实验完成..."
echo "当前时间: $(date '+%H:%M:%S')"
echo

# 每30秒检查一次
while pgrep -f "run_attention_ablation_fixed.py agent" > /dev/null; do
    # 显示最新进度
    latest_epoch=$(tail -100 logs/attention_agent_fixed.log | grep -E "Epoch [0-9]+/50" | tail -1)
    latest_acc=$(tail -100 logs/attention_agent_fixed.log | grep "Val Acc:" | tail -1 | grep -oP "Val Acc: \K[0-9.]+")
    
    if [ -n "$latest_epoch" ]; then
        echo "[$(date '+%H:%M:%S')] $latest_epoch | 最新Val Acc: ${latest_acc}%"
    fi
    
    sleep 30
done

echo
echo "✓ 实验已完成! 时间: $(date '+%H:%M:%S')"
echo

# 检查结果文件
if [ -f checkpoints/attention_ablation_fixed/attention_agent/results.json ]; then
    echo "Agent Attention修复版结果:"
    cat checkpoints/attention_ablation_fixed/attention_agent/results.json | grep -E "(best_acc|best_epoch|total_params|training_time)" | head -4
    echo
    
    # 复制结果到原目录用于对比
    echo "复制结果到原消融目录..."
    cp checkpoints/attention_ablation_fixed/attention_agent/results.json checkpoints/attention_ablation/attention_agent/results.json
    cp checkpoints/attention_ablation_fixed/attention_agent/best_model.pth checkpoints/attention_ablation/attention_agent/best_model.pth
    echo "✓ 结果已更新"
    echo
    
    # 重新绘图
    echo "重新绘制对比图..."
    python scripts/plot_attention_results.py
    
    echo
    echo "=========================================="
    echo "✓ 所有任务完成!"
    echo "图表位置: docs/attention_ablation_results.png"
    echo "=========================================="
else
    echo "❌ 未找到结果文件"
fi
