#!/bin/bash
# 实时监控Agent Attention修复版实验

while true; do
    clear
    echo "========================================"
    echo "Agent Attention 修复版实验监控"
    echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"
    echo
    
    # 检查进程状态
    if pgrep -f "run_attention_ablation_fixed.py agent" > /dev/null; then
        echo "✓ 实验正在运行中"
    else
        echo "✗ 实验已完成或未运行"
    fi
    echo
    
    # 显示最近的训练日志
    echo "最近训练进度:"
    echo "----------------------------------------"
    tail -40 logs/attention_agent_fixed.log | grep -E "(Epoch [0-9]+/50|Val Acc:|✓ 保存|训练完成)" | tail -15
    echo "========================================"
    echo
    echo "按 Ctrl+C 退出监控"
    
    sleep 10
done
