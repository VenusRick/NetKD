#!/bin/bash

# 监控消融实验进度

LOG_DIR="logs/ablation"

echo "========================================"
echo "消融实验进度监控"
echo "========================================"
echo

# 检查日志文件是否存在
for exp in s_ce s_kl s_kl2; do
    LOG_FILE="$LOG_DIR/${exp}.log"
    if [ -f "$LOG_FILE" ]; then
        echo "[$exp]"
        # 提取最后一次epoch信息
        tail -50 "$LOG_FILE" | grep -i "epoch" | tail -3 || echo "  (训练中...)"
        echo
    else
        echo "[$exp] - 未开始"
        echo
    fi
done

echo "========================================"
echo "实时监控 (按Ctrl+C退出):"
echo "tail -f $LOG_DIR/s_ce.log"
