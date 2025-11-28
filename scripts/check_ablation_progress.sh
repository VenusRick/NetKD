#!/bin/bash

# 快速检查消融实验进度

echo "========================================"
echo "消融实验进度检查"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo

# 检查进程状态
if pgrep -f "run_ablation_student.py" > /dev/null; then
    echo "✓ 实验进程正在运行"
    echo
else
    echo "✗ 未检测到运行中的实验进程"
    echo
fi

# 检查各个实验的进度
for mode in ce kl kl2; do
    log_file="logs/ablation/s_${mode}.log"
    if [ -f "$log_file" ]; then
        echo "[$mode]"
        # 获取最新的epoch信息
        latest_epoch=$(grep -i "\[Student\] epoch=" "$log_file" | tail -1)
        if [ ! -z "$latest_epoch" ]; then
            echo "  $latest_epoch"
        else
            echo "  (训练尚未开始或日志为空)"
        fi
        echo
    else
        echo "[$mode] - 未开始 (日志文件不存在)"
        echo
    fi
done

echo "========================================"
echo "查看实时日志: tail -f logs/ablation/s_ce.log"
echo "========================================"
