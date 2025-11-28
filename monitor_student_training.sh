#!/bin/bash
# 学生模型训练监控脚本

LOG_FILE="/walnut_data/yqm/NetKD/logs/student_direct_training.log"
PID_FILE="/tmp/train_pid.txt"

echo "========================================"
echo "学生模型训练监控"
echo "========================================"

# 检查进程
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ 训练进程正在运行 (PID: $PID)"
    else
        echo "❌ 训练进程已结束 (PID: $PID)"
    fi
else
    ps aux | grep "train_student_direct.py" | grep -v grep || echo "⚠️  未找到训练进程"
fi
echo ""

# 显示最新进度
echo "最新训练进度:"
echo "----------------------------------------"
tail -20 "$LOG_FILE" 2>/dev/null | grep -E "Monitor.*epoch|最终测试精度|训练完成"
echo "----------------------------------------"
echo ""

# 显示最高精度
echo "历史最高验证精度:"
grep -o "val_acc=[0-9.]*" "$LOG_FILE" 2>/dev/null | cut -d= -f2 | sort -rn | head -1 | awk '{printf "%.2f%%\n", $1*100}'
echo ""

echo "使用命令实时查看:"
echo "  tail -f $LOG_FILE"
