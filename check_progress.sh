#!/bin/bash
echo "========================================"
echo "实验进度快照 ($(date '+%Y-%m-%d %H:%M:%S'))"
echo "========================================"
echo ""

# 检查进程状态
if pgrep -f "run_attention_experiments.sh" > /dev/null; then
    echo "✓ 实验正在运行中"
else
    echo "✗ 实验已完成或未运行"
fi

echo ""
echo "当前训练状态:"
echo "----------------------------------------"
tail -5 logs/attention_experiments_full.log 2>/dev/null | grep -E "(Epoch|Val Acc)"

echo ""
echo "已完成实验列表:"
echo "----------------------------------------"
for att in agent cbam eca simam none; do
    RESULT_FILE="checkpoints/attention_ablation/attention_$att/results.json"
    if [ -f "$RESULT_FILE" ]; then
        ACC=$(python3 -c "import json; d=json.load(open('$RESULT_FILE')); print(f\"{d['best_acc']*100:.2f}% (Epoch {d['best_epoch']}, {d['training_time']/60:.1f}min)\")" 2>/dev/null)
        PARAMS=$(python3 -c "import json; print(f\"{json.load(open('$RESULT_FILE'))['total_params']:,}\")" 2>/dev/null)
        printf "  ✓ %-10s %s, 参数: %s\n" "$att" "$ACC" "$PARAMS"
    else
        printf "  ⏳ %-10s 待运行或运行中...\n" "$att"
    fi
done

echo ""
echo "========================================"
