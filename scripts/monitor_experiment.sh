#!/bin/bash
# 实验监控脚本
LOG_FILE="/walnut_data/yqm/NetKD/results/teacher_search/experiment.log"
RESULTS_DIR="/walnut_data/yqm/NetKD/results/teacher_search"

echo "========================================"
echo "Teacher Search Experiment Monitor"
echo "========================================"
echo ""

# 检查进程是否在运行
PID=$(pgrep -f "run_teacher_search_experiment.py")
if [ -n "$PID" ]; then
    echo "✓ Experiment is running (PID: $PID)"
else
    echo "✗ Experiment is not running"
fi
echo ""

# 显示GPU使用情况
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "No GPU info available"
echo ""

# 显示最近的日志
echo "Recent Log (last 30 lines):"
echo "----------------------------------------"
tail -30 "$LOG_FILE" 2>/dev/null || echo "No log file found"
echo ""

# 检查中间结果
if [ -f "$RESULTS_DIR/eca_ablation/eca_ablation_results.json" ]; then
    echo "ECA Ablation Results:"
    echo "----------------------------------------"
    python3 -c "
import json
with open('$RESULTS_DIR/eca_ablation/eca_ablation_results.json') as f:
    results = json.load(f)
for name, r in results.items():
    acc = r.get('test_accuracy', r.get('best_val_acc', 'N/A'))
    if isinstance(acc, float):
        print(f'  {name}: {acc:.4f}')
    else:
        print(f'  {name}: {acc}')
"
    echo ""
fi

if [ -f "$RESULTS_DIR/stacking/stacking_results.json" ]; then
    echo "Stacking Results:"
    echo "----------------------------------------"
    python3 -c "
import json
with open('$RESULTS_DIR/stacking/stacking_results.json') as f:
    results = json.load(f)
for name, r in results.items():
    acc = r.get('test_accuracy', r.get('best_val_acc', 'N/A'))
    if isinstance(acc, float):
        print(f'  {name}: {acc:.4f}')
    else:
        print(f'  {name}: {acc}')
"
    echo ""
fi

echo "========================================"
echo "Use: tail -f $LOG_FILE to follow live"
echo "========================================"
