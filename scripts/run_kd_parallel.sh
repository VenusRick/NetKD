#!/bin/bash
# 并行运行 CE+KL 实验 - 充分利用3个GPU

OUTPUT_DIR="results/kd_simple_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR logs

echo "=========================================="
echo "CE+KL 并行实验 - 3个GPU同时运行"
echo "输出目录: $OUTPUT_DIR"
echo "=========================================="

# GPU 0: repvit_m0_9 全部实验
(
for FRAC in 1.0 0.5 0.2; do
    echo "[GPU0] repvit_m0_9 | frac=$FRAC"
    python scripts/run_kd_simple.py --student repvit_m0_9 --use_kd --train_fraction $FRAC --gpu 0 --output_dir $OUTPUT_DIR 2>&1 | tee -a "$OUTPUT_DIR/log_gpu0.txt"
done
echo "[GPU0] ✅ 完成!"
) &
PID0=$!

# GPU 1: mobilenetv3_small 全部实验
(
for FRAC in 1.0 0.5 0.2; do
    echo "[GPU1] mobilenetv3_small | frac=$FRAC"
    python scripts/run_kd_simple.py --student mobilenetv3_small --use_kd --train_fraction $FRAC --gpu 1 --output_dir $OUTPUT_DIR 2>&1 | tee -a "$OUTPUT_DIR/log_gpu1.txt"
done
echo "[GPU1] ✅ 完成!"
) &
PID1=$!

echo "GPU 0 PID: $PID0"
echo "GPU 1 PID: $PID1"
echo "等待所有实验完成..."

wait $PID0 $PID1

echo ""
echo "=========================================="
echo "🎉 所有 CE+KL 实验完成!"
echo "结果保存在: $OUTPUT_DIR"
echo "=========================================="

# 生成汇总
python analysis/analyze_step3_step4.py
