#!/bin/bash

# 注意力机制消融实验批量运行脚本

set -e

echo "========================================"
echo "注意力机制消融实验"
echo "时间: $(date)"
echo "========================================"
echo

# 注意力机制列表
ATTENTIONS=("agent" "cbam" "eca" "simam" "none")

# 记录开始时间
START_TIME=$(date +%s)

# 依次运行每个实验
for i in "${!ATTENTIONS[@]}"; do
    ATT="${ATTENTIONS[$i]}"
    echo ""
    echo "========================================" 
    echo "[$((i+1))/${#ATTENTIONS[@]}] 运行 $ATT 实验..."
    echo "========================================"
    
    EXP_START=$(date +%s)
    python run_attention_ablation.py "$ATT"
    EXP_END=$(date +%s)
    EXP_DURATION=$((EXP_END - EXP_START))
    
    echo "✓ $ATT 完成 (耗时: ${EXP_DURATION}s = $((EXP_DURATION/60))min)"
done

# 总结
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo ""
echo "========================================"
echo "所有实验完成!"
echo "总耗时: ${TOTAL_DURATION}s ($((TOTAL_DURATION / 60))分钟)"
echo "========================================"
echo ""
echo "运行以下命令生成可视化结果:"
echo "python scripts/plot_attention_results.py"
