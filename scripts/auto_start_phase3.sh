#!/bin/bash
# 等待 Phase 2 完成后自动启动 Phase 3

RESULTS_FILE="/walnut_data/yqm/NetKD/results/teacher_search_bs128/eca_ablation/eca_ablation_results.json"
OUTPUT_DIR="/walnut_data/yqm/NetKD/results/teacher_search_bs128"

echo "$(date): 开始监控 Phase 2 完成状态..."

while true; do
    # 检查是否有10个模型结果
    COUNT=$(python3 -c "import json; f=open('$RESULTS_FILE'); print(len(json.load(f)))" 2>/dev/null || echo "0")
    
    if [ "$COUNT" -eq "10" ]; then
        echo "$(date): Phase 2 完成! 共 $COUNT 个模型"
        echo "$(date): 启动 Phase 3 Stacking..."
        cd /walnut_data/yqm/NetKD
        python scripts/run_teacher_search_experiment.py --dataset ISCXVPN2016 --phase 3 --output-dir $OUTPUT_DIR >> ${OUTPUT_DIR}/experiment.log 2>&1
        echo "$(date): Phase 3 完成!"
        
        # 运行 Phase 4 分析
        echo "$(date): 启动 Phase 4 分析..."
        python scripts/run_teacher_search_experiment.py --dataset ISCXVPN2016 --phase 4 --output-dir $OUTPUT_DIR >> ${OUTPUT_DIR}/experiment.log 2>&1
        echo "$(date): Phase 4 完成!"
        break
    else
        echo "$(date): Phase 2 进行中... ($COUNT/10 模型完成)"
        sleep 30
    fi
done

echo "$(date): 全部实验完成!"
