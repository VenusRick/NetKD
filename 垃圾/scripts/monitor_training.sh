#!/bin/bash
while true; do
    clear
    echo "================================================================================"
    echo "                         🚀 训练监控面板"
    echo "================================================================================"
    echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # 进程状态
    if ps aux | grep -q "[t]rain_complete_pipeline.py"; then
        echo "✅ 训练进程: 运行中"
        PID=$(ps aux | grep "[t]rain_complete_pipeline.py" | awk '{print $2}')
        echo "   PID: $PID"
        CPU=$(ps aux | grep "[t]rain_complete_pipeline.py" | awk '{print $3}')
        MEM=$(ps aux | grep "[t]rain_complete_pipeline.py" | awk '{print $4}')
        echo "   资源: CPU ${CPU}% | MEM ${MEM}%"
    else
        echo "❌ 训练进程: 已停止"
    fi
    
    echo ""
    echo "================================================================================"
    echo "                         📊 最新训练日志 (最后30行)"
    echo "================================================================================"
    tail -30 logs/complete_pipeline.log
    
    echo ""
    echo "================================================================================"
    echo "按 Ctrl+C 退出监控"
    echo "================================================================================"
    sleep 5
done
