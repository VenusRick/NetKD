#!/bin/bash
# 自动监控实验进度并更新报告

RESULT_DIR="results/complete_experiment/20251210_220550"
SLEEP_INTERVAL=300  # 5分钟检查一次

echo "🚀 启动自动监控..."
echo "检查间隔: ${SLEEP_INTERVAL}秒"
echo "结果目录: ${RESULT_DIR}"
echo ""

last_count=0

while true; do
    current_count=$(find ${RESULT_DIR} -name "results.json" 2>/dev/null | wc -l)
    
    echo "=================================================="
    echo "$(date '+%Y-%m-%d %H:%M:%S') - 检查进度"
    echo "=================================================="
    
    if [ $current_count -gt $last_count ]; then
        echo "✨ 新实验完成！ ($last_count -> $current_count)"
        
        # 更新监控报告
        python scripts/monitor_and_summarize.py
        
        # 重新生成完整报告
        echo ""
        echo "📝 重新生成报告..."
        python scripts/generate_final_report.py
        
        last_count=$current_count
    else
        echo "⏳ 实验进行中... (${current_count}/72)"
    fi
    
    # 检查进程是否还在运行
    if ! ps aux | grep "run_complete_experiment.py" | grep -v grep > /dev/null; then
        echo ""
        echo "⚠️  主实验进程已停止"
        echo "最终完成实验数: ${current_count}/72"
        break
    fi
    
    echo "下次检查时间: $(date -d "+${SLEEP_INTERVAL} seconds" '+%H:%M:%S')"
    echo ""
    
    sleep ${SLEEP_INTERVAL}
done

echo ""
echo "🎉 监控结束"
echo "请查看最终报告: results/complete_experiment/FINAL_REPORT.md"
