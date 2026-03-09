#!/usr/bin/env python3
"""
并行实验启动器 - 充分利用3个GPU
GPU 0: SimCLR预训练实验 (efficientnetv2_rw_s)
GPU 1: SimCLR预训练实验 (convnextv2_tiny)
GPU 2: SimCLR预训练实验 (mobilenetv3_large_100)
"""
import subprocess
import os
from datetime import datetime

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir = f'logs/parallel_{timestamp}'
os.makedirs(log_dir, exist_ok=True)

experiments = [
    {'gpu': 0, 'backbone': 'efficientnetv2_rw_s', 'log': f'{log_dir}/gpu0_efficientnet.log'},
    {'gpu': 1, 'backbone': 'convnextv2_tiny', 'log': f'{log_dir}/gpu1_convnext.log'},
    {'gpu': 2, 'backbone': 'mobilenetv3_large_100', 'log': f'{log_dir}/gpu2_mobilenet.log'},
]

processes = []
for exp in experiments:
    cmd = f"python scripts/run_simclr_experiment.py --gpu {exp['gpu']} --backbone {exp['backbone']} > {exp['log']} 2>&1"
    print(f"Starting GPU {exp['gpu']}: {exp['backbone']}")
    p = subprocess.Popen(cmd, shell=True)
    processes.append((p, exp))

print(f"\n✅ 已启动3个GPU实验，日志目录: {log_dir}")
print("使用 'tail -f logs/parallel_*/gpu*.log' 监控进度")
