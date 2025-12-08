#!/usr/bin/env python3
"""
并行实验执行脚本 - 充分利用3块RTX 4090

实验计划:
1. GPU 0: 学生模型实验 (MobileNetV2, MobileNetV3-Small)
2. GPU 1: 学生模型实验 (GhostNetV3, RepViT)  
3. GPU 2: 教师模型训练 (新增GhostNetV3, RepViT教师)
"""

import subprocess
import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)

# 实验配置
EXPERIMENTS = {
    # GPU 0: MobileNetV2 和 MobileNetV3-Small 学生蒸馏
    "gpu0": [
        {
            "name": "mobilenetv2_baseline_traffic_kd",
            "cmd": [
                "python", "experiments/student_kd/train_student_kd.py",
                "--student_name", "mobilenetv2",
                "--teacher_set_id", "baseline",
                "--kd_config_id", "traffic_kd_full",
                "--epochs", "100",
                "--batch_size", "128",
                "--device", "cuda:0",
            ],
        },
        {
            "name": "mobilenetv3_small_baseline_traffic_kd",
            "cmd": [
                "python", "experiments/student_kd/train_student_kd.py",
                "--student_name", "mobilenetv3_small",
                "--teacher_set_id", "baseline",
                "--kd_config_id", "traffic_kd_full",
                "--epochs", "100",
                "--batch_size", "128",
                "--device", "cuda:0",
            ],
        },
    ],
    # GPU 1: GhostNet 和 RepViT 学生蒸馏
    "gpu1": [
        {
            "name": "ghostnet_baseline_traffic_kd",
            "cmd": [
                "python", "experiments/student_kd/train_student_kd.py",
                "--student_name", "ghostnet_v1_1_0x",
                "--teacher_set_id", "baseline",
                "--kd_config_id", "traffic_kd_full",
                "--epochs", "100",
                "--batch_size", "128",
                "--device", "cuda:1",
            ],
        },
    ],
    # GPU 2: 消融实验
    "gpu2": [
        {
            "name": "mobilenetv2_ce_only",
            "cmd": [
                "python", "experiments/student_kd/train_student_kd.py",
                "--student_name", "mobilenetv2",
                "--teacher_set_id", "baseline",
                "--kd_config_id", "ce_only",
                "--epochs", "100",
                "--batch_size", "128",
                "--device", "cuda:2",
            ],
        },
        {
            "name": "mobilenetv2_ce_kl",
            "cmd": [
                "python", "experiments/student_kd/train_student_kd.py",
                "--student_name", "mobilenetv2",
                "--teacher_set_id", "baseline",
                "--kd_config_id", "ce_kl",
                "--epochs", "100",
                "--batch_size", "128",
                "--device", "cuda:2",
            ],
        },
    ],
}


def run_experiment(exp_config, gpu_id):
    """运行单个实验"""
    name = exp_config["name"]
    cmd = exp_config["cmd"]
    
    log_dir = PROJECT_ROOT / "logs" / "parallel_experiments"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    print(f"[GPU {gpu_id}] Starting: {name}")
    start_time = time.time()
    
    with open(log_file, "w") as f:
        process = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=PROJECT_ROOT,
        )
    
    elapsed = time.time() - start_time
    status = "SUCCESS" if process.returncode == 0 else "FAILED"
    print(f"[GPU {gpu_id}] {status}: {name} ({elapsed/60:.1f} min)")
    
    return {
        "name": name,
        "gpu": gpu_id,
        "status": status,
        "elapsed_min": elapsed / 60,
        "log_file": str(log_file),
    }


def run_gpu_queue(gpu_id, experiments):
    """顺序运行一个GPU上的所有实验"""
    results = []
    for exp in experiments:
        result = run_experiment(exp, gpu_id)
        results.append(result)
    return results


def main():
    print("=" * 60)
    print("NetKD 并行实验执行器")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GPU数量: 3 x RTX 4090")
    print("=" * 60)
    
    all_results = []
    
    # 使用进程池并行执行各GPU的实验队列
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = {}
        for gpu_id, experiments in EXPERIMENTS.items():
            gpu_num = int(gpu_id.replace("gpu", ""))
            future = executor.submit(run_gpu_queue, gpu_num, experiments)
            futures[future] = gpu_id
        
        for future in as_completed(futures):
            gpu_id = futures[future]
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as e:
                print(f"[{gpu_id}] Error: {e}")
    
    # 保存汇总结果
    summary_file = PROJECT_ROOT / "results" / "parallel_experiment_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "experiments": all_results,
        }, f, indent=2)
    
    print("\n" + "=" * 60)
    print("实验完成汇总:")
    for r in all_results:
        print(f"  [{r['gpu']}] {r['name']}: {r['status']} ({r['elapsed_min']:.1f} min)")
    print(f"\n结果保存至: {summary_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
