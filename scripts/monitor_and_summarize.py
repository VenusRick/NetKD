#!/usr/bin/env python3
"""
监控完整实验进度并生成实时结果汇总
"""
import json
import os
import glob
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np

def load_results(result_dir):
    """加载所有实验结果"""
    all_results = []
    
    result_files = glob.glob(f"{result_dir}/**/results.json", recursive=True)
    
    for file in result_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                all_results.append(data)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    return all_results

def summarize_by_dataset(results):
    """按数据集汇总结果"""
    summary = defaultdict(lambda: defaultdict(list))
    
    for r in results:
        dataset = r['dataset']
        ratio = int(r['data_ratio'] * 100)
        
        # 学生模型结果
        if 'student' in r and 'test_acc' in r['student']:
            summary[dataset][ratio].append({
                'student_acc': r['student']['test_acc'],
                'student_f1': r['student']['test_f1'],
                'stacking_acc': r.get('stacking', {}).get('test_acc', 0),
                'train_samples': r.get('train_samples_used', 0),
            })
    
    return summary

def print_summary_table(summary):
    """打印汇总表格"""
    print("\n" + "="*80)
    print("实验进度汇总")
    print("="*80)
    
    datasets = sorted(summary.keys())
    ratios = [100, 75, 50, 25]
    
    for dataset in datasets:
        print(f"\n📊 {dataset}")
        print("-" * 80)
        print(f"{'比例':<10} {'完成':<8} {'学生准确率':<20} {'Stacking准确率':<20} {'训练样本':<15}")
        print("-" * 80)
        
        for ratio in ratios:
            if ratio in summary[dataset]:
                results = summary[dataset][ratio]
                n_runs = len(results)
                
                student_accs = [r['student_acc'] for r in results]
                stacking_accs = [r['stacking_acc'] for r in results]
                train_samples = results[0]['train_samples'] if results else 0
                
                student_mean = np.mean(student_accs) * 100
                student_std = np.std(student_accs) * 100
                stacking_mean = np.mean(stacking_accs) * 100
                stacking_std = np.std(stacking_accs) * 100
                
                print(f"{ratio}%{'':<7} {n_runs}/3{'':<4} "
                      f"{student_mean:.2f}±{student_std:.2f}%{'':<8} "
                      f"{stacking_mean:.2f}±{stacking_std:.2f}%{'':<8} "
                      f"{train_samples}")
            else:
                print(f"{ratio}%{'':<7} {'0/3':<8} {'N/A':<20} {'N/A':<20} {'N/A':<15}")
    
    print("\n" + "="*80)

def count_total_progress(summary):
    """统计总体进度"""
    total_expected = 0
    total_completed = 0
    
    for dataset in summary:
        for ratio in summary[dataset]:
            total_expected += 3  # 每个比例3次运行
            total_completed += len(summary[dataset][ratio])
    
    # 6个数据集 × 4个比例 × 3次运行 = 72
    total_expected = 6 * 4 * 3
    
    return total_completed, total_expected

def main():
    result_dir = "results/complete_experiment/20251210_220550"
    
    if not os.path.exists(result_dir):
        print(f"❌ 结果目录不存在: {result_dir}")
        return
    
    results = load_results(result_dir)
    
    if not results:
        print("⚠️  还没有完成的实验结果")
        return
    
    summary = summarize_by_dataset(results)
    print_summary_table(summary)
    
    completed, total = count_total_progress(summary)
    progress_pct = completed / total * 100
    
    print(f"\n🎯 总体进度: {completed}/{total} ({progress_pct:.1f}%)")
    print(f"⏱️  预计剩余: {total - completed} 个实验")
    
    if completed == total:
        print("\n🎉 所有实验已完成！")
    else:
        print(f"\n⏳ 实验进行中...")

if __name__ == "__main__":
    main()
