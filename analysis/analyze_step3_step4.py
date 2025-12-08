#!/usr/bin/env python3
"""分析 STEP 3 & 4 实验结果"""
import json
from pathlib import Path
import pandas as pd

def load_all_results(results_dir):
    results = []
    for json_file in Path(results_dir).rglob("metrics.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
                results.append(data)
        except:
            pass
    return results

def generate_analysis(results_dir="results/kd_ablation_20251208_191348"):
    results = load_all_results(results_dir)
    
    if not results:
        print("未找到实验结果")
        return
    
    df = pd.DataFrame(results)
    
    # STEP 3 分析: CE-only vs CE+KL (100% data)
    print("=" * 80)
    print("STEP 3: KD 配置消融分析 (100% 训练数据)")
    print("=" * 80)
    
    full_data = df[df['train_fraction'] == 1.0].copy()
    if not full_data.empty:
        full_data_sorted = full_data.sort_values(['student_name', 'kd_config_id'])
        print("\n完整训练数据结果:")
        print(full_data_sorted[['student_name', 'kd_config_id', 'test_acc', 'macro_f1', 'params_m']].to_string(index=False))
    
    # STEP 4 分析: 数据效率
    print("\n" + "=" * 80)
    print("STEP 4: 数据效率分析 (不同训练数据比例)")
    print("=" * 80)
    
    for student in df['student_name'].unique():
        student_data = df[df['student_name'] == student].copy()
        student_data_sorted = student_data.sort_values(['kd_config_id', 'train_fraction'])
        
        print(f"\n{student}:")
        print(student_data_sorted[['kd_config_id', 'train_fraction', 'test_acc', 'macro_f1']].to_string(index=False))
    
    # 关键洞察
    print("\n" + "=" * 80)
    print("关键发现")
    print("=" * 80)
    
    # 最佳模型
    best_model = df.loc[df['macro_f1'].idxmax()]
    print(f"\n1. 最佳模型:")
    print(f"   模型: {best_model['student_name']}")
    print(f"   配置: {best_model['kd_config_id']}")
    print(f"   数据比例: {best_model['train_fraction']:.1%}")
    print(f"   Test Acc: {best_model['test_acc']:.4f}")
    print(f"   Macro F1: {best_model['macro_f1']:.4f}")
    print(f"   参数量: {best_model['params_m']:.2f}M")
    
    # 数据效率洞察
    print("\n2. 数据效率洞察:")
    for student in df['student_name'].unique():
        for kd_config in df['kd_config_id'].unique():
            subset = df[(df['student_name'] == student) & (df['kd_config_id'] == kd_config)]
            if len(subset) >= 2:
                full = subset[subset['train_fraction'] == 1.0]
                half = subset[subset['train_fraction'] == 0.5]
                if not full.empty and not half.empty:
                    f1_drop = full['macro_f1'].values[0] - half['macro_f1'].values[0]
                    print(f"   {student} ({kd_config}): 50%数据 → F1下降 {f1_drop:.4f}")
    
    # 保存汇总
    output_dir = Path("analysis/summary")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # CSV汇总
    df_sorted = df.sort_values(['student_name', 'kd_config_id', 'train_fraction'])
    df_sorted[['student_name', 'kd_config_id', 'train_fraction', 'test_acc', 'macro_f1', 'params_m']].to_csv(
        output_dir / "step3_step4_results.csv", index=False)
    
    print(f"\n✅ 汇总已保存到: {output_dir}/step3_step4_results.csv")

if __name__ == "__main__":
    generate_analysis()
