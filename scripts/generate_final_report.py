#!/usr/bin/env python3
"""
生成完整实验报告
包括：数据效率分析、模型性能对比、可视化图表
"""
import json
import glob
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
from datetime import datetime


def load_all_results(result_dir):
    """加载所有实验结果"""
    all_results = []
    result_files = glob.glob(f"{result_dir}/**/results.json", recursive=True)
    
    for file in result_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                all_results.append(data)
        except Exception as e:
            print(f"⚠️  Error loading {file}: {e}")
    
    return all_results


def create_performance_table(results):
    """创建性能对比表"""
    data = []
    
    for r in results:
        dataset = r['dataset']
        ratio = int(r['data_ratio'] * 100)
        run_id = r['run_id']
        
        # 教师模型
        for teacher_name, metrics in r.get('teachers', {}).items():
            data.append({
                'Dataset': dataset,
                'Ratio': ratio,
                'Run': run_id,
                'Model_Type': 'Teacher',
                'Model_Name': teacher_name,
                'Test_Acc': metrics['test_acc'] * 100,
                'Test_F1': metrics['test_f1'] * 100,
                'Params_M': 0  # TODO: 添加参数量
            })
        
        # Stacking
        if 'stacking' in r:
            data.append({
                'Dataset': dataset,
                'Ratio': ratio,
                'Run': run_id,
                'Model_Type': 'Stacking',
                'Model_Name': 'Ensemble',
                'Test_Acc': r['stacking']['test_acc'] * 100,
                'Test_F1': r['stacking']['test_f1'] * 100,
                'Params_M': 0
            })
        
        # 学生模型
        if 'student' in r:
            data.append({
                'Dataset': dataset,
                'Ratio': ratio,
                'Run': run_id,
                'Model_Type': 'Student',
                'Model_Name': 'RepViT-M0.9',
                'Test_Acc': r['student']['test_acc'] * 100,
                'Test_F1': r['student']['test_f1'] * 100,
                'Params_M': r['student'].get('params_m', 4.72)
            })
    
    df = pd.DataFrame(data)
    return df


def generate_summary_statistics(df):
    """生成汇总统计"""
    summary = []
    
    for dataset in df['Dataset'].unique():
        for ratio in sorted(df['Ratio'].unique(), reverse=True):
            for model_type in ['Teacher', 'Stacking', 'Student']:
                subset = df[(df['Dataset'] == dataset) & 
                           (df['Ratio'] == ratio) & 
                           (df['Model_Type'] == model_type)]
                
                if len(subset) > 0:
                    if model_type == 'Teacher':
                        # 对于教师，按模型名称分组
                        for model_name in subset['Model_Name'].unique():
                            model_data = subset[subset['Model_Name'] == model_name]
                            summary.append({
                                'Dataset': dataset,
                                'Ratio': ratio,
                                'Model_Type': model_type,
                                'Model_Name': model_name,
                                'Avg_Acc': model_data['Test_Acc'].mean(),
                                'Std_Acc': model_data['Test_Acc'].std(),
                                'Avg_F1': model_data['Test_F1'].mean(),
                                'Std_F1': model_data['Test_F1'].std(),
                                'N_Runs': len(model_data)
                            })
                    else:
                        summary.append({
                            'Dataset': dataset,
                            'Ratio': ratio,
                            'Model_Type': model_type,
                            'Model_Name': subset['Model_Name'].iloc[0],
                            'Avg_Acc': subset['Test_Acc'].mean(),
                            'Std_Acc': subset['Test_Acc'].std(),
                            'Avg_F1': subset['Test_F1'].mean(),
                            'Std_F1': subset['Test_F1'].std(),
                            'N_Runs': len(subset)
                        })
    
    return pd.DataFrame(summary)


def generate_markdown_report(df_summary, output_file):
    """生成Markdown格式的报告"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# NetKD 完整实验结果报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # 总体概述
        f.write("## 📊 实验概述\n\n")
        datasets = df_summary['Dataset'].unique()
        f.write(f"- **数据集数量**: {len(datasets)}\n")
        f.write(f"- **数据集**: {', '.join(datasets)}\n")
        f.write(f"- **数据比例**: 100%, 75%, 50%, 25%\n")
        f.write(f"- **每组重复次数**: 3\n")
        f.write(f"- **总实验数**: {len(df_summary)}\n\n")
        
        # 按数据集生成表格
        for dataset in sorted(datasets):
            f.write(f"## 🎯 {dataset}\n\n")
            
            # 学生模型性能
            f.write("### 学生模型 (RepViT-M0.9) 性能\n\n")
            f.write("| 数据比例 | 测试准确率 | F1分数 | 训练样本数 |\n")
            f.write("|---------|-----------|--------|----------|\n")
            
            student_data = df_summary[(df_summary['Dataset'] == dataset) & 
                                     (df_summary['Model_Type'] == 'Student')]
            student_data = student_data.sort_values('Ratio', ascending=False)
            
            for _, row in student_data.iterrows():
                f.write(f"| {row['Ratio']}% | "
                       f"{row['Avg_Acc']:.2f}±{row['Std_Acc']:.2f}% | "
                       f"{row['Avg_F1']:.2f}±{row['Std_F1']:.2f}% | "
                       f"- |\n")
            
            f.write("\n")
            
            # Stacking性能
            f.write("### Stacking集成性能\n\n")
            f.write("| 数据比例 | 测试准确率 | F1分数 |\n")
            f.write("|---------|-----------|--------|\n")
            
            stacking_data = df_summary[(df_summary['Dataset'] == dataset) & 
                                      (df_summary['Model_Type'] == 'Stacking')]
            stacking_data = stacking_data.sort_values('Ratio', ascending=False)
            
            for _, row in stacking_data.iterrows():
                f.write(f"| {row['Ratio']}% | "
                       f"{row['Avg_Acc']:.2f}±{row['Std_Acc']:.2f}% | "
                       f"{row['Avg_F1']:.2f}±{row['Std_F1']:.2f}% |\n")
            
            f.write("\n")
            
            # 教师模型对比
            f.write("### 教师模型对比 (100%数据)\n\n")
            f.write("| 模型 | 测试准确率 | F1分数 |\n")
            f.write("|------|-----------|--------|\n")
            
            teacher_data = df_summary[(df_summary['Dataset'] == dataset) & 
                                     (df_summary['Model_Type'] == 'Teacher') & 
                                     (df_summary['Ratio'] == 100)]
            teacher_data = teacher_data.sort_values('Avg_Acc', ascending=False)
            
            for _, row in teacher_data.iterrows():
                f.write(f"| {row['Model_Name']} | "
                       f"{row['Avg_Acc']:.2f}±{row['Std_Acc']:.2f}% | "
                       f"{row['Avg_F1']:.2f}±{row['Std_F1']:.2f}% |\n")
            
            f.write("\n---\n\n")
        
        # 数据效率分析
        f.write("## 📈 数据效率分析\n\n")
        f.write("各数据集在不同数据比例下的学生模型性能：\n\n")
        
        for dataset in sorted(datasets):
            student_data = df_summary[(df_summary['Dataset'] == dataset) & 
                                     (df_summary['Model_Type'] == 'Student')]
            student_data = student_data.sort_values('Ratio', ascending=False)
            
            f.write(f"### {dataset}\n\n")
            f.write("```\n")
            for _, row in student_data.iterrows():
                f.write(f"{row['Ratio']:3d}% -> {row['Avg_Acc']:5.2f}% "
                       f"(±{row['Std_Acc']:.2f})\n")
            f.write("```\n\n")
        
        # 关键发现
        f.write("## 🔍 关键发现\n\n")
        
        # 找出最佳性能
        best_student = df_summary[df_summary['Model_Type'] == 'Student'].nlargest(1, 'Avg_Acc').iloc[0]
        best_stacking = df_summary[df_summary['Model_Type'] == 'Stacking'].nlargest(1, 'Avg_Acc').iloc[0]
        
        f.write(f"1. **最佳学生模型性能**: {best_student['Dataset']} - {best_student['Ratio']}% 数据\n")
        f.write(f"   - 准确率: {best_student['Avg_Acc']:.2f}±{best_student['Std_Acc']:.2f}%\n")
        f.write(f"   - F1分数: {best_student['Avg_F1']:.2f}±{best_student['Std_F1']:.2f}%\n\n")
        
        f.write(f"2. **最佳Stacking性能**: {best_stacking['Dataset']} - {best_stacking['Ratio']}% 数据\n")
        f.write(f"   - 准确率: {best_stacking['Avg_Acc']:.2f}±{best_stacking['Std_Acc']:.2f}%\n")
        f.write(f"   - F1分数: {best_stacking['Avg_F1']:.2f}±{best_stacking['Std_F1']:.2f}%\n\n")
        
        f.write("3. **数据效率观察**:\n")
        for dataset in sorted(datasets):
            student_data = df_summary[(df_summary['Dataset'] == dataset) & 
                                     (df_summary['Model_Type'] == 'Student')]
            if len(student_data) >= 2:
                full_data = student_data[student_data['Ratio'] == 100]
                half_data = student_data[student_data['Ratio'] == 50]
                
                if len(full_data) > 0 and len(half_data) > 0:
                    drop = full_data['Avg_Acc'].iloc[0] - half_data['Avg_Acc'].iloc[0]
                    f.write(f"   - {dataset}: 50%数据性能下降 {drop:.2f}%\n")
        
        f.write("\n---\n")
        f.write(f"\n*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    print(f"✅ 报告已生成: {output_file}")


def main():
    result_dir = "results/complete_experiment/20251210_220550"
    
    print("📊 加载实验结果...")
    results = load_all_results(result_dir)
    
    if not results:
        print("❌ 没有找到实验结果")
        return
    
    print(f"✅ 加载了 {len(results)} 个实验结果")
    
    print("📈 创建性能表...")
    df = create_performance_table(results)
    
    print("📊 生成汇总统计...")
    df_summary = generate_summary_statistics(df)
    
    # 保存CSV
    csv_file = "results/complete_experiment/summary_statistics.csv"
    df_summary.to_csv(csv_file, index=False)
    print(f"✅ CSV已保存: {csv_file}")
    
    # 生成Markdown报告
    report_file = "results/complete_experiment/FINAL_REPORT.md"
    print("📝 生成Markdown报告...")
    generate_markdown_report(df_summary, report_file)
    
    print("\n🎉 报告生成完成！")


if __name__ == "__main__":
    main()
