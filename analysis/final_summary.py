#!/usr/bin/env python3
"""
最终实验结果汇总和分析
"""
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent

def load_all_results():
    """加载所有实验结果"""
    results = []
    for json_file in PROJECT_ROOT.rglob("**/metrics.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
                data['source_file'] = str(json_file)
                results.append(data)
        except:
            pass
    return results

def generate_final_report():
    """生成最终报告"""
    results = load_all_results()
    
    print("=" * 80)
    print("NetKD 项目最终实验结果汇总")
    print("=" * 80)
    print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总实验数: {len(results)}")
    print()
    
    # 按学生模型分组
    by_student = {}
    for r in results:
        student = r.get('student_name', 'unknown')
        if student not in by_student:
            by_student[student] = []
        by_student[student].append(r)
    
    print("=" * 80)
    print("一、各学生模型最佳结果")
    print("=" * 80)
    print()
    
    best_results = []
    for student, student_results in sorted(by_student.items()):
        # 找到该学生模型的最佳结果
        best = max(student_results, key=lambda x: x.get('macro_f1', 0))
        best_results.append(best)
        
        print(f"【{student}】")
        print(f"  参数量: {best.get('params_m', 0):.2f}M")
        print(f"  最佳 Test Acc: {best.get('test_acc', 0):.4f}")
        print(f"  最佳 Macro F1: {best.get('macro_f1', 0):.4f}")
        print(f"  配置: {best.get('kd_config_id', '-')}")
        print(f"  数据比例: {best.get('train_fraction', 1.0):.0%}")
        print()
    
    # 数据效率分析
    print("=" * 80)
    print("二、数据效率分析")
    print("=" * 80)
    print()
    print(f"{'模型':<25} {'100%数据':<12} {'50%数据':<12} {'20%数据':<12} {'下降幅度(50%)':<15}")
    print("-" * 80)
    
    for student in sorted(by_student.keys()):
        student_results = by_student[student]
        f1_by_frac = {}
        for r in student_results:
            frac = r.get('train_fraction', 1.0)
            f1 = r.get('macro_f1', 0)
            if frac not in f1_by_frac or f1 > f1_by_frac[frac]:
                f1_by_frac[frac] = f1
        
        f1_100 = f1_by_frac.get(1.0, 0)
        f1_50 = f1_by_frac.get(0.5, 0)
        f1_20 = f1_by_frac.get(0.2, 0)
        drop = f1_100 - f1_50 if f1_100 and f1_50 else 0
        
        print(f"{student:<25} {f1_100:<12.4f} {f1_50:<12.4f} {f1_20:<12.4f} {drop:<15.4f}")
    
    print()
    print("=" * 80)
    print("三、关键发现")
    print("=" * 80)
    print()
    
    # 最佳模型
    overall_best = max(results, key=lambda x: x.get('macro_f1', 0))
    print(f"1. 最佳整体性能:")
    print(f"   模型: {overall_best.get('student_name', 'unknown')}")
    print(f"   F1 Score: {overall_best.get('macro_f1', 0):.4f}")
    print(f"   参数量: {overall_best.get('params_m', 0):.2f}M")
    print()
    
    # 最轻量但性能好的
    light_models = [r for r in results if r.get('params_m', 0) < 2 and r.get('train_fraction', 1.0) == 1.0]
    if light_models:
        best_light = max(light_models, key=lambda x: x.get('macro_f1', 0))
        print(f"2. 最佳轻量模型 (<2M参数):")
        print(f"   模型: {best_light.get('student_name', 'unknown')}")
        print(f"   F1 Score: {best_light.get('macro_f1', 0):.4f}")
        print(f"   参数量: {best_light.get('params_m', 0):.2f}M")
        print()
    
    # 压缩效率
    teacher_f1 = 0.98  # 教师集成估计性能
    teacher_params = 55.6  # 教师集成参数量
    
    print(f"3. 知识蒸馏压缩效率:")
    for r in best_results:
        if r.get('train_fraction', 1.0) == 1.0:
            student = r.get('student_name', 'unknown')
            params = r.get('params_m', 0)
            f1 = r.get('macro_f1', 0)
            compression = teacher_params / params if params > 0 else 0
            f1_retention = f1 / teacher_f1 * 100 if teacher_f1 > 0 else 0
            print(f"   {student}: {compression:.1f}x压缩, 保留{f1_retention:.1f}%性能")
    
    return results

def save_final_csv(results):
    """保存最终CSV"""
    import csv
    
    output_path = PROJECT_ROOT / "analysis" / "FINAL_RESULTS.csv"
    
    fields = ['student_name', 'kd_config_id', 'train_fraction', 'params_m', 
              'test_acc', 'macro_f1', 'precision', 'recall', 'best_epoch']
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for r in sorted(results, key=lambda x: (x.get('student_name', ''), x.get('train_fraction', 0))):
            writer.writerow(r)
    
    print(f"\n✅ CSV已保存: {output_path}")

if __name__ == "__main__":
    results = generate_final_report()
    save_final_csv(results)
