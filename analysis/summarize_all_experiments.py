#!/usr/bin/env python3
"""
汇总所有实验结果的脚本

生成:
1. 教师模型汇总表
2. 学生模型汇总表  
3. Pareto 前沿分析
4. 数据效率对比
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import csv


def load_json_results(results_dir: Path, pattern: str = "**/results.json") -> List[Dict]:
    """加载所有 JSON 结果文件"""
    results = []
    for json_file in results_dir.rglob(pattern):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
                data["_source_file"] = str(json_file)
                results.append(data)
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
    return results


def generate_student_table(results: List[Dict]) -> str:
    """生成学生模型汇总表"""
    lines = [
        "# 学生模型实验汇总",
        "",
        f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 性能对比",
        "",
        "| 学生模型 | 教师组合 | KD配置 | 验证精度 | 测试精度 | Macro-F1 | 参数量(M) |",
        "|----------|----------|--------|----------|----------|----------|-----------|",
    ]
    
    for r in sorted(results, key=lambda x: x.get("test_metrics", {}).get("accuracy", 0), reverse=True):
        student = r.get("student_name", "N/A")
        teacher_set = r.get("teacher_set_id", "N/A")
        kd_config = r.get("kd_config_id", "N/A")
        val_acc = r.get("best_val_accuracy", 0)
        test_metrics = r.get("test_metrics", {})
        test_acc = test_metrics.get("accuracy", 0)
        f1_macro = test_metrics.get("f1_macro", 0)
        params = r.get("num_params_m", 0)
        
        lines.append(
            f"| {student} | {teacher_set} | {kd_config} | "
            f"{val_acc:.2f}% | {test_acc:.2f}% | {f1_macro:.4f} | {params:.2f} |"
        )
    
    return "\n".join(lines)


def generate_csv(results: List[Dict], output_path: Path):
    """生成 CSV 文件"""
    if not results:
        return
    
    fieldnames = [
        "student_name", "teacher_set_id", "kd_config_id",
        "val_accuracy", "test_accuracy", "f1_macro", "f1_weighted",
        "params_m", "epochs", "batch_size", "lr"
    ]
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for r in results:
            test_metrics = r.get("test_metrics", {})
            config = r.get("config", {})
            writer.writerow({
                "student_name": r.get("student_name", ""),
                "teacher_set_id": r.get("teacher_set_id", ""),
                "kd_config_id": r.get("kd_config_id", ""),
                "val_accuracy": r.get("best_val_accuracy", 0),
                "test_accuracy": test_metrics.get("accuracy", 0),
                "f1_macro": test_metrics.get("f1_macro", 0),
                "f1_weighted": test_metrics.get("f1_weighted", 0),
                "params_m": r.get("num_params_m", 0),
                "epochs": config.get("epochs", 0),
                "batch_size": config.get("batch_size", 0),
                "lr": config.get("lr", 0),
            })


def main():
    parser = argparse.ArgumentParser(description="Summarize All Experiments")
    parser.add_argument("--results_root", type=str, default="results")
    parser.add_argument("--output_dir", type=str, default="analysis/summary")
    args = parser.parse_args()
    
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning results from: {results_root}")
    
    # 加载学生模型结果
    student_results = load_json_results(results_root / "student_kd")
    if student_results:
        print(f"Found {len(student_results)} student experiment results")
        
        # 生成 Markdown 表格
        md_content = generate_student_table(student_results)
        md_path = output_dir / "students_summary.md"
        with open(md_path, "w") as f:
            f.write(md_content)
        print(f"Saved: {md_path}")
        
        # 生成 CSV
        csv_path = output_dir / "students_summary.csv"
        generate_csv(student_results, csv_path)
        print(f"Saved: {csv_path}")
    else:
        print("No student results found")
    
    # 加载教师搜索结果
    teacher_results = load_json_results(results_root / "teacher_search")
    if teacher_results:
        print(f"Found {len(teacher_results)} teacher experiment results")
    
    print(f"\nSummary files saved to: {output_dir}")


if __name__ == "__main__":
    main()
