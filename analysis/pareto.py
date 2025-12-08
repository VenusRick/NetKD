"""
Pareto 前沿分析工具

用于计算和可视化模型的 Pareto 前沿，
在多个目标之间寻找最优权衡（如精度 vs 参数量/FLOPs）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import json
from pathlib import Path


@dataclass
class ModelPoint:
    """模型数据点"""
    name: str
    macro_f1: float
    accuracy: float
    params_m: float
    flops_g: Optional[float] = None
    latency_ms: Optional[float] = None
    
    def dominates(self, other: "ModelPoint") -> bool:
        """检查当前点是否支配另一个点
        
        支配条件: 所有目标都不差，且至少一个目标更好
        目标: 最大化 macro_f1, 最小化 params_m
        """
        # 当前点在所有目标上都不差
        f1_not_worse = self.macro_f1 >= other.macro_f1
        params_not_worse = self.params_m <= other.params_m
        
        # 且至少有一个目标更好
        f1_better = self.macro_f1 > other.macro_f1
        params_better = self.params_m < other.params_m
        
        return (f1_not_worse and params_not_worse) and (f1_better or params_better)


def compute_pareto_front(points: List[ModelPoint]) -> List[ModelPoint]:
    """计算 Pareto 前沿
    
    返回非支配点的子集，这些点在精度和参数量之间达到最优权衡
    """
    pareto_front = []
    
    for point in points:
        is_dominated = False
        for other in points:
            if other.dominates(point):
                is_dominated = True
                break
        
        if not is_dominated:
            pareto_front.append(point)
    
    # 按参数量排序
    pareto_front.sort(key=lambda p: p.params_m)
    
    return pareto_front


def load_results_as_points(results_dir: Path) -> List[ModelPoint]:
    """从结果目录加载模型点"""
    points = []
    
    # 遍历所有结果 JSON 文件
    for json_file in results_dir.rglob("results.json"):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            
            point = ModelPoint(
                name=data.get("student_name", json_file.parent.name),
                macro_f1=data.get("test_metrics", {}).get("f1_macro", 0),
                accuracy=data.get("test_metrics", {}).get("accuracy", 0),
                params_m=data.get("num_params_m", 0),
                flops_g=data.get("flops_g"),
                latency_ms=data.get("latency_ms"),
            )
            points.append(point)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load {json_file}: {e}")
    
    return points


def generate_pareto_table(
    points: List[ModelPoint],
    pareto_front: List[ModelPoint],
) -> str:
    """生成 Pareto 分析的 Markdown 表格"""
    lines = [
        "# Pareto 前沿分析",
        "",
        "## 所有模型",
        "",
        "| 模型 | Macro-F1 | 准确率 | 参数量(M) | Pareto最优 |",
        "|------|----------|--------|-----------|------------|",
    ]
    
    pareto_names = {p.name for p in pareto_front}
    
    for point in sorted(points, key=lambda p: -p.macro_f1):
        is_pareto = "✓" if point.name in pareto_names else ""
        lines.append(
            f"| {point.name} | {point.macro_f1:.4f} | {point.accuracy:.2f}% | "
            f"{point.params_m:.2f} | {is_pareto} |"
        )
    
    lines.extend([
        "",
        "## Pareto 前沿",
        "",
        "| 模型 | Macro-F1 | 参数量(M) | 效率比 |",
        "|------|----------|-----------|--------|",
    ])
    
    for point in pareto_front:
        efficiency = point.macro_f1 / point.params_m if point.params_m > 0 else 0
        lines.append(
            f"| {point.name} | {point.macro_f1:.4f} | {point.params_m:.2f} | {efficiency:.4f} |"
        )
    
    return "\n".join(lines)


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pareto Front Analysis")
    parser.add_argument("--results_dir", type=str, default="results/student_kd")
    parser.add_argument("--output", type=str, default="analysis/summary/pareto_analysis.md")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 加载结果
    points = load_results_as_points(results_dir)
    if not points:
        print(f"No results found in {results_dir}")
        return
    
    print(f"Loaded {len(points)} model points")
    
    # 计算 Pareto 前沿
    pareto_front = compute_pareto_front(points)
    print(f"Pareto front contains {len(pareto_front)} models:")
    for p in pareto_front:
        print(f"  - {p.name}: F1={p.macro_f1:.4f}, Params={p.params_m:.2f}M")
    
    # 生成报告
    report = generate_pareto_table(points, pareto_front)
    with open(output_path, "w") as f:
        f.write(report)
    
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
