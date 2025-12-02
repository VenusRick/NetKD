#!/usr/bin/env python3
"""Summarize Teacher Search Experiment Results - Generate Comparison Tables"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.teacher_search.result_schema import ExperimentResult, TeacherResult, StackingResult

def load_all_results(results_dir: Path) -> Dict[str, Any]:
    """Load all JSON result files from directory"""
    results = {"teachers": {}, "stacking": {}}
    for json_file in results_dir.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        name = json_file.stem
        if name.startswith("stacking_"):
            set_name = name.replace("stacking_", "").replace("_result", "")
            results["stacking"][set_name] = StackingResult.from_dict(data)
        elif name.endswith("_result"):
            teacher_name = name.replace("_result", "")
            results["teachers"][teacher_name] = TeacherResult.from_dict(data)
    return results

def generate_teacher_table(teachers: Dict[str, TeacherResult]) -> str:
    """Generate Markdown table for individual teacher results"""
    if not teachers:
        return "No teacher results found.\n"
    lines = ["## Individual Teacher Results\n", "| Teacher | Params(M) | Best Val Acc | Test Acc | F1-macro | F1-weighted | Time(s) |",
             "|---------|-----------|--------------|----------|----------|-------------|---------|"]
    sorted_teachers = sorted(teachers.items(), key=lambda x: x[1].test_accuracy, reverse=True)
    for name, t in sorted_teachers:
        lines.append(f"| {name} | {t.params_millions:.2f} | {t.best_val_acc:.4f} | {t.test_accuracy:.4f} | {t.test_f1_macro:.4f} | {t.test_f1_weighted:.4f} | {t.training_time_seconds:.1f} |")
    return "\n".join(lines) + "\n"

def generate_stacking_table(stacking: Dict[str, StackingResult]) -> str:
    """Generate Markdown table for stacking results"""
    if not stacking:
        return "No stacking results found.\n"
    lines = ["## Stacking Ensemble Results\n", "| Teacher Set | Teachers | Test Acc | F1-macro | Disagreement | Oracle Acc | Diversity |",
             "|-------------|----------|----------|----------|--------------|------------|-----------|"]
    sorted_stacking = sorted(stacking.items(), key=lambda x: x[1].test_accuracy, reverse=True)
    for name, s in sorted_stacking:
        teachers_str = ", ".join(s.teachers[:3]) + ("..." if len(s.teachers) > 3 else "")
        lines.append(f"| {name} | {teachers_str} | {s.test_accuracy:.4f} | {s.test_f1_macro:.4f} | {s.disagreement_rate:.4f} | {s.oracle_accuracy:.4f} | {s.diversity_score:.4f} |")
    return "\n".join(lines) + "\n"

def generate_latex_table(teachers: Dict[str, TeacherResult], stacking: Dict[str, StackingResult]) -> str:
    """Generate LaTeX tables for paper"""
    lines = ["## LaTeX Tables\n", "### Individual Teachers\n", "```latex", "\\begin{table}[h]", "\\centering",
             "\\caption{Individual Teacher Model Performance}", "\\label{tab:teacher_results}",
             "\\begin{tabular}{lcccc}", "\\toprule", "Model & Params(M) & Test Acc & F1-macro & F1-weighted \\\\", "\\midrule"]
    sorted_teachers = sorted(teachers.items(), key=lambda x: x[1].test_accuracy, reverse=True)
    for name, t in sorted_teachers:
        display_name = name.replace("_", "\\_")
        lines.append(f"{display_name} & {t.params_millions:.2f} & {t.test_accuracy:.4f} & {t.test_f1_macro:.4f} & {t.test_f1_weighted:.4f} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", "```\n"])
    lines.extend(["### Stacking Ensembles\n", "```latex", "\\begin{table}[h]", "\\centering",
                  "\\caption{Stacking Ensemble Performance}", "\\label{tab:stacking_results}",
                  "\\begin{tabular}{lccccc}", "\\toprule", "Teacher Set & Test Acc & F1-macro & Disagreement & Oracle Acc & Diversity \\\\", "\\midrule"])
    sorted_stacking = sorted(stacking.items(), key=lambda x: x[1].test_accuracy, reverse=True)
    for name, s in sorted_stacking:
        display_name = name.replace("_", "\\_")
        lines.append(f"{display_name} & {s.test_accuracy:.4f} & {s.test_f1_macro:.4f} & {s.disagreement_rate:.4f} & {s.oracle_accuracy:.4f} & {s.diversity_score:.4f} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", "```\n"])
    return "\n".join(lines)

def generate_summary_stats(teachers: Dict[str, TeacherResult], stacking: Dict[str, StackingResult]) -> str:
    """Generate summary statistics"""
    lines = ["## Summary Statistics\n"]
    if teachers:
        best_teacher = max(teachers.values(), key=lambda x: x.test_accuracy)
        avg_teacher_acc = sum(t.test_accuracy for t in teachers.values()) / len(teachers)
        total_params = sum(t.params_millions for t in teachers.values())
        lines.extend([f"### Teachers ({len(teachers)} total)", f"- Best Teacher: **{best_teacher.name}** ({best_teacher.test_accuracy:.4f})",
                      f"- Average Test Accuracy: {avg_teacher_acc:.4f}", f"- Total Parameters: {total_params:.2f}M\n"])
    if stacking:
        best_stacking = max(stacking.values(), key=lambda x: x.test_accuracy)
        avg_stacking_acc = sum(s.test_accuracy for s in stacking.values()) / len(stacking)
        lines.extend([f"### Stacking Ensembles ({len(stacking)} total)", f"- Best Ensemble: **{best_stacking.teacher_set_name}** ({best_stacking.test_accuracy:.4f})",
                      f"- Average Test Accuracy: {avg_stacking_acc:.4f}", f"- Best Ensemble Teachers: {', '.join(best_stacking.teachers)}\n"])
    if teachers and stacking:
        best_teacher = max(teachers.values(), key=lambda x: x.test_accuracy)
        best_stacking = max(stacking.values(), key=lambda x: x.test_accuracy)
        improvement = best_stacking.test_accuracy - best_teacher.test_accuracy
        lines.extend(["### Ensemble Improvement", f"- Improvement over best single teacher: **{improvement*100:.2f}%**\n"])
    return "\n".join(lines)

def generate_recommendations(teachers: Dict[str, TeacherResult], stacking: Dict[str, StackingResult]) -> str:
    """Generate recommendations based on results"""
    lines = ["## Recommendations\n"]
    if stacking:
        best_stacking = max(stacking.values(), key=lambda x: x.test_accuracy)
        high_diversity = [s for s in stacking.values() if s.diversity_score > 0.3]
        lines.append(f"1. **Best Overall Ensemble**: {best_stacking.teacher_set_name}")
        lines.append(f"   - Teachers: {', '.join(best_stacking.teachers)}")
        lines.append(f"   - Test Accuracy: {best_stacking.test_accuracy:.4f}\n")
        if high_diversity:
            lines.append("2. **High Diversity Ensembles** (diversity > 0.3):")
            for s in sorted(high_diversity, key=lambda x: x.diversity_score, reverse=True):
                lines.append(f"   - {s.teacher_set_name}: diversity={s.diversity_score:.4f}, acc={s.test_accuracy:.4f}")
    if teachers:
        efficient = [(n, t) for n, t in teachers.items() if t.params_millions < 20 and t.test_accuracy > 0.9]
        if efficient:
            lines.append("\n3. **Efficient Teachers** (< 20M params, > 90% acc):")
            for n, t in sorted(efficient, key=lambda x: x[1].test_accuracy, reverse=True):
                lines.append(f"   - {n}: {t.params_millions:.2f}M params, {t.test_accuracy:.4f} acc")
    return "\n".join(lines) + "\n"

def main():
    parser = argparse.ArgumentParser(description="Summarize teacher search results")
    parser.add_argument("--results-dir", type=str, default=str(PROJECT_ROOT / "results/teacher_search"), help="Directory containing result JSON files")
    parser.add_argument("--output", type=str, default=None, help="Output Markdown file (default: stdout)")
    parser.add_argument("--format", choices=["markdown", "latex", "both"], default="both", help="Output format")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return
    
    results = load_all_results(results_dir)
    teachers, stacking = results["teachers"], results["stacking"]
    
    output_lines = ["# Teacher Search Experiment Summary\n", f"Generated from: `{results_dir}`\n"]
    output_lines.append(generate_summary_stats(teachers, stacking))
    output_lines.append(generate_teacher_table(teachers))
    output_lines.append(generate_stacking_table(stacking))
    if args.format in ["latex", "both"]:
        output_lines.append(generate_latex_table(teachers, stacking))
    output_lines.append(generate_recommendations(teachers, stacking))
    
    output_text = "\n".join(output_lines)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output_text)
        print(f"Summary saved to {output_path}")
    else:
        print(output_text)

if __name__ == "__main__":
    main()
