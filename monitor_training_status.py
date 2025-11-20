"""Console monitor for reading live training metrics from JSONL logs."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def load_records(path: Path) -> Dict[str, List[dict]]:
    stages: Dict[str, List[dict]] = defaultdict(list)
    if not path.exists():
        return stages
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            stages[record.get("stage", "unknown")].append(record)
    return stages


def render(records: Dict[str, List[dict]], history: int) -> str:
    lines: List[str] = []
    if not records:
        return "等待训练日志写入..."

    for stage, entries in sorted(records.items()):
        last = entries[-1]
        lines.append(
            f"阶段 {stage}: epoch {last.get('epoch')} | train_loss={last.get('train_loss', 'NA')} "
            f"| val_loss={last.get('val_loss', 'NA')} | val_acc={last.get('val_acc', 'NA')}"
        )
        if history > 1:
            acc_values = [e.get("val_acc") for e in entries[-history:] if e.get("val_acc") is not None]
            if acc_values:
                trend = " -> ".join(f"{acc:.4f}" for acc in acc_values)
                lines.append(f"    最近准确率: {trend}")
    return "\n".join(lines)


def clear_terminal() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def main():
    parser = argparse.ArgumentParser(description="Live monitor for NetKD training status logs")
    parser.add_argument("--status_file", type=str, default="checkpoints/training_status.jsonl", help="Path to JSONL status file")
    parser.add_argument("--refresh", type=float, default=2.0, help="Refresh interval in seconds")
    parser.add_argument("--history", type=int, default=5, help="Number of trailing accuracy records to display")
    args = parser.parse_args()

    status_path = Path(args.status_file)
    print(f"Monitoring {status_path}. 按 Ctrl+C 退出。")

    try:
        while True:
            records = load_records(status_path)
            clear_terminal()
            print(time.strftime("%Y-%m-%d %H:%M:%S"))
            print("=" * 60)
            print(render(records, args.history))
            print("=" * 60)
            time.sleep(max(0.5, args.refresh))
    except KeyboardInterrupt:
        print("\n停止监控。")


if __name__ == "__main__":
    main()
