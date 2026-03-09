#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import subprocess
import time
from pathlib import Path

try:
    import psutil  # type: ignore
except ImportError:
    psutil = None


def query_gpus() -> list[dict[str, str]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,"         "memory.used,memory.total,pstate,power.draw",
        "--format=csv,noheader",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    rows: list[dict[str, str]] = []
    reader = csv.reader(line.strip() for line in res.stdout.strip().splitlines())
    for row in reader:
        if not row:
            continue
        ts, idx, name, util_gpu, util_mem, mem_used, mem_total, pstate, power = [c.strip() for c in row]
        rows.append(
            {
                "timestamp": ts,
                "index": idx,
                "name": name,
                "util_gpu": util_gpu,
                "util_mem": util_mem,
                "mem_used": mem_used,
                "mem_total": mem_total,
                "pstate": pstate,
                "power": power,
            }
        )
    return rows


def query_cpu() -> dict[str, str]:
    if psutil is None:
        return {"cpu_pct": "n/a", "mem_pct": "n/a"}
    return {
        "cpu_pct": f"{psutil.cpu_percent(interval=None):.1f}%",
        "mem_pct": f"{psutil.virtual_memory().percent:.1f}%",
    }


def log_sample(path: Path) -> None:
    rows = query_gpus()
    cpu_stats = query_cpu()
    now = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(
                f"{now} | GPU{r['index']} {r['name']} | util={r['util_gpu']} mem={r['mem_used']}/{r['mem_total']} "
                f"pstate={r['pstate']} power={r['power']} | CPU {cpu_stats['cpu_pct']} MEM {cpu_stats['mem_pct']}\n"
            )


def tail(path: Path, lines: int) -> None:
    try:
        subprocess.run(["tail", f"-{lines}", str(path)], check=True)
    except subprocess.CalledProcessError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous GPU/CPU sampler")
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--log", type=Path, default=Path("logs/gpu_monitor.log"))
    parser.add_argument("--tail-lines", type=int, default=5)
    parser.add_argument("--no-tail", action="store_true")
    args = parser.parse_args()

    print(f"Logging GPU stats to {args.log} every {args.interval}s")
    try:
        while True:
            log_sample(args.log)
            if not args.no_tail:
                tail(args.log, args.tail_lines)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("
Monitoring stopped.")


if __name__ == "__main__":
    main()
