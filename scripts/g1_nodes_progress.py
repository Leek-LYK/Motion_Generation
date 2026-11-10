"""Display the current conversion progress without importing ML packages."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def duration(seconds):
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/g1_30hz_nodes_v1")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("interval must be positive")
    while True:
        path = args.output / "status.json"
        if not path.exists():
            print("等待转换任务创建进度文件……", flush=True)
        else:
            data = json.loads(path.read_text())
            try:
                os.kill(data["pid"], 0)
                alive = True
            except ProcessLookupError:
                alive = False
            except PermissionError:
                alive = True
            updated = datetime.fromisoformat(data["updated_at_utc"])
            local = updated.astimezone(timezone(timedelta(hours=8)))
            age = (datetime.now(timezone.utc) - updated).total_seconds()
            total = data["total"]
            done = data["complete"]
            filled = int(30 * done / total)
            eta = data.get("eta_seconds")
            estimate = duration(eta) if eta is not None and data["processed_this_run"] >= 100 else "统计不足/不适用"
            warning = "；进程已退出，若未完成请检查日志并续跑" if not alive and data["phase"] in ("running", "stopping") else ""
            print(f"\n更新时间 {local:%Y-%m-%d %H:%M:%S} 北京时间 | {data['phase']} | 心跳 {age:.0f}s 前{warning}", flush=True)
            print(f"[{'#' * filled}{'-' * (30 - filled)}] {done:,}/{total:,} ({100 * done / total:.2f}%)")
            print(f"成功 {done:,} | 失败 {data['failed']:,} | 待处理 {data['pending']:,} | CPU workers {data['workers']} | PID {data['pid']}")
            print(f"输出 {data['output_frames']:,} 帧，{data['output_bytes']/1024**3:.2f} GiB | 本次平均 {data['motions_per_second']:.2f} 条/s | 预计剩余 {estimate}", flush=True)
            if data.get("last_error"):
                print("最近失败：", data["last_error"], flush=True)
        if not args.watch:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
