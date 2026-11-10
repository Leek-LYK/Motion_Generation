"""Display motion-feature and text-embedding cache progress."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=5)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("interval must be positive")
    directory = ROOT / "data/processed/g1_model_inputs_v1"
    while True:
        print("\n模型输入准备进度", flush=True)
        for name, label, complete, total in [
            ("motion_status.json", "动作特征", "completed_samples", "total_samples"),
            ("text_status.json", "文本向量", "completed_texts", "total_texts"),
        ]:
            path = directory / name
            if not path.exists():
                print(label, "尚未启动")
                continue
            r = json.loads(path.read_text())
            updated = datetime.fromisoformat(r["updated_at_utc"]).astimezone(timezone(timedelta(hours=8)))
            done, count = r[complete], r[total]
            warning = ""
            try:
                os.kill(r["pid"], 0)
            except ProcessLookupError:
                if r["phase"] not in ("complete", "pilot_complete", "stopped", "failed"):
                    warning = " [进程已退出，检查日志后续跑]"
            print(f"{label}: {r['phase']} {done:,}/{count:,} ({100*done/count:.2f}%) | {updated:%H:%M:%S} 北京时间{warning}")
            if "failed_shards" in r:
                print(f"  已完成分片 {r['completed_shards']}/{r['total_shards']}，失败分片 {r['failed_shards']}")
            eta = r.get("eta_seconds")
            if eta is not None and done > 100 and r["phase"] in ("running", "encoding"):
                print(f"  估算剩余 {eta/60:.1f} 分钟；初期估算受加载/预热影响")
        print("最终审计:", "READY 已生成" if (directory / "READY").exists() else "尚未完成", flush=True)
        if not args.watch:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
