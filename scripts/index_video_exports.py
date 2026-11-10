#!/usr/bin/env python3
"""Rebuild the local export catalogue from immutable per-export manifests."""
from pathlib import Path
import json
import os
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]


def build_index(root):
    rows = []
    for path in root.rglob("manifest.json"):
        record = json.loads(path.read_text())
        if record.get("schema") != "kimodo-video-export-v1" or record["status"] != "complete":
            continue
        rows.append((record, path.parent.relative_to(root)))
    rows.sort(key=lambda item: item[0]["exported_at"], reverse=True)
    lines = ["# G1 测试视频目录", "", "由各导出目录的 manifest.json 自动生成。时间为 Asia/Shanghai。",
             "视频为 MuJoCo 运动学回放，不代表动力学控制器跟踪通过。", "",
             "| 导出时间 | 训练批次 | checkpoint | 文本 | seed / 时长 | 文件 |",
             "|---|---|---|---|---|---|"]
    for record, relative in rows:
        video = quote(str(relative / "replay.mp4"))
        meta = quote(str(relative / "manifest.json"))
        prompt = record['prompt'].replace('|', '\\|').replace('\n', ' ')
        lines.append(f"| {record['exported_at']} | {record['training_run_id']} | "
                     f"{record['checkpoint_step']} / {record['weights']} / {record['checkpoint_sha256'][:8]} | "
                     f"{prompt} | {record['seed']} / {record['duration_s']:.2f}s | [视频]({video}) · [参数]({meta}) |")
    temp = root / f"INDEX.{os.getpid()}.tmp"
    temp.write_text("\n".join(lines) + "\n")
    os.replace(temp, root / "INDEX.md")


if __name__ == "__main__":
    build_index(ROOT / "exports")
