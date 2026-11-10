#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
mkdir -p "$project_dir/logs/tensorboard_tmp"
export TMPDIR="$project_dir/logs/tensorboard_tmp"
export PYTHONDONTWRITEBYTECODE=1
exec "$project_dir/.venv/bin/python" -B -m tensorboard.main \
  --logdir "$project_dir/runs/m3_20261101" --host 0.0.0.0 \
  --port "${M3_TENSORBOARD_PORT:-6006}" --reload_interval 10 --load_fast false
