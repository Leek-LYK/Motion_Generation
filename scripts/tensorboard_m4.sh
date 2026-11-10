#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
experiment_dir="$project_dir/runs/m4_20261102"
mkdir -p "$project_dir/logs/tensorboard_m4_tmp"
export TMPDIR="$project_dir/logs/tensorboard_m4_tmp"
export PYTHONDONTWRITEBYTECODE=1
exec "$project_dir/.venv/bin/python" -B -m tensorboard.main \
  --logdir_spec "M4-A:$experiment_dir/A_256/tensorboard,M4-B:$experiment_dir/B_1024/tensorboard,M4-C:$experiment_dir/C_1024_textdrop/tensorboard,M4-D:$experiment_dir/D_1024_heading/tensorboard" \
  --host 0.0.0.0 --port "${M4_TENSORBOARD_PORT:-6007}" --reload_interval 10 --load_fast false
