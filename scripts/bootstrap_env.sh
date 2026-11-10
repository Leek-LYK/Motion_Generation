#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
workspace_dir="$(cd -- "${project_dir}/../.." && pwd -P)"
kimodo_dir="${project_dir}/../kimodo"
uv_bin="${UV_BIN:-${workspace_dir}/.local/bin/uv}"
python_bin="${KIMODO_REPRO_PYTHON:-/usr/bin/python3.10}"
native_dir="${kimodo_dir}/source/MotionCorrection/python"

if [[ ! -x "${uv_bin}" ]]; then
    echo "uv executable not found: ${uv_bin}; set UV_BIN to your uv path" >&2
    exit 1
fi
if [[ ! -f "${native_dir}/motion_correction/_motion_correction.cpython-310-x86_64-linux-gnu.so" ]]; then
    echo "The existing Python 3.10 motion_correction extension is missing: ${native_dir}" >&2
    exit 1
fi

cd "${project_dir}"
if [[ ! -d "${project_dir}/.venv" ]]; then
    "${uv_bin}" venv --python "${python_bin}" --prompt mg_repro "${project_dir}/.venv"
fi
UV_CACHE_DIR="${UV_CACHE_DIR:-${workspace_dir}/.cache/uv}" \
UV_LINK_MODE="${UV_LINK_MODE:-copy}" \
SKIP_MOTION_CORRECTION_IN_SETUP=1 \
    "${uv_bin}" sync --locked --python "${python_bin}"

site_dir="$("${project_dir}/.venv/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
printf '%s\n' "${native_dir}" > "${site_dir}/kimodo_motion_correction_workspace.pth"
if [[ ! -e "${project_dir}/.venv/bin/uv" && ! -L "${project_dir}/.venv/bin/uv" ]]; then
    ln -s "${uv_bin}" "${project_dir}/.venv/bin/uv"
fi
"${project_dir}/.venv/bin/python" -c 'import motion_correction; print("motion_correction:", motion_correction.__file__)'
