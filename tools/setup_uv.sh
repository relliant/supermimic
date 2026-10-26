#!/bin/bash
# Build the full SuperMimic environment on the host with uv, without Docker.
#
# The kit-less image (IsaacLab/docker/Dockerfile.kitless) is itself a uv project: it creates a
# Python 3.12 environment with `uv sync --frozen --extra all --extra importers` from the pinned
# Isaac Lab checkout's uv.lock. This script does the same thing, but puts the environment at
# <repo>/.venv instead of IsaacLab/.venv: Isaac Lab stays the project that owns the lock, only the
# environment's location moves (via UV_PROJECT_ENVIRONMENT). The resulting versions (torch 2.11.0 /
# CUDA 13, newton, warp-lang 1.17, mujoco 3.15, isaaclab 3.0.0) are therefore the ones the
# reproduction was run with. MuJoCo, pillow and matplotlib come from the Isaac Lab lock;
# SuperMimic's SwanLab logging dependency is installed afterwards from docker/requirements.txt.
#
# Requirements: Linux x86_64, a working NVIDIA driver for GPU execution, ~10 GB of
# disk, and uv >= 0.12 (https://docs.astral.sh/uv/getting-started/installation/).
#
# Usage: tools/setup_uv.sh [ISAACLAB_DIR]     (default: <repo>/IsaacLab)
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
ISAACLAB_COMMIT=46d6cc3d7b115906fad12efd372fbf69f4e3d3ed
ISAACLAB=${1:-$REPO/IsaacLab}

command -v uv >/dev/null || {
  echo "uv not found; install it first: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
}

if [ ! -d "$ISAACLAB/.git" ]; then
  git clone https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB"
fi
git -C "$ISAACLAB" checkout "$ISAACLAB_COMMIT"

# `python-preference = "only-managed"` in Isaac Lab's pyproject makes uv download its own CPython
# 3.12, so the host interpreter (3.13/3.14, conda, ...) is irrelevant. `--frozen` is what makes the
# install match the image exactly: it consumes Isaac Lab's uv.lock instead of re-resolving it.
export UV_PROJECT_ENVIRONMENT="$REPO/.venv"
uv sync --project "$ISAACLAB" --frozen --extra all --extra importers

VENV="$REPO/.venv"
# `uv sync` matches the lock exactly and uninstalls anything else, so SwanLab (absent from Isaac
# Lab's lock) only survives until the next sync. Re-run this script instead of syncing by hand.
uv pip install --python "$VENV/bin/python" -r "$REPO/docker/requirements.txt"
"$VENV/bin/python" - <<'PY'
import torch, mujoco, warp

import isaaclab, isaaclab_newton, newton

print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
print("mujoco", mujoco.__version__, "| warp", warp.__version__)
print("isaaclab / isaaclab_newton / newton import OK")
PY

# mujoco's offscreen renderer (tools/render.py) needs the GLVND driver libraries and ffmpeg; the
# image installs them with apt, the host is the user's job.
render_deps_ok=true
ldconfig_output=$(ldconfig -p 2>/dev/null || true)
for lib in libEGL.so.1 libGL.so.1 libOpenGL.so.0; do
  [[ "$ldconfig_output" == *"$lib"* ]] || render_deps_ok=false
done
command -v ffmpeg >/dev/null || render_deps_ok=false
if [ "$render_deps_ok" = false ]; then
  echo "note: for tools/render.py install the GL libraries and ffmpeg, e.g." >&2
  echo "      sudo apt-get install -y libegl1 libgl1 libopengl0 ffmpeg" >&2
fi

cat <<EOF

Environment ready: $VENV

Isaac Lab checkout: $ISAACLAB

Run things from the repository root (the code is used in place, not installed):

  $VENV/bin/python supermimic/train.py --log_dir logs/lafan1_main
  $VENV/bin/python supermimic/evaluate.py --run_dir logs/lafan1_main --watch
  $VENV/bin/python tools/test_env.py
  $VENV/bin/python tools/render.py traj.npz out.mp4 --clips dance2_subject1

Use the interpreter above from the SuperMimic root to keep repository-relative paths explicit.

The environment used to live in $ISAACLAB/.venv; that directory is no longer used and can be
removed (rm -rf $ISAACLAB/.venv) once this script has finished successfully.

To keep build caches inside the repository, like run_container.sh does:

  export WARP_CACHE_PATH=$REPO/.cache/warp
  export TORCHINDUCTOR_CACHE_DIR=$REPO/.cache/inductor
  export MPLCONFIGDIR=/tmp/mpl
EOF
