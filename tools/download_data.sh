#!/bin/bash
# Download the G1-retargeted LAFAN1 motions and the G1 description, then build the MJCF/USD assets.
# LAFAN1 (Ubisoft La Forge) is CC BY-NC-ND 4.0; the Unitree G1 description is BSD-3-Clause.
#
# Works both inside the 'supermimic' container (conversion via `docker exec`) and directly on a
# host/cluster node with the environment already active. Overrides:
#   HF_ENDPOINT   mirror for huggingface.co, e.g. https://hf-mirror.com (the H100 cluster's only
#                 working HF egress)
#   PYTHON        interpreter that has mujoco + the kit-less USD importers (default: python3)
#   CONVERSION_MODE  auto (default), host, or docker
set -euo pipefail
cd "$(dirname "$0")/.."
HF_BASE=${HF_ENDPOINT:-https://huggingface.co}
PYTHON=${PYTHON:-python3}
CONVERSION_MODE=${CONVERSION_MODE:-auto}
HF=$HF_BASE/datasets/lvhaidong/LAFAN1_Retargeting_Dataset
API=$HF_BASE/api/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/tree/main
# Minimal images (conda-based cluster pods, for instance) may not ship curl, so fall back to
# urllib. Preserve binary STL bytes as well as text CSV/URDF files.
fetch() {
  if command -v curl >/dev/null; then
    curl --retry 3 --fail --silent --show-error --location "$1"
  else
    "$PYTHON" -c "import sys,urllib.request; sys.stdout.buffer.write(urllib.request.urlopen(sys.argv[1], timeout=120).read())" "$1"
  fi
}
download() { fetch "$1" > "$2"; }
list() { fetch "$API/$1" | "$PYTHON" -c "import json,sys; [print(x['path']) for x in json.load(sys.stdin)]"; }
mkdir -p data/lafan1_g1 data/g1_description/meshes
for f in $(list g1); do download "$HF/resolve/main/$f" "data/lafan1_g1/$(basename "$f")"; done
download "$HF/resolve/main/robot_description/g1/g1_29dof_rev_1_0.urdf" data/g1_description/g1_29dof_rev_1_0.urdf
for f in $(list robot_description/g1/meshes); do download "$HF/resolve/main/$f" "data/g1_description/meshes/$(basename "$f")"; done
echo "motions: $(ls data/lafan1_g1 | wc -l) clips"
# Asset conversion needs mujoco + the kit-less USD importers, so it runs in the environment. Inside
# the container the repository is mounted at /workspace/supermimic; on a host we are already in it.
container_names=$(docker ps --format '{{.Names}}' 2>/dev/null || true)
if [[ "$CONVERSION_MODE" == docker ]] || { [[ "$CONVERSION_MODE" == auto ]] && [[ $'\n'"$container_names"$'\n' == *$'\nsupermimic\n'* ]]; }; then
  docker exec supermimic bash -c "cd /workspace/supermimic && python tools/urdf_to_mjcf.py && python tools/mjcf_to_usd.py"
else
  "$PYTHON" tools/urdf_to_mjcf.py && "$PYTHON" tools/mjcf_to_usd.py
fi
