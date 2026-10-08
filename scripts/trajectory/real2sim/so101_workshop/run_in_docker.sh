#!/usr/bin/env bash
# Run a script of this repo inside the Sim-to-Real SO-101 Workshop image (Isaac Sim 5.1 /
# Isaac Lab), headless. The Workshop stack stays in its own image; this repo is mounted and
# put on PYTHONPATH, nothing is pip-installed from it.
#
#   WORKSHOP_ROOT=~/sim2real/Sim-to-Real-SO-101-Workshop \
#     bash scripts/trajectory/real2sim/so101_workshop/run_in_docker.sh \
#     scripts/trajectory/real2sim/so101_workshop/verify_motion.py --out /workspace/out/verify
#
# Prerequisites (docs/simulators.md, "SO-101 (Isaac Lab Workshop)"): a checkout of
# https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop at tag v1.0, and its sim image
# built from that checkout: docker build -t teleop-docker -f docker/sim/Dockerfile .
#
# Env: WORKSHOP_ROOT (required), IMAGE (teleop-docker:latest), OUT_ROOT (./rollouts/so101_workshop,
# mounted at /workspace/out), CACHE (~/docker/isaac-sim, the Workshop README's cache dirs).
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../../../.." && pwd)
: "${WORKSHOP_ROOT:?set WORKSHOP_ROOT to a Sim-to-Real-SO-101-Workshop v1.0 checkout}"
IMAGE=${IMAGE:-teleop-docker:latest}
OUT_ROOT=${OUT_ROOT:-$REPO/rollouts/so101_workshop}
CACHE=${CACHE:-$HOME/docker/isaac-sim}
NAME=${NAME:-showharness_so101_$(date +%H%M%S)}

[ $# -ge 1 ] || { echo "usage: $0 <script.py> [args...]" >&2; exit 2; }
SCRIPT=$1; shift
# Quote every argument for the inner `bash -lc` (an unquoted "a|b" there would be a pipe).
ARGS=""
[ $# -gt 0 ] && ARGS=$(printf '%q ' "$@")
mkdir -p "$OUT_ROOT"

docker run --rm --name "$NAME" --gpus all --network=host \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -e PYTHONPATH=/workspace/Show-Harness \
  -v "$WORKSHOP_ROOT/docker/env:/root/env" \
  -v "$WORKSHOP_ROOT:/workspace/Sim-to-Real-SO-101-Workshop" \
  -v "$REPO:/workspace/Show-Harness:ro" \
  -v "$OUT_ROOT:/workspace/out" \
  -v "$CACHE/cache/kit:/isaac-sim/kit/cache:rw" \
  -v "$CACHE/cache/ov:/root/.cache/ov:rw" \
  -v "$CACHE/cache/pip:/root/.cache/pip:rw" \
  -v "$CACHE/cache/glcache:/root/.cache/nvidia/GLCache:rw" \
  -v "$CACHE/cache/computecache:/root/.nv/ComputeCache:rw" \
  -v "$CACHE/logs:/root/.nvidia-omniverse/logs:rw" \
  -v "$CACHE/data:/root/.local/share/ov/data:rw" \
  -v "$CACHE/documents:/root/Documents:rw" \
  "$IMAGE" \
  bash -lc "
    cd /workspace/Sim-to-Real-SO-101-Workshop
    python3 /workspace/Show-Harness/$SCRIPT --headless $ARGS
    rc=\$?
    chown -R $(id -u):$(id -g) /workspace/out || true
    exit \$rc
  "
