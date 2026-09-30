#!/usr/bin/env bash
# Batch-evaluate a served MVTOKEN LoRA on the LIBERO family: N episodes per suite, then print
# the per-suite and overall success rate. Each episode is a full closed-loop rollout
# (scripts/run_libero_mvtoken.py), so this is the real task metric, not token accuracy.
#
#   bash scripts/libero/eval_batch.sh <model> <n_episodes> [suite ...]
#
# The three LIBERO checkouts install as the same `libero` package and differ only in which
# scenes and protocol they carry, so ONE runner serves all of them: pick the checkout with PY
# and the protocol with RUN_ARGS. That is also why suites are the positional argument here
# while RoboLab takes tasks -- a LIBERO suite already is a set of tasks.
#
#   # LIBERO (original), 10 tasks x 10 episodes
#   PY=../LIBERO/.venv/bin/python \
#     bash scripts/libero/eval_batch.sh <adapter> 10 libero_spatial
#
#   # LIBERO-PRO, base suite + its four shipped perturbations, one process each
#   PY=../LIBERO-PRO/.venv/bin/python \
#     bash scripts/libero/eval_batch.sh <adapter> 10 \
#       libero_spatial libero_spatial_object libero_spatial_swap libero_spatial_lan libero_spatial_task
#
#   # LIBERO-plus: every task in the perturbed suite, ONE trial each -- its own protocol flag,
#   # so N is 1 and the suite list is just the base name.
#   PY=../LIBERO-plus/.venv/bin/python RUN_ARGS="--libero-plus" \
#     bash scripts/libero/eval_batch.sh <adapter> 1 libero_spatial
#
#   PY=          interpreter of the LIBERO checkout to evaluate in (required -- there is no
#                sensible default, the checkout IS the benchmark)
#   GPUS="0"     render cards, handed out round-robin across suites (one process per suite)
#   PORT=8000    the served model's port
#   OUT=         rollout output root (default <repo>/../results_libero/<model>); these hold
#                the per-step PNGs, so expect tens of GB per full run
#   MAX_STEPS=200   decision limit per episode
#   RUN_ARGS=    extra flags passed to the runner verbatim, e.g. "--libero-plus",
#                "--ignore-done" (use it when comparing against published numbers: the other
#                VLAs have no stop action)
#
# Requires the action model already served (scripts/serve_vlm.sh) and a LIBERO venv built by
# `bash scripts/setup.sh libero <checkout>`.
set -uo pipefail

MODEL="${1:?usage: eval_batch.sh <model> <n_episodes> [suite ...]}"
N="${2:-10}"
shift 2 || true
SUITES=("$@")
[ ${#SUITES[@]} -eq 0 ] && SUITES=(libero_spatial)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:?set PY to the interpreter of the LIBERO checkout to evaluate, e.g. PY=../LIBERO-PRO/.venv/bin/python}"
PORT="${PORT:-8000}"
GPUS="${GPUS:-${GPU:-0}}"
MAX_STEPS="${MAX_STEPS:-200}"
OUT="${OUT:-$(dirname "${ROOT}")/results_libero/${MODEL}}"

cd "$ROOT"
# Allow a PY given relative to the repo root (the examples above are), resolved after the cd.
if [ ! -x "$PY" ]; then
  _d="$(cd "$(dirname "$PY")" 2>/dev/null && pwd)"
  [ -n "$_d" ] && PY="${_d%/}/$(basename "$PY")"
fi
[ -x "$PY" ] || { echo "ERROR: not an interpreter: $PY" >&2
                  echo "  build one: bash scripts/setup.sh libero <checkout>" >&2; exit 1; }

# A port serving a DIFFERENT model answers every request just as happily and the run silently
# measures that other model, so check before spending hours.
curl -s -m 5 "http://127.0.0.1:${PORT}/v1/models" >/dev/null || {
  echo "ERROR: nothing serving on ${PORT} -- start scripts/serve_vlm.sh first" >&2; exit 1; }

read -r -a EXTRA <<<"${RUN_ARGS:-}"
read -ra gpu_list <<< "${GPUS}"
mkdir -p "$OUT"

# --tasks is required whenever we are NOT on the --libero-plus protocol: without it the runner
# picks tasks by looking for *_follow directories under --dataset-root, which exist only on a
# machine that generated the training data. Perturbation suites carry their base suite's task
# names, so reading them from the benchmark once covers every suite in the list.
TASK_NAMES_PY='import sys
from pathlib import Path
from libero.libero import benchmark
s = benchmark.get_benchmark_dict()[sys.argv[1]]()
print(",".join(sorted(Path(s.get_task(i).bddl_file).stem for i in range(s.n_tasks))))'

task_arg() {
  case " ${RUN_ARGS:-} " in *" --libero-plus "*) return 0 ;; esac
  local names
  names="$(PYTHONPATH="$ROOT" "$PY" -c "$TASK_NAMES_PY" "$1" 2>/dev/null | tail -1)"
  [ -n "$names" ] && printf -- '--tasks\n%s\n' "$names"
}

echo "[eval] model=$MODEL | $N episodes x ${#SUITES[@]} suite(s) | max_steps=$MAX_STEPS${RUN_ARGS:+ | $RUN_ARGS}"
echo "[eval] out=$OUT"
n=0
for suite in "${SUITES[@]}"; do
  g="${gpu_list[$((n % ${#gpu_list[@]}))]}"; n=$((n + 1))
  mapfile -t TASK_ARGS < <(task_arg "$suite")
  echo "[run] $suite (render GPU $g)"
  ( CUDA_VISIBLE_DEVICES="$g" MUJOCO_EGL_DEVICE_ID="$g" PYTHONPATH="$ROOT" \
      "$PY" -u scripts/run_libero_mvtoken.py \
        --suite "$suite" --episodes "$N" --max-steps "$MAX_STEPS" \
        --out "$OUT/$suite" --vlm-url "http://127.0.0.1:${PORT}/v1" --model "$MODEL" \
        ${TASK_ARGS[@]+"${TASK_ARGS[@]}"} ${EXTRA[@]+"${EXTRA[@]}"} \
      > "$OUT/$suite.log" 2>&1 ) &
done
wait

# Re-read the per-task summary.json files rather than parsing the logs: on a sharded or
# parallel run each process only ever sees its own slice.
total_ok=0; total_n=0
for suite in "${SUITES[@]}"; do
  line=$(PYTHONPATH="$ROOT" "$PY" -u scripts/run_libero_mvtoken.py \
           --out "$OUT/$suite" --summary-only 2>/dev/null | grep -E "^Success rate:")
  ok=$(sed -n 's|^Success rate: \([0-9]*\)/.*|\1|p' <<<"$line")
  tot=$(sed -n 's|^Success rate: [0-9]*/\([0-9]*\).*|\1|p' <<<"$line")
  ok="${ok:-0}"; tot="${tot:-0}"
  total_ok=$((total_ok + ok)); total_n=$((total_n + tot))
  printf '  %-42s %s/%s\n' "$suite" "$ok" "$tot"
done
[ "$total_n" -gt 0 ] && \
  echo "[eval] $MODEL RESULT: $total_ok/$total_n ($((100 * total_ok / total_n))%)"
