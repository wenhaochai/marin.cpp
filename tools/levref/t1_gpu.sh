#!/bin/bash
# T1 levanter fixtures on GPU, one preset per call (run through tools/devq; the runner sets CUDA_VISIBLE_DEVICES and TMPDIR).
#   t1_gpu.sh precision130   4 GPUs: T1.0 facts (XLA dump + LD_DEBUG) and T1.6 inputs of 130m, one full-N step
#   t1_gpu.sh precision1_2b  1 GPU:  T1.6 inputs of 1_2b (u of update 0, NS default/highest), no train step
#   t1_gpu.sh units          1 GPU:  T1.2-T1.5 at 130m, T1.4 at 520m, T1.2-T1.5 at 1_2b
# After all three: tools/levref/t1_finish.sh on della-vis1 (f64 oracles on CPU, then cache/levref/t1/ with README.md).
set -u
: "${NANOMARIN_ROOT:?set NANOMARIN_ROOT}"
PY="$NANOMARIN_ROOT/project/marin/.venv/bin/python"
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 2
export HF_HUB_OFFLINE=1 WANDB_MODE=disabled PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export MARIN_PREFIX="${MARIN_PREFIX:-$NANOMARIN_ROOT/marin_store_big}"
ulimit -c 0
echo "host $(hostname) $(date -u +%FT%TZ) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} TMPDIR=${TMPDIR:-unset}"
nvidia-smi -L
run() { echo "=== $* $(date -u +%T)"; "$PY" -m "$@"; local rc=$?; echo "=== exit $rc $(date -u +%T)"; return $rc; }
rc=0
case "${1:-}" in
  precision130) run tools.levref.precision --size 130m --dump --ld-debug || rc=1 ;;
  precision1_2b) run tools.levref.precision --size 1_2b --no-step || rc=1 ;;
  units)
    run tools.levref.units --size 130m || rc=1
    run tools.levref.units --size 520m || rc=1
    run tools.levref.units --size 1_2b || rc=1 ;;
  *) echo "unknown preset ${1:-}"; exit 2 ;;
esac
exit $rc
