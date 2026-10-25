#!/bin/bash
# T1 after the GPU presets of t1_gpu.sh have written <size>/t1_precision and <size>/t1_units: the f64 oracles (numpy, CPU)
# and the report (cache/levref/t1/{results.jsonl,table.md,README.md}). CPU only: run it on della-vis1, never on a GPU.
#   ssh della-vis1 'NANOMARIN_ROOT=<root> nice -n 19 bash <repo>/tools/levref/t1_finish.sh'
# 1_2b needs about 17 GB of RAM and about 2e14 f64 FLOP (the CE oracle over 131072 rows); 130m and 520m take minutes.
set -u
: "${NANOMARIN_ROOT:?set NANOMARIN_ROOT}"
PY="$NANOMARIN_ROOT/project/marin/.venv/bin/python"
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 2
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=
echo "host $(hostname) $(date -u +%FT%TZ)"
rc=0
for size in 130m 520m 1_2b; do
  parts=norm,attn,ce,ns
  [ "$size" = 520m ] && parts=attn
  fx="$NANOMARIN_ROOT/project/nanomarin/cache/levref/$size"
  if [ ! -f "$fx/t1_units/manifest.json" ] && [ ! -f "$fx/t1_precision/manifest.json" ]; then
    echo "=== oracle $size: no GPU fixture yet, skipped"; rc=1; continue
  fi
  echo "=== oracle $size ($parts) $(date -u +%T)"
  "$PY" -m tools.levref.oracle --size "$size" --parts "$parts" || rc=1
done
echo "=== report $(date -u +%T)"
"$PY" -m tools.levref.t1_report || rc=1
echo "=== exit $rc $(date -u +%T)"
exit $rc
