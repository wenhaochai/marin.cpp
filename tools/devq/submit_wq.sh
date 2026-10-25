#!/bin/bash
# Queue N devq runners (4 H100 each, <= 3 h) as wq jobs at --prio 20 (owner rule for marin.cpp GPU work).
# Usage: NANOMARIN_ROOT=<scratch root> tools/devq/submit_wq.sh [N] [GPUS]   (GPUS per runner, default 4)
set -euo pipefail
: "${NANOMARIN_ROOT:?set NANOMARIN_ROOT}"
n=${1:-1}
g=${2:-4}
here=$(cd "$(dirname "$0")" && pwd)
out=$NANOMARIN_ROOT/tmp/devq/wq_runner_g$g.sh
mkdir -p "$(dirname "$out")"
sed -e "s|@ROOT@|$NANOMARIN_ROOT|" -e "s|@GPUS@|$g|g" -e "s|@MEM@|$((64 * g))G|" "$here/wq_runner.sh.in" > "$out"
for _ in $(seq "$n"); do "$NANOMARIN_ROOT/tools/wq/wq" submit --prio 20 "$out"; done
