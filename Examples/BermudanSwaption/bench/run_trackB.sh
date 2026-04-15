#!/usr/bin/env bash
# Track B benchmark sweep — parallel LM Jacobian.
# Outputs CSV to stdout; human-readable lines to stderr.
# Usage: bash bench/run_trackB.sh > trackB_results.csv

set -euo pipefail

BIN="$(dirname "$0")/../../../build-release/Examples/BermudanSwaption/BermudanSwaptionOMP"
STEPS=300
G2PTS=128
THREADS="1 2 4 8"

echo "model,threads,mode,wall_s,jac_s,jac_calls,params" >&2
echo "model,threads,mode,wall_s,jac_s,jac_calls,params"

# Sequential baseline (OMP_NUM_THREADS=1 to suppress inner TreeLattice pragma)
echo "--- sequential baseline ---" >&2
OMP_NUM_THREADS=1 OMP_STACKSIZE=64M \
    "$BIN" -seq -steps "$STEPS" -g2pts "$G2PTS" 2>&1 | \
    grep "^CSV," | sed 's/^CSV,//' | tee /dev/stderr

# Parallel sweep
for T in $THREADS; do
    echo "--- parallel T=$T ---" >&2
    OMP_NUM_THREADS="$T" OMP_STACKSIZE=64M \
        "$BIN" -t "$T" -steps "$STEPS" -g2pts "$G2PTS" 2>&1 | \
        grep "^CSV," | sed 's/^CSV,//' | tee /dev/stderr
done

echo "--- done ---" >&2
