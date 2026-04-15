#!/bin/bash
# run_all.sh — Regenerate all benchmark CSVs and figures.
#
# Usage (from build-release/Examples/PortfolioVaR/):
#   bash ../../Examples/PortfolioVaR/bench/run_all.sh
#
# Prerequisites: PortfolioVaR, PortfolioVaROMP, PortfolioVaRValidate built.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="bench_results"
mkdir -p "$OUTDIR"

echo "=== Step 1: Validation + convergence (E1) ==="
./PortfolioVaRValidate 10000
cp convergence_var.csv validation_parametric_var.csv \
   validation_bs_spotcheck.csv "$OUTDIR/" 2>/dev/null || true

echo ""
echo "=== Step 2: Sequential stage timing ==="
./PortfolioVaR 10000 > /tmp/seq_out.txt
cp stage_timing_seq.csv pnl_distribution.csv var_summary.csv "$OUTDIR/"
grep -E "Stage|Total" /tmp/seq_out.txt

echo ""
echo "=== Step 3: E3/E4/E5/E6 (parallel experiments) ==="
bash "$SCRIPT_DIR/../bench.sh" "$OUTDIR"

echo ""
echo "=== Step 4: Parallel stage timing (T=8) ==="
./PortfolioVaROMP 10000 8 0 4 > /tmp/par_out.txt
cp stage_timing_omp.csv thread_busy_omp.csv "$OUTDIR/"
grep -E "Stage|Thread eff" /tmp/par_out.txt

echo ""
echo "=== Step 5: Generate figures ==="
cp "$OUTDIR"/*.csv .
python3 "$SCRIPT_DIR/plot.py" --csv-dir . --out-dir figures

echo ""
echo "All done. CSVs in $OUTDIR/, figures in figures/"
