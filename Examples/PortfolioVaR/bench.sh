#!/bin/bash
# bench.sh — Phase 2c experiments: E3, E4, E5, E6
# Run from build-release/Examples/PortfolioVaR/
# Usage: ./bench.sh [output_dir]

set -euo pipefail
OUTDIR="${1:-bench_results}"
mkdir -p "$OUTDIR"

SEQ=./PortfolioVaR
PAR=./PortfolioVaROMP

echo "=== Phase 2c Benchmark ==="
echo "Output directory: $OUTDIR"
echo ""

# ---------------------------------------------------------------------------
# E3 — Strong scaling: fixed N=10000, vary threads
# ---------------------------------------------------------------------------
echo "--- E3: Strong scaling (N=10000) ---"
N_STRONG=10000

# Sequential baseline
echo -n "  seq N=$N_STRONG ... "
T_SEQ=$(TIMEFORMAT='%R'; { time $SEQ $N_STRONG > /tmp/bench_seq.txt 2>&1; } 2>&1)
VAR99_SEQ=$(grep "VaR(99%)" /tmp/bench_seq.txt | awk '{print $2}')
echo "${T_SEQ}s  VaR99=${VAR99_SEQ}"

E3_CSV="$OUTDIR/strong_scaling.csv"
echo "threads,wall_s,speedup,efficiency_pct,var99" > "$E3_CSV"
echo "1_seq,${T_SEQ},1.00,100.00,${VAR99_SEQ}" >> "$E3_CSV"

for T in 1 2 4 8 16 32 64; do
    echo -n "  T=$T ... "
    $PAR $N_STRONG $T 0 16 > /tmp/bench_par.txt 2>&1
    WALL=$(grep "Wall time:" /tmp/bench_par.txt | awk '{print $3}')
    EFF=$(grep "Thread efficiency:" /tmp/bench_par.txt | awk '{print $3}' | tr -d '%')
    VAR99=$(grep "VaR(99%):" /tmp/bench_par.txt | awk '{print $2}')
    SPEEDUP=$(awk "BEGIN{printf \"%.2f\", ${T_SEQ}/${WALL}}")
    echo "${WALL}s  speedup=${SPEEDUP}x  eff=${EFF}%  VaR99=${VAR99}"
    echo "${T},${WALL},${SPEEDUP},${EFF},${VAR99}" >> "$E3_CSV"
done
echo "  Written $E3_CSV"

# ---------------------------------------------------------------------------
# E4 — Weak scaling: per-thread work = 1000 scenarios
# ---------------------------------------------------------------------------
echo ""
echo "--- E4: Weak scaling (1000 scenarios/thread) ---"
N_PER_THREAD=1000

E4_CSV="$OUTDIR/weak_scaling.csv"
echo "threads,n_scenarios,wall_s,var99" > "$E4_CSV"

for T in 1 2 4 8 16 32 64; do
    N=$(( N_PER_THREAD * T ))
    echo -n "  T=$T N=$N ... "
    $PAR $N $T 0 16 > /tmp/bench_par.txt 2>&1
    WALL=$(grep "Wall time:" /tmp/bench_par.txt | awk '{print $3}')
    VAR99=$(grep "VaR(99%):" /tmp/bench_par.txt | awk '{print $2}')
    echo "${WALL}s  VaR99=${VAR99}"
    echo "${T},${N},${WALL},${VAR99}" >> "$E4_CSV"
done
echo "  Written $E4_CSV"

# ---------------------------------------------------------------------------
# E5 — Schedule comparison: 8 threads, N=10000
# ---------------------------------------------------------------------------
echo ""
echo "--- E5: Schedule comparison (T=8, N=10000) ---"
N_SCHED=10000
T_SCHED=8

E5_CSV="$OUTDIR/schedule_comparison.csv"
echo "schedule,chunk,wall_s,efficiency_pct,var99" > "$E5_CSV"

# static
echo -n "  static ... "
$PAR $N_SCHED $T_SCHED 1 1 > /tmp/bench_par.txt 2>&1
WALL=$(grep "Wall time:" /tmp/bench_par.txt | awk '{print $3}')
EFF=$(grep "Thread efficiency:" /tmp/bench_par.txt | awk '{print $3}' | tr -d '%')
VAR99=$(grep "VaR(99%):" /tmp/bench_par.txt | awk '{print $2}')
echo "${WALL}s  eff=${EFF}%"
echo "static,N/A,${WALL},${EFF},${VAR99}" >> "$E5_CSV"

# dynamic with various chunk sizes
for CHUNK in 1 4 16 64; do
    echo -n "  dynamic chunk=$CHUNK ... "
    $PAR $N_SCHED $T_SCHED 0 $CHUNK > /tmp/bench_par.txt 2>&1
    WALL=$(grep "Wall time:" /tmp/bench_par.txt | awk '{print $3}')
    EFF=$(grep "Thread efficiency:" /tmp/bench_par.txt | awk '{print $3}' | tr -d '%')
    VAR99=$(grep "VaR(99%):" /tmp/bench_par.txt | awk '{print $2}')
    echo "${WALL}s  eff=${EFF}%"
    echo "dynamic,${CHUNK},${WALL},${EFF},${VAR99}" >> "$E5_CSV"
done

# guided
echo -n "  guided ... "
$PAR $N_SCHED $T_SCHED 2 16 > /tmp/bench_par.txt 2>&1
WALL=$(grep "Wall time:" /tmp/bench_par.txt | awk '{print $3}')
EFF=$(grep "Thread efficiency:" /tmp/bench_par.txt | awk '{print $3}' | tr -d '%')
VAR99=$(grep "VaR(99%):" /tmp/bench_par.txt | awk '{print $2}')
echo "${WALL}s  eff=${EFF}%"
echo "guided,16,${WALL},${EFF},${VAR99}" >> "$E5_CSV"

echo "  Written $E5_CSV"

# ---------------------------------------------------------------------------
# E6 — Per-thread busy time: save the CSV from the 8-thread N=10000 run
# ---------------------------------------------------------------------------
echo ""
echo "--- E6: Per-thread busy time (T=8, N=10000, dynamic/16) ---"
$PAR $N_STRONG 8 0 16 > /tmp/bench_par.txt 2>&1
cp thread_busy_omp.csv "$OUTDIR/thread_busy_8t.csv"
echo "  Wall: $(grep 'Wall time:' /tmp/bench_par.txt | awk '{print $3}')s"
echo "  Efficiency: $(grep 'Thread efficiency:' /tmp/bench_par.txt | awk '{print $3}')%"
echo "  Written $OUTDIR/thread_busy_8t.csv"

echo ""
echo "=== All experiments complete ==="
echo "CSVs in $OUTDIR/"
ls -lh "$OUTDIR/"
