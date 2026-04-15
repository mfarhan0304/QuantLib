Portfolio Monte Carlo VaR — OpenMP Parallelization
==================================================

Prerequisites
-------------
  - Boost >= 1.71 (headers only for QuantLib)
  - CMake >= 3.15
  - GCC >= 9 with OpenMP support (libgomp)
  - Python >= 3.8 with numpy, pandas, matplotlib (for plots)

Install Python dependencies:
    pip3 install --user numpy pandas matplotlib


Build
-----
From the QuantLib root directory:

  # Release build (benchmarks)
  cmake -B build-release \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_FLAGS="-O3 -march=native" \
        -DQL_ENABLE_OPENMP=OFF
  cmake --build build-release --target PortfolioVaR PortfolioVaROMP PortfolioVaRValidate -j$(nproc)

Note: QL_ENABLE_OPENMP=OFF is intentional. The build system already
applies ${OpenMP_CXX_FLAGS} to ql_library unconditionally; the
PortfolioVaROMP target receives -fopenmp via its own CMakeLists.txt.


Run
---
From build-release/Examples/PortfolioVaR/:

  # Sequential baseline (N=10000 scenarios)
  ./PortfolioVaR 10000

  # Parallel (N=10000, 8 threads, dynamic scheduling, chunk=4)
  ./PortfolioVaROMP 10000 8 0 4

  # Parallel — schedule options:
  #   argv[3]:  0=dynamic (default), 1=static, 2=guided
  #   argv[4]:  chunk size (default 16)
  ./PortfolioVaROMP 10000 8 1       # static
  ./PortfolioVaROMP 10000 8 2 16    # guided, chunk=16

  # Control thread count via argv[2] (0 = omp_get_max_threads):
  OMP_NUM_THREADS=8 ./PortfolioVaROMP 10000 0 0 4

  # Validation suite (parametric VaR + B-S spot check + convergence)
  ./PortfolioVaRValidate 10000


Regenerate benchmark CSVs
-------------------------
From build-release/Examples/PortfolioVaR/:

  bash ../../Examples/PortfolioVaR/bench/run_all.sh

This runs all experiments (E1-E7) and writes CSVs to bench_results/.
Estimated runtime: ~15 minutes on an 8-core machine.


Regenerate plots
----------------
From build-release/Examples/PortfolioVaR/:

  python3 ../../Examples/PortfolioVaR/bench/plot.py \
      --csv-dir . \
      --out-dir figures

Figures are written to figures/ as PNG files (fig1_convergence.png
through fig7_cache_table.png).


Directory map
-------------
  Examples/PortfolioVaR/
    PortfolioVaR.cpp          Sequential driver
    PortfolioVaROMP.cpp       Parallel driver (uses ScenarioEvaluator)
    portfolio.{hpp,cpp}       Portfolio construction (shared)
    scenarios.{hpp,cpp}       Scenario generator (shared)
    var_stats.hpp             VaR/ES statistics (shared)
    validate.cpp              Validation suite driver
    bench.sh                  Phase 2c benchmark script
    bench/
      run_all.sh              Full experiment regeneration script
      plot.py                 Figure generation script
    report/
      report.tex              LaTeX source
      refs.bib                BibTeX references
      report.pdf              Compiled report
      figures/                PNG figures (copied from build)
    readme.txt                This file

  ql/experimental/risk/
    scenarioevaluator.hpp     ScenarioEvaluator class declaration
    scenarioevaluator.cpp     Implementation (OpenMP + sequential fallback)

  build-release/Examples/PortfolioVaR/
    PortfolioVaR              Sequential binary
    PortfolioVaROMP           Parallel binary
    PortfolioVaRValidate      Validation binary
    bench_results/            Benchmark CSVs (E1-E7)
    figures/                  Generated PNG plots
    stage_timing_seq.csv      Dixon stage breakdown (sequential)
    stage_timing_omp.csv      Dixon stage breakdown (parallel)


Key results
-----------
  Sequential (N=10000):        126.1 s
  Parallel T=8  (N=10000):     17.7 s  -> 7.1x speedup, 89% efficiency
  Parallel T=64 (N=10000):     3.9 s   -> 32x speedup, 50% efficiency
  Best schedule: dynamic, chunk=4 (21% faster than static at T=8)
  Revaluation stage share: 99.997% of total runtime
  Correctness: bit-identical P&L vs sequential for all thread counts
