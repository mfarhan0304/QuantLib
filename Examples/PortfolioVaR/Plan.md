# Portfolio Monte Carlo VaR — Parallelization Project Plan

## 1. Objective

Build a portfolio-level **Monte Carlo Value-at-Risk (VaR) and Expected Shortfall (ES)** engine on top of QuantLib, parallelize it with OpenMP, and measure strong/weak scaling on an 8-core Linux VM. Produce a report with literature survey, methodology, scaling curves, and a discussion of the tradeoffs encountered.

## 2. Why this project

- **Trading relevance.** Every trading floor at every bank computes portfolio VaR and ES daily. It's a regulatory requirement under Basel III and the FRTB Internal Models Approach (FRTB-IMA), and it directly gates how much risk traders are allowed to take. "Market Risk Engineer" / "VaR Quant" is a real, well-paid job.
- **Compute-bound.** A realistic VaR run revalues every position in a portfolio under thousands of market scenarios. Inner pricers (especially American Monte Carlo) dominate runtime.
- **Rich parallelization shape.** Three nested axes of parallelism (scenarios × positions × MC paths), real load imbalance between asset classes (a bond reprices in microseconds; an American option MC takes milliseconds), and a meaningful aggregation/reduction step for tail statistics. This is *not* embarrassingly parallel — there's a story to tell.
- **LinkedIn keywords earned.** Monte Carlo VaR, Expected Shortfall, FRTB, market risk, OpenMP, nested parallelism, HPC, QuantLib.

## 3. Why QuantLib

- Mature C++17 codebase used in production at banks
- Pricing engines for every instrument we need (bonds, swaps, European/American options) ready to use
- `Handle<>` / `RelinkableHandle<>` market-data plumbing makes "shock the market and reprice" tractable
- CMake build, builds cleanly on Linux, GCC + OpenMP work natively
- Existing `Examples/CVAIRS/` provides a structural template (path-based revaluation of a swap)

## 4. Scope guardrails

- **In scope:** sequential baseline, OpenMP parallel version, strong/weak scaling experiments, schedule comparison, correctness validation, report.
- **Out of scope:** GPU (CUDA/OpenCL), distributed-memory MPI, AAD/AD Greeks, regulatory edge cases (FRTB stressed-period overlay, P&L attribution test), real production market data feeds.
- **Stretch goals (only if time permits):** nested parallelism (inner MC paths inside outer scenario loop), historical-simulation VaR comparison, calibration of factor model from real SPY + UST data via qstrader.

## 5. Architecture

### 5.1 Pipeline (sequential view)

```
                   ┌─────────────────┐
                   │ Portfolio       │  100 bonds, 50 swaps, 20 EU opts, 10 AM opts
                   └────────┬────────┘
                            │
                   ┌────────▼────────┐
                   │ Base Market     │  yield curve, equity spots, vol surface
                   └────────┬────────┘
                            │
                   ┌────────▼────────┐
                   │ Scenario Gen    │  10K draws from 3-factor Normal model
                   └────────┬────────┘
                            │
              ┌─────────────▼─────────────┐
              │ Revaluation Loop          │  per scenario: shift market, reprice all
              │   for s in 1..N_sc:       │
              │     for inst in book:     │
              │       pnl[s] += npv_shocked - npv_base
              └─────────────┬─────────────┘
                            │
                   ┌────────▼────────┐
                   │ Tail Statistics │  sort, VaR(95/99), ES(95/99)
                   └─────────────────┘
```

### 5.2 Where compute lives

Profile expectation (to be verified with `gprof`):
- ~70-80% in `Instrument::NPV()` — dominated by American MC LSM pricer
- ~10-15% in scenario application (curve rebuild, vol surface bump)
- ~5-10% in RNG + statistics

The dominant axis to parallelize is the outer scenario loop, with possible nested parallelism inside the American MC pricer.

## 6. Portfolio composition (V1)

Synthetic but realistic mixed book:

| Instrument | Count | QuantLib type | Pricer | Per-call cost |
|---|---|---|---|---|
| Fixed-rate bond | 100 | `FixedRateBond` | `DiscountingBondEngine` | ~µs |
| Interest rate swap | 50 | `VanillaSwap` | `DiscountingSwapEngine` | ~µs |
| European equity option | 20 | `VanillaOption` | `AnalyticEuropeanEngine` | ~µs |
| American equity option | 10 | `VanillaOption` | `MCAmericanEngine` (LSM) | ~ms |

**Total:** 180 instruments. The 10 American options dominate cost — exactly the load-imbalance source we want to demonstrate `schedule(dynamic)` on.

## 7. Scenario model (V1)

10,000 scenarios drawn from a 3-factor Normal model:

| Factor | Shock | Calibration source |
|---|---|---|
| Rates parallel shift | `Δr ~ N(0, σ_r²)` | UST 10Y daily changes |
| Equity index return | `r_S ~ N(0, σ_S²)` | SPY daily log returns (via qstrader CSV loader) |
| Equity vol shock | `Δσ ~ N(0, σ_v²)` | VIX daily changes |

Factor correlation matrix `Σ` calibrated from joint history. Cholesky factor applied to independent normals to produce correlated shocks.

V1 simplification: assume zero correlation; V2 adds Cholesky.

## 8. Build system

### 8.1 New files in the QuantLib library (`ql/`)
The parallel engine is a first-class library class, not an Examples-only driver. This way `PortfolioVaR` (sequential) and `PortfolioVaROMP` (parallel) are **both thin drivers** that differ only in which library code path they call — apples-to-apples timing.

- `ql/experimental/risk/scenarioevaluator.hpp` — class declaration
- `ql/experimental/risk/scenarioevaluator.cpp` — implementation, including the `#pragma omp parallel for` dispatch and per-thread context machinery
- Register both in `ql/CMakeLists.txt`: `.hpp` in `QL_HEADERS`, `.cpp` in `QL_SOURCES`
- Add `find_package(OpenMP)` gate at the top of `ql/CMakeLists.txt`; if found, link `OpenMP::OpenMP_CXX` to the `ql_library` target and add `-DQL_ENABLE_OPENMP` so the class can guard its OMP calls with `#ifdef`
- If OpenMP is **not** found, the class falls back to a sequential loop so the library still builds

### 8.2 Example driver files (`Examples/PortfolioVaR/`)
- `CMakeLists.txt` (mirrors `Examples/CVAIRS/CMakeLists.txt`)
- `PortfolioVaR.cpp` — sequential driver (directly iterates scenarios, no library engine)
- `PortfolioVaROMP.cpp` — parallel driver (constructs a `ScenarioEvaluator`, passes a factory + shock function)
- `portfolio.hpp` / `portfolio.cpp` — portfolio construction (shared between both drivers)
- `scenarios.hpp` / `scenarios.cpp` — scenario generator (shared)
- `var_stats.hpp` — VaR/ES reduction (shared)
- Add to top-level `Examples/CMakeLists.txt`

### 8.3 Build types
- `Release` — `-O3 -DNDEBUG -march=native` for benchmarking
- `RelWithDebInfo` — `-O2 -g -pg` for `gprof` and `perf`

## 9. Sequential baseline (Phase 1)

### 9.1 Implementation order

1. CMake skeleton + hello-world that links QuantLib
2. Build base market data (yield curve, equity spot, vol surface) using `RelinkableHandle`
3. Build portfolio of 180 instruments with engines attached
4. Compute base NPV — sanity check (sum, individual prices look sane)
5. Implement scenario generator (single 3-factor draw)
6. Implement market-state shifter (apply factor draw → relink curves/quotes)
7. Single-scenario revaluation: shift, reprice, restore, record P&L
8. Loop over N scenarios → P&L vector
9. VaR/ES statistics (sort + percentile + tail mean)
10. CSV output of P&L distribution + summary stats

### 9.2 Validation

- **Bond-only subset, parametric VaR cross-check.** Bond P&L under parallel rate shifts is well-approximated by `−Duration × Δr × NPV`. The MC VaR should converge to the analytic VaR as N → ∞.
- **Black-Scholes spot check** for European option NPVs in the base case.
- **Convergence plot:** VaR estimate vs N from 1K → 100K scenarios. Should be O(1/√N).

### 9.3 Profiling

- Build `RelWithDebInfo` with `-pg`
- Run on 10K-scenario portfolio
- `gprof` flat profile + call graph → confirm hot path is `Instrument::NPV()` chain
- `perf stat -e cycles,instructions,cache-misses,cache-references` for IPC + miss rate
- `perf record -g` + `perf report` for sampled call graph
- Document hotspots in report — this justifies the parallel axis choice

## 10. Parallelization strategy (Phase 2)

### 10.0 Where the parallel code lives

The parallel engine is a new class in the QuantLib library: `ql/experimental/risk/scenarioevaluator.hpp/.cpp`. It is **not** an Examples-only driver. Motivation:
- Apples-to-apples benchmark: sequential vs parallel paths both go through the library on a single shared build, so any observed speedup is attributable to OpenMP rather than unrelated code differences between two hand-written drivers
- The class becomes a reusable library feature — any QuantLib consumer can parallelize portfolio revaluation without reinventing the harness
- Keeps the Examples/ code small: `PortfolioVaROMP.cpp` shrinks to a factory + shock lambda + a single `.run()` call

Sketch of the public API:

```cpp
// ql/experimental/risk/scenarioevaluator.hpp
namespace QuantLib {

    class ScenarioEvaluator {
      public:
        enum class Schedule { Static, Dynamic, Guided };

        struct Config {
            Size nThreads     = 0;                  // 0 = omp_get_max_threads()
            Schedule schedule = Schedule::Dynamic;
            Size chunkSize    = 16;
        };

        // One thread's independent view of the market + portfolio.
        // The factory produces one of these per thread; no state is shared.
        struct ThreadContext {
            std::vector<ext::shared_ptr<SimpleQuote>> quotes;       // shockable
            std::vector<ext::shared_ptr<Instrument>>  instruments;  // engines linked to quotes
        };

        using ContextFactory = std::function<ThreadContext()>;
        using ShockFn        = std::function<void(ThreadContext&, Size scenarioIdx)>;

        // Pre-builds nThreads independent ThreadContexts via factory().
        ScenarioEvaluator(ContextFactory factory,
                          const std::vector<Real>& baseNPVs,
                          Config cfg = {});

        // Parallel P&L evaluation across scenarios.
        // shockFn applies scenario i to a thread-local context before NPV().
        std::vector<Real> run(Size nScenarios, ShockFn shockFn);

        // Diagnostics for E3/E5/E6 experiments
        Real wallTime() const;
        const std::vector<Real>& threadBusyTime() const;
    };

}
```

### 10.1 Primary axis — scenario-parallel (inside `ScenarioEvaluator::run`)

```cpp
// Inside ql/experimental/risk/scenarioevaluator.cpp
std::vector<Real> ScenarioEvaluator::run(Size nScenarios, ShockFn shockFn) {
    std::vector<Real> pnl(nScenarios);
    auto& ctxs = threadContexts_;   // pre-built, one per thread
    auto& base = baseNPVs_;

#ifdef QL_ENABLE_OPENMP
    const int nt = static_cast<int>(cfg_.nThreads ? cfg_.nThreads : omp_get_max_threads());
    const Real t0 = omp_get_wtime();

    if (cfg_.schedule == Schedule::Dynamic) {
        #pragma omp parallel for num_threads(nt) schedule(dynamic, cfg_.chunkSize)
        for (Size s = 0; s < nScenarios; ++s) {
            int tid = omp_get_thread_num();
            Real t = omp_get_wtime();
            auto& ctx = ctxs[tid];
            shockFn(ctx, s);
            Real v = 0.0;
            for (Size i = 0; i < ctx.instruments.size(); ++i)
                v += ctx.instruments[i]->NPV() - base[i];
            pnl[s] = v;
            threadBusy_[tid] += omp_get_wtime() - t;
        }
    } else if (cfg_.schedule == Schedule::Static) {
        #pragma omp parallel for num_threads(nt) schedule(static)
        for (Size s = 0; s < nScenarios; ++s) { /* same body */ }
    } else /* Guided */ {
        #pragma omp parallel for num_threads(nt) schedule(guided)
        for (Size s = 0; s < nScenarios; ++s) { /* same body */ }
    }
    wallTime_ = omp_get_wtime() - t0;
#else
    // Sequential fallback when built without OpenMP
    for (Size s = 0; s < nScenarios; ++s) { /* same body with tid=0 */ }
#endif
    return pnl;
}
```

The sequential `PortfolioVaR.cpp` driver keeps its own plain `for` loop — that is the baseline. The parallel `PortfolioVaROMP.cpp` driver constructs a `ScenarioEvaluator` and calls `.run(...)`. Both use the same `portfolio.cpp` factory functions, same scenarios, same base NPVs — so the only difference is the code path inside the library.

### 10.2 Thread-safety hazards in QuantLib

QuantLib was designed before threading was a first-class concern. Critical hazards:

- **`Settings::evaluationDate()`** is a global singleton. Changing it from one thread breaks the others.
- **`Handle<>` / `RelinkableHandle<>`** observers fire across threads if shared. Each thread needs its **own** handles/curves/quotes.
- **Pricing engine internals** may cache state (e.g. lazily computed yield curve segments).
- **RNG**: `MersenneTwisterUniformRng` is per-instance, but if shared between threads the state is corrupted.

**Mitigation strategy:** each thread owns a complete *clone* of the market data graph + instruments + engines. Built once in a `#pragma omp parallel` region's `firstprivate` or via `omp_get_thread_num()` index into a pre-built array of N\_threads markets. Trade memory for safety.

### 10.3 RNG strategy

Two options:
1. **Mersenne Twister with jump-ahead.** Each thread seeds a separate MT instance with a seed derived from a counter; assign a 2^64-step jump per thread. Reproducible but jump-ahead code is non-trivial.
2. **Counter-based RNG (Philox/Threefry).** Lock-free, stateless, reproducible across thread counts. ~50-line implementation if no external library allowed.

V1: per-thread MT19937_64 with widely-spaced seeds (good enough; document as a limitation).
V2: Philox-4×32 if time permits.

### 10.4 Aggregation

Each thread accumulates a private P&L vector. After the parallel region:
- Concatenate into a single global vector
- Sort
- Take percentile → VaR, tail mean → ES

The sort itself is sequential in V1 (not the bottleneck). V2 stretch: parallel sort via `__gnu_parallel::sort` or a manual parallel merge.

### 10.5 Schedule comparison (the load-balancing story)

The portfolio has heterogeneous per-instrument cost (American options dominate). With `schedule(static)`, a thread that draws an unlucky scenario pays the same cost as any other — there is no per-scenario imbalance because every scenario reprices the *same* portfolio. **So the scenario loop alone won't show load imbalance.**

Where load imbalance *does* appear:
- **Per-instrument loop inside each scenario.** If we collapse to `for (s, i) in scenarios × instruments` and parallelize the joint index space, dynamic scheduling will demonstrably outperform static because instrument cost varies 1000×.
- **Mixed scenario types** (V2): if some scenarios trigger barrier knockouts or American early exercise paths that take longer, dynamic helps.

V1 plan: collapse the (scenario, instrument) loop with `collapse(2)` and benchmark static vs dynamic vs guided. This gives the load-balancing chart for the report.

### 10.6 Stretch — nested parallelism

If single-axis parallelism doesn't saturate 8 cores (it should, with 10K × 180 = 1.8M tasks), enable inner parallelism inside `MCAmericanEngine` over its MC paths. Requires `omp_set_nested(1)` and careful thread budget (e.g. 4 outer × 2 inner). Worth attempting only after Phase 2.5 is solid.

## 11. Experiments

| ID | Experiment | What it shows | Output |
|---|---|---|---|
| E1 | Convergence vs N | MC error O(1/√N) | log-log plot |
| E2 | Sequential profile | Hot path identification | gprof flat + call graph |
| E3 | Strong scaling | Speedup, efficiency, Amdahl serial fraction | speedup curve, efficiency curve |
| E4 | Weak scaling | Synchronization cost as work grows | flat-ish line; deviation = sync cost |
| E5 | Schedule comparison | Load-imbalance gain from `dynamic`/`guided` | bar chart of wall-clock per schedule |
| E6 | Per-thread busy time | Imbalance visualization | histogram |
| E7 | Cache behavior | Miss rate vs threads | `perf stat` table |
| E8 (stretch) | Nested parallelism | Outer × inner trade-off | matrix of times |
| E9 (stretch) | Sequential vs OpenMP correctness | VaR ± MC noise within tolerance | numerical table |

All experiments output CSV. Plotting via Python / matplotlib in a separate `bench/plot.py` script.

## 12. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| QuantLib thread-safety bugs | High | High | Per-thread market clones, no shared mutable state |
| `Settings::evaluationDate` corruption | High | High | Set once, never mutate inside parallel region |
| American MC randomness across threads breaks reproducibility | Medium | Medium | Per-thread RNG with deterministic seeding |
| Build errors in new Examples subdir | Medium | Low | Copy `Examples/CVAIRS/CMakeLists.txt` verbatim, change names |
| Scaling curve flat (Amdahl bound) | Medium | High | Profile first, justify axis choice in report |
| Underestimating American MC cost | Low | Medium | Start with 1K scenarios for iteration, scale up |
| Time overrun on Phase 2 | Medium | High | Ship Phase 1 + simple flat parallel by week N; collapse + nested are stretch |

## 13. Deliverables

- [ ] `ql/experimental/risk/scenarioevaluator.hpp` + `.cpp` — parallel engine inside the library
- [ ] `ql/CMakeLists.txt` updated: OpenMP gate on `ql_library`, new files registered
- [ ] Source code in `Examples/PortfolioVaR/`
- [ ] Sequential binary `PortfolioVaR` (direct loop, no `ScenarioEvaluator`)
- [ ] Parallel binary `PortfolioVaROMP` (uses `ScenarioEvaluator` via the library)
- [ ] Bench harness scripts (shell + python)
- [ ] CSVs of all benchmark runs
- [ ] Plots: convergence, strong scaling, weak scaling, schedule comparison, per-thread busy time
- [ ] Report PDF: literature survey + methodology + results + discussion
- [ ] LinkedIn post draft

## 14. Milestone checklist

### Phase 0 — Setup
- [x] All deps present on Linux VM (Boost, CMake, gcc, gprof, perf) — user confirmed
- [x] Clone QuantLib (done)
- [x] CMake configure + build QuantLib `Release`
- [x] CMake configure + build QuantLib `RelWithDebInfo`
- [x] Run `Examples/BermudanSwaption` end-to-end as smoke test
- [x] Run `Examples/CVAIRS` end-to-end as structural reference

### Phase 1 — Sequential baseline
- [x] Create `Examples/PortfolioVaR/` directory
- [x] Write `CMakeLists.txt` (copy CVAIRS template)
- [x] Hook into `Examples/CMakeLists.txt`
- [x] `PortfolioVaR.cpp` skeleton — links + prints "hello QuantLib"
- [x] `portfolio.{hpp,cpp}` — build 100 bonds
- [x] Add 50 swaps
- [x] Add 20 European options
- [x] Add 10 American options
- [x] Compute and print base NPV totals (sanity check)
- [x] `scenarios.{hpp,cpp}` — 3-factor Normal generator (no correlation V1)
- [x] Market-state shifter — apply factor draw to handles
- [x] Single-scenario revaluation function
- [x] N-scenario sequential loop
- [x] `var_stats.hpp` — sort + percentile + tail mean
- [x] CSV output of P&L distribution + VaR/ES summary
- [x] **Validation:** parametric VaR check on bond-only subset
- [x] **Validation:** Black-Scholes spot check on European options
- [x] **Validation:** convergence plot (E1)
- [x] gprof profile (E2) → confirm hotspot
      - gprof captured 5.3% of runtime (sampling blind to <10ms calls). Top symbols: `shared_ptr::release` 38%, `TermStructure::dayCounter` 27%, `FlatForward::discountImpl` 7%. FD solver invisible due to 10ms tick granularity. Conclusion: runtime is dominated by FD pricing + shared_ptr teardown; `perf` needed for accurate call graph.
- [x] perf cache profile (E7 sequential baseline)
      - Release build, 1000 scenarios: **36.96B cycles / 23.44B instructions → IPC 0.63** (memory-bound). Cache miss rate 4.69% (15.5M misses / 331M refs). Branch misses 37.9M. Wall time 17.65s. Low IPC corroborates that per-scenario pointer-chasing through the QuantLib observer graph stalls the pipeline.
- [x] Document Phase 1 results

### Phase 2 — OpenMP parallelization (in `ql/experimental/risk/`)

#### 2a. Library class scaffolding
- [ ] Audit thread-safety: list all global state and shared handles touched by the sequential path
- [ ] Add `find_package(OpenMP)` in `ql/CMakeLists.txt`; link `OpenMP::OpenMP_CXX` to `ql_library`
- [ ] Add `-DQL_ENABLE_OPENMP` compile definition when OpenMP is found
- [ ] Create `ql/experimental/risk/scenarioevaluator.hpp` — declare `ScenarioEvaluator`, `Config`, `ThreadContext`, `ContextFactory`, `ShockFn`, `Schedule`
- [ ] Create `ql/experimental/risk/scenarioevaluator.cpp` — implementation with `#ifdef QL_ENABLE_OPENMP` guards and sequential fallback
- [ ] Register both files in `ql/CMakeLists.txt` (`QL_HEADERS`, `QL_SOURCES`)
- [ ] Rebuild QuantLib Release; verify `nm libQuantLib.so | grep ScenarioEvaluator`

#### 2b. Parallel loop correctness
- [ ] Constructor: pre-build `nThreads` independent `ThreadContext`s via the factory
- [ ] `run()`: `#pragma omp parallel for` over scenarios, each thread reading its own `ctxs[tid]`
- [ ] Per-thread busy-time instrumentation with `omp_get_wtime()` into a thread-local accumulator
- [ ] Per-thread RNG is a *driver* concern — scenarios are generated once, sequentially, then the evaluator only reads them (deterministic, reproducible)
- [ ] Wire up `Examples/PortfolioVaR/PortfolioVaROMP.cpp` to use the new class: factory lambda calls `buildMarket()` + `buildPortfolio()`; shock lambda applies `scenarios[s]` to the context's quotes
- [ ] **Correctness:** parallel run produces *bit-identical* P&L to sequential for the same seed (E9)

#### 2c. Experiments
- [ ] Strong-scaling sweep: `nThreads = 1, 2, 4, 8, 16, 32, 64` at fixed 10K scenarios (E3)
- [ ] Weak-scaling sweep: `nScenarios = 1K × nThreads` so per-thread work stays constant (E4)
- [ ] Schedule comparison at 8 threads: static / dynamic / guided, chunk 1 / 16 / 64 (E5)
- [ ] Per-thread busy-time histogram from `ScenarioEvaluator::threadBusyTime()` (E6)
- [ ] `perf stat` cache profile under parallel run, compare to sequential (E7)

#### 2d. Stretch
- [ ] **(Stretch)** Alternative parallel form: `collapse(2)` over (scenario, instrument) — exposes intra-scenario load imbalance more sharply
- [ ] **(Stretch)** Nested parallelism: swap American options back to `MCAmericanEngine`, enable `omp_set_nested(1)`, parallelize inner MC paths (E8)
- [ ] **(Stretch)** Philox-4×32 counter-based RNG in the scenario generator
- [ ] **(Stretch)** Cholesky-correlated factor scenarios (V2 model)

### Phase 3 — Experiments + report
- [ ] Re-run all experiments with final code, fresh CSVs
- [ ] `bench/plot.py` produces all figures
- [ ] Write literature survey section (refs in §15)
- [ ] Write methodology section (this plan, condensed)
- [ ] Write results section (figures + tables)
- [ ] Write discussion: what worked, what didn't, what was surprising
- [ ] Write conclusion + future work
- [ ] Compile to PDF
- [ ] Draft LinkedIn post (1 paragraph + headline figure)

## 15. References to pursue

### Assignment-supplied
- "Parallelizing Monte Carlo applications" (assignment PDF link 1 — Springer)
- "Parallel Monte Carlo techniques" (assignment PDF link 2)
- 2014 CSE633 student examples (assignment PDF link 3)

### Domain — Monte Carlo VaR / market risk
- Glasserman, *Monte Carlo Methods in Financial Engineering*, Springer, 2004 — the standard reference
- Jorion, *Value at Risk*, McGraw-Hill — VaR methodology
- BCBS, *Minimum capital requirements for market risk* (FRTB), 2019 — regulatory context
- Longstaff & Schwartz (2001), *Valuing American Options by Simulation: A Simple Least-Squares Approach* — LSM original

### Parallelization of finance codes
- Intel MKL Vector Statistics — *Monte Carlo simulating European options pricing* cookbook
- Several papers on parallelizing QuantLib pricing engines with OpenMP (search ACM/IEEE)
- PARSEC `swaptions` benchmark whitepaper — for technique comparison
- FinanceBench paper (Cavazos lab) — for kernel-level comparison points

### RNG for parallel MC
- Salmon et al. (2011), *Parallel Random Numbers: As Easy as 1, 2, 3* — Philox/Threefry (Random123)
- Matsumoto & Nishimura (1998), MT19937 original

## 16. Open questions — answered by testing

- **Does `MCAmericanEngine` cache calibration state across calls?**
  Sidestepped in V1 by switching to `FdBlackScholesVanillaEngine`. From source inspection: `Instrument` inherits `LazyObject`; each `setValue()` on a linked `SimpleQuote` propagates `update()` through the observer chain, setting `calculated_ = false` on every downstream object. So caching is invalidated correctly on each shock — but that also means every `NPV()` call re-runs the full LSM simulation from scratch. No cross-call state leaks.

- **Cheapest way to shock a yield curve — rebuild from quotes or `ZeroSpreadedTermStructure`?**
  Measured directly (10K iterations each, `-O3`):
  - `SimpleQuote::setValue()` on the base `FlatForward` rate: **0.61 ns/call**
  - `ZeroSpreadedTermStructure` spread `setValue()`: **0.98 ns/call**
  
  **Decision: keep the `SimpleQuote` on the base rate.** It is 1.6× cheaper, and the sum matches (`ZeroSpreadedTermStructure` just adds an extra indirection). Use `ZeroSpreadedTermStructure` only if we later need to keep the unshocked base curve unchanged (e.g. for relative spread attribution), which we don't need for V1 VaR.

- **Best way to time per-thread busy intervals?**
  `omp_get_wtime()` is the correct tool — it returns wall-clock seconds with sub-microsecond resolution and is thread-safe by the OpenMP standard. Pattern: capture `double t = omp_get_wtime()` before the body and add `omp_get_wtime() - t` into a `thread_local` (or `threadBusy_[tid]`) accumulator after. No mutex needed since each thread writes its own slot.

- **European option vol surface: `BlackConstantVol` vs `BlackVarianceSurface`?**
  `BlackConstantVol` (flat) is the right V1 choice. It links directly to the `equityVol` `SimpleQuote`, so a single `setValue()` shocks vol across the whole surface at essentially zero cost. `BlackVarianceSurface` would require a full 2-D grid re-interpolation on every shock, adding complexity and cost with no benefit for a flat-vol scenario model. V2 can upgrade if we add a vol-surface skew factor.

- **Is `Settings::evaluationDate()` thread-local in this build?**
  **No. Verified by both source and test.**
  `QL_ENABLE_SESSIONS` is `#undef`'d in `build-release/ql/config.hpp`. The `Singleton<T>` template therefore uses the non-sessions branch: `static T instance;` — a single global object. A 4-thread test writing different dates produced **147 corruptions** out of 1000 iterations, confirming shared mutable state.
  **Mitigation:** set `Settings::instance().evaluationDate()` exactly once before the parallel region and never touch it inside. The evaluation date is fixed for the whole VaR run anyway, so this is not a constraint in practice.

- **Does `ql_library` build as shared, and is OpenMP compatible with it?**
  **Yes, shared (`libQuantLib.so.1.43.0`).** The current Release build does **not** link `libgomp` (`ldd` shows no OMP dependency). Adding `OpenMP::OpenMP_CXX` to `ql_library` in `ql/CMakeLists.txt` will add `-fopenmp` to the compile flags and cause `libgomp.so` to be recorded as a dependency in the shared library. Any executable linking `ql_library` will automatically pull in `libgomp` — including both `PortfolioVaR` and `PortfolioVaROMP`. The OpenMP runtime initialises lazily on first `#pragma omp` entry, which is well after static init, so there is no ordering hazard. The existing `#pragma omp` code already in `ql/` (gaussian1dswaptionengine, triplebandlinearop) confirms the pattern works.

---

**Status:** Phase 1 complete. Sequential run 131.8s (10K scenarios, base NPV 517,806, VaR99=1,008,960). Validations: parametric VaR PASS (6.4% gap explained by convexity), B-S spot check PASS (machine precision), convergence CSV written (E1). Profiling: IPC 0.63 (memory-bound), 4.69% cache-miss rate, FD solver invisible to gprof — per-scenario pointer-chasing through observer graph stalls pipeline. Phase 2a next: create `ql/experimental/risk/scenarioevaluator.hpp/.cpp`, wire OpenMP into `ql_library`.

