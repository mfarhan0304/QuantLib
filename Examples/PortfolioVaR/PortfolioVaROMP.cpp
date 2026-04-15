// PortfolioVaROMP.cpp — OpenMP parallel version using ScenarioEvaluator
//
// Architecture:
//   - Scenarios are generated once, sequentially, with the same seed as the
//     sequential driver → identical P&L values (E9 correctness check).
//   - ScenarioEvaluator pre-builds one independent (market + book) clone per
//     thread.  No shared mutable state inside the parallel region.
//   - Settings::evaluationDate() is set once before ScenarioEvaluator is
//     constructed and never changed again.

#include <ql/quantlib.hpp>
#include <ql/experimental/risk/scenarioevaluator.hpp>

#include "portfolio.hpp"
#include "scenarios.hpp"
#include "var_stats.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace QuantLib;

// ---------------------------------------------------------------------------
// Build one ScenarioThreadContext (a self-contained market + book clone)
// ---------------------------------------------------------------------------
static ScenarioThreadContext makeContext(const Date& evalDate) {
    // Each context has its own MarketData with independent quotes/handles.
    MarketData mkt = buildMarket(evalDate);

    auto book = buildPortfolio(mkt);

    ScenarioThreadContext ctx;
    ctx.quotes = { mkt.rateQuote, mkt.equitySpot, mkt.equityVol };

    ctx.instruments.reserve(book.size());
    for (auto& e : book)
        ctx.instruments.push_back(e.instrument);

    return ctx;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    const int N_SCENARIOS = (argc > 1) ? std::atoi(argv[1]) : 10000;
    const int N_THREADS   = (argc > 2) ? std::atoi(argv[2]) : 0; // 0=max
    // schedule: 0=dynamic (default), 1=static, 2=guided
    const int SCHED_IDX   = (argc > 3) ? std::atoi(argv[3]) : 0;
    // chunk size for dynamic/guided
    const int CHUNK       = (argc > 4) ? std::atoi(argv[4]) : 16;

#ifdef _OPENMP
    std::cout << "OpenMP enabled. Available threads: " << omp_get_max_threads() << "\n";
#else
    std::cout << "OpenMP NOT enabled — running sequential fallback.\n";
#endif

    const Date evalDate(15, April, 2026);
    Settings::instance().evaluationDate() = evalDate;

    // --- Base NPVs (needed by ScenarioEvaluator) ---
    // Build one reference book to compute base NPVs.
    MarketData refMkt  = buildMarket(evalDate);
    auto       refBook = buildPortfolio(refMkt);
    computeBaseNPVs(refBook);

    double totalBaseNPV = 0.0;
    for (auto& e : refBook) totalBaseNPV += e.baseNPV;
    std::cout << "Portfolio base NPV: " << std::fixed << std::setprecision(2)
              << totalBaseNPV << "\n";

    std::vector<Real> baseNPVs;
    baseNPVs.reserve(refBook.size());
    for (auto& e : refBook) baseNPVs.push_back(e.baseNPV);

    // --- Scenarios (generated sequentially, same seed as PortfolioVaR) ---
    std::cout << "Generating " << N_SCENARIOS << " scenarios...\n";
    auto scenarios = generateScenarios(N_SCENARIOS);

    // --- Build ScenarioEvaluator ---
    ScenarioEvaluatorConfig cfg;
    cfg.nThreads  = static_cast<Size>(N_THREADS);
    cfg.chunkSize = static_cast<Size>(CHUNK);
    switch (SCHED_IDX) {
        case 1:  cfg.schedule = ScenarioSchedule::Static;  break;
        case 2:  cfg.schedule = ScenarioSchedule::Guided;  break;
        default: cfg.schedule = ScenarioSchedule::Dynamic; break;
    }

    const char* schedNames[] = { "dynamic", "static", "guided" };
    std::cout << "Schedule: " << schedNames[SCHED_IDX < 3 ? SCHED_IDX : 0]
              << "  chunk=" << CHUNK
              << "  nThreads=" << (N_THREADS ? N_THREADS : -1) << " (−1 = max)\n";

    // evalDate must be set before factory runs (called inside constructor)
    ScenarioEvaluator ev(
        [&]() { return makeContext(evalDate); },
        baseNPVs,
        cfg);

    std::cout << "Threads used: " << ev.nThreads() << "\n";
    std::cout << "Running " << N_SCENARIOS << " scenarios in parallel...\n";
    std::cout.flush();

    // --- Shock function (thread-safe: only touches ctx.quotes) ---
    const double baseRate = refMkt.baseRate;
    const double baseSpot = refMkt.baseSpot;
    const double baseVol  = refMkt.baseVol;

    auto shockFn = [&](ScenarioThreadContext& ctx, Size s) {
        // quotes[0]=rateQuote, [1]=equitySpot, [2]=equityVol
        ctx.quotes[0]->setValue(baseRate + scenarios[s].deltaRate);
        ctx.quotes[1]->setValue(baseSpot * std::exp(scenarios[s].returnEquity));
        ctx.quotes[2]->setValue(std::max(0.001, baseVol + scenarios[s].deltaVol));
    };

    // --- Parallel run ---
    auto pnlReal = ev.run(static_cast<Size>(N_SCENARIOS), shockFn);

    std::cout << "Wall time: " << std::setprecision(4) << ev.wallTime() << " s\n";
    std::cout << "Throughput: " << std::setprecision(1)
              << N_SCENARIOS / ev.wallTime() << " scenarios/s\n";

    // --- Per-thread busy time ---
    const auto& busy = ev.threadBusyTime();
    double totalBusy = std::accumulate(busy.begin(), busy.end(), 0.0);
    double efficiency = (ev.wallTime() > 0)
                        ? totalBusy / (ev.wallTime() * ev.nThreads()) * 100.0
                        : 0.0;
    std::cout << "Thread efficiency: " << std::setprecision(1)
              << efficiency << "%\n";

    // Convert to double for VaR stats
    std::vector<double> pnl(pnlReal.begin(), pnlReal.end());
    VaRResult result = computeVaRES(pnl);

    std::cout << "\n=== VaR / ES Results ===\n";
    std::cout << "VaR(95%): " << std::fixed << std::setprecision(0)
              << result.var95 << "\n";
    std::cout << "VaR(99%): " << result.var99 << "\n";
    std::cout << "ES(95%):  " << result.es95  << "\n";
    std::cout << "ES(99%):  " << result.es99  << "\n";

    // --- CSV output ---
    {
        std::ofstream f("pnl_distribution_omp.csv");
        f << "pnl\n";
        for (double v : pnl) f << v << "\n";
    }
    {
        std::ofstream f("var_summary_omp.csv");
        f << "metric,value\n";
        f << "nScenarios,"  << N_SCENARIOS       << "\n";
        f << "nThreads,"    << ev.nThreads()      << "\n";
        f << "schedule,"    << schedNames[SCHED_IDX < 3 ? SCHED_IDX : 0] << "\n";
        f << "chunkSize,"   << CHUNK              << "\n";
        f << "wallTime,"    << ev.wallTime()       << "\n";
        f << "var95,"       << result.var95        << "\n";
        f << "var99,"       << result.var99        << "\n";
        f << "es95,"        << result.es95         << "\n";
        f << "es99,"        << result.es99         << "\n";
        f << "efficiency_pct," << efficiency       << "\n";
    }
    {
        std::ofstream f("thread_busy_omp.csv");
        f << "thread_id,busy_s\n";
        for (Size t = 0; t < busy.size(); ++t)
            f << t << "," << busy[t] << "\n";
    }

    std::cout << "Output written to pnl_distribution_omp.csv, "
                 "var_summary_omp.csv, thread_busy_omp.csv\n";
    return 0;
}
