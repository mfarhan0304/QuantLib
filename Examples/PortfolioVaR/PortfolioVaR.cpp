#include <ql/quantlib.hpp>
#include "portfolio.hpp"
#include "scenarios.hpp"
#include "var_stats.hpp"

#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <numeric>

using namespace QuantLib;

// Apply a scenario shock to the market via the SimpleQuotes,
// then compute portfolio P&L vs base NPVs.
// Restores market state before returning.
static double evalScenario(const Scenario& sc,
                            MarketData& mkt,
                            std::vector<BookEntry>& book) {
    // Shock
    mkt.rateQuote->setValue(mkt.baseRate + sc.deltaRate);
    mkt.equitySpot->setValue(mkt.baseSpot * std::exp(sc.returnEquity));
    mkt.equityVol->setValue(std::max(0.001, mkt.baseVol + sc.deltaVol));

    double pnl = 0.0;
    for (auto& e : book)
        pnl += e.instrument->NPV() - e.baseNPV;

    // Restore
    mkt.rateQuote->setValue(mkt.baseRate);
    mkt.equitySpot->setValue(mkt.baseSpot);
    mkt.equityVol->setValue(mkt.baseVol);

    return pnl;
}

int main(int argc, char* argv[]) {
    const int N_SCENARIOS = (argc > 1) ? std::atoi(argv[1]) : 10000;

    // --- Setup ---
    Date evalDate(15, April, 2026);
    Settings::instance().evaluationDate() = evalDate;

    std::cout << "Building market and portfolio..." << std::flush;
    MarketData market = buildMarket(evalDate);
    auto book = buildPortfolio(market);
    computeBaseNPVs(book);
    std::cout << " done.\n";

    double totalBaseNPV = 0.0;
    for (auto& e : book) totalBaseNPV += e.baseNPV;
    std::cout << "Portfolio base NPV: " << totalBaseNPV << "\n";
    std::cout << "  Instruments: " << book.size() << "\n";

    using clk = std::chrono::steady_clock;

    // --- Stage 1: Scenario generation (RNG + distribution transform) ---
    auto ts0 = clk::now();
    auto scenarios = generateScenarios(N_SCENARIOS);
    double t_rng = std::chrono::duration<double>(clk::now() - ts0).count();
    std::cout << "Stage 1 (scenario gen):    " << t_rng << " s\n";

    // --- Stage 2: Portfolio revaluation ---
    std::cout << "Running sequential revaluation..." << std::flush;
    auto ts1 = clk::now();

    std::vector<double> pnl(N_SCENARIOS);
    for (int s = 0; s < N_SCENARIOS; ++s) {
        pnl[s] = evalScenario(scenarios[s], market, book);
        if ((s + 1) % 100 == 0)
            std::cout << "  " << (s + 1) << "/" << N_SCENARIOS << "\n" << std::flush;
    }

    double t_reval = std::chrono::duration<double>(clk::now() - ts1).count();
    double elapsed  = t_rng + t_reval;
    std::cout << "Stage 2 (revaluation):     " << t_reval << " s\n";

    // --- Stage 3: Tail statistics (sort + percentile) ---
    auto ts2 = clk::now();
    VaRResult r = computeVaRES(pnl);
    double t_stats = std::chrono::duration<double>(clk::now() - ts2).count();
    std::cout << "Stage 3 (tail statistics): " << t_stats << " s\n";
    std::cout << "Total time:                " << elapsed  << " s\n";
    std::cout << "\n=== VaR / ES Results ===\n";
    std::cout << "VaR(95%): " << r.var95 << "\n";
    std::cout << "VaR(99%): " << r.var99 << "\n";
    std::cout << "ES(95%):  " << r.es95  << "\n";
    std::cout << "ES(99%):  " << r.es99  << "\n";

    // --- CSV output ---
    {
        std::ofstream csv("pnl_distribution.csv");
        csv << "pnl\n";
        for (double v : pnl) csv << v << "\n";
    }
    {
        std::ofstream csv("var_summary.csv");
        csv << "metric,value\n";
        csv << "var95," << r.var95 << "\n";
        csv << "var99," << r.var99 << "\n";
        csv << "es95,"  << r.es95  << "\n";
        csv << "es99,"  << r.es99  << "\n";
        csv << "time_s," << elapsed << "\n";
        csv << "n_scenarios," << N_SCENARIOS << "\n";
        csv << "n_instruments," << book.size() << "\n";
    }
    {
        std::ofstream csv("stage_timing_seq.csv");
        csv << "stage,time_s,pct\n";
        double total = t_rng + t_reval + t_stats;
        csv << "rng_scenario_gen," << t_rng   << "," << 100.0*t_rng/total   << "\n";
        csv << "portfolio_reval,"  << t_reval << "," << 100.0*t_reval/total << "\n";
        csv << "tail_statistics,"  << t_stats << "," << 100.0*t_stats/total << "\n";
        csv << "total,"            << total   << ",100\n";
    }
    std::cout << "Output written to pnl_distribution.csv, var_summary.csv, stage_timing_seq.csv\n";

    return 0;
}
