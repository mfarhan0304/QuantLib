// PortfolioVaROMP.cpp — OpenMP parallel version (Phase 2)
// Placeholder: currently builds and runs the sequential path to verify
// the build target compiles with OpenMP flags before Phase 2 implementation.

#include <ql/quantlib.hpp>
#include "portfolio.hpp"
#include "scenarios.hpp"
#include "var_stats.hpp"

#include <iostream>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace QuantLib;

int main() {
#ifdef _OPENMP
    std::cout << "OpenMP enabled. Max threads: " << omp_get_max_threads() << "\n";
#else
    std::cout << "OpenMP NOT enabled — recheck CMake flags.\n";
#endif

    Date evalDate(15, April, 2026);
    Settings::instance().evaluationDate() = evalDate;

    MarketData market = buildMarket(evalDate);
    auto book = buildPortfolio(market);
    computeBaseNPVs(book);

    double totalBaseNPV = 0.0;
    for (auto& e : book) totalBaseNPV += e.baseNPV;
    std::cout << "Portfolio base NPV: " << totalBaseNPV << "\n";
    std::cout << "Phase 2 (parallel revaluation) not yet implemented.\n";
    return 0;
}
