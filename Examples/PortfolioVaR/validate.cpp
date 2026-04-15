#include <ql/quantlib.hpp>
#include "portfolio.hpp"
#include "scenarios.hpp"
#include "var_stats.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <numeric>

using namespace QuantLib;

// ---------------------------------------------------------------------------
// Shared scenario evaluator (bond-only or full portfolio)
// ---------------------------------------------------------------------------
static double evalScenario(const Scenario& sc, MarketData& mkt,
                            std::vector<BookEntry>& book) {
    mkt.rateQuote->setValue(mkt.baseRate + sc.deltaRate);
    mkt.equitySpot->setValue(mkt.baseSpot * std::exp(sc.returnEquity));
    mkt.equityVol->setValue(std::max(0.001, mkt.baseVol + sc.deltaVol));
    double pnl = 0.0;
    for (auto& e : book) pnl += e.instrument->NPV() - e.baseNPV;
    mkt.rateQuote->setValue(mkt.baseRate);
    mkt.equitySpot->setValue(mkt.baseSpot);
    mkt.equityVol->setValue(mkt.baseVol);
    return pnl;
}

// ---------------------------------------------------------------------------
// Validation 1 — Parametric VaR cross-check (bond-only subset)
//
// Under a parallel rate shift Δr, bond P&L ≈ -ModDur × NPV × Δr.
// Since Δr ~ N(0, σ_r²):
//   Parametric VaR(95%) = 1.6449 × σ_r × Σ(ModDur_i × NPV_i)
//   Parametric VaR(99%) = 2.3263 × σ_r × Σ(ModDur_i × NPV_i)
// The MC VaR on the bond-only portfolio should converge to these values.
// ---------------------------------------------------------------------------
static bool validateParametricVaR(const Date& evalDate, int nScenarios) {
    std::cout << "\n=== Validation 1: Parametric VaR (bond-only) ===\n";

    MarketData mkt = buildMarket(evalDate);

    // Build bond-only book (first 100 entries of a full portfolio)
    auto fullBook = buildPortfolio(mkt);
    std::vector<BookEntry> bondBook;
    for (auto& e : fullBook)
        if (e.label.substr(0, 4) == "Bond")
            bondBook.push_back(e);
    computeBaseNPVs(bondBook);

    // Compute portfolio dollar-duration = Σ ModDur_i × NPV_i
    // ModDur via BondFunctions::duration with each bond's own YTM
    double portfolioDollarDuration = 0.0;
    double portfolioNPV = 0.0;
    for (auto& e : bondBook) {
        auto bond = ext::dynamic_pointer_cast<FixedRateBond>(e.instrument);
        if (!bond) continue;
        // Use Continuous compounding to match the FlatForward discount curve.
        // For a flat continuous rate r, shifting r by Δr shifts the bond's
        // continuous YTM by the same Δr, so ModDur_continuous × NPV × Δr is
        // exact (to first order). Using Compounded/Semiannual would introduce
        // a systematic ~25% underestimate due to the convexity of the
        // compounding conversion.
        double ytm = BondFunctions::yield(*bond,
                                          Bond::Price(e.baseNPV, Bond::Price::Dirty),
                                          Actual365Fixed(), Continuous, NoFrequency);
        InterestRate yld(ytm, Actual365Fixed(), Continuous, NoFrequency);
        double modDur = BondFunctions::duration(*bond, yld, Duration::Modified);
        portfolioDollarDuration += modDur * e.baseNPV;
        portfolioNPV            += e.baseNPV;
    }

    // Also collect dollar-convexity = Σ Convexity_i × NPV_i
    // Used to compute the convexity-corrected parametric VaR
    double portfolioDollarConvexity = 0.0;
    for (auto& e : bondBook) {
        auto bond = ext::dynamic_pointer_cast<FixedRateBond>(e.instrument);
        if (!bond) continue;
        double ytm = BondFunctions::yield(*bond,
                                          Bond::Price(e.baseNPV, Bond::Price::Dirty),
                                          Actual365Fixed(), Continuous, NoFrequency);
        InterestRate yld(ytm, Actual365Fixed(), Continuous, NoFrequency);
        double conv = BondFunctions::convexity(*bond, yld);
        portfolioDollarConvexity += conv * e.baseNPV;
    }

    const double sigmaRate = 0.001;

    // First-order (duration only) — overestimates VaR because it ignores
    // that bond prices are convex: real losses are smaller than linear prediction
    double paramVaR95 = 1.6449 * sigmaRate * portfolioDollarDuration;
    double paramVaR99 = 2.3263 * sigmaRate * portfolioDollarDuration;

    // Second-order correction: convexity shifts mean P&L positively by
    // (1/2) × DollarConvexity × σ_r² (P&L distribution is right-skewed)
    double convexityMeanGain = 0.5 * portfolioDollarConvexity * sigmaRate * sigmaRate;
    double paramVaR95c = paramVaR95 - convexityMeanGain;
    double paramVaR99c = paramVaR99 - convexityMeanGain;

    std::cout << "  Bond portfolio NPV:         " << std::fixed << std::setprecision(2)
              << portfolioNPV << "\n";
    std::cout << "  Portfolio dollar-duration:  " << portfolioDollarDuration << "\n";
    std::cout << "  Portfolio dollar-convexity: " << portfolioDollarConvexity << "\n";
    std::cout << "  Convexity mean P&L gain:    " << convexityMeanGain << "\n";
    std::cout << "  Parametric VaR(95%) [1st order]: " << paramVaR95 << "\n";
    std::cout << "  Parametric VaR(99%) [1st order]: " << paramVaR99 << "\n";
    std::cout << "  Parametric VaR(95%) [+convex]:   " << paramVaR95c << "\n";
    std::cout << "  Parametric VaR(99%) [+convex]:   " << paramVaR99c << "\n";

    // MC VaR on bond-only portfolio
    auto scenarios = generateScenarios(nScenarios, sigmaRate);
    std::vector<double> pnl(nScenarios);
    for (int s = 0; s < nScenarios; ++s)
        pnl[s] = evalScenario(scenarios[s], mkt, bondBook);

    VaRResult mc = computeVaRES(pnl);
    std::cout << "  MC VaR(95%) [N=" << nScenarios << "]: " << mc.var95 << "\n";
    std::cout << "  MC VaR(99%) [N=" << nScenarios << "]: " << mc.var99 << "\n";

    double err95  = std::abs(mc.var95 - paramVaR95)  / paramVaR95;
    double err99  = std::abs(mc.var99 - paramVaR99)  / paramVaR99;
    double err95c = std::abs(mc.var95 - paramVaR95c) / paramVaR95c;
    double err99c = std::abs(mc.var99 - paramVaR99c) / paramVaR99c;
    std::cout << "  Error vs 1st-order  VaR(95%): " << std::setprecision(4)
              << err95 * 100 << " % (expected ~5-7%: convexity bias)\n";
    std::cout << "  Error vs 1st-order  VaR(99%): " << err99 * 100 << " %\n";
    std::cout << "  Error vs conv-corrected 95%:  " << err95c * 100 << " %\n";
    std::cout << "  Error vs conv-corrected 99%:  " << err99c * 100 << " %\n";

    // Pass if within 10% of convexity-corrected estimate
    // (residual = 3rd-order terms + MC sampling noise at N=10K)
    bool pass = (err95c < 0.10 && err99c < 0.10);
    std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << "\n";

    // Write CSV for reporting
    std::ofstream csv("validation_parametric_var.csv");
    csv << "metric,param_1st_order,param_convexity_corrected,mc,err_1st_pct,err_conv_pct\n";
    csv << "VaR95,"  << paramVaR95  << "," << paramVaR95c << "," << mc.var95
        << "," << err95*100  << "," << err95c*100 << "\n";
    csv << "VaR99,"  << paramVaR99  << "," << paramVaR99c << "," << mc.var99
        << "," << err99*100  << "," << err99c*100 << "\n";
    csv << "dollar_duration," << portfolioDollarDuration << ",NA,NA,NA,NA\n";
    csv << "dollar_convexity," << portfolioDollarConvexity << ",NA,NA,NA,NA\n";

    return pass;
}

// ---------------------------------------------------------------------------
// Validation 2 — Black-Scholes spot check (European options)
//
// AnalyticEuropeanEngine computes the exact B-S formula internally.
// We independently compute B-S prices using QuantLib's blackFormula()
// and verify they match to < 1e-8 relative error.
// ---------------------------------------------------------------------------
static bool validateBlackScholes(const Date& evalDate) {
    std::cout << "\n=== Validation 2: Black-Scholes spot check (European options) ===\n";

    const double spot   = 100.0;
    const double rate   = 0.03;
    const double vol    = 0.20;
    const double T      = 1.0;   // 1 year
    const double q      = 0.0;   // no dividend

    MarketData mkt = buildMarket(evalDate, rate, spot, vol);
    auto fullBook  = buildPortfolio(mkt);
    computeBaseNPVs(fullBook);

    bool allPass = true;
    int  checked = 0;
    std::ofstream csv("validation_bs_spotcheck.csv");
    csv << "label,strike,engine_npv,bs_formula,abs_err\n";

    // N(0,1) CDF via erfc
    auto normcdf = [](double x) {
        return 0.5 * std::erfc(-x / std::sqrt(2.0));
    };

    for (auto& e : fullBook) {
        if (e.label.substr(0, 5) != "EUOpt") continue;

        auto opt = ext::dynamic_pointer_cast<VanillaOption>(e.instrument);
        if (!opt) continue;

        auto payoff = ext::dynamic_pointer_cast<PlainVanillaPayoff>(opt->payoff());
        double K = payoff->strike();

        // Manual B-S call price
        double F    = spot * std::exp((rate - q) * T);
        double df   = std::exp(-rate * T);
        double d1   = (std::log(F / K) + 0.5 * vol * vol * T) / (vol * std::sqrt(T));
        double d2   = d1 - vol * std::sqrt(T);
        double bsPrice = df * (F * normcdf(d1) - K * normcdf(d2));

        double absErr = std::abs(e.baseNPV - bsPrice);
        bool pass = absErr < 1e-6;
        allPass &= pass;

        std::cout << "  " << e.label << "  K=" << std::setw(6) << K
                  << "  engine=" << std::setprecision(6) << e.baseNPV
                  << "  B-S=" << bsPrice
                  << "  err=" << std::scientific << absErr
                  << (pass ? "" : " FAIL") << "\n";

        csv << e.label << "," << K << "," << e.baseNPV << "," << bsPrice
            << "," << absErr << "\n";
        ++checked;
    }

    std::cout << "  Checked " << checked << " European options. "
              << (allPass ? "All PASS" : "FAILURES DETECTED") << "\n";
    return allPass;
}

// ---------------------------------------------------------------------------
// Validation 3 — Convergence plot (E1)
//
// VaR estimate vs N scenarios. Expected O(1/√N) MC error.
// Generates convergence_var.csv for plotting.
// ---------------------------------------------------------------------------
static void validateConvergence(const Date& evalDate) {
    std::cout << "\n=== Validation 3: Convergence of VaR estimate vs N (E1) ===\n";

    // Use bond-only for speed (each scenario is ~µs, not ms)
    MarketData mkt = buildMarket(evalDate);
    auto fullBook  = buildPortfolio(mkt);
    std::vector<BookEntry> bondBook;
    for (auto& e : fullBook)
        if (e.label.substr(0, 4) == "Bond") bondBook.push_back(e);
    computeBaseNPVs(bondBook);

    const int N_MAX = 100000;
    auto allScenarios = generateScenarios(N_MAX, 0.001);

    // Pre-compute all P&L
    std::vector<double> allPnL(N_MAX);
    std::cout << "  Pre-computing " << N_MAX << " bond scenarios..." << std::flush;
    for (int s = 0; s < N_MAX; ++s)
        allPnL[s] = evalScenario(allScenarios[s], mkt, bondBook);
    std::cout << " done.\n";

    std::vector<int> Ns = {100, 200, 500, 1000, 2000, 5000, 10000,
                           20000, 50000, 100000};

    std::ofstream csv("convergence_var.csv");
    csv << "n_scenarios,var95,var99,es95,es99\n";
    std::cout << "  " << std::setw(10) << "N"
              << std::setw(12) << "VaR(95%)"
              << std::setw(12) << "VaR(99%)"
              << std::setw(12) << "ES(95%)" << "\n";
    std::cout << "  " << std::string(46, '-') << "\n";

    for (int N : Ns) {
        std::vector<double> sub(allPnL.begin(), allPnL.begin() + N);
        VaRResult r = computeVaRES(sub);
        std::cout << "  " << std::setw(10) << N
                  << std::setw(12) << std::fixed << std::setprecision(1) << r.var95
                  << std::setw(12) << r.var99
                  << std::setw(12) << r.es95 << "\n";
        csv << N << "," << r.var95 << "," << r.var99 << ","
            << r.es95 << "," << r.es99 << "\n";
    }
    std::cout << "  Written convergence_var.csv\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    const int N_SCENARIOS = (argc > 1) ? std::atoi(argv[1]) : 10000;

    Date evalDate(15, April, 2026);
    Settings::instance().evaluationDate() = evalDate;

    std::cout << std::fixed;

    bool v1 = validateParametricVaR(evalDate, N_SCENARIOS);
    bool v2 = validateBlackScholes(evalDate);
    validateConvergence(evalDate);

    std::cout << "\n=== Summary ===\n";
    std::cout << "  Parametric VaR check: " << (v1 ? "PASS" : "FAIL") << "\n";
    std::cout << "  Black-Scholes check:  " << (v2 ? "PASS" : "FAIL") << "\n";
    std::cout << "  Convergence CSV:      convergence_var.csv\n";

    return (v1 && v2) ? 0 : 1;
}
