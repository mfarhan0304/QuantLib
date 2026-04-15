#include "portfolio.hpp"
#include <ql/quantlib.hpp>
#include <cmath>
#include <string>

using namespace QuantLib;

// ---------------------------------------------------------------------------
// buildMarket
// ---------------------------------------------------------------------------
MarketData buildMarket(const Date& evalDate,
                       double flatRate,
                       double equitySpot,
                       double equityVol) {
    MarketData m;
    m.evalDate   = evalDate;
    m.baseRate   = flatRate;
    m.baseSpot   = equitySpot;
    m.baseVol    = equityVol;

    // Rate quote — single shockable handle
    m.rateQuote = ext::make_shared<SimpleQuote>(flatRate);
    auto flatCurve = ext::make_shared<FlatForward>(
        evalDate,
        Handle<Quote>(m.rateQuote),
        Actual365Fixed());
    flatCurve->enableExtrapolation();
    m.discountCurve = Handle<YieldTermStructure>(flatCurve);
    m.forecastCurve = m.discountCurve;
    m.riskFreeRate  = m.discountCurve;

    // Equity
    m.equitySpot = ext::make_shared<SimpleQuote>(equitySpot);
    m.equityVol  = ext::make_shared<SimpleQuote>(equityVol);

    auto volTs = ext::make_shared<BlackConstantVol>(
        evalDate, NullCalendar(),
        Handle<Quote>(m.equityVol),
        Actual365Fixed());
    m.volSurface = Handle<BlackVolTermStructure>(volTs);

    return m;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static ext::shared_ptr<Instrument> makeBond(int i, const MarketData& m) {
    // Fixed-rate bullet bond, 5-year, semi-annual, face 1000
    Calendar cal = TARGET();
    Date issue = m.evalDate;
    Date maturity = issue + Period(5, Years);
    Schedule sch(issue, maturity, Period(Semiannual), cal,
                 ModifiedFollowing, ModifiedFollowing,
                 DateGeneration::Forward, false);

    double coupon = 0.03 + 0.001 * (i % 5); // 3.0 – 3.4 %
    auto bond = ext::make_shared<FixedRateBond>(
        0, 1000.0, sch, std::vector<Rate>{coupon}, Actual365Fixed());

    auto engine = ext::make_shared<DiscountingBondEngine>(m.discountCurve);
    bond->setPricingEngine(engine);
    return bond;
}

static ext::shared_ptr<Instrument> makeSwap(int i, const MarketData& m) {
    Calendar cal = TARGET();
    Date start = m.evalDate + 2;
    Date end   = start + Period(10, Years);
    Schedule fixedSch(start, end, Period(Annual),   cal,
                      ModifiedFollowing, ModifiedFollowing,
                      DateGeneration::Forward, false);
    Schedule floatSch(start, end, Period(Semiannual), cal,
                      ModifiedFollowing, ModifiedFollowing,
                      DateGeneration::Forward, false);

    double fixedRate = 0.025 + 0.001 * (i % 10);
    auto idx = ext::make_shared<Euribor6M>(m.forecastCurve);

    auto swap = ext::make_shared<VanillaSwap>(
        VanillaSwap::Payer, 1.0e6,
        fixedSch, fixedRate, Thirty360(Thirty360::BondBasis),
        floatSch, idx, 0.0, Actual360());

    auto engine = ext::make_shared<DiscountingSwapEngine>(m.discountCurve);
    swap->setPricingEngine(engine);
    return swap;
}

static ext::shared_ptr<Instrument> makeEuropeanOption(int i, const MarketData& m) {
    auto spot = Handle<Quote>(m.equitySpot);
    auto qYield = Handle<YieldTermStructure>(
        ext::make_shared<FlatForward>(m.evalDate, 0.0, Actual365Fixed()));
    auto divYield = qYield;

    auto process = ext::make_shared<BlackScholesMertonProcess>(
        spot, divYield, m.riskFreeRate, m.volSurface);

    double strike = m.baseSpot * (0.90 + 0.01 * (i % 20)); // 90–109% moneyness
    Date expiry   = m.evalDate + Period(1, Years);
    auto option = ext::make_shared<VanillaOption>(
        ext::make_shared<PlainVanillaPayoff>(Option::Call, strike),
        ext::make_shared<EuropeanExercise>(expiry));

    option->setPricingEngine(
        ext::make_shared<AnalyticEuropeanEngine>(process));
    return option;
}

static ext::shared_ptr<Instrument> makeAmericanOption(int i, const MarketData& m) {
    auto spot = Handle<Quote>(m.equitySpot);
    auto qYield = Handle<YieldTermStructure>(
        ext::make_shared<FlatForward>(m.evalDate, 0.0, Actual365Fixed()));

    auto process = ext::make_shared<BlackScholesMertonProcess>(
        spot, qYield, m.riskFreeRate, m.volSurface);

    double strike = m.baseSpot * (0.95 + 0.01 * (i % 10));
    Date expiry   = m.evalDate + Period(1, Years);
    auto option = ext::make_shared<VanillaOption>(
        ext::make_shared<PlainVanillaPayoff>(Option::Put, strike),
        ext::make_shared<AmericanExercise>(m.evalDate, expiry));

    // FD Black-Scholes: ~1ms per reprice — creates clear load imbalance vs
    // bonds/swaps (~µs) without the per-call MC simulation overhead of LSM.
    // For Phase 2 stretch, swap back to MCAmericanEngine to add a nested-
    // parallelism axis over MC paths.
    // 50×50 FD grid: ~0.5ms per reprice, ~500× slower than bonds/swaps.
    // Increase to 200×200 for final accuracy benchmarks.
    option->setPricingEngine(
        ext::make_shared<FdBlackScholesVanillaEngine>(process, 50, 50));
    return option;
}

// ---------------------------------------------------------------------------
// buildPortfolio
// ---------------------------------------------------------------------------
std::vector<BookEntry> buildPortfolio(const MarketData& market) {
    std::vector<BookEntry> book;
    book.reserve(180);

    for (int i = 0; i < 100; ++i) {
        BookEntry e;
        e.instrument = makeBond(i, market);
        e.label = "Bond_" + std::to_string(i);
        book.push_back(std::move(e));
    }
    for (int i = 0; i < 50; ++i) {
        BookEntry e;
        e.instrument = makeSwap(i, market);
        e.label = "Swap_" + std::to_string(i);
        book.push_back(std::move(e));
    }
    for (int i = 0; i < 20; ++i) {
        BookEntry e;
        e.instrument = makeEuropeanOption(i, market);
        e.label = "EUOpt_" + std::to_string(i);
        book.push_back(std::move(e));
    }
    for (int i = 0; i < 10; ++i) {
        BookEntry e;
        e.instrument = makeAmericanOption(i, market);
        e.label = "AMOpt_" + std::to_string(i);
        book.push_back(std::move(e));
    }
    return book;
}

// ---------------------------------------------------------------------------
// computeBaseNPVs
// ---------------------------------------------------------------------------
void computeBaseNPVs(std::vector<BookEntry>& book) {
    for (auto& e : book)
        e.baseNPV = e.instrument->NPV();
}
