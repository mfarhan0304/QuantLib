#pragma once
#include <ql/quantlib.hpp>
#include <vector>
#include <memory>

using namespace QuantLib;

struct MarketData {
    // Evaluation date
    Date evalDate;

    // Rates market: flat yield curve shockable via this quote
    ext::shared_ptr<SimpleQuote> rateQuote;         // base flat rate (e.g. 0.03)
    Handle<YieldTermStructure> discountCurve;
    Handle<YieldTermStructure> forecastCurve;

    // Equity market
    ext::shared_ptr<SimpleQuote> equitySpot;        // S0
    ext::shared_ptr<SimpleQuote> equityVol;         // flat Black vol
    Handle<BlackVolTermStructure> volSurface;

    // Risk-free rate handle (also used as equity discount/div)
    Handle<YieldTermStructure> riskFreeRate;

    // Base factor values (for restoring after shock)
    double baseRate;
    double baseSpot;
    double baseVol;
};

struct BookEntry {
    ext::shared_ptr<Instrument> instrument;
    double baseNPV = 0.0;
    std::string label;
};

// Build the base market data object (flat curves, constant vol)
MarketData buildMarket(const Date& evalDate,
                       double flatRate   = 0.03,
                       double equitySpot = 100.0,
                       double equityVol  = 0.20);

// Build the portfolio: 100 bonds, 50 swaps, 20 EU opts, 10 AM opts
// All instruments are priced against the handles inside market.
std::vector<BookEntry> buildPortfolio(const MarketData& market);

// Price all instruments and store base NPVs
void computeBaseNPVs(std::vector<BookEntry>& book);
