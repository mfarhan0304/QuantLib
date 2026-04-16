/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*!
 Copyright (C) 2002, 2003 Sadruddin Rejeb
 Copyright (C) 2004 Ferdinando Ametrano
 Copyright (C) 2005, 2006, 2007 StatPro Italia srl

 This file is part of QuantLib, a free-software/open-source library
 for financial quantitative analysts and developers - http://quantlib.org/

 QuantLib is free software: you can redistribute it and/or modify it
 under the terms of the QuantLib license.  You should have received a
 copy of the license along with this program; if not, please email
 <quantlib-dev@lists.sf.net>. The license is also available online at
 <https://www.quantlib.org/license.shtml>.

 This program is distributed in the hope that it will be useful, but WITHOUT
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE.  See the license for more details.
*/

/*
 Parallel Jacobian variant of the shipped BermudanSwaption example.

 The program is structurally identical to Examples/BermudanSwaption.cpp:
 it calibrates G2, HullWhite (analytic), HullWhite (numerical) and
 BlackKarasinski to the co-terminal swaption grid, prints implied
 Black vols per helper, reports calibrated parameters, and then prices
 ATM/OTM/ITM Bermudan swaptions with both tree and FDM engines across
 all four models.

 The only change is the calibration step itself: instead of
 `model->calibrate(helpers, LevenbergMarquardt, endCriteria)` (which
 drives the shipped serial `fdjac2`), each call is routed through
 `calibrateModelParallel<>()`.  The latter sets up N independent
 (curve, model, helpers, cost-fn, problem) clones and registers them on
 `LevenbergMarquardt::setParallelProblems()` so the finite-difference
 Jacobian inside `lmdif` is evaluated by `fdjac2_parallel` across
 OpenMP threads.  When `-seq` is passed, or when OMP is unavailable,
 the serial code path is used instead.

 After calibration, the caller-supplied model has the same parameters
 it would have under the shipped serial path (bit-identical at every
 thread count tested), so the downstream pretty-printing and Bermudan
 pricing blocks see a correctly calibrated state.
*/

#include <ql/qldefines.hpp>
#if !defined(BOOST_ALL_NO_LIB) && defined(BOOST_MSVC)
#  include <ql/auto_link.hpp>
#endif
#include <ql/instruments/vanillaswap.hpp>
#include <ql/instruments/swaption.hpp>
#include <ql/pricingengines/swap/discountingswapengine.hpp>
#include <ql/pricingengines/swaption/treeswaptionengine.hpp>
#include <ql/pricingengines/swaption/jamshidianswaptionengine.hpp>
#include <ql/pricingengines/swaption/g2swaptionengine.hpp>
#include <ql/pricingengines/swaption/fdhullwhiteswaptionengine.hpp>
#include <ql/pricingengines/swaption/fdg2swaptionengine.hpp>
#include <ql/models/shortrate/calibrationhelpers/swaptionhelper.hpp>
#include <ql/models/shortrate/twofactormodels/g2.hpp>
#include <ql/models/shortrate/onefactormodels/hullwhite.hpp>
#include <ql/models/shortrate/onefactormodels/blackkarasinski.hpp>
#include <ql/math/optimization/levenbergmarquardt.hpp>
#include <ql/math/optimization/lmdif.hpp>
#include <ql/math/optimization/problem.hpp>
#include <ql/math/optimization/constraint.hpp>
#include <ql/math/optimization/costfunction.hpp>
#include <ql/math/optimization/endcriteria.hpp>
#include <ql/indexes/ibor/euribor.hpp>
#include <ql/cashflows/coupon.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/thirty360.hpp>
#include <ql/utilities/dataformatters.hpp>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <memory>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace QuantLib;

// Number of swaptions to be calibrated to...

Size numRows = 5;
Size numCols = 5;

Integer swapLengths[] = {
      1,     2,     3,     4,     5};
Volatility swaptionVols[] = {
  0.1490, 0.1340, 0.1228, 0.1189, 0.1148,
  0.1290, 0.1201, 0.1146, 0.1108, 0.1040,
  0.1149, 0.1112, 0.1070, 0.1010, 0.0957,
  0.1047, 0.1021, 0.0980, 0.0951, 0.1270,
  0.1000, 0.0950, 0.0900, 0.1230, 0.1160};

namespace {

// Public-API mirror of CalibratedModel::CalibrationFunction — the
// library's version is a private nested class, but the public
// CostFunction interface is enough for us to build an LM problem.
class PublicCalibrationFunction : public CostFunction {
  public:
    PublicCalibrationFunction(
        CalibratedModel* model,
        std::vector<ext::shared_ptr<CalibrationHelper>> helpers,
        std::vector<Real> weights)
    : model_(model), helpers_(std::move(helpers)), weights_(std::move(weights)) {}

    Real value(const Array& params) const override {
        model_->setParams(params);
        Real v = 0.0;
        for (Size i = 0; i < helpers_.size(); ++i) {
            Real diff = helpers_[i]->calibrationError();
            v += diff * diff * weights_[i];
        }
        return std::sqrt(v);
    }

    Array values(const Array& params) const override {
        model_->setParams(params);
        Array out(helpers_.size());
        for (Size i = 0; i < helpers_.size(); ++i)
            out[i] = helpers_[i]->calibrationError() * std::sqrt(weights_[i]);
        return out;
    }

  private:
    CalibratedModel* model_;
    std::vector<ext::shared_ptr<CalibrationHelper>> helpers_;
    std::vector<Real> weights_;
};

// Market-data factory for cloning. Each call yields a disjoint
// SimpleQuote + FlatForward pair so the per-thread observer graphs
// don't alias.
struct MarketFactory {
    Date settlementDate;
    Rate flatRate;

    Handle<YieldTermStructure> buildCurve() const {
        auto quote = ext::make_shared<SimpleQuote>(flatRate);
        return Handle<YieldTermStructure>(
            ext::make_shared<FlatForward>(settlementDate,
                                          Handle<Quote>(quote),
                                          Actual365Fixed()));
    }
};

// One thread-local calibration context: owns its own yield curve,
// model clone, swaption helpers (with engines bound to the clone),
// cost function and Problem.
template <typename Model>
struct CalibContext {
    Handle<YieldTermStructure> curve;
    ext::shared_ptr<Model> model;
    std::vector<ext::shared_ptr<CalibrationHelper>> helpers;
    std::unique_ptr<PublicCalibrationFunction> costFn;
    ext::shared_ptr<Constraint> constraint;
    std::unique_ptr<Problem> problem;
};

template <typename EngineFactory>
std::vector<ext::shared_ptr<CalibrationHelper>>
buildSwaptionHelpers(const Handle<YieldTermStructure>& curve,
                     EngineFactory makeEngine) {
    auto index = ext::make_shared<Euribor6M>(curve);
    std::vector<ext::shared_ptr<CalibrationHelper>> out;
    for (Size i = 0; i < numRows; ++i) {
        Size j = numCols - i - 1;
        Size k = i * numCols + j;
        auto vol = ext::make_shared<SimpleQuote>(swaptionVols[k]);
        auto h = ext::make_shared<SwaptionHelper>(
            Period(Integer(i + 1), Years),
            Period(swapLengths[j], Years),
            Handle<Quote>(vol),
            index,
            index->tenor(),
            index->dayCounter(),
            index->dayCounter(),
            curve);
        h->setPricingEngine(makeEngine(curve));
        out.push_back(h);
    }
    return out;
}

template <typename Model, typename ModelFactory, typename EngineFactory>
CalibContext<Model> buildCloneContext(const MarketFactory& mkt,
                                      ModelFactory makeModel,
                                      EngineFactory makeEngine) {
    CalibContext<Model> ctx;
    ctx.curve = mkt.buildCurve();
    ctx.model = makeModel(ctx.curve);
    ctx.helpers = buildSwaptionHelpers(ctx.curve, [&](const Handle<YieldTermStructure>& c) {
        return makeEngine(ctx.model, c);
    });
    std::vector<Real> weights(ctx.helpers.size(), 1.0);
    ctx.costFn = std::make_unique<PublicCalibrationFunction>(
        ctx.model.get(), ctx.helpers, std::move(weights));
    ctx.constraint = ctx.model->constraint();
    Array x0 = ctx.model->params();
    ctx.problem = std::make_unique<Problem>(*ctx.costFn, *ctx.constraint, x0);
    return ctx;
}

// Parallel-aware replacement for `calibrateModel` in the shipped example.
// The caller's `model` and `swaptions` play the role of the "main" problem
// during `lm.minimize`, so calibrated parameters land directly on the
// model passed in and the post-calibration pretty-print and Bermudan
// pricing see the correct state.
template <typename Model, typename ModelFactory, typename EngineFactory>
void calibrateModelParallel(
    const std::string& label,
    const ext::shared_ptr<Model>& model,
    const std::vector<ext::shared_ptr<BlackCalibrationHelper>>& swaptions,
    const MarketFactory& mkt,
    ModelFactory makeModel,
    EngineFactory makeEngine,
    int nThreads,
    bool useSequential)
{
    const int effectiveThreads = useSequential ? 1 : nThreads;

    // Main cost function + problem on the caller's model and helpers.
    std::vector<ext::shared_ptr<CalibrationHelper>> mainHelpers(
        swaptions.begin(), swaptions.end());
    std::vector<Real> mainWeights(mainHelpers.size(), 1.0);
    PublicCalibrationFunction mainCost(model.get(), mainHelpers, mainWeights);
    auto mainConstraint = model->constraint();
    Array x0 = model->params();
    Problem mainProblem(mainCost, *mainConstraint, x0);

    std::vector<CalibContext<Model>> clones;
    std::vector<Problem*> cloneProblems;
    if (!useSequential) {
        clones.reserve(nThreads);
        for (int t = 0; t < nThreads; ++t)
            clones.push_back(buildCloneContext<Model>(mkt, makeModel, makeEngine));
        for (auto& c : clones)
            cloneProblems.push_back(c.problem.get());
    }

    LevenbergMarquardt lm;
    if (!useSequential)
        lm.setParallelProblems(cloneProblems);

    EndCriteria endCriteria(400, 100, 1.0e-8, 1.0e-8, 1.0e-8);

    MINPACK::resetJacobianStats();
    auto t0 = std::chrono::steady_clock::now();
    lm.minimize(mainProblem, endCriteria);
    auto t1 = std::chrono::steady_clock::now();
    double jacSeconds = MINPACK::jacobianSeconds();
    long long jacCalls = MINPACK::jacobianCalls();

    // Write calibrated parameters back to the caller's model — mirrors
    // what CalibratedModel::calibrate() does at the end of its loop.
    model->setParams(mainProblem.currentValue());

    double wall = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "=== " << label
              << (useSequential ? " (sequential)" : " (parallel)")
              << ", threads=" << effectiveThreads << " ===\n";
    std::cout << "  wall = " << std::fixed << std::setprecision(4) << wall
              << " s,  jacobian = " << jacSeconds
              << " s (" << std::setprecision(1)
              << (wall > 0.0 ? 100.0 * jacSeconds / wall : 0.0)
              << "%), jac_calls = " << jacCalls << "\n";

    const Array& p = model->params();
    // Machine-readable line for bench harness: parseable with split(',').
    std::cout << "CSV," << label << "," << effectiveThreads << ","
              << (useSequential ? "seq" : "par") << ","
              << std::setprecision(6) << wall << ","
              << jacSeconds << "," << jacCalls;
    for (Size i = 0; i < p.size(); ++i)
        std::cout << "," << std::setprecision(10) << p[i];
    std::cout << std::endl;

    // Original per-helper implied-vol pretty-print preserved from the
    // shipped example — runs against the caller's helpers and caller's
    // model, both now reflecting the calibrated params.
    for (Size i = 0; i < numRows; ++i) {
        Size j = numCols - i - 1;
        Size k = i * numCols + j;
        Real npv = swaptions[i]->modelValue();
        Volatility implied = swaptions[i]->impliedVolatility(npv, 1e-4,
                                                             1000, 0.05, 0.50);
        Volatility diff = implied - swaptionVols[k];

        std::cout << i+1 << "x" << swapLengths[j]
                  << std::setprecision(5) << std::noshowpos
                  << ": model " << std::setw(7) << io::volatility(implied)
                  << ", market " << std::setw(7)
                  << io::volatility(swaptionVols[k])
                  << " (" << std::setw(7) << std::showpos
                  << io::volatility(diff) << std::noshowpos << ")\n";
    }
}

} // namespace

int main(int argc, char* argv[]) {

    try {
        bool useSequential = false;
        int nThreads = 1;
        int treeSteps = 50;
        int g2IntPts = 16;
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "-seq") == 0)
                useSequential = true;
            else if (std::strcmp(argv[i], "-t") == 0 && i + 1 < argc)
                nThreads = std::atoi(argv[++i]);
            else if (std::strcmp(argv[i], "-steps") == 0 && i + 1 < argc)
                treeSteps = std::atoi(argv[++i]);
            else if (std::strcmp(argv[i], "-g2pts") == 0 && i + 1 < argc)
                g2IntPts = std::atoi(argv[++i]);
        }
#ifdef _OPENMP
        if (nThreads <= 0)
            nThreads = omp_get_max_threads();
        // Prevent nested-parallel oversubscription: TreeLattice::stepback
        // contains an unguarded `#pragma omp parallel for`, which fires
        // at every LM cost-function evaluation. Without this each of the
        // nt Jacobian threads would spawn its own inner team of nt
        // threads — nt^2 oversubscription.
        omp_set_max_active_levels(1);
        omp_set_dynamic(0);
        omp_set_num_threads(nThreads);
#else
        nThreads = 1;
        useSequential = true;
#endif

        std::cout << std::endl;
        std::cout << "workload: treeSteps=" << treeSteps
                  << ", g2IntPts=" << g2IntPts << "\n\n";

        Date todaysDate(15, February, 2002);
        Calendar calendar = TARGET();
        Date settlementDate(19, February, 2002);
        Settings::instance().evaluationDate() = todaysDate;

        // flat yield term structure impling 1x5 swap at 5%
        auto flatRate = ext::make_shared<SimpleQuote>(0.04875825);
        Handle<YieldTermStructure> rhTermStructure(
            ext::make_shared<FlatForward>(
                      settlementDate, Handle<Quote>(flatRate),
                                      Actual365Fixed()));

        MarketFactory mkt;
        mkt.settlementDate = settlementDate;
        mkt.flatRate = 0.04875825;

        // Define the ATM/OTM/ITM swaps
        Frequency fixedLegFrequency = Annual;
        BusinessDayConvention fixedLegConvention = Unadjusted;
        BusinessDayConvention floatingLegConvention = ModifiedFollowing;
        DayCounter fixedLegDayCounter = Thirty360(Thirty360::European);
        Frequency floatingLegFrequency = Semiannual;
        Swap::Type type = Swap::Payer;
        Rate dummyFixedRate = 0.03;
        auto indexSixMonths = ext::make_shared<Euribor6M>(rhTermStructure);

        Date startDate = calendar.advance(settlementDate,1,Years,
                                          floatingLegConvention);
        Date maturity = calendar.advance(startDate,5,Years,
                                         floatingLegConvention);
        Schedule fixedSchedule(startDate,maturity,Period(fixedLegFrequency),
                               calendar,fixedLegConvention,fixedLegConvention,
                               DateGeneration::Forward,false);
        Schedule floatSchedule(startDate,maturity,Period(floatingLegFrequency),
                               calendar,floatingLegConvention,floatingLegConvention,
                               DateGeneration::Forward,false);

        auto swap = ext::make_shared<VanillaSwap>(
            type, 1000.0,
            fixedSchedule, dummyFixedRate, fixedLegDayCounter,
            floatSchedule, indexSixMonths, 0.0,
            indexSixMonths->dayCounter());
        swap->setPricingEngine(ext::make_shared<DiscountingSwapEngine>(rhTermStructure));
        Rate fixedATMRate = swap->fairRate();
        Rate fixedOTMRate = fixedATMRate * 1.2;
        Rate fixedITMRate = fixedATMRate * 0.8;

        auto atmSwap = ext::make_shared<VanillaSwap>(
            type, 1000.0,
            fixedSchedule, fixedATMRate, fixedLegDayCounter,
            floatSchedule, indexSixMonths, 0.0,
            indexSixMonths->dayCounter());
        auto otmSwap = ext::make_shared<VanillaSwap>(
            type, 1000.0,
            fixedSchedule, fixedOTMRate, fixedLegDayCounter,
            floatSchedule, indexSixMonths, 0.0,
            indexSixMonths->dayCounter());
        auto itmSwap = ext::make_shared<VanillaSwap>(
            type, 1000.0,
            fixedSchedule, fixedITMRate, fixedLegDayCounter,
            floatSchedule, indexSixMonths, 0.0,
            indexSixMonths->dayCounter());

        // defining the swaptions to be used in model calibration
        std::vector<Period> swaptionMaturities;
        swaptionMaturities.emplace_back(1, Years);
        swaptionMaturities.emplace_back(2, Years);
        swaptionMaturities.emplace_back(3, Years);
        swaptionMaturities.emplace_back(4, Years);
        swaptionMaturities.emplace_back(5, Years);

        std::vector<ext::shared_ptr<BlackCalibrationHelper>> swaptions;

        Size i;
        for (i=0; i<numRows; i++) {
            Size j = numCols - i -1; // 1x5, 2x4, 3x3, 4x2, 5x1
            Size k = i*numCols + j;
            auto vol = ext::make_shared<SimpleQuote>(swaptionVols[k]);

            swaptions.push_back(ext::make_shared<SwaptionHelper>(
                               swaptionMaturities[i],
                               Period(swapLengths[j], Years),
                               Handle<Quote>(vol),
                               indexSixMonths,
                               indexSixMonths->tenor(),
                               indexSixMonths->dayCounter(),
                               indexSixMonths->dayCounter(),
                               rhTermStructure));
        }

        // defining the models
        auto modelG2 = ext::make_shared<G2>(rhTermStructure);
        auto modelHW = ext::make_shared<HullWhite>(rhTermStructure);
        auto modelHW2 = ext::make_shared<HullWhite>(rhTermStructure);
        auto modelBK = ext::make_shared<BlackKarasinski>(rhTermStructure);


        // model calibrations

        std::cout << "G2 (analytic formulae) calibration" << std::endl;
        for (i=0; i<swaptions.size(); i++)
            swaptions[i]->setPricingEngine(ext::make_shared<G2SwaptionEngine>(modelG2, 6.0, g2IntPts));

        calibrateModelParallel<G2>(
            "G2 analytic",
            modelG2, swaptions, mkt,
            [](const Handle<YieldTermStructure>& c) {
                return ext::make_shared<G2>(c);
            },
            [g2IntPts](const ext::shared_ptr<G2>& m,
                       const Handle<YieldTermStructure>&) {
                return ext::shared_ptr<PricingEngine>(
                    ext::make_shared<G2SwaptionEngine>(m, 6.0, g2IntPts));
            },
            nThreads, useSequential);

        std::cout << "calibrated to:\n"
                  << "a     = " << modelG2->params()[0] << ", "
                  << "sigma = " << modelG2->params()[1] << "\n"
                  << "b     = " << modelG2->params()[2] << ", "
                  << "eta   = " << modelG2->params()[3] << "\n"
                  << "rho   = " << modelG2->params()[4]
                  << std::endl << std::endl;



        std::cout << "Hull-White (analytic formulae) calibration" << std::endl;
        for (i=0; i<swaptions.size(); i++)
            swaptions[i]->setPricingEngine(ext::make_shared<JamshidianSwaptionEngine>(modelHW));

        calibrateModelParallel<HullWhite>(
            "HullWhite analytic",
            modelHW, swaptions, mkt,
            [](const Handle<YieldTermStructure>& c) {
                return ext::make_shared<HullWhite>(c);
            },
            [](const ext::shared_ptr<HullWhite>& m,
               const Handle<YieldTermStructure>&) {
                return ext::shared_ptr<PricingEngine>(
                    ext::make_shared<JamshidianSwaptionEngine>(m));
            },
            nThreads, useSequential);

        std::cout << "calibrated to:\n"
                  << "a = " << modelHW->params()[0] << ", "
                  << "sigma = " << modelHW->params()[1]
                  << std::endl << std::endl;

        std::cout << "Hull-White (numerical) calibration" << std::endl;
        for (i=0; i<swaptions.size(); i++)
            swaptions[i]->setPricingEngine(ext::make_shared<TreeSwaptionEngine>(modelHW2, treeSteps));

        calibrateModelParallel<HullWhite>(
            "HullWhite numerical",
            modelHW2, swaptions, mkt,
            [](const Handle<YieldTermStructure>& c) {
                return ext::make_shared<HullWhite>(c);
            },
            [treeSteps](const ext::shared_ptr<HullWhite>& m,
                        const Handle<YieldTermStructure>&) {
                return ext::shared_ptr<PricingEngine>(
                    ext::make_shared<TreeSwaptionEngine>(m, treeSteps));
            },
            nThreads, useSequential);

        std::cout << "calibrated to:\n"
                  << "a = " << modelHW2->params()[0] << ", "
                  << "sigma = " << modelHW2->params()[1]
                  << std::endl << std::endl;

        std::cout << "Black-Karasinski (numerical) calibration" << std::endl;
        for (i=0; i<swaptions.size(); i++)
            swaptions[i]->setPricingEngine(ext::make_shared<TreeSwaptionEngine>(modelBK, treeSteps));

        calibrateModelParallel<BlackKarasinski>(
            "BlackKarasinski numerical",
            modelBK, swaptions, mkt,
            [](const Handle<YieldTermStructure>& c) {
                return ext::make_shared<BlackKarasinski>(c);
            },
            [treeSteps](const ext::shared_ptr<BlackKarasinski>& m,
                        const Handle<YieldTermStructure>&) {
                return ext::shared_ptr<PricingEngine>(
                    ext::make_shared<TreeSwaptionEngine>(m, treeSteps));
            },
            nThreads, useSequential);

        std::cout << "calibrated to:\n"
                  << "a = " << modelBK->params()[0] << ", "
                  << "sigma = " << modelBK->params()[1]
                  << std::endl << std::endl;


        // ATM Bermudan swaption pricing

        std::cout << "Payer bermudan swaption "
                  << "struck at " << io::rate(fixedATMRate)
                  << " (ATM)" << std::endl;

        std::vector<Date> bermudanDates;
        const std::vector<ext::shared_ptr<CashFlow>>& leg =
            swap->fixedLeg();
        for (i=0; i<leg.size(); i++) {
            auto coupon = ext::dynamic_pointer_cast<Coupon>(leg[i]);
            bermudanDates.push_back(coupon->accrualStartDate());
        }

        auto bermudanExercise = ext::make_shared<BermudanExercise>(bermudanDates);

        Swaption bermudanSwaption(atmSwap, bermudanExercise);

        // Do the pricing for each model

        // G2 price the European swaption here, it should switch to bermudan
        bermudanSwaption.setPricingEngine(ext::make_shared<TreeSwaptionEngine>(modelG2, 50));
        std::cout << "G2 (tree):      " << bermudanSwaption.NPV() << std::endl;
        bermudanSwaption.setPricingEngine(ext::make_shared<FdG2SwaptionEngine>(modelG2));
        std::cout << "G2 (fdm) :      " << bermudanSwaption.NPV() << std::endl;

        bermudanSwaption.setPricingEngine(ext::make_shared<TreeSwaptionEngine>(modelHW, 50));
        std::cout << "HW (tree):      " << bermudanSwaption.NPV() << std::endl;
        bermudanSwaption.setPricingEngine(ext::make_shared<FdHullWhiteSwaptionEngine>(modelHW));
        std::cout << "HW (fdm) :      " << bermudanSwaption.NPV() << std::endl;

        bermudanSwaption.setPricingEngine(ext::make_shared<TreeSwaptionEngine>(modelHW2, 50));
        std::cout << "HW (num, tree): " << bermudanSwaption.NPV() << std::endl;
        bermudanSwaption.setPricingEngine(ext::make_shared<FdHullWhiteSwaptionEngine>(modelHW2));
        std::cout << "HW (num, fdm) : " << bermudanSwaption.NPV() << std::endl;

        bermudanSwaption.setPricingEngine(ext::make_shared<TreeSwaptionEngine>(modelBK, 50));
        std::cout << "BK:             " << bermudanSwaption.NPV() << std::endl;


        // OTM Bermudan swaption pricing

        std::cout << "Payer bermudan swaption "
                  << "struck at " << io::rate(fixedOTMRate)
                  << " (OTM)" << std::endl;

        Swaption otmBermudanSwaption(otmSwap,bermudanExercise);

        // Do the pricing for each model
        otmBermudanSwaption.setPricingEngine(ext::make_shared<TreeSwaptionEngine>(modelG2, 300));
        std::cout << "G2 (tree):       " << otmBermudanSwaption.NPV()
                  << std::endl;
        otmBermudanSwaption.setPricingEngine(ext::make_shared<FdG2SwaptionEngine>(modelG2));
        std::cout << "G2 (fdm) :       " << otmBermudanSwaption.NPV()
                  << std::endl;

        otmBermudanSwaption.setPricingEngine(ext::make_shared<TreeSwaptionEngine>(modelHW, 50));
        std::cout << "HW (tree):       " << otmBermudanSwaption.NPV()
                  << std::endl;
        otmBermudanSwaption.setPricingEngine(ext::make_shared<FdHullWhiteSwaptionEngine>(modelHW));
        std::cout << "HW (fdm) :       " << otmBermudanSwaption.NPV()
                  << std::endl;

        otmBermudanSwaption.setPricingEngine(ext::make_shared<TreeSwaptionEngine>(modelHW2, 50));
        std::cout << "HW (num, tree):  " << otmBermudanSwaption.NPV()
                  << std::endl;
        otmBermudanSwaption.setPricingEngine(ext::make_shared<FdHullWhiteSwaptionEngine>(modelHW2));
        std::cout << "HW (num, fdm):   " << otmBermudanSwaption.NPV()
                  << std::endl;

        otmBermudanSwaption.setPricingEngine(ext::make_shared<TreeSwaptionEngine>(modelBK, 50));
        std::cout << "BK:              " << otmBermudanSwaption.NPV()
                  << std::endl;


        // ITM Bermudan swaption pricing

        std::cout << "Payer bermudan swaption "
                  << "struck at " << io::rate(fixedITMRate)
                  << " (ITM)" << std::endl;

        Swaption itmBermudanSwaption(itmSwap,bermudanExercise);

        // Do the pricing for each model
        itmBermudanSwaption.setPricingEngine(ext::make_shared<TreeSwaptionEngine>(modelG2, 50));
        std::cout << "G2 (tree):       " << itmBermudanSwaption.NPV()
                  << std::endl;
        itmBermudanSwaption.setPricingEngine(ext::make_shared<FdG2SwaptionEngine>(modelG2));
        std::cout << "G2 (fdm) :       " << itmBermudanSwaption.NPV()
                  << std::endl;

        itmBermudanSwaption.setPricingEngine(ext::make_shared<TreeSwaptionEngine>(modelHW, 50));
        std::cout << "HW (tree):       " << itmBermudanSwaption.NPV()
                  << std::endl;
        itmBermudanSwaption.setPricingEngine(ext::make_shared<FdHullWhiteSwaptionEngine>(modelHW));
        std::cout << "HW (fdm) :       " << itmBermudanSwaption.NPV()
                  << std::endl;

        itmBermudanSwaption.setPricingEngine(ext::make_shared<TreeSwaptionEngine>(modelHW2, 50));
        std::cout << "HW (num, tree):  " << itmBermudanSwaption.NPV()
                  << std::endl;
        itmBermudanSwaption.setPricingEngine(ext::make_shared<FdHullWhiteSwaptionEngine>(modelHW2));
        std::cout << "HW (num, fdm) :  " << itmBermudanSwaption.NPV()
                  << std::endl;

        itmBermudanSwaption.setPricingEngine(ext::make_shared<TreeSwaptionEngine>(modelBK, 50));
        std::cout << "BK:              " << itmBermudanSwaption.NPV()
                  << std::endl;

        return 0;
    } catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "unknown error" << std::endl;
        return 1;
    }
}
