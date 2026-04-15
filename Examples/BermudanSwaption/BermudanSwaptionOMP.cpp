/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Parallel Jacobian variant of BermudanSwaption calibration — Track B driver.

 Builds N independent (model, helpers, cost-function, problem) clones and
 registers them on LevenbergMarquardt::setParallelProblems() so the finite
 difference Jacobian inside lmdif is evaluated by fdjac2_parallel across
 OpenMP threads. Each thread hits its own model clone — no shared mutable
 state across cost-function evaluations.

 Sequential fallback when OMP unavailable or when -seq is passed.
*/

#include <ql/qldefines.hpp>
#include <ql/instruments/vanillaswap.hpp>
#include <ql/pricingengines/swap/discountingswapengine.hpp>
#include <ql/pricingengines/swaption/treeswaptionengine.hpp>
#include <ql/pricingengines/swaption/jamshidianswaptionengine.hpp>
#include <ql/pricingengines/swaption/g2swaptionengine.hpp>
#include <ql/models/shortrate/calibrationhelpers/swaptionhelper.hpp>
#include <ql/models/shortrate/twofactormodels/g2.hpp>
#include <ql/models/shortrate/onefactormodels/hullwhite.hpp>
#include <ql/math/optimization/levenbergmarquardt.hpp>
#include <ql/math/optimization/lmdif.hpp>
#include <ql/math/optimization/problem.hpp>
#include <ql/math/optimization/constraint.hpp>
#include <ql/math/optimization/costfunction.hpp>
#include <ql/math/optimization/endcriteria.hpp>
#include <ql/math/optimization/projectedconstraint.hpp>
#include <ql/math/optimization/projection.hpp>
#include <ql/indexes/ibor/euribor.hpp>
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

static Size numRows = 5;
static Size numCols = 5;
static Integer swapLengths[] = {1, 2, 3, 4, 5};
static Volatility swaptionVols[] = {
    0.1490, 0.1340, 0.1228, 0.1189, 0.1148,
    0.1290, 0.1201, 0.1146, 0.1108, 0.1040,
    0.1149, 0.1112, 0.1070, 0.1010, 0.0957,
    0.1047, 0.1021, 0.0980, 0.0951, 0.1270,
    0.1000, 0.0950, 0.0900, 0.1230, 0.1160};

namespace {

// A self-contained copy of CalibratedModel::CalibrationFunction. We can't use
// the library one because it's a private nested class; the public CostFunction
// interface is enough.
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

// Market-data factory shared by main + per-thread clones. Each call creates
// fresh SimpleQuote + FlatForward instances so observer graphs stay disjoint.
struct MarketFactory {
    Date todaysDate;
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

// One thread-local calibration context: owns its own yield curve, swaption
// helpers (with engines bound to its own model clone), the cost function,
// and the Problem handed to LevenbergMarquardt.
template <typename Model>
struct CalibContext {
    Handle<YieldTermStructure> curve;
    ext::shared_ptr<Model> model;
    std::vector<ext::shared_ptr<CalibrationHelper>> helpers;
    std::unique_ptr<PublicCalibrationFunction> costFn;
    ext::shared_ptr<Constraint> constraint;
    std::unique_ptr<Problem> problem;
};

// Build fresh swaption helpers for the given curve + engine factory.
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
CalibContext<Model> buildContext(const MarketFactory& mkt,
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

template <typename Model, typename ModelFactory, typename EngineFactory>
void calibrateParallel(const std::string& label,
                       const MarketFactory& mkt,
                       ModelFactory makeModel,
                       EngineFactory makeEngine,
                       int nThreads,
                       bool useSequential) {
    const int effectiveThreads = useSequential ? 1 : nThreads;

    auto main = buildContext<Model>(mkt, makeModel, makeEngine);

    std::vector<CalibContext<Model>> clones;
    std::vector<Problem*> cloneProblems;
    if (!useSequential) {
        clones.reserve(nThreads);
        for (int t = 0; t < nThreads; ++t)
            clones.push_back(buildContext<Model>(mkt, makeModel, makeEngine));
        for (auto& c : clones)
            cloneProblems.push_back(c.problem.get());
    }

    LevenbergMarquardt lm;
    if (!useSequential)
        lm.setParallelProblems(cloneProblems);

    EndCriteria endCriteria(400, 100, 1.0e-8, 1.0e-8, 1.0e-8);

    MINPACK::resetJacobianStats();
    auto t0 = std::chrono::steady_clock::now();
    lm.minimize(*main.problem, endCriteria);
    auto t1 = std::chrono::steady_clock::now();
    double jacSeconds = MINPACK::jacobianSeconds();
    long long jacCalls = MINPACK::jacobianCalls();

    // Pull the solution back into the main model (mirror CalibratedModel::calibrate).
    main.model->setParams(main.problem->currentValue());

    double wall = std::chrono::duration<double>(t1 - t0).count();
    const Array& p = main.model->params();

    std::cout << "=== " << label
              << (useSequential ? " (sequential)" : " (parallel)")
              << ", threads=" << effectiveThreads << " ===\n";
    std::cout << "  wall = " << std::fixed << std::setprecision(4) << wall
              << " s,  jacobian = " << jacSeconds
              << " s (" << std::setprecision(1)
              << (wall > 0.0 ? 100.0 * jacSeconds / wall : 0.0)
              << "%), jac_calls = " << jacCalls << "\n";
    std::cout << "  params =";
    for (Size i = 0; i < p.size(); ++i)
        std::cout << " " << std::setprecision(8) << p[i];
    std::cout << "\n";

    // Machine-readable line for bench harness: parseable with split(',')
    std::cout << "CSV," << label << "," << effectiveThreads << ","
              << (useSequential ? "seq" : "par") << ","
              << std::setprecision(6) << wall << ","
              << jacSeconds << "," << jacCalls;
    for (Size i = 0; i < p.size(); ++i)
        std::cout << "," << std::setprecision(10) << p[i];
    std::cout << std::endl;
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
        // Suppress nested parallelism so inner pragmas (e.g. TreeLattice::stepback)
        // don't oversubscribe cores. Only our fdjac2_parallel outer region uses
        // num_threads(nt) to get real parallelism.
        omp_set_max_active_levels(1);
#else
        nThreads = 1;
        useSequential = true;
#endif

        MarketFactory mkt;
        mkt.todaysDate = Date(15, February, 2002);
        mkt.settlementDate = Date(19, February, 2002);
        mkt.flatRate = 0.04875825;
        Settings::instance().evaluationDate() = mkt.todaysDate;

        std::cout << "workload: treeSteps=" << treeSteps
                  << ", g2IntPts=" << g2IntPts << "\n\n";

        // --- G2 (n = 5) ---
        calibrateParallel<G2>(
            "G2 analytic",
            mkt,
            [](const Handle<YieldTermStructure>& c) {
                return ext::make_shared<G2>(c);
            },
            [g2IntPts](const ext::shared_ptr<G2>& model,
                       const Handle<YieldTermStructure>&) {
                return ext::shared_ptr<PricingEngine>(
                    ext::make_shared<G2SwaptionEngine>(model, 6.0, g2IntPts));
            },
            nThreads, useSequential);

        // --- Hull-White analytic (n = 2) — no inner OMP, clean fdjac2 test ---
        calibrateParallel<HullWhite>(
            "HullWhite analytic",
            mkt,
            [](const Handle<YieldTermStructure>& c) {
                return ext::make_shared<HullWhite>(c);
            },
            [](const ext::shared_ptr<HullWhite>& model,
               const Handle<YieldTermStructure>&) {
                return ext::shared_ptr<PricingEngine>(
                    ext::make_shared<JamshidianSwaptionEngine>(model));
            },
            nThreads, useSequential);

        // --- Hull-White numerical (n = 2) ---
        calibrateParallel<HullWhite>(
            "HullWhite numerical",
            mkt,
            [](const Handle<YieldTermStructure>& c) {
                return ext::make_shared<HullWhite>(c);
            },
            [treeSteps](const ext::shared_ptr<HullWhite>& model,
                        const Handle<YieldTermStructure>&) {
                return ext::shared_ptr<PricingEngine>(
                    ext::make_shared<TreeSwaptionEngine>(model, treeSteps));
            },
            nThreads, useSequential);

        return 0;
    } catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
