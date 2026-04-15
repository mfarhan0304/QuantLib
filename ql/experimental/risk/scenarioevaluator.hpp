/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Copyright (C) 2026 QuantLib contributors

 This file is part of QuantLib, a free-software/open-source library
 for financial quantitative analysts and developers - http://quantlib.org/

 QuantLib is free software: you can redistribute it and/or modify it
 under the terms of the QuantLib license.  You should have received a
 copy of the license along with this program; if not, please email
 <quantlib-dev@lists.sf.net>. The license is also available online at
 <http://quantlib.org/license.shtml>.

 This program is distributed in the hope that it will be useful, but WITHOUT
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE.  See the license for more details.
*/

/*! \file scenarioevaluator.hpp
    \brief Parallel portfolio revaluation under a set of market scenarios.

    ScenarioEvaluator parallelises the outer scenario loop of a Monte Carlo
    VaR engine using OpenMP.  Each OpenMP thread owns an independent clone of
    the market data graph and the instrument book so that no shared mutable
    state is accessed inside the parallel region.

    If OpenMP is not available the implementation falls back to a sequential
    single-threaded loop, so the class compiles and produces correct results
    in all configurations.
*/

#ifndef quantlib_scenario_evaluator_hpp
#define quantlib_scenario_evaluator_hpp

#include <ql/types.hpp>
#include <ql/shared_ptr.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/instrument.hpp>

#include <functional>
#include <vector>

namespace QuantLib {

    //! OpenMP loop schedule selector for ScenarioEvaluator.
    enum class ScenarioSchedule { Static, Dynamic, Guided };

    //! Configuration for ScenarioEvaluator's parallel loop.
    struct ScenarioEvaluatorConfig {
        Size             nThreads  = 0;                        //!< 0 = omp_get_max_threads()
        ScenarioSchedule schedule  = ScenarioSchedule::Dynamic;
        Size             chunkSize = 16;                       //!< chunk for Dynamic/Guided
    };

    //! One thread's independent market-data and instrument universe.
    /*!  The factory function supplied to ScenarioEvaluator must produce one
         of these per thread.  Every SimpleQuote in \c quotes must be linked
         (directly or via curves/surfaces) to the instruments in \c instruments.
         No object in one ThreadContext may be shared with another.
    */
    struct ScenarioThreadContext {
        std::vector<ext::shared_ptr<SimpleQuote>>  quotes;       //!< shockable inputs
        std::vector<ext::shared_ptr<Instrument>>   instruments;  //!< linked pricers
    };

    //! Parallel portfolio revaluation across Monte Carlo scenarios.
    /*!  Each call to run() evaluates \c nScenarios profit-and-loss values
         in parallel using OpenMP.  Thread safety is achieved by giving every
         OpenMP thread its own independent copy of the market data graph and
         instrument book (a ScenarioThreadContext).

         \ingroup experimental
    */
    class ScenarioEvaluator {
      public:
        using Config        = ScenarioEvaluatorConfig;
        using ThreadContext = ScenarioThreadContext;

        //! Factory that constructs one independent ThreadContext.
        using ContextFactory = std::function<ThreadContext()>;

        //! Applies scenario \c scenarioIdx to a thread-local context.
        /*!  The function sets the relevant quotes in \c ctx to their shocked
             values for scenario \c scenarioIdx.  Because each thread owns its
             own context, the next call will overwrite the quotes completely,
             so no explicit restore step is required.
        */
        using ShockFn = std::function<void(ThreadContext&, Size scenarioIdx)>;

        //! Pre-builds \c nThreads ThreadContexts via the factory.
        ScenarioEvaluator(ContextFactory factory,
                          std::vector<Real> baseNPVs,
                          Config cfg = Config());

        //! Evaluate P&L across \c nScenarios scenarios in parallel.
        std::vector<Real> run(Size nScenarios, const ShockFn& shockFn);

        //! Wall-clock time of the most recent run() call (seconds).
        Real wallTime() const { return wallTime_; }

        //! Accumulated busy time per thread across the most recent run().
        const std::vector<Real>& threadBusyTime() const { return threadBusy_; }

        //! Number of threads actually used.
        Size nThreads() const { return static_cast<Size>(contexts_.size()); }

      private:
        std::vector<ThreadContext> contexts_;
        std::vector<Real>          baseNPVs_;
        Config                     cfg_;
        Real                       wallTime_ = 0.0;
        std::vector<Real>          threadBusy_;
    };

}

#endif
