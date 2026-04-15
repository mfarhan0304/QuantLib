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

#include <ql/experimental/risk/scenarioevaluator.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <chrono>
#include <stdexcept>

namespace QuantLib {

    ScenarioEvaluator::ScenarioEvaluator(ContextFactory factory,
                                         std::vector<Real> baseNPVs,
                                         Config cfg)
    : baseNPVs_(std::move(baseNPVs)), cfg_(cfg) {

#ifdef _OPENMP
        const int nt = (cfg_.nThreads > 0)
                       ? static_cast<int>(cfg_.nThreads)
                       : omp_get_max_threads();
#else
        const int nt = 1;
#endif
        contexts_.reserve(static_cast<Size>(nt));
        for (int t = 0; t < nt; ++t)
            contexts_.push_back(factory());

        threadBusy_.assign(static_cast<Size>(nt), 0.0);
    }

    std::vector<Real> ScenarioEvaluator::run(Size nScenarios,
                                             const ShockFn& shockFn) {
        std::vector<Real> pnl(nScenarios, 0.0);

        const auto& base = baseNPVs_;
        const int   nt   = static_cast<int>(contexts_.size());

        // Reset per-thread accumulators
        std::fill(threadBusy_.begin(), threadBusy_.end(), 0.0);

#ifdef _OPENMP
        const int cs  = static_cast<int>(cfg_.chunkSize > 0 ? cfg_.chunkSize : 1);
        const double t0 = omp_get_wtime();

        if (cfg_.schedule == ScenarioSchedule::Dynamic) {
            #pragma omp parallel for num_threads(nt) schedule(dynamic, cs)
            for (int s = 0; s < static_cast<int>(nScenarios); ++s) {
                const int  tid = omp_get_thread_num();
                const double ts = omp_get_wtime();
                auto& ctx = contexts_[static_cast<Size>(tid)];

                shockFn(ctx, static_cast<Size>(s));

                Real v = 0.0;
                for (Size i = 0; i < ctx.instruments.size(); ++i)
                    v += ctx.instruments[i]->NPV() - base[i];
                pnl[static_cast<Size>(s)] = v;

                threadBusy_[static_cast<Size>(tid)] += omp_get_wtime() - ts;
            }
        } else if (cfg_.schedule == ScenarioSchedule::Static) {
            #pragma omp parallel for num_threads(nt) schedule(static)
            for (int s = 0; s < static_cast<int>(nScenarios); ++s) {
                const int  tid = omp_get_thread_num();
                const double ts = omp_get_wtime();
                auto& ctx = contexts_[static_cast<Size>(tid)];

                shockFn(ctx, static_cast<Size>(s));

                Real v = 0.0;
                for (Size i = 0; i < ctx.instruments.size(); ++i)
                    v += ctx.instruments[i]->NPV() - base[i];
                pnl[static_cast<Size>(s)] = v;

                threadBusy_[static_cast<Size>(tid)] += omp_get_wtime() - ts;
            }
        } else { // Guided
            #pragma omp parallel for num_threads(nt) schedule(guided, cs)
            for (int s = 0; s < static_cast<int>(nScenarios); ++s) {
                const int  tid = omp_get_thread_num();
                const double ts = omp_get_wtime();
                auto& ctx = contexts_[static_cast<Size>(tid)];

                shockFn(ctx, static_cast<Size>(s));

                Real v = 0.0;
                for (Size i = 0; i < ctx.instruments.size(); ++i)
                    v += ctx.instruments[i]->NPV() - base[i];
                pnl[static_cast<Size>(s)] = v;

                threadBusy_[static_cast<Size>(tid)] += omp_get_wtime() - ts;
            }
        }

        wallTime_ = omp_get_wtime() - t0;

#else
        // Sequential fallback — single thread, tid always 0
        using clock = std::chrono::steady_clock;
        auto t0 = clock::now();

        auto& ctx = contexts_[0];
        for (Size s = 0; s < nScenarios; ++s) {
            auto ts = clock::now();
            shockFn(ctx, s);

            Real v = 0.0;
            for (Size i = 0; i < ctx.instruments.size(); ++i)
                v += ctx.instruments[i]->NPV() - base[i];
            pnl[s] = v;

            threadBusy_[0] += std::chrono::duration<Real>(clock::now() - ts).count();
        }

        wallTime_ = std::chrono::duration<Real>(clock::now() - t0).count();
#endif
        return pnl;
    }

}
