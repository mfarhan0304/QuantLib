/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Copyright (C) 2006 Klaus Spanderen
 Copyright (C) 2015 Peter Caspers

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

/*! \file levenbergmarquardt.hpp
    \brief Levenberg-Marquardt optimization method
*/

#ifndef quantlib_optimization_levenberg_marquardt_hpp
#define quantlib_optimization_levenberg_marquardt_hpp

#include <ql/math/optimization/problem.hpp>
#include <vector>

namespace QuantLib {

    //! Levenberg-Marquardt optimization method
    /*! This implementation is based on MINPACK
        (<http://www.netlib.org/minpack>,
        <http://www.netlib.org/cephes/linalg.tgz>)
        It has a built in fd scheme to compute
        the jacobian, which is used by default.
        If useCostFunctionsJacobian is true the
        corresponding method in the cost function
        of the problem is used instead. Note that
        the default implementation of the jacobian
        in CostFunction uses a central difference
        (order 2, but requiring more function
        evaluations) compared to the forward
        difference implemented here (order 1).

        \ingroup optimizers
    */
    class LevenbergMarquardt : public OptimizationMethod {
      public:
        LevenbergMarquardt(Real epsfcn = 1.0e-8,
                           Real xtol = 1.0e-8,
                           Real gtol = 1.0e-8,
                           bool useCostFunctionsJacobian = false);
        EndCriteria::Type minimize(Problem& P,
                                   const EndCriteria& endCriteria) override;
<<<<<<< HEAD

        // Enable OpenMP-parallel forward-difference Jacobian.
        // Each entry must be an independent Problem clone owning its own
        // model/instrument state; one is consumed per OpenMP thread during
        // the Jacobian phase. The vector size sets the thread count.
        // Ignored when useCostFunctionsJacobian=true.
        void setParallelProblems(std::vector<Problem*> problems);
=======
>>>>>>> 8aef029c02935baf52c93391eb70dcdbd9ab88aa

      private:
        void fcn(int m, int n, Real* x, Real* fvec);
        void jacFcn(int m, int n, Real* x, Real* fjac);
<<<<<<< HEAD
        void fcnForProblem(Problem& problem,
                           int m, int n, Real* x, Real* fvec);
=======
>>>>>>> 8aef029c02935baf52c93391eb70dcdbd9ab88aa

        Problem* currentProblem_;
        std::vector<Problem*> parallelProblems_;
        Array initCostValues_;
        Matrix initJacobian_;
        mutable Integer info_ = 0; // remove together with getInfo
        const Real epsfcn_, xtol_, gtol_;
        const bool useCostFunctionsJacobian_;
    };

}


#endif
