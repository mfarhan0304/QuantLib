#pragma once
#include <algorithm>
#include <numeric>
#include <vector>
#include <stdexcept>

struct VaRResult {
    double var95, var99;
    double es95, es99;
};

// Compute VaR and ES from a P&L vector (losses are negative P&L).
// VaR at level alpha is the -(alpha-th percentile of P&L).
// ES at level alpha is the mean of P&L values below the VaR threshold.
inline VaRResult computeVaRES(std::vector<double> pnl) {
    if (pnl.empty())
        throw std::invalid_argument("P&L vector is empty");

    std::sort(pnl.begin(), pnl.end());
    auto n = static_cast<int>(pnl.size());

    auto varAtLevel = [&](double alpha) -> double {
        // Floor index so we take the conservative (larger loss) side
        int idx = static_cast<int>(alpha * n);
        if (idx >= n) idx = n - 1;
        return -pnl[idx];
    };

    auto esAtLevel = [&](double alpha) -> double {
        int idx = static_cast<int>(alpha * n);
        if (idx >= n) idx = n - 1;
        double sum = std::accumulate(pnl.begin(), pnl.begin() + idx + 1, 0.0);
        return -sum / (idx + 1);
    };

    VaRResult r;
    r.var95 = varAtLevel(0.05);
    r.var99 = varAtLevel(0.01);
    r.es95  = esAtLevel(0.05);
    r.es99  = esAtLevel(0.01);
    return r;
}
