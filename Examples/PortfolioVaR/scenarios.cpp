#include "scenarios.hpp"
#include <random>
#include <cmath>

std::vector<Scenario> generateScenarios(int N,
                                        double sigmaRate,
                                        double sigmaEquity,
                                        double sigmaVol,
                                        uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normRate(0.0, sigmaRate);
    std::normal_distribution<double> normEquity(0.0, sigmaEquity);
    std::normal_distribution<double> normVol(0.0, sigmaVol);

    std::vector<Scenario> scenarios(N);
    for (auto& s : scenarios) {
        s.deltaRate    = normRate(rng);
        s.returnEquity = normEquity(rng);
        s.deltaVol     = normVol(rng);
    }
    return scenarios;
}
