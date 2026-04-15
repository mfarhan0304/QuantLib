#pragma once
#include <vector>
#include <cstdint>

struct Scenario {
    double deltaRate;   // parallel rate shift (additive, e.g. 0.001 = 1bp)
    double returnEquity; // equity log-return
    double deltaVol;    // vol shock (additive)
};

// Generate N scenarios from independent 3-factor Normal model (V1: zero correlation).
// sigmaRate, sigmaEquity, sigmaVol are daily 1-sigma values.
std::vector<Scenario> generateScenarios(int N,
                                        double sigmaRate   = 0.001,
                                        double sigmaEquity = 0.01,
                                        double sigmaVol    = 0.02,
                                        uint64_t seed      = 42);
