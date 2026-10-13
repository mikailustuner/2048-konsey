#pragma once
#include <memory>
#include <string>
#include <vector>

#include "ntuple/td.hpp"

namespace k2048 {

// Population based training (Jaderberg 2017): evolution steers the TD hyper-parameters,
// TD learns the weights. Per generation every agent trains `train_games` games, then all
// agents play the same `tourney_games` seeds (paired tournament, fitness = mean score);
// the bottom quarter is replaced by copies (weights + genome) of random top-quarter agents
// whose genome is then mutated.
struct Genome {
    float alpha = 0.1f;
    float lambda = 0.0f;
    float oi = 0.0f;  // optimistic initial value; only acts on generation-0 weights
};

struct PBTConfig {
    std::string dir;  // population state, nets and logs; resumes if state exists
    int population = 8;
    int generations = 20;
    int64_t train_games = 20'000;  // per agent per generation
    int tourney_games = 300;
    int threads = 8;
    uint64_t seed = 7;
    std::string net = "4x6";
    std::string stages;  // e.g. "14;14,13"
    bool tc = false;
    float alpha_lo = 0.02f, alpha_hi = 0.5f;  // generation-0 sampling (log-uniform)
    float lambda_hi = 0.5f;                   // uniform [0, lambda_hi]
    float oi_hi = 0.0f;                       // uniform [0, oi_hi]
    std::string run_id = "pbt";
};

void run_pbt(const PBTConfig& cfg);

Genome mutate(const Genome& g, Rng& rng);

}  // namespace k2048
