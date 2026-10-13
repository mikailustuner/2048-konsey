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
    int save_every = 1;  // population checkpoint every N generations (and after the last)
};

struct PBTAgent {
    int uid = 0, parent = -1;
    Genome g;
    int64_t games = 0;
    double fitness = 0;
    std::unique_ptr<NTupleNet> net;
};

// Where training and tournaments run. The default (nullptr) is the CPU backend; the GPU
// backend (gpu/gpu_pbt.cpp) keeps the whole population resident on the device.
class PBTBackend {
public:
    virtual ~PBTBackend() = default;
    virtual void begin(std::vector<PBTAgent>& pop) = 0;
    // Train every agent cfg.train_games more games (updates a.games); logs to train.jsonl.
    virtual void train(std::vector<PBTAgent>& pop, const PBTConfig& cfg, int gen) = 0;
    virtual std::vector<EvalReport> tournament(std::vector<PBTAgent>& pop, const EvalConfig& ec) = 0;
    // Copy src's weights into dst.
    virtual void exploit(std::vector<PBTAgent>& pop, size_t dst, size_t src) = 0;
    // Make host nets current before they are saved.
    virtual void sync_to_host(std::vector<PBTAgent>& pop) = 0;
};

void run_pbt(const PBTConfig& cfg, PBTBackend* backend = nullptr);
std::vector<std::vector<int>> parse_stages(const std::string& s);

// lambda is perturbed only when mutate_lambda (the GPU path trains TD(0), lambda stays 0).
Genome mutate(const Genome& g, Rng& rng, bool mutate_lambda = true);

}  // namespace k2048
