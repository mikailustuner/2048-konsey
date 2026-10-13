#pragma once
// GPU (CUDA) path: batched self-play TD training and batched evaluation for n-tuple networks.
// Built only with -DUSE_CUDA=ON. The CPU code is the reference; tests/gpu_tests.cu checks the
// GPU engine, value function and updates against it.
//
// Layout: one warp plays one game. Its 32 lanes split the network's features (tuple x symmetry)
// and sum them with warp shuffles; board logic runs uniformly on every lane. Many warps =
// many games in flight, which hides the random-access latency of the weight tables.
// Weights live on the device in exactly the host NTupleNet layout (w, or w/E/A interleaved), so
// checkpoints, the GUI and the CPU tools read GPU-trained networks unchanged.
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "engine/eval.hpp"
#include "ntuple/network.hpp"

namespace k2048 {

std::string gpu_info();

struct GpuAgentParams {
    float alpha = 0.1f;  // total step, split over the features (as on the CPU)
    // Batched updates: per launch a weight takes at most kappa/alpha sequential updates (with the
    // mean TD error of its hits). 5 measured best (30k games, 768 concurrent: no cap diverges).
    float kappa = 5.0f;
};

// A set of networks resident on the GPU (agents of a population, ablation arms, council members).
// Shapes may differ (up to 16 distinct shapes).
class GpuPopulation {
public:
    // slots = concurrent games (warps) on the device, 0 = sized from the device.
    GpuPopulation(const std::vector<NTupleNet*>& nets, int slots = 0);
    ~GpuPopulation();

    int size() const;
    int slots() const;
    // Training moves per game between weight updates (default 32; 1 = online TD, as the CPU).
    void set_moves_per_launch(int m);
    // Device -> host copy of agent a's weights (and promoted-stage flags).
    void download(int a, NTupleNet& net) const;
    // Device-side copy of agent src's weights into agent dst (PBT exploit; same shape).
    void copy_agent(int dst, int src);

    // Trains every agent a until done[a] == target[a] games; training game g of agent a uses
    // seed seed_base[a] + g. TD(0) on afterstates with greedy moves (TC if the nets carry
    // accumulators). Games run in parallel and their TD errors are applied in batches (see
    // GpuAgentParams::kappa): with thousands of concurrent games, plain Hogwild updates of the
    // shared early-game features add up and diverge (measured: 768 games x alpha 0.1). on_window(a, report, done) fires each time `window` games of agent a end;
    // on_tick() runs after every kernel launch (checkpoint hook).
    void train(const std::vector<GpuAgentParams>& params, std::vector<int64_t>& done,
               const std::vector<int64_t>& target, const std::vector<uint64_t>& seed_base, int64_t window,
               const std::function<void(int, const EvalReport&, int64_t)>& on_window,
               const std::function<void()>& on_tick = {});

    // Plays cfg.games games (seeds cfg.seed0 + i) with expectimax of `depth` (1 = greedy, max 3)
    // on the weighted mean value of the member agents. Fast board path only (no 65536 merge).
    EvalReport evaluate(const std::vector<int>& members, const std::vector<float>& weights, int depth,
                        const EvalConfig& cfg) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace k2048
