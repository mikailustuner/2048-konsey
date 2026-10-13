#pragma once
#include <string>

#include "engine/eval.hpp"
#include "network.hpp"

namespace k2048 {

struct TDParams {
    float alpha = 0.1f;   // total step, split evenly over the network's features
    float lambda = 0.0f;  // 0: online TD(0); > 0: TD(lambda) by a backward pass at game end
};

struct TrainOptions {
    int64_t games = 100'000;         // total target, including games already trained
    int threads = 1;
    uint64_t seed = 1ULL << 40;      // training game g uses seed + g (disjoint from eval seeds)
    std::string ckpt;                // saved every ckpt_sec and at the end; "" = never
    int ckpt_sec = 600;
    int64_t log_every = 10'000;      // stats window over training games
    std::string log_path;            // JSONL; "" = stdout only
    LogMeta meta;
    bool quiet = false;
};

// One self-play game with greedy (1-ply) moves on afterstates, learning as it goes.
GameResult td_episode(NTupleNet& net, const TDParams& p, uint64_t seed);

// Hogwild self-play training (threads share the weights without locks). Continues from
// `done` games; returns the number of games trained in total.
int64_t train(NTupleNet& net, const TDParams& p, const TrainOptions& o, int64_t done);

std::string ckpt_header(const TDParams& p, int64_t games);
// Reads "games=" back from a checkpoint's extra header.
int64_t ckpt_games(const std::string& header);

}  // namespace k2048
