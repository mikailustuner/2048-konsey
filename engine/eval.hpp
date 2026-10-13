#pragma once
#include <array>
#include <functional>
#include <string>
#include <vector>

#include "game.hpp"

namespace k2048 {

struct Wilson {
    double p, lo, hi;
};
// Wilson score interval, 95% (z = 1.96).
Wilson wilson(uint64_t k, uint64_t n);

// Shared evaluation harness: games use seeds seed0 .. seed0+games-1, so two agents evaluated
// with the same seed0 face identical spawn sequences (paired comparison).
struct EvalConfig {
    uint64_t seed0 = 1'000'000;
    int games = 1000;
    int threads = 1;
    bool record = false;
};

struct EvalReport {
    std::vector<GameResult> results;  // in seed order
    double seconds = 0;

    int games() const { return int(results.size()); }
    double mean_score() const;
    uint64_t max_score() const;
    uint64_t total_moves() const;
    // Games whose max tile reached 2^e.
    uint64_t reached(int e) const;
};

// make_policy is called once per worker thread (policies may keep per-thread state).
EvalReport evaluate(const std::function<Policy()>& make_policy, const EvalConfig& cfg);

// Identification of a measurement for the JSONL log (schema in README / PLAN Faz 0).
struct LogMeta {
    std::string run_id;
    std::string faz;
    std::string ajan_id;
    int nesil = 0;
    int64_t egitim_oyunu = -1;  // training games so far; logged when >= 0
};

// One JSONL line: run_id, faz, ajan_id, nesil, oyun_sayisi, hamle_sayisi, oranlar{2048..65536},
// guven{2048..65536: [lo, hi]}, ort_skor, maks_skor, sure (+ egitim_oyunu when set).
std::string log_line(const LogMeta& meta, const EvalReport& r);

// Human readable summary table.
std::string summary(const EvalReport& r);

void append_line(const std::string& path, const std::string& line);

}  // namespace k2048
