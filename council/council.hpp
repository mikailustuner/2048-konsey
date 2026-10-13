#pragma once
#include <memory>
#include <string>
#include <vector>

#include "ntuple/td.hpp"

namespace k2048 {

// A council of value networks.
//  mean:     afterstate value = average of members (one search on the combined value)
//  weighted: weighted average (weights normalised)
//  vote:     every member searches on its own; majority move, ties broken by summed weight
struct Council {
    enum class Mode { Mean, Weighted, Vote };
    std::vector<std::shared_ptr<const NTupleNet>> nets;
    std::vector<float> weights;  // empty = equal
    Mode mode = Mode::Mean;

    float value(Board a) const;
    int choose(Board b, int depth) const;
};

Council::Mode parse_mode(const std::string& s);

// Distillation: trains `student` to regress the council's (mean/weighted) values on the
// afterstates met while the council plays greedily. Returns mean council score of those games.
double distill(NTupleNet& student, const Council& teacher, int64_t games, int threads, uint64_t seed, float alpha,
               const std::string& log_path, const LogMeta& meta);

}  // namespace k2048
