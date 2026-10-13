#pragma once
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "engine/board.hpp"

namespace k2048 {

// N-tuple value network over afterstates with 8-fold symmetric sampling.
//
// Multi-stage: stage k is active when the board holds every exponent in stage_conds[k]
// (the highest matching k wins; stage 0 has no condition). Each stage has its own weights.
// Optional temporal coherence (TC, Beal & Smith / Jaśkowski): per-weight accumulators E (sum of
// errors) and A (sum of |errors|) give each weight its own step |E|/A. Weights and their
// accumulators are interleaved so one feature touches one cache line.
class NTupleNet {
public:
    NTupleNet(std::vector<std::vector<int>> patterns, std::vector<std::vector<int>> stage_conds, bool tc,
              float init_value = 0);

    // Named presets: "4x6", "8x6".
    static std::vector<std::vector<int>> preset(const std::string& name);

    float value(Board b) const;
    // Adds step * coherence * delta to every active feature; TC accumulators see `delta`.
    void update(Board b, float delta, float step);

    int stage(Board b) const;
    int features() const { return int(patterns_.size()) * 8; }
    int stages() const { return int(stage_conds_.size()); }
    bool tc() const { return stride_ == 3; }
    size_t weight_count() const { return weights_per_stage_ * stage_conds_.size(); }

    // Copies stage k-1 weights into stage k the first time stage k is seen (weight promotion,
    // Jaśkowski 2018). Safe to call from many threads.
    void promote_if_new(int stage);

    // Copies weights (and TC accumulators, stage flags) from a network of the same shape.
    void copy_from(const NTupleNet& o);

    void save(const std::string& path, const std::string& extra_header) const;
    // Returns the extra header lines given to save().
    static std::unique_ptr<NTupleNet> load(const std::string& path, std::string* extra_header = nullptr);

    // Raw storage for the GPU path: per stage, per tuple, index * stride floats (w, E, A).
    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }
    size_t data_size() const { return data_.size(); }
    size_t weights_per_stage() const { return weights_per_stage_; }
    size_t tuple_offset(size_t t) const { return offset_[t]; }
    uint32_t stage_mask(int k) const { return stage_mask_[size_t(k)]; }
    bool seen(int k) const { return seen_[k]; }
    void set_seen(int k, bool v) { seen_[k] = v; }
    // Cell that pattern cell `cell` maps to under symmetry s (0..7).
    static int sym_cell(int cell, int s);

    const std::vector<std::vector<int>>& patterns() const { return patterns_; }
    const std::vector<std::vector<int>>& stage_conds() const { return stage_conds_; }

private:
    template <class F>
    void for_each_feature(Board b, F&& f) const;

    std::vector<std::vector<int>> patterns_, stage_conds_;
    std::vector<uint32_t> stage_mask_;      // bit e set = exponent e required
    std::vector<std::vector<uint8_t>> shifts_;  // [tuple*8 + sym][j] = 4 * cell
    std::vector<size_t> offset_;           // per tuple, within one stage
    size_t weights_per_stage_ = 0;
    int stride_ = 1;
    std::vector<float> data_;  // [stage][tuple][index] * stride: w (, E, A)
    std::unique_ptr<std::atomic<bool>[]> seen_;
    std::mutex promote_mu_;
};

}  // namespace k2048
