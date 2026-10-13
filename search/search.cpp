#include "search.hpp"

namespace k2048 {

namespace {

float chance_node(Board a, int depth, const ValueFn& v);

float max_node(Board b, int depth, const ValueFn& v, int* best_dir) {
    float best = 0;
    int dir = -1;
    for (int d = 0; d < 4; ++d) {
        uint32_t r = 0;
        Board a = move(b, d, r);
        if (a == b) continue;
        float val = float(r) + (depth == 1 ? v(a) : chance_node(a, depth - 1, v));
        if (dir < 0 || val > best) dir = d, best = val;
    }
    if (best_dir) *best_dir = dir;
    return best;  // 0 when no move is legal
}

float chance_node(Board a, int depth, const ValueFn& v) {
    float sum = 0;
    int n = 0;
    for (int i = 0; i < 16; ++i) {
        if (cell(a, i)) continue;
        sum += 0.9f * max_node(set_cell(a, i, 1), depth, v, nullptr) +
               0.1f * max_node(set_cell(a, i, 2), depth, v, nullptr);
        n++;
    }
    return n ? sum / float(n) : 0;
}

}  // namespace

int greedy_move(Board b, const ValueFn& v) { return expectimax_move(b, 1, v); }

int expectimax_move(Board b, int depth, const ValueFn& v) {
    int dir;
    max_node(b, depth, v, &dir);
    return dir;
}

}  // namespace k2048

#include <chrono>

namespace k2048 {

Searcher::Searcher(ValueFn v, SearchConfig c) : v_(std::move(v)), c_(c) {
    if (c_.tt_bits > 0) tt_.resize(size_t(1) << c_.tt_bits);
}

float Searcher::max_node(Board b, int depth, float prob, int* best_dir) {
    ++nodes_;
    // Unlikely branches are worth only a 1-ply look.
    if (prob < c_.prob_cut) depth = 1;
    float best = 0;
    int dir = -1;
    for (int d = 0; d < 4; ++d) {
        uint32_t r = 0;
        Board a = move(b, d, r);
        if (a == b) continue;
        float val = float(r) + (depth == 1 ? v_(a) : chance_node(a, depth - 1, prob));
        if (dir < 0 || val > best) dir = d, best = val;
    }
    if (best_dir) *best_dir = dir;
    return best;
}

float Searcher::chance_node(Board a, int depth, float prob) {
    Entry* e = nullptr;
    if (!tt_.empty()) {
        e = &tt_[size_t((a * 0x9E3779B97F4A7C15ULL) >> (64 - c_.tt_bits))];
        // Same position at the same depth: the value is exactly what a recomputation gives.
        if (e->key == a && e->depth == depth) return e->value;
    }
    const int n = count_empty(a);
    float sum = 0;
    for (int i = 0; i < 16; ++i) {
        if (cell(a, i)) continue;
        sum += 0.9f * max_node(set_cell(a, i, 1), depth, prob * 0.9f / float(n), nullptr) +
               0.1f * max_node(set_cell(a, i, 2), depth, prob * 0.1f / float(n), nullptr);
    }
    float v = n ? sum / float(n) : 0;
    if (e) *e = {a, depth, v};
    return v;
}

int Searcher::choose(Board b0) {
    Board b = c_.downgrade > 0 ? tile_downgrade(b0, c_.downgrade) : b0;
    int dir = -1;
    if (c_.time_ms <= 0) {
        max_node(b, c_.depth, 1.0f, &dir);
        last_depth_ = c_.depth;
        return dir;
    }
    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();
    double prev = 0;
    for (int d = 1; d <= c_.depth; ++d) {
        auto ts = clock::now();
        max_node(b, d, 1.0f, &dir);
        last_depth_ = d;
        double took = std::chrono::duration<double, std::milli>(clock::now() - ts).count();
        double elapsed = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
        double growth = prev > 0.01 ? std::max(4.0, took / prev) : 16.0;
        if (dir < 0 || elapsed + took * growth > c_.time_ms) break;
        prev = took;
    }
    return dir;
}

Board tile_downgrade(Board b, int threshold) {
    for (;;) {
        int mx = max_exp(b);
        if (mx < threshold) return b;
        uint32_t present = 0;
        for (int i = 0; i < 16; ++i) present |= 1u << cell(b, i);
        int gap = mx - 1;  // the largest missing exponent: only the big tiles move down
        while (gap >= 6 && (present >> gap & 1)) --gap;
        if (gap < 6) return b;
        Board r = 0;
        for (int i = 0; i < 16; ++i) {
            int v = cell(b, i);
            r = set_cell(r, i, v > gap ? v - 1 : v);
        }
        b = r;
    }
}

}  // namespace k2048
