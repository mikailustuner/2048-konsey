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
