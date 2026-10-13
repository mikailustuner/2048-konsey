#pragma once
#include <functional>

#include "engine/board.hpp"

namespace k2048 {

// Value of an afterstate (board right after a move, before the spawn).
using ValueFn = std::function<float(Board)>;

// 1-ply: argmax over legal moves of reward + V(afterstate). -1 if no legal move.
int greedy_move(Board b, const ValueFn& v);

// Expectimax with `depth` move layers (depth 1 == greedy): move -> chance (all empty cells,
// 2 with 0.9 / 4 with 0.1) -> move ... leaves are V(afterstate). Positions with no legal
// move are worth 0.
int expectimax_move(Board b, int depth, const ValueFn& v);

}  // namespace k2048

#include <vector>

namespace k2048 {

// Faz 6a: fast expectimax.
struct SearchConfig {
    int depth = 2;         // move layers (1 = greedy)
    double time_ms = 0;    // > 0: iterative deepening up to `depth`, stopping when the next depth
                           // is predicted to exceed this budget per move
    float prob_cut = 0;    // branches whose probability falls below this are cut to 1-ply
    int tt_bits = 20;      // chance-node transposition table: 2^bits entries (0 = off)
    int downgrade = 0;     // > 0: tile-downgrading once the board holds 2^downgrade (see tile_downgrade)
};

class Searcher {
public:
    Searcher(ValueFn v, SearchConfig c);
    int choose(Board b);
    uint64_t nodes() const { return nodes_; }
    int last_depth() const { return last_depth_; }

private:
    float max_node(Board b, int depth, float prob, int* best_dir);
    float chance_node(Board a, int depth, float prob);

    struct Entry {
        Board key = 0;
        int depth = 0;
        float value = 0;
    };
    ValueFn v_;
    SearchConfig c_;
    std::vector<Entry> tt_;
    uint64_t nodes_ = 0;
    int last_depth_ = 0;
};

// Tile-downgrading (Guei et al. 2021): while the board holds a tile >= 2^threshold and some
// exponent between 2^6 and its largest tile is missing, every tile above the largest such gap
// is lowered by one. Equal neighbours stay equal and small tiles are untouched, so the moves are
// the same, while the network sees the large-tile region it was trained on.
Board tile_downgrade(Board b, int threshold);

}  // namespace k2048
