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
