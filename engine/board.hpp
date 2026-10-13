#pragma once
#include <array>
#include <cstdint>

#include "rng.hpp"

namespace k2048 {

// Fast board: 16 cells x 4 bits (log2 of the tile, 0 = empty), cell i = row i/4, column i%4,
// stored at bits 4i..4i+3. Holds tiles up to 32768 (=15). Two 32768s would merge into 65536,
// which does not fit: the fast tables treat 15+15 as non-merging and the game driver switches
// to the exact Grid path before that can matter (see game.cpp).
using Board = uint64_t;

// Exact board for any tile (exponents up to 17 = 131072). Slow; used as the reference
// implementation in tests and for the rare positions holding 65536+.
using Grid = std::array<uint8_t, 16>;

enum Dir : int { UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3 };
inline constexpr char kDirChar[4] = {'U', 'R', 'D', 'L'};
inline constexpr int kMaxFastExp = 15;

inline int cell(Board b, int i) { return int(b >> (4 * i)) & 0xF; }
inline Board set_cell(Board b, int i, int v) {
    return (b & ~(Board(0xF) << (4 * i))) | (Board(v) << (4 * i));
}

// Applies a move. Returns the new board (== b if the move is illegal) and adds the merge score.
Board move(Board b, int dir, uint32_t& score);
inline Board move(Board b, int dir) {
    uint32_t s = 0;
    return move(b, dir, s);
}

Board transpose(Board b);
int count_empty(Board b);
int max_exp(Board b);
bool has_exp(Board b, int e);
bool can_move(Board b);
// Count of cells holding exponent e.
int count_exp(Board b, int e);

// Places a 2 (90%) or 4 (10%) on a uniformly chosen empty cell. Consumes exactly two RNG
// draws; empties are enumerated in ascending cell order. Board must have an empty cell.
Board spawn(Board b, Rng& rng);

// Starting position: two spawns on an empty board.
Board initial_board(Rng& rng);

Grid to_grid(Board b);
// Requires every exponent <= 15.
Board from_grid(const Grid& g);

// Reference (slow, obviously-correct) implementation over Grid, any exponent.
namespace ref {
// Returns true if anything moved; adds merge score.
bool move(Grid& g, int dir, uint64_t& score);
void spawn(Grid& g, Rng& rng);
bool can_move(const Grid& g);
}  // namespace ref

// Tile-downgrading view of a Grid with 65536+ tiles: while some exponent exceeds 15, the
// smallest missing exponent above 0 is removed by lowering every larger tile by one. Keeps
// emptiness and equal-neighbour relations, so the set of legal moves is unchanged.
Board downgrade(const Grid& g);

// Fixed-cell mask API for endgame tables: cells in `fixed` must hold the values in
// `fixed_vals`; the remaining cells range over exponents 0..max_exp and are indexed in mixed
// radix (ascending cell order).
struct MaskedSpace {
    uint16_t fixed = 0;
    Board fixed_vals = 0;
    int max_exp = 0;

    int free_cells() const { return 16 - __builtin_popcount(fixed); }
    uint64_t size() const;
    // True if fixed cells match and every free cell is <= max_exp.
    bool contains(Board b) const;
    uint64_t index(Board b) const;
    Board board(uint64_t idx) const;
};

}  // namespace k2048
