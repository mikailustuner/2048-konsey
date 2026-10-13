#pragma once
// Faz 6b: endgame tablebase.
//
// A pattern locks big tiles on a set of cells (`fixed`) and asks for the tile 2^target to be
// built in the remaining free cells. The table stores, for every reachable position, the
// probability of building it under optimal play, where a move that would move or merge a locked
// tile is not allowed. Locked tiles are represented by placeholders 15, 14, ... (ascending cell
// order): they are larger than any free tile, distinct, and never move in an allowed move, so a
// real board whose locked cells hold distinct tiles >= 2^target behaves exactly the same.
//
// Positions are layered by the sum of their free tiles: a move keeps it, a spawn adds 2 or 4.
// build() walks the layers forward from the smallest positions (all reachable positions), then
// backward from the largest sums, keeping at most three layers in memory; every layer is a
// sorted file on disk (states + probabilities) that the reader maps into memory.
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "engine/board.hpp"

namespace k2048 {

struct Pattern {
    uint16_t fixed = 0;  // bit i: cell i holds a locked tile
    int target = 0;      // exponent to build in the free cells

    int locked() const { return __builtin_popcount(fixed); }
    // "0,1,2,3:11" -> cells 0..3 locked, build 2048.
    static Pattern parse(const std::string& spec);
    std::string str() const;
};

struct BuildStats {
    uint64_t states = 0;
    int layers = 0;
    double seconds = 0;
};

// Builds the table into `dir` (created). Progress lines go to stdout unless quiet. Throws, before
// the backward pass, once more than max_states positions have been generated (0 = no limit), so
// an oversized pattern cannot fill the disk.
BuildStats build_tablebase(const Pattern& p, const std::string& dir, int threads, bool quiet = false,
                           uint64_t max_states = 0);

class Tablebase {
public:
    explicit Tablebase(const std::string& dir);
    ~Tablebase();
    Tablebase(const Tablebase&) = delete;
    Tablebase& operator=(const Tablebase&) = delete;

    const Pattern& pattern() const { return pattern_; }
    uint64_t states() const;

    // The table position of a real board, if the pattern applies: locked cells hold distinct
    // tiles whose smallest is exactly 2^target (the table builds the tile that merges into that
    // chain end; with larger locked tiles building the target serves nothing and the table would
    // chase a useless goal), free cells only smaller tiles. Returns false otherwise.
    bool to_table(Board real, Board& t) const;
    // Pattern rule of to_table() for a pattern without a table (coverage measurements).
    static bool applies(const Pattern& p, Board real);
    // Success probability of a table position (player to move); -1 if it is not in the table.
    float prob(Board t) const;
    // Best allowed move for a real board and its success probability; -1 if the table does not
    // know the position (pattern does not apply, or the position was never reachable).
    int best_move(Board real, float* p = nullptr) const;
    // best_move over the 8 symmetries of the board (the pattern may sit in any corner or edge);
    // returns the real direction with the highest success probability, or -1.
    int best_move_any(Board real, float* p = nullptr) const;
    // Success probability of each real move (-1: not allowed / unknown), over the 8 symmetries.
    // Returns false if the table does not apply to the board in any orientation.
    bool move_probs_any(Board real, float probs[4]) const;

private:
    struct Layer;
    float lookup(const Layer& l, Board t) const;
    Pattern pattern_;
    std::vector<std::unique_ptr<Layer>> layers_;  // index = (sum of free tiles) / 2
};

// Shared rules (also used by the tests' brute-force solver).
namespace tb {
// The table board for free-cell contents `b` (locked cells set to the placeholders).
Board with_placeholders(const Pattern& p, Board b);
// Sum of the free tiles / 2.
int layer_of(const Pattern& p, Board t);
// Allowed move: legal and the locked cells unchanged. Returns the afterstate or t if not allowed.
Board allowed_move(const Pattern& p, Board t, int dir);
bool has_target(const Pattern& p, Board t);
}  // namespace tb

}  // namespace k2048
