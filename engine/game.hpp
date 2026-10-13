#pragma once
#include <functional>
#include <string>

#include "board.hpp"

namespace k2048 {

// A policy sees a fast board and returns a direction; it must return a legal move whenever
// one exists. Positions holding 65536+ are shown through downgrade().
using Policy = std::function<int(Board)>;

struct GameResult {
    uint64_t seed = 0;
    uint64_t score = 0;
    int max_exp = 0;  // exact, may exceed 15
    int moves = 0;
    std::string record;  // one char per move (U/R/D/L), filled when recording
};

// Plays one full game from `seed`. Spawns come from Rng(seed) only, so a game is fully
// determined by its seed and move sequence. `start` != 0 replaces the two initial spawns
// (used for tests and endgame positions).
GameResult play_game(uint64_t seed, const Policy& policy, bool record = false, Board start = 0);

// Replays a recorded game; throws std::runtime_error on an illegal move.
GameResult replay_game(uint64_t seed, const std::string& record);

int dir_from_char(char c);

}  // namespace k2048
