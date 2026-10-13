#include "game.hpp"

#include <algorithm>
#include <stdexcept>

namespace k2048 {

int dir_from_char(char c) {
    for (int d = 0; d < 4; ++d)
        if (kDirChar[d] == c) return d;
    throw std::runtime_error(std::string("bad move char: ") + c);
}

namespace {

// Exact slow path, entered once two 32768s are on the board (the only way to make 65536).
void play_grid(Grid g, Rng& rng, const Policy& policy, bool record, GameResult& res) {
    while (ref::can_move(g)) {
        int dir = policy(downgrade(g));
        Grid next = g;
        uint64_t score = 0;
        if (dir < 0 || !ref::move(next, dir, score)) {
            // Downgraded view may disagree on 15+15 merges; fall back to the first legal move.
            for (dir = 0; dir < 4; ++dir) {
                next = g;
                score = 0;
                if (ref::move(next, dir, score)) break;
            }
        }
        g = next;
        res.score += score;
        res.moves++;
        if (record) res.record += kDirChar[dir];
        ref::spawn(g, rng);
    }
    for (auto v : g) res.max_exp = std::max<int>(res.max_exp, v);
}

}  // namespace

GameResult play_game(uint64_t seed, const Policy& policy, bool record) {
    GameResult res;
    res.seed = seed;
    Rng rng(seed);
    Board b = initial_board(rng);
    while (can_move(b)) {
        if (count_exp(b, kMaxFastExp) >= 2) {
            play_grid(to_grid(b), rng, policy, record, res);
            return res;
        }
        int dir = policy(b);
        uint32_t score = 0;
        Board next = dir < 0 ? b : move(b, dir, score);
        if (next == b) throw std::runtime_error("policy returned an illegal move");
        res.score += score;
        res.moves++;
        if (record) res.record += kDirChar[dir];
        b = spawn(next, rng);
    }
    res.max_exp = max_exp(b);
    return res;
}

GameResult replay_game(uint64_t seed, const std::string& record) {
    size_t i = 0;
    Policy p = [&](Board) {
        if (i >= record.size()) throw std::runtime_error("record ended before the game");
        return dir_from_char(record[i++]);
    };
    GameResult r = play_game(seed, p, true);
    if (i != record.size()) throw std::runtime_error("game ended before the record");
    return r;
}

}  // namespace k2048
