#include <set>

#include "doctest.h"
#include "ntuple/td.hpp"
#include "search/search.hpp"

using namespace k2048;

// Positions from real (random-play) games.
static std::vector<Board> positions(int n, uint64_t seed) {
    std::vector<Board> out;
    Rng r(seed);
    while (int(out.size()) < n) {
        Board b = initial_board(r);
        while (can_move(b) && int(out.size()) < n) {
            if (r.below(8) == 0) out.push_back(b);
            Board x;
            while ((x = move(b, int(r.below(4)))) == b) {}
            b = spawn(x, r);
        }
    }
    return out;
}

static std::unique_ptr<NTupleNet> trained() {
    auto n = std::make_unique<NTupleNet>(std::vector<std::vector<int>>{{0, 1, 2, 3}, {4, 5, 6, 7}, {0, 1, 4, 5}},
                                         std::vector<std::vector<int>>{{}}, false);
    for (uint64_t g = 0; g < 500; ++g) td_episode(*n, {}, 900 + g);
    return n;
}

TEST_CASE("Searcher with the transposition table equals plain expectimax and visits fewer nodes") {
    auto net = trained();
    ValueFn v = [&](Board a) { return net->value(a); };
    for (int depth : {2, 3}) {
        SearchConfig with, without;
        with.depth = without.depth = depth;
        without.tt_bits = 0;
        Searcher s1(v, with), s0(v, without);
        for (Board b : positions(depth == 2 ? 300 : 40, 11)) {
            int d0 = expectimax_move(b, depth, v);
            CHECK(s1.choose(b) == d0);
            CHECK(s0.choose(b) == d0);
        }
        MESSAGE("depth " << depth << ": nodes with TT " << s1.nodes() << ", without " << s0.nodes());
        CHECK(s1.nodes() < s0.nodes());
    }
}

TEST_CASE("probability cut and time budget still return legal moves") {
    auto net = trained();
    SearchConfig c;
    c.depth = 4;
    c.prob_cut = 1e-3f;
    c.time_ms = 5;
    Searcher s([&](Board a) { return net->value(a); }, c);
    for (Board b : positions(50, 12)) {
        int d = s.choose(b);
        REQUIRE(d >= 0);
        CHECK(move(b, d) != b);
        CHECK(s.last_depth() >= 1);
        CHECK(s.last_depth() <= 4);
    }
}

TEST_CASE("tile downgrading keeps the moves and fills the largest gap") {
    // 32768 16384 4096 | 2 ... : 8192 missing -> 16384 8192 4096
    Board b = Board(15) | Board(14) << 4 | Board(12) << 8 | Board(1) << 16;
    Board d = tile_downgrade(b, 15);
    CHECK(cell(d, 0) == 14);
    CHECK(cell(d, 1) == 13);
    CHECK(cell(d, 2) == 12);
    CHECK(cell(d, 4) == 1);
    CHECK(tile_downgrade(b, 16) == b);  // below threshold: unchanged
    Rng r(3);
    for (int i = 0; i < 20000; ++i) {
        Board x = 0;
        for (int c = 0; c < 16; ++c) x = set_cell(x, c, int(r.below(16)));
        if (count_exp(x, 15) >= 2) continue;  // 15+15 does not merge on the fast path (65536 is exact-path only)
        Board y = tile_downgrade(x, 14);
        for (int dir = 0; dir < 4; ++dir) CHECK((move(x, dir) != x) == (move(y, dir) != y));
        for (int c = 0; c < 16; ++c)
            if (cell(x, c) <= 6) CHECK(cell(y, c) == cell(x, c));  // small tiles untouched
    }
}
