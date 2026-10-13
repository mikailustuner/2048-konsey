#include <algorithm>
#include <cmath>
#include <set>

#include "doctest.h"
#include "engine/game.hpp"

using namespace k2048;

static Board row_board(int a, int b, int c, int d) {
    return Board(a) | Board(b) << 4 | Board(c) << 8 | Board(d) << 12;
}

// Random board with exponents 0..max_e.
static Board random_board(Rng& r, int max_e) {
    Board b = 0;
    for (int i = 0; i < 16; ++i) b = set_cell(b, i, int(r.below(uint32_t(max_e + 1))));
    return b;
}

TEST_CASE("row merges") {
    uint32_t s = 0;
    CHECK(move(row_board(1, 1, 1, 1), LEFT, s) == row_board(2, 2, 0, 0));
    CHECK(s == 8);
    s = 0;
    CHECK(move(row_board(2, 1, 1, 0), LEFT, s) == row_board(2, 2, 0, 0));
    CHECK(s == 4);
    CHECK(move(row_board(0, 2, 2, 2), RIGHT) == row_board(0, 0, 2, 3));
    CHECK(move(row_board(1, 2, 1, 2), LEFT) == row_board(1, 2, 1, 2));  // illegal: unchanged
    CHECK(move(row_board(15, 15, 0, 0), LEFT) == row_board(15, 15, 0, 0));  // fast path cap
}

TEST_CASE("transpose is an involution and swaps rows/cols") {
    Rng r(1);
    for (int i = 0; i < 1000; ++i) {
        Board b = r.next();
        CHECK(transpose(transpose(b)) == b);
        Board t = transpose(b);
        for (int c = 0; c < 16; ++c) CHECK(cell(t, (c % 4) * 4 + c / 4) == cell(b, c));
    }
}

TEST_CASE("fast moves match the reference on 4M random boards") {
    Rng r(2);
    long checked = 0, mismatch = 0;
    for (int i = 0; i < 1'000'000; ++i) {
        Board b = random_board(r, i % 2 ? 15 : 4);  // dense (0..4) boards exercise merges
        for (int d = 0; d < 4; ++d) {
            Grid g = to_grid(b);
            uint64_t rs = 0;
            bool moved = ref::move(g, d, rs);
            if (*std::max_element(g.begin(), g.end()) > 15) continue;  // 65536: fast path n/a
            uint32_t fs = 0;
            Board f = move(b, d, fs);
            mismatch += f != from_grid(g) || fs != rs || (f != b) != moved;
            checked++;
        }
    }
    CHECK(mismatch == 0);
    CHECK(checked > 3'900'000);
}

TEST_CASE("spawn and whole games match the reference") {
    for (uint64_t seed = 0; seed < 2000; ++seed) {
        Rng rf(seed), rr(seed), moves(seed + 99);
        Board b = initial_board(rf);
        Grid g{};
        ref::spawn(g, rr);
        ref::spawn(g, rr);
        REQUIRE(from_grid(g) == b);
        while (can_move(b)) {
            REQUIRE(ref::can_move(g));
            int d;
            while (move(b, d = int(moves.below(4))) == b) {}
            uint32_t fs = 0;
            uint64_t rs = 0;
            b = spawn(move(b, d, fs), rf);
            ref::move(g, d, rs);
            ref::spawn(g, rr);
            REQUIRE(fs == rs);
            REQUIRE(from_grid(g) == b);
        }
        CHECK_FALSE(ref::can_move(g));
    }
}

TEST_CASE("spawn distribution: ~10% fours, uniform cells") {
    Rng r(3);
    int fours = 0, n = 200'000;
    int where[16] = {};
    for (int i = 0; i < n; ++i) {
        Board b = spawn(0, r);
        for (int c = 0; c < 16; ++c)
            if (cell(b, c)) {
                where[c]++;
                fours += cell(b, c) == 2;
            }
    }
    CHECK(std::abs(fours / double(n) - 0.1) < 0.005);
    for (int c : where) CHECK(std::abs(c / double(n) - 1 / 16.0) < 0.005);
}

TEST_CASE("replay reproduces the game exactly") {
    Rng moves(5);
    auto pol = [&](Board b) {
        int d;
        while (move(b, d = int(moves.below(4))) == b) {}
        return d;
    };
    for (uint64_t seed = 10; seed < 60; ++seed) {
        GameResult g = play_game(seed, pol, true);
        CHECK(g.record.size() == size_t(g.moves));
        GameResult h = replay_game(seed, g.record);
        CHECK(h.score == g.score);
        CHECK(h.max_exp == g.max_exp);
        CHECK(h.record == g.record);
    }
    CHECK_THROWS(replay_game(10, "UUUU"));  // ends before the game does / illegal
}

TEST_CASE("reference merges past 32768 and downgrade keeps legal moves") {
    Grid g{};
    g[0] = 15;
    g[1] = 15;
    uint64_t s = 0;
    CHECK(ref::move(g, LEFT, s));
    CHECK(g[0] == 16);
    CHECK(s == 65536);

    Rng r(4);
    for (int i = 0; i < 100'000; ++i) {
        Grid h{};
        for (auto& v : h) v = uint8_t(r.below(18));
        Board v = downgrade(h);
        CHECK(max_exp(v) <= 15);
        std::set<int> before, after;
        for (int c = 0; c < 16; ++c) before.insert(h[c]), after.insert(cell(v, c));
        if (before.size() < 16 && after.size() == before.size()) {  // a gap existed
            for (int d = 0; d < 4; ++d) {
                Grid t = h;
                uint64_t sc = 0;
                bool legal = ref::move(t, d, sc);
                // equal-neighbour structure is preserved except 15+15, which fast tables refuse
                if (count_exp(v, 15) < 2) CHECK((move(v, d) != v) == legal);
            }
        }
    }
}

TEST_CASE("masked space index round trip") {
    MaskedSpace m;
    m.fixed = 0b1000'1000'1000'1111;  // top row + right column fixed
    m.fixed_vals = row_board(15, 14, 13, 12);
    m.fixed_vals = set_cell(set_cell(set_cell(m.fixed_vals, 7, 11), 11, 10), 15, 9);
    m.max_exp = 6;
    CHECK(m.free_cells() == 9);
    CHECK(m.size() == 40'353'607ULL);  // 7^9
    Rng r(6);
    for (int i = 0; i < 100'000; ++i) {
        uint64_t idx = r.next() % m.size();
        Board b = m.board(idx);
        CHECK(m.contains(b));
        CHECK(m.index(b) == idx);
    }
    CHECK_FALSE(m.contains(0));
}
