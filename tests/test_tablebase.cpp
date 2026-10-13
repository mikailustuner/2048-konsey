#include <algorithm>
#include <filesystem>
#include <fstream>
#include <unordered_map>

#include "doctest.h"
#include "ntuple/network.hpp"
#include "tablebase/tablebase.hpp"

using namespace k2048;

// Independent solver: top-down recursion with a memo, same rules (locked cells must not change).
static double brute(const Pattern& p, Board t, std::unordered_map<Board, double>& memo) {
    auto it = memo.find(t);
    if (it != memo.end()) return it->second;
    double best = 0;
    for (int d = 0; d < 4; ++d) {
        Board a = tb::allowed_move(p, t, d);
        if (a == t) continue;
        double v = 0;
        if (tb::has_target(p, a)) {
            v = 1;
        } else {
            int n = 0;
            for (int i = 0; i < 16; ++i)
                if (!cell(a, i)) {
                    v += 0.9 * brute(p, set_cell(a, i, 1), memo) + 0.1 * brute(p, set_cell(a, i, 2), memo);
                    n++;
                }
            v /= n;
        }
        best = std::max(best, v);
    }
    return memo[t] = best;
}

TEST_CASE("pattern parsing and placeholder rules") {
    Pattern p = Pattern::parse("0,1,2,3:11");
    CHECK(p.locked() == 4);
    CHECK(p.str() == "0,1,2,3:11");
    CHECK_THROWS(Pattern::parse("0,1,2,3,4,5:11"));  // placeholders would fall below the target
    Board t = tb::with_placeholders(p, 0);
    CHECK(cell(t, 0) == 15);
    CHECK(cell(t, 3) == 12);
    // A move that shifts locked tiles is not allowed: top row full, 'down' would move it.
    CHECK(tb::allowed_move(p, t, DOWN) == t);
    CHECK(tb::allowed_move(p, t, UP) == t);  // nothing moves
}

TEST_CASE("tablebase equals an independent brute-force solver on every position") {
    for (const char* spec : {"3,7,8,9,10,11,12,13,14,15:6", "2,3,6,7,10,11,14,15:5"}) {
        Pattern p = Pattern::parse(spec);
        std::string dir = (std::filesystem::temp_directory_path() / "k2048_tb_test").string();
        std::filesystem::remove_all(dir);
        BuildStats st = build_tablebase(p, dir, 4, true);
        Tablebase t(dir);
        CHECK(t.states() == st.states);
        std::unordered_map<Board, double> memo;
        size_t checked = 0;
        double maxdiff = 0;
        // Every stored position: walk the layer files through the public API.
        for (const auto& e : std::filesystem::directory_iterator(dir)) {
            if (e.path().extension() != ".states") continue;
            std::ifstream f(e.path(), std::ios::binary);
            for (Board b; f.read(reinterpret_cast<char*>(&b), sizeof b);) {
                maxdiff = std::max(maxdiff, std::abs(double(t.prob(b)) - brute(p, b, memo)));
                checked++;
            }
        }
        MESSAGE(std::string(spec) << ": " << checked << " durum, max fark " << maxdiff);
        CHECK(checked == st.states);
        CHECK(maxdiff < 1e-6);
        std::filesystem::remove_all(dir);
    }
}

TEST_CASE("best_move on a real board uses the table and refuses boards outside the pattern") {
    Pattern p = Pattern::parse("2,3,6,7,10,11,14,15:5");
    std::string dir = (std::filesystem::temp_directory_path() / "k2048_tb_test2").string();
    std::filesystem::remove_all(dir);
    build_tablebase(p, dir, 4, true);
    Tablebase t(dir);
    // Locked cells: distinct big tiles in any order; free cells: a small position.
    Board real = 0;
    int big[8] = {9, 14, 7, 12, 11, 5, 8, 13};
    int k = 0;
    for (int i = 0; i < 16; ++i)
        if (p.fixed >> i & 1) real = set_cell(real, i, big[k++]);
    real = set_cell(set_cell(real, 0, 1), 4, 1);
    float pr = -1;
    int d = t.best_move(real, &pr);
    REQUIRE(d >= 0);
    Board tbb;
    REQUIRE(t.to_table(real, tbb));
    CHECK(pr == doctest::Approx(t.prob(tbb)).epsilon(1e-6));
    CHECK(pr > 0);
    CHECK(move(real, d) != real);
    std::filesystem::remove_all(dir);
}

TEST_CASE("best_move_any finds the pattern in every orientation and maps the move back") {
    // An asymmetric pattern (right column + two bottom rows), so exactly one frame matches.
    Pattern p = Pattern::parse("3,7,8,9,10,11,12,13,14,15:6");
    std::string dir = (std::filesystem::temp_directory_path() / "k2048_tb_test3").string();
    std::filesystem::remove_all(dir);
    build_tablebase(p, dir, 4, true);
    Tablebase t(dir);
    Board real = 0;
    int big[10] = {9, 14, 7, 12, 11, 6, 8, 13, 15, 10};
    int k = 0;
    for (int i = 0; i < 16; ++i)
        if (p.fixed >> i & 1) real = set_cell(real, i, big[k++]);
    real = set_cell(set_cell(real, 0, 1), 5, 2);
    float pr = -1;
    int d = t.best_move(real, &pr);
    REQUIRE(d >= 0);
    for (int sym = 0; sym < 8; ++sym) {
        Board m = 0;
        for (int i = 0; i < 16; ++i) m = set_cell(m, NTupleNet::sym_cell(i, sym), cell(real, i));
        float q = -1;
        int dm = t.best_move_any(m, &q);
        REQUIRE(dm >= 0);
        CHECK(q == doctest::Approx(pr).epsilon(1e-6));
        Board back = 0, moved = move(m, dm);  // undo the symmetry: same afterstate as the table's move
        for (int i = 0; i < 16; ++i) back = set_cell(back, i, cell(moved, NTupleNet::sym_cell(i, sym)));
        CHECK(back == move(real, d));
        float pr4[4];
        REQUIRE(t.move_probs_any(m, pr4));
        CHECK(*std::max_element(pr4, pr4 + 4) == doctest::Approx(pr).epsilon(1e-6));
        CHECK(pr4[dm] == doctest::Approx(pr).epsilon(1e-6));
    }
    std::filesystem::remove_all(dir);
}

TEST_CASE("table refuses boards outside its pattern") {
    Pattern p = Pattern::parse("2,3,6,7,10,11,14,15:5");
    std::string dir = (std::filesystem::temp_directory_path() / "k2048_tb_test4").string();
    std::filesystem::remove_all(dir);
    build_tablebase(p, dir, 4, true);
    Tablebase t(dir);
    Board real = 0;
    int big[8] = {9, 14, 7, 12, 11, 5, 8, 13};
    int k = 0;
    for (int i = 0; i < 16; ++i)
        if (p.fixed >> i & 1) real = set_cell(real, i, big[k++]);
    real = set_cell(set_cell(real, 0, 1), 4, 1);
    CHECK(t.best_move(real) >= 0);
    CHECK(t.best_move(set_cell(real, 1, 6)) == -1);     // free tile >= target: not this table
    CHECK(t.best_move(set_cell(real, 2, 13)) == -1);    // locked tiles not distinct
    // Smallest locked tile above the target: building the target would not feed the chain.
    Board up = real;
    for (int i = 0; i < 16; ++i)
        if (p.fixed >> i & 1) up = set_cell(up, i, cell(real, i) + 1);
    CHECK(t.best_move(up) == -1);
    std::filesystem::remove_all(dir);
}
