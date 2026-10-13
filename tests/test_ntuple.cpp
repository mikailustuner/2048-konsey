#include <cstdio>
#include <filesystem>

#include "doctest.h"
#include "ntuple/td.hpp"
#include "search/search.hpp"

using namespace k2048;

static Board flip_cols(Board b) {
    Board r = 0;
    for (int i = 0; i < 16; ++i) r = set_cell(r, (i / 4) * 4 + 3 - i % 4, cell(b, i));
    return r;
}

// Small 4-cell tuples keep the tests light.
static NTupleNet small_net(bool tc, std::vector<std::vector<int>> stages = {{}}) {
    return NTupleNet({{0, 1, 2, 3}, {4, 5, 6, 7}, {0, 1, 4, 5}}, stages, tc, 0);
}

TEST_CASE("value is symmetric and update moves it towards the target") {
    NTupleNet net = small_net(false);
    Rng r(1);
    for (int i = 0; i < 200; ++i) {
        Board b = r.next() & 0x3333333333333333ULL;
        net.update(b, 100.0f - net.value(b), 0.5f / float(net.features()));
    }
    for (int i = 0; i < 200; ++i) {
        Board b = r.next() & 0x3333333333333333ULL;
        CHECK(net.value(b) == doctest::Approx(net.value(transpose(b))).epsilon(1e-4));
        CHECK(net.value(b) == doctest::Approx(net.value(flip_cols(b))).epsilon(1e-4));
    }
    Board b = 0x0000000000001234ULL;
    float before = std::abs(50.0f - net.value(b));
    net.update(b, 50.0f - net.value(b), 0.5f / float(net.features()));
    CHECK(std::abs(50.0f - net.value(b)) < before);
}

TEST_CASE("optimistic init and multi-stage promotion") {
    // Border-row tuple: its 8 symmetric images never see the interior cell 5.
    NTupleNet net({{0, 1, 2, 3}}, {{}, {5}}, false, 1000.0f);
    Board low = 0x1, high = low | (Board(5) << 20);
    CHECK(net.stage(low) == 0);
    CHECK(net.stage(high) == 1);
    CHECK(net.value(low) == doctest::Approx(1000.0f));
    net.update(low, 1.0f, 100.0f);
    CHECK(net.value(low) != doctest::Approx(1000.0f));
    CHECK(net.value(high) == doctest::Approx(1000.0f));  // stage 1 untouched
    net.promote_if_new(1);
    CHECK(net.value(high) == doctest::Approx(net.value(low)));  // stage 0 copied into stage 1
}

TEST_CASE("save/load round trip keeps values, stages and TC state") {
    NTupleNet net = small_net(true, {{}, {6}});
    Rng r(2);
    for (int i = 0; i < 500; ++i) net.update(r.next() & 0x7777777777777777ULL, float(i % 7) - 3, 0.01f);
    std::string path = (std::filesystem::temp_directory_path() / "k2048_net_test.bin").string();
    net.save(path, "games=42\n");
    std::string h;
    auto back = NTupleNet::load(path, &h);
    CHECK(ckpt_games(h) == 42);
    CHECK(back->tc());
    CHECK(back->stages() == 2);
    for (int i = 0; i < 100; ++i) {
        Board b = r.next() & 0x7777777777777777ULL;
        CHECK(back->value(b) == net.value(b));
    }
    std::filesystem::remove(path);
}

TEST_CASE("TD training improves a small network") {
    NTupleNet net = small_net(false);
    TDParams p;
    auto mean = [&](uint64_t seed0, bool learn) {
        double s = 0;
        for (uint64_t g = 0; g < 300; ++g) {
            if (learn) {
                s += double(td_episode(net, p, seed0 + g).score);
            } else {
                ValueFn v = [&](Board a) { return net.value(a); };
                s += double(play_game(seed0 + g, [&](Board b) { return greedy_move(b, v); }).score);
            }
        }
        return s / 300;
    };
    double before = mean(1000, false);
    for (int i = 0; i < 10; ++i) mean(uint64_t(i) * 1000 + 50000, true);
    double after = mean(1000, false);
    CHECK(after > before * 1.5);
}

TEST_CASE("expectimax depth 2 is at least as strong as greedy on a crude evaluator") {
    // Empty-cell count as value: search should help.
    ValueFn v = [](Board a) { return float(count_empty(a)) * 100.0f; };
    double g1 = 0, g2 = 0;
    for (uint64_t s = 0; s < 40; ++s) {
        g1 += double(play_game(s, [&](Board b) { return greedy_move(b, v); }).score);
        g2 += double(play_game(s, [&](Board b) { return expectimax_move(b, 2, v); }).score);
    }
    CHECK(g2 > g1);
}
