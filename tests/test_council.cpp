#include <filesystem>

#include "council/council.hpp"
#include "doctest.h"
#include "evolve/pbt.hpp"
#include "search/search.hpp"

#include <fstream>

using namespace k2048;

static std::shared_ptr<NTupleNet> trained(uint64_t seed) {
    auto n = std::make_shared<NTupleNet>(std::vector<std::vector<int>>{{0, 1, 2, 3}, {0, 1, 4, 5}},
                                         std::vector<std::vector<int>>{{}}, false);
    for (uint64_t g = 0; g < 300; ++g) td_episode(*n, {}, seed + g);
    return n;
}

TEST_CASE("council combines values and votes") {
    auto a = trained(100), b = trained(9000);
    Council c;
    c.nets = {a, b};
    Board x = 0x0000000012341234ULL;
    CHECK(c.value(x) == doctest::Approx((a->value(x) + b->value(x)) / 2));
    c.mode = Council::Mode::Weighted;
    c.weights = {3, 1};
    CHECK(c.value(x) == doctest::Approx((3 * a->value(x) + b->value(x)) / 4));
    // A single-member vote equals that member's greedy move.
    Council one;
    one.nets = {a};
    one.mode = Council::Mode::Vote;
    ValueFn v = [&](Board s) { return a->value(s); };
    Rng r(1);
    for (int i = 0; i < 200; ++i) {
        Board bd = r.next() & 0x3333333333333333ULL;
        CHECK(one.choose(bd, 1) == greedy_move(bd, v));
    }
}

TEST_CASE("distillation moves the student towards the teacher") {
    Council c;
    c.nets = {trained(100)};
    NTupleNet student({{0, 1, 2, 3}, {0, 1, 4, 5}}, {{}}, false);
    Rng r(2);
    std::vector<Board> probe;
    for (int i = 0; i < 300; ++i) probe.push_back(r.next() & 0x3333333333333333ULL);
    auto err = [&] {
        double e = 0;
        for (Board p : probe) e += std::abs(student.value(p) - c.value(p));
        return e;
    };
    double before = err();
    distill(student, c, 300, 1, 77, 0.1f, "", {});
    CHECK(err() < 0.5 * before);
}

TEST_CASE("mutation stays in range and PBT runs and resumes") {
    Rng r(3);
    Genome g{0.1f, 0.85f, 0};
    for (int i = 0; i < 50; ++i) {
        g = mutate(g, r);
        CHECK(g.lambda >= 0);
        CHECK(g.lambda <= 0.9f);
        CHECK(g.alpha > 0);
    }
    PBTConfig c;
    c.dir = (std::filesystem::temp_directory_path() / "k2048_pbt_test").string();
    std::filesystem::remove_all(c.dir);
    c.population = 4;
    c.generations = 2;
    c.train_games = 50;
    c.tourney_games = 20;
    c.threads = 4;
    c.net = "4x6";
    run_pbt(c);
    c.generations = 3;  // resume for one more generation
    run_pbt(c);
    std::ifstream f(c.dir + "/population.txt");
    int gen = 0;
    f >> gen;
    CHECK(gen == 3);
    int lines = 0;
    std::ifstream gl(c.dir + "/generations.jsonl");
    for (std::string s; std::getline(gl, s);) lines++;
    CHECK(lines == 12);
    std::filesystem::remove_all(c.dir);
}
