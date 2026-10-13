#include "doctest.h"
#include "engine/eval.hpp"
#include "tools/agents.hpp"

using namespace k2048;

TEST_CASE("wilson interval") {
    Wilson w = wilson(50, 100);
    CHECK(w.p == doctest::Approx(0.5));
    CHECK(w.lo == doctest::Approx(0.4038).epsilon(0.001));
    CHECK(w.hi == doctest::Approx(0.5962).epsilon(0.001));
    w = wilson(0, 1000);
    CHECK(w.lo == 0);
    CHECK(w.hi == doctest::Approx(0.00383).epsilon(0.01));
}

TEST_CASE("evaluation is deterministic across thread counts") {
    const char* argv[] = {"x", "--agent", "random"};
    Args args(3, const_cast<char**>(argv));
    auto agent = make_agent(args);
    EvalConfig c;
    c.games = 200;
    c.threads = 1;
    EvalReport a = evaluate(agent, c);
    c.threads = 4;
    EvalReport b = evaluate(agent, c);
    REQUIRE(a.games() == 200);
    for (int i = 0; i < 200; ++i) {
        CHECK(a.results[size_t(i)].score == b.results[size_t(i)].score);
        CHECK(a.results[size_t(i)].seed == c.seed0 + uint64_t(i));
    }
    std::string line = log_line({"r", "0", "random", 0}, a);
    CHECK(line.find("\"oyun_sayisi\":200") != std::string::npos);
    CHECK(line.find("\"oranlar\":{\"2048\":") != std::string::npos);
}
