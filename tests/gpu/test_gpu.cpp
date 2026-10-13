// GPU path vs the CPU reference. Run on a machine with a CUDA device (built with -DUSE_CUDA=ON).
#include <cmath>
#include <cstdio>

#include "doctest.h"
#include "gpu/gpu.hpp"
#include "ntuple/td.hpp"
#include "search/search.hpp"

using namespace k2048;

// 5-cell tuples (16 MB) where the network size does not matter: the test GPU has ~2 GB free.
static const std::vector<std::vector<int>> k4x5 = {{0, 1, 2, 3, 4}, {4, 5, 6, 7, 8}, {0, 1, 2, 4, 5}, {4, 5, 6, 8, 9}};
static const std::vector<std::vector<int>> k8x5 = {{0, 1, 2, 3, 4}, {4, 5, 6, 7, 8},  {0, 1, 2, 4, 5}, {4, 5, 6, 8, 9},
                                                   {0, 1, 5, 6, 7}, {0, 1, 2, 5, 9}, {0, 1, 5, 9, 13}, {0, 1, 5, 8, 9}};

// Integer weights keep every value sum exact, so GPU and CPU must agree bit for bit.
static void integer_weights(NTupleNet& n, uint64_t seed) {
    Rng r(seed);
    size_t stride = n.tc() ? 3 : 1;
    for (size_t i = 0; i < n.data_size(); i += stride) n.data()[i] = float(int(r.below(201)) - 100);
}

static EvalReport cpu_eval(const NTupleNet& n, int depth, const EvalConfig& c) {
    return evaluate(
        [&] {
            ValueFn v = [&](Board a) { return n.value(a); };
            return Policy([v, depth](Board b) { return expectimax_move(b, depth, v); });
        },
        c);
}

TEST_CASE("device info") { std::printf("GPU: %s\n", gpu_info().c_str()); }

TEST_CASE("greedy games match the CPU exactly (engine, RNG, features, symmetries, stages)") {
    NTupleNet net(k4x5, {{}, {7}, {7, 8}}, false);
    integer_weights(net, 1);
    for (int k = 1; k < 3; ++k) net.set_seen(k, true);
    GpuPopulation gp({&net}, 64);
    EvalConfig c;
    c.games = 300;
    c.threads = 8;
    EvalReport g = gp.evaluate({0}, {}, 1, c), h = cpu_eval(net, 1, c);
    REQUIRE(g.games() == 300);
    int same = 0;
    for (int i = 0; i < 300; ++i) {
        const auto &a = g.results[size_t(i)], &b = h.results[size_t(i)];
        same += a.score == b.score && a.moves == b.moves && a.max_exp == b.max_exp && a.seed == b.seed;
    }
    CHECK(same == 300);
}

TEST_CASE("depth-2 expectimax matches the CPU exactly") {
    NTupleNet net(NTupleNet::preset("4x6"), {{}}, false);  // the real 6-cell index path
    integer_weights(net, 2);
    GpuPopulation gp({&net}, 64);
    EvalConfig c;
    c.games = 40;
    c.threads = 8;
    EvalReport g = gp.evaluate({0}, {}, 2, c), h = cpu_eval(net, 2, c);
    int same = 0;
    for (int i = 0; i < c.games; ++i) same += g.results[size_t(i)].score == h.results[size_t(i)].score;
    CHECK(same == c.games);
}

TEST_CASE("council mean on the GPU equals one net holding the averaged weights") {
    NTupleNet a(k4x5, {{}}, false), b(k4x5, {{}}, false), avg(k4x5, {{}}, false);
    integer_weights(a, 3);
    integer_weights(avg, 4);
    // b chosen so (3a + b) / 4 == avg exactly: every sum stays an exact integer.
    for (size_t i = 0; i < a.data_size(); ++i) b.data()[i] = 4 * avg.data()[i] - 3 * a.data()[i];
    GpuPopulation gp({&a, &b, &avg}, 64);
    EvalConfig c;
    c.games = 100;
    EvalReport m = gp.evaluate({0, 1}, {3, 1}, 1, c), s = gp.evaluate({2}, {}, 1, c);
    for (int i = 0; i < c.games; ++i) CHECK(m.results[size_t(i)].score == s.results[size_t(i)].score);
}

TEST_CASE("single-slot TD training follows the CPU trajectory") {
    for (bool tc : {false, true}) {
        NTupleNet cpu(k4x5, {{}}, tc), gpu_net(k4x5, {{}}, tc);
        TDParams p;
        p.alpha = tc ? 1.0f : 0.1f;
        std::vector<GameResult> cpu_games;
        for (uint64_t g = 0; g < 3; ++g) cpu_games.push_back(td_episode(cpu, p, 500 + g));
        GpuPopulation gp({&gpu_net}, 1);
        gp.set_moves_per_launch(1);  // weights updated after every move, unlimited batch: online TD
        std::vector<int64_t> done{0};
        std::vector<GameResult> gpu_games;
        gp.train({{p.alpha, 1e9f}}, done, {3}, {500}, 1, [&](int, const EvalReport& r, int64_t) { gpu_games.push_back(r.results[0]); });
        gp.download(0, gpu_net);
        REQUIRE(gpu_games.size() == 3);
        // One slot, one move per launch, no cap: sequential TD (TC included), so the games match.
        CHECK(gpu_games[0].score == cpu_games[0].score);
        double maxdiff = 0, maxw = 0;
        for (size_t i = 0; i < cpu.data_size(); ++i) {
            maxdiff = std::max(maxdiff, double(std::fabs(cpu.data()[i] - gpu_net.data()[i])));
            maxw = std::max(maxw, double(std::fabs(cpu.data()[i])));
        }
        std::printf("tc=%d: CPU %llu/%llu/%llu  GPU %llu/%llu/%llu  max |dw| %.3g (max |w| %.3g)\n", tc,
                    (unsigned long long)cpu_games[0].score, (unsigned long long)cpu_games[1].score,
                    (unsigned long long)cpu_games[2].score, (unsigned long long)gpu_games[0].score,
                    (unsigned long long)gpu_games[1].score, (unsigned long long)gpu_games[2].score, maxdiff, maxw);
        CHECK(maxdiff < 1e-3 * maxw);
    }
}

TEST_CASE("batched GPU training learns; copy_agent and multi-stage promotion work") {
    NTupleNet a(k4x5, {{}, {9}}, false), b(k4x5, {{}, {9}}, false);
    GpuPopulation gp({&a, &b});
    EvalConfig c;
    c.games = 500;
    double before = gp.evaluate({0}, {}, 1, c).mean_score();
    std::vector<int64_t> done{0, 0};
    int windows = 0;
    gp.train({{0.1f}, {0.1f}}, done, {20000, 1000}, {1ULL << 40, 2ULL << 40}, 5000,
             [&](int, const EvalReport&, int64_t) { windows++; });
    CHECK(done[0] == 20000);
    CHECK(done[1] == 1000);
    CHECK(windows == 4);
    EvalReport after = gp.evaluate({0}, {}, 1, c);
    std::printf("GPU egitim: once %.0f, 20k oyun sonra %.0f (512: %llu/500)\n", before, after.mean_score(),
                (unsigned long long)after.reached(9));
    CHECK(after.mean_score() > 4 * before);
    gp.download(0, a);
    CHECK(a.seen(1));  // 512 reached during training -> stage 1 promoted
    gp.copy_agent(1, 0);
    EvalReport copy = gp.evaluate({1}, {}, 1, c);
    CHECK(copy.mean_score() == after.mean_score());
    // Downloaded weights give the same games on the CPU (greedy, non-integer weights: allow rare ties).
    EvalReport cpu = cpu_eval(a, 1, c);
    int same = 0;
    for (int i = 0; i < c.games; ++i) same += cpu.results[size_t(i)].score == after.results[size_t(i)].score;
    CHECK(same == c.games);  // bit-identical values
}

TEST_CASE("mixed shapes in one population (4x5 TC + 8x5 multi-stage) match the CPU") {
    NTupleNet a(k4x5, {{}}, true), b(k8x5, {{}, {8}}, false);
    integer_weights(a, 5);
    integer_weights(b, 6);
    b.set_seen(1, true);
    GpuPopulation gp({&a, &b}, 64);
    EvalConfig c;
    c.games = 100;
    c.threads = 8;
    for (int i : {0, 1}) {
        const NTupleNet& n = i ? b : a;
        EvalReport g = gp.evaluate({i}, {}, 1, c), h = cpu_eval(n, 1, c);
        int same = 0;
        for (int k = 0; k < c.games; ++k) same += g.results[size_t(k)].score == h.results[size_t(k)].score;
        CHECK(same == c.games);
    }
    CHECK_THROWS(gp.copy_agent(0, 1));
}

TEST_CASE("one network on every slot stays stable (batched updates)") {
    NTupleNet n(NTupleNet::preset("4x6"), {{}}, false);
    GpuPopulation gp({&n});  // all device slots on one net: Hogwild diverged here (768 x 0.1)
    std::vector<int64_t> done{0};
    double last = 0;
    gp.train({{0.1f}}, done, {40000}, {7ULL << 40}, 10000, [&](int, const EvalReport& w, int64_t d) {
        last = w.mean_score();
        std::printf("tum slotlar: %lld oyun, ort %.0f\n", (long long)d, last);
    });
    CHECK(last > 20000);
}
