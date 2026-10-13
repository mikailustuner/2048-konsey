// play: plays one game live and prints one JSON line per position (for the GUI gateway).
//   play --agent expectimax --depth 2 --net model.bin --seed 42
// Lines: {"t":"pos","i":move_index,"cells":[16 exponents],"score":S,"dir":"L","ms":think_time}
//        {"t":"end","cells":[...],"score":S,"max_tile":T,"moves":N,"record":"..."}
#include <chrono>
#include <cstdio>
#include <string>

#include "engine/game.hpp"
#include "tools/agents.hpp"

using namespace k2048;

static std::string cells_json(const Grid& g) {
    std::string s = "[";
    for (int i = 0; i < 16; ++i) s += (i ? "," : "") + std::to_string(g[size_t(i)]);
    return s + "]";
}

int main(int argc, char** argv) {
    return run_main([&] {
        Args a(argc, argv);
        uint64_t seed = uint64_t(a.num("seed", 1));
        auto agent = make_agent(a);
        a.check();

        Policy inner = agent();
        uint64_t score = 0;
        int i = 0;
        // The policy sees every position before its move; that is the live stream.
        Policy live = [&](Board b) {
            auto t0 = std::chrono::steady_clock::now();
            int d = inner(b);
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            std::printf("{\"t\":\"pos\",\"i\":%d,\"cells\":%s,\"score\":%llu,\"dir\":\"%c\",\"ms\":%.2f}\n", i,
                        cells_json(to_grid(b)).c_str(), (unsigned long long)score, d >= 0 ? kDirChar[d] : '-', ms);
            std::fflush(stdout);
            uint32_t r = 0;
            if (d >= 0) move(b, d, r);
            score += r;
            ++i;
            return d;
        };
        GameResult g = play_game(seed, live, true);
        std::printf("{\"t\":\"end\",\"cells\":%s,\"score\":%llu,\"max_tile\":%llu,\"moves\":%d,\"seed\":%llu,\"record\":\"%s\"}\n",
                    cells_json(g.final).c_str(), (unsigned long long)g.score, 1ULL << g.max_exp, g.moves,
                    (unsigned long long)seed, g.record.c_str());
        return 0;
    });
}
