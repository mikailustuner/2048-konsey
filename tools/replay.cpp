// replay: re-plays every game in a replays.jsonl file from its seed and move string and checks
// that score and max tile match the recorded values.
#include <cstdio>
#include <fstream>

#include "../engine/game.hpp"
#include "cli.hpp"

using namespace k2048;

static std::string field(const std::string& line, const std::string& key) {
    size_t p = line.find("\"" + key + "\":");
    if (p == std::string::npos) throw std::runtime_error("missing field " + key);
    p += key.size() + 3;
    if (line[p] == '"') return line.substr(p + 1, line.find('"', p + 1) - p - 1);
    return line.substr(p, line.find_first_of(",}", p) - p);
}

int main(int argc, char** argv) {
    return run_main([&] {
        Args a(argc, argv);
        std::string path = a.str("file", "runs/eval/replays.jsonl");
        a.check();
        std::ifstream f(path);
        if (!f) throw std::runtime_error("cannot open " + path);
        int ok = 0, bad = 0;
        for (std::string line; std::getline(f, line);) {
            if (line.empty()) continue;
            uint64_t seed = std::stoull(field(line, "seed"));
            GameResult g = replay_game(seed, field(line, "moves"));
            bool same = std::to_string(g.score) == field(line, "score") &&
                        std::to_string(1ULL << g.max_exp) == field(line, "max_tile");
            (same ? ok : bad)++;
            if (!same) std::printf("UYUSMAZLIK seed %llu\n", (unsigned long long)seed);
        }
        std::printf("tekrar: %d eslesti, %d uyusmadi\n", ok, bad);
        return bad ? 1 : 0;
    });
}
