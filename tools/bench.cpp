// bench: single-thread move throughput on positions from real (random-play) games.
#include <chrono>
#include <cstdio>
#include <vector>

#include "../engine/game.hpp"
#include "cli.hpp"

using namespace k2048;

int main(int argc, char** argv) {
    return run_main([&] {
        Args a(argc, argv);
        long long n_moves = a.num("moves", 200'000'000);
        a.check();

        // Collect positions from random games.
        std::vector<Board> pos;
        Rng rng(42);
        while (pos.size() < (1u << 20)) {
            Board b = initial_board(rng);
            while (can_move(b)) {
                pos.push_back(b);
                Board n;
                while ((n = move(b, int(rng.below(4)))) == b) {}
                b = spawn(n, rng);
            }
        }
        const size_t mask = (1u << 20) - 1;

        auto t0 = std::chrono::steady_clock::now();
        uint64_t sink = 0;
        uint32_t score = 0;
        for (long long i = 0; i < n_moves; ++i) sink ^= move(pos[size_t(i >> 2) & mask], int(i & 3), score);
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("hamle: %.1f M/sn (%lld hamle, %.2f sn) [sink %llx %u]\n", n_moves / sec / 1e6, n_moves, sec,
                    (unsigned long long)sink, score);

        t0 = std::chrono::steady_clock::now();
        long long games = 0, moves = 0;
        Rng prng(7);
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 2.0) {
            GameResult g = play_game(uint64_t(games), [&](Board b) {
                int d;
                while (move(b, d = int(prng.below(4))) == b) {}
                return d;
            });
            games++;
            moves += g.moves;
        }
        sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("rastgele oyun: %.0f oyun/sn, %.1f M hamle/sn (spawn + oyun sonu kontrolu dahil)\n",
                    games / sec, moves / sec / 1e6);
        return 0;
    });
}
