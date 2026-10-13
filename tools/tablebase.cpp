// tablebase: builds or inspects an endgame table (Faz 6b).
//   tablebase --build --pattern "0,1,2,3:11" --out tables/top4_2048 --threads 8
//   tablebase --info --dir tables/top4_2048
//   tablebase --coverage 50 --net net.bin --pattern "0,1,2,3,7:9" [--dir table]
//     plays greedy games with the net and counts the positions where the pattern applies
//     (any symmetry) and, with --dir, where the table knows the position: picks useful patterns.
#include <cstdio>
#include <thread>

#include "engine/game.hpp"
#include "ntuple/network.hpp"
#include "search/search.hpp"
#include "tablebase/tablebase.hpp"
#include "tools/cli.hpp"

using namespace k2048;

int main(int argc, char** argv) {
    return run_main([&] {
        Args a(argc, argv);
        bool build = a.has("build");
        a.has("info");  // default action: print the table summary
        std::string pattern = a.str("pattern"), dir = a.str("out", a.str("dir"));
        int threads = int(a.num("threads", std::thread::hardware_concurrency()));
        int coverage = int(a.num("coverage", 0));
        uint64_t max_states = uint64_t(a.num("max-states", 0));
        std::string net_path = a.str("net");
        a.check();
        if (coverage > 0) {
            Pattern p = Pattern::parse(pattern);
            auto net = NTupleNet::load(net_path);
            std::unique_ptr<Tablebase> table = dir.empty() ? nullptr : std::make_unique<Tablebase>(dir);
            uint64_t positions = 0, shape = 0, known = 0, games_hit = 0;
            for (int g = 0; g < coverage; ++g) {
                bool hit = false;
                play_game(5'000'000 + uint64_t(g), [&](Board b) {
                    positions++;
                    for (int sym = 0; sym < 8; ++sym) {
                        Board m = 0, t = 0;
                        for (int i = 0; i < 16; ++i) m = set_cell(m, NTupleNet::sym_cell(i, sym), cell(b, i));
                        if (!Tablebase::applies(p, m)) continue;
                        shape++, hit = true;
                        if (table && table->to_table(m, t) && table->prob(t) >= 0) known++;
                        break;
                    }
                    return greedy_move(b, [&](Board x) { return net->value(x); });
                });
                games_hit += hit;
            }
            std::printf("desen %s: %llu konumun %llu'unda uyuyor (%.2f%%), %llu/%d oyunda en az bir kez; tabloda bilinen: %llu\n",
                        p.str().c_str(), (unsigned long long)positions, (unsigned long long)shape,
                        100.0 * double(shape) / double(positions), (unsigned long long)games_hit, coverage,
                        (unsigned long long)known);
            return 0;
        }
        if (dir.empty()) throw std::runtime_error("--out/--dir required");
        if (build) build_tablebase(Pattern::parse(pattern), dir, threads, false, max_states);
        Tablebase t(dir);
        std::printf("desen %s (%d kilitli hucre, hedef %d), %llu durum\n", t.pattern().str().c_str(), t.pattern().locked(),
                    1 << t.pattern().target, (unsigned long long)t.states());
        // Success probability from the smallest positions: two 2s in the first free cells.
        Board start = 0;
        int placed = 0;
        for (int i = 0; i < 16 && placed < 2; ++i)
            if (!(t.pattern().fixed >> i & 1)) start = set_cell(start, i, 1), placed++;
        Board tb_start = tb::with_placeholders(t.pattern(), start);
        std::printf("iki 2'den baslayarak basari olasiligi: %.6f\n", t.prob(tb_start));
        return 0;
    });
}
