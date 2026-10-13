// evolve: population based training of TD agents (Faz 4). Resumes from <dir>/population.txt.
#include <thread>

#include "evolve/pbt.hpp"
#include "tools/cli.hpp"

using namespace k2048;

int main(int argc, char** argv) {
    return run_main([&] {
        Args a(argc, argv);
        PBTConfig c;
        c.dir = a.str("dir", "runs/faz4");
        c.population = int(a.num("population", 8));
        c.generations = int(a.num("generations", 20));
        c.train_games = a.num("train-games", 20'000);
        c.tourney_games = int(a.num("tourney-games", 300));
        c.threads = int(a.num("threads", std::thread::hardware_concurrency()));
        c.seed = uint64_t(a.num("seed", 7));
        c.net = a.str("net", "4x6");
        c.stages = a.str("stages", "");
        c.tc = a.has("tc");
        c.alpha_lo = float(a.real("alpha-lo", 0.02));
        c.alpha_hi = float(a.real("alpha-hi", 0.5));
        c.lambda_hi = float(a.real("lambda-hi", 0.5));
        c.oi_hi = float(a.real("oi-hi", 0));
        c.run_id = a.str("run-id", "pbt");
        a.check();
        run_pbt(c);
        return 0;
    });
}
