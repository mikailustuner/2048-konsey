// evolve: population based training of TD agents (Faz 4). Resumes from <dir>/population.txt.
#include <thread>

#include "evolve/pbt.hpp"
#include "tools/cli.hpp"
#ifdef K2048_CUDA
#include "gpu/gpu.hpp"
#include "gpu/gpu_pbt.hpp"
#endif

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
        c.save_every = int(a.num("save-every", 1));
        bool gpu = a.has("gpu");
        int slots = int(a.num("slots", 0));
        a.check();
        if (!gpu) {
            run_pbt(c);
            return 0;
        }
#ifdef K2048_CUDA
        if (c.lambda_hi > 0) throw std::runtime_error("--gpu trains TD(0): use --lambda-hi 0");
        std::printf("GPU: %s\n", gpu_info().c_str());
        auto backend = make_gpu_pbt_backend(slots);
        run_pbt(c, backend.get());
#else
        (void)slots;
        throw std::runtime_error("built without CUDA (-DUSE_CUDA=ON)");
#endif
        return 0;
    });
}
