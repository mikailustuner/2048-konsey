// train: TD self-play training of an n-tuple network, resumable from its checkpoint.
//   train --ckpt runs/x/net.bin --net 4x6 [--stages "14;14,13"] [--tc] [--oi 320000]
//         --alpha 0.1 [--lambda 0.5] --games 1000000 --threads 8 --out runs/x
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <thread>

#include "ntuple/td.hpp"
#include "tools/cli.hpp"
#ifdef K2048_CUDA
#include "gpu/gpu.hpp"
#endif

using namespace k2048;

static std::vector<std::vector<int>> parse_stages(const std::string& s) {
    // "14;14,13" -> {{}, {14}, {14,13}}
    std::vector<std::vector<int>> v(1);
    if (s.empty()) return v;
    v.emplace_back();
    std::string num;
    for (char c : s + ";") {
        if (c == ';' || c == ',') {
            if (!num.empty()) v.back().push_back(std::stoi(num));
            num.clear();
            if (c == ';') v.emplace_back();
        } else {
            num += c;
        }
    }
    v.pop_back();
    return v;
}

int main(int argc, char** argv) {
    return run_main([&] {
        Args a(argc, argv);
        std::string out = a.str("out", "runs/train");
        TrainOptions o;
        o.ckpt = a.str("ckpt", out + "/net.bin");
        o.games = a.num("games", 100'000);
        o.threads = int(a.num("threads", std::thread::hardware_concurrency()));
        o.seed = uint64_t(a.num("seed", 1LL << 40));
        o.ckpt_sec = int(a.num("ckpt-sec", 600));
        o.log_every = a.num("log-every", 10'000);
        o.log_path = out + "/train.jsonl";
        o.meta = {a.str("run-id", "train"), a.str("faz", "2"), a.str("ajan-id", "td"), int(a.num("nesil", 0))};
        TDParams p;
        p.alpha = float(a.real("alpha", 0.1));
        p.lambda = float(a.real("lambda", 0.0));
        std::string preset = a.str("net", "4x6"), stages = a.str("stages", "");
        bool tc = a.has("tc");
        float oi = float(a.real("oi", 0.0));
        bool gpu = a.has("gpu");
        int slots = int(a.num("slots", 0)), moves = int(a.num("moves", 32));
        float kappa = float(a.real("kappa", 5.0));
        a.check();

        std::filesystem::create_directories(out);
        std::unique_ptr<NTupleNet> net;
        int64_t done = 0;
        if (std::filesystem::exists(o.ckpt)) {
            std::string h;
            net = NTupleNet::load(o.ckpt, &h);
            done = ckpt_games(h);
            std::printf("devam: %s (%lld oyun egitilmis)\n", o.ckpt.c_str(), (long long)done);
        } else {
            net = std::make_unique<NTupleNet>(NTupleNet::preset(preset), parse_stages(stages), tc, oi);
            std::printf("yeni ag: %s, %d asama, tc=%d, oi=%.0f, %.0f MB\n", preset.c_str(), net->stages(), tc, oi,
                        double(net->weight_count()) * (tc ? 12 : 4) / 1e6);
        }
        std::printf("alpha=%g lambda=%g, hedef %lld oyun, %d is parcacigi\n", p.alpha, p.lambda, (long long)o.games,
                    o.threads);
        std::fflush(stdout);
        if (!gpu) {
            train(*net, p, o, done);
            return 0;
        }
#ifdef K2048_CUDA
        if (p.lambda > 0) throw std::runtime_error("--gpu trains TD(0) only (--lambda 0)");
        GpuPopulation gp({net.get()}, slots);
        gp.set_moves_per_launch(moves);
        std::printf("GPU: %s, %d eszamanli oyun\n", gpu_info().c_str(), gp.slots());
        std::vector<int64_t> games{done};
        auto last = std::chrono::steady_clock::now();
        auto save = [&] {
            gp.download(0, *net);
            net->save(o.ckpt, ckpt_header(p, games[0]));
        };
        gp.train({{p.alpha, kappa}}, games, {o.games}, {o.seed}, o.log_every,
                 [&](int, const EvalReport& w, int64_t d) { report_window(o, d, w); },
                 [&] {
                     auto now = std::chrono::steady_clock::now();
                     if (std::chrono::duration<double>(now - last).count() >= o.ckpt_sec) save(), last = now;
                 });
        save();
#else
        (void)slots, (void)moves, (void)kappa;
        throw std::runtime_error("built without CUDA (-DUSE_CUDA=ON)");
#endif
        return 0;
    });
}
