// train: TD self-play training of an n-tuple network, resumable from its checkpoint.
//   train --ckpt runs/x/net.bin --net 4x6 [--stages "14;14,13"] [--tc] [--oi 320000]
//         --alpha 0.1 [--lambda 0.5] --games 1000000 --threads 8 --out runs/x
#include <cstdio>
#include <filesystem>
#include <thread>

#include "ntuple/td.hpp"
#include "tools/cli.hpp"

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
        train(*net, p, o, done);
        return 0;
    });
}
