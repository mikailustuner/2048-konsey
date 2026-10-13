// train_many: trains many networks at once on the GPU (Faz 3 ablation arms, replicas of the final
// configuration, Faz 4 re-training...). One job per line of the jobs file:
//   name=E_8x6 net=8x6 stages=14;14,13 tc=1 oi=320000 alpha=0.1 games=20000000
// Each job lives in <dir>/<name>/ (net.bin checkpoint + train.jsonl) and resumes from its
// checkpoint. Optional per job: seed=N (default: (job index + 1) << 40), kappa=K (batched-update
// limit, default 5; see GpuAgentParams).
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>

#include "evolve/pbt.hpp"
#include "gpu/gpu.hpp"
#include "tools/cli.hpp"

using namespace k2048;

struct Job {
    std::map<std::string, std::string> kv;
    std::string get(const std::string& k, const std::string& def = "") const {
        auto it = kv.find(k);
        return it == kv.end() ? def : it->second;
    }
};

int main(int argc, char** argv) {
    return run_main([&] {
        Args a(argc, argv);
        std::string jobs_path = a.str("jobs"), dir = a.str("dir", "runs/many");
        int slots = int(a.num("slots", 0)), moves = int(a.num("moves", 32)), ckpt_sec = int(a.num("ckpt-sec", 900));
        int64_t log_every = a.num("log-every", 100'000);
        std::string run_id = a.str("run-id", "many"), faz = a.str("faz", "3");
        a.check();

        std::vector<Job> jobs;
        std::ifstream jf(jobs_path);
        if (!jf) throw std::runtime_error("cannot open --jobs " + jobs_path);
        for (std::string line; std::getline(jf, line);) {
            if (line.empty() || line[0] == '#') continue;
            Job j;
            std::stringstream ss(line);
            for (std::string tok; ss >> tok;) {
                size_t eq = tok.find('=');
                if (eq == std::string::npos) throw std::runtime_error("bad token " + tok);
                j.kv[tok.substr(0, eq)] = tok.substr(eq + 1);
            }
            if (j.get("name").empty()) throw std::runtime_error("job without name: " + line);
            jobs.push_back(j);
        }
        if (jobs.empty()) throw std::runtime_error("no jobs");

        std::vector<std::unique_ptr<NTupleNet>> nets;
        std::vector<TDParams> params;
        std::vector<int64_t> done, target;
        std::vector<uint64_t> seeds;
        std::vector<TrainOptions> opts;
        for (size_t i = 0; i < jobs.size(); ++i) {
            const Job& j = jobs[i];
            std::string out = dir + "/" + j.get("name");
            std::filesystem::create_directories(out);
            TrainOptions o;
            o.ckpt = out + "/net.bin";
            o.log_path = out + "/train.jsonl";
            o.meta = {run_id, faz, j.get("name"), 0};
            o.quiet = true;
            TDParams p;
            p.alpha = std::stof(j.get("alpha", "0.1"));
            int64_t d = 0;
            if (std::filesystem::exists(o.ckpt)) {
                std::string h;
                nets.push_back(NTupleNet::load(o.ckpt, &h));
                d = ckpt_games(h);
            } else {
                nets.push_back(std::make_unique<NTupleNet>(NTupleNet::preset(j.get("net", "4x6")),
                                                           parse_stages(j.get("stages")), j.get("tc", "0") == "1",
                                                           std::stof(j.get("oi", "0"))));
            }
            params.push_back(p);
            done.push_back(d);
            target.push_back(std::stoll(j.get("games", "1000000")));
            seeds.push_back(j.kv.count("seed") ? std::stoull(j.get("seed")) : (uint64_t(i) + 1) << 40);
            opts.push_back(o);
            std::printf("%-16s %s tc=%s alpha=%g: %lld / %lld oyun (%.0f MB)\n", j.get("name").c_str(),
                        j.get("net", "4x6").c_str(), j.get("tc", "0").c_str(), p.alpha, (long long)d,
                        (long long)target.back(), double(nets.back()->data_size()) * 4 / 1e6);
        }

        std::vector<NTupleNet*> ptrs;
        for (auto& n : nets) ptrs.push_back(n.get());
        GpuPopulation gp(ptrs, slots);
        gp.set_moves_per_launch(moves);
        std::printf("GPU: %s, %d slot\n", gpu_info().c_str(), gp.slots());
        std::fflush(stdout);

        auto save_all = [&] {
            for (size_t i = 0; i < nets.size(); ++i) {
                gp.download(int(i), *nets[i]);
                nets[i]->save(opts[i].ckpt, ckpt_header(params[i], done[i]));
            }
        };
        auto last = std::chrono::steady_clock::now();
        std::vector<GpuAgentParams> gparams;
        for (size_t i = 0; i < params.size(); ++i) gparams.push_back({params[i].alpha, std::stof(jobs[i].get("kappa", "5"))});
        gp.train(gparams, done, target, seeds, log_every,
                 [&](int i, const EvalReport& w, int64_t d) {
                     report_window(opts[size_t(i)], d, w);
                     std::printf("%-16s [%lld oyun, %.0f oyun/sn] ort %.0f | 2048 %.1f%% 8192 %.1f%% 16384 %.2f%% 32768 %.2f%%\n",
                                 jobs[size_t(i)].get("name").c_str(), (long long)d, double(w.games()) / w.seconds,
                                 w.mean_score(), 100.0 * double(w.reached(11)) / w.games(),
                                 100.0 * double(w.reached(13)) / w.games(), 100.0 * double(w.reached(14)) / w.games(),
                                 100.0 * double(w.reached(15)) / w.games());
                     std::fflush(stdout);
                 },
                 [&] {
                     auto now = std::chrono::steady_clock::now();
                     if (std::chrono::duration<double>(now - last).count() >= ckpt_sec) save_all(), last = now;
                 });
        save_all();
        std::printf("bitti\n");
        return 0;
    });
}
