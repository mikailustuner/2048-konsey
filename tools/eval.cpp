// eval: plays N games with fixed seeds and reports mean score, tile reach rates and 95% Wilson
// intervals. Appends one JSONL line to <out>/log.jsonl and (with --record) replays to
// <out>/replays.jsonl.
#include <cstdio>
#include <thread>

#include "../engine/eval.hpp"
#include "agents.hpp"
#ifdef K2048_CUDA
#include <sstream>

#include "council/council.hpp"
#include "gpu/gpu.hpp"
#endif

using namespace k2048;

int main(int argc, char** argv) {
    return run_main([&] {
        Args a(argc, argv);
        EvalConfig cfg;
        cfg.games = int(a.num("games", 1000));
        cfg.seed0 = uint64_t(a.num("seed", 1'000'000));
        cfg.threads = int(a.num("threads", std::thread::hardware_concurrency()));
        cfg.record = a.has("record");
        std::string per_game = a.str("per-game", "");
        LogMeta meta{a.str("run-id", "eval"), a.str("faz", "0"), a.str("ajan-id", a.str("agent", "random")),
                     int(a.num("nesil", 0))};
        std::string out = a.str("out", "runs/eval");
        EvalReport r;
        if (a.has("gpu")) {
#ifdef K2048_CUDA
            // greedy / expectimax / council (mean|weighted) over same-shaped nets, all on the GPU.
            std::string kind = a.str("agent", "greedy");
            int depth = int(a.num("depth", kind == "expectimax" ? 2 : 1));
            if (kind == "greedy") depth = 1;
            std::vector<float> weights;
            if (kind == "council") {
                if (parse_mode(a.str("combine", "mean")) == Council::Mode::Vote) throw std::runtime_error("--gpu: vote is CPU only");
                std::stringstream ws(a.str("weights", ""));
                for (std::string w; std::getline(ws, w, ',');) weights.push_back(std::stof(w));
            }
            int slots = int(a.num("slots", 0));
            std::vector<std::unique_ptr<NTupleNet>> nets;
            std::stringstream ns(a.str("net"));
            for (std::string pth; std::getline(ns, pth, ',');) nets.push_back(NTupleNet::load(pth));
            a.check();
            std::vector<NTupleNet*> ptrs;
            std::vector<int> members;
            for (auto& n : nets) members.push_back(int(ptrs.size())), ptrs.push_back(n.get());
            GpuPopulation gp(ptrs, slots);
            std::printf("GPU: %s\n", gpu_info().c_str());
            r = gp.evaluate(members, weights, depth, cfg);
#else
            throw std::runtime_error("built without CUDA (-DUSE_CUDA=ON)");
#endif
        } else {
            auto agent = make_agent(a);
            a.check();
            r = evaluate(agent, cfg);
        }
        std::fputs(summary(r).c_str(), stdout);
        append_line(out + "/log.jsonl", log_line(meta, r));
        if (cfg.record)
            for (auto& g : r.results)
                append_line(out + "/replays.jsonl",
                            "{\"run_id\":\"" + meta.run_id + "\",\"seed\":" + std::to_string(g.seed) +
                                ",\"score\":" + std::to_string(g.score) + ",\"max_tile\":" +
                                std::to_string(1ULL << g.max_exp) + ",\"moves\":\"" + g.record + "\"}");
        if (!per_game.empty())  // one line per game, for paired comparisons (same seeds)
            for (auto& g : r.results)
                append_line(per_game, "{\"ajan_id\":\"" + meta.ajan_id + "\",\"seed\":" + std::to_string(g.seed) +
                                          ",\"score\":" + std::to_string(g.score) + ",\"max_exp\":" +
                                          std::to_string(g.max_exp) + ",\"moves\":" + std::to_string(g.moves) + "}");
        std::printf("log: %s/log.jsonl\n", out.c_str());
        return 0;
    });
}
