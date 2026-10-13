// eval: plays N games with fixed seeds and reports mean score, tile reach rates and 95% Wilson
// intervals. Appends one JSONL line to <out>/log.jsonl and (with --record) replays to
// <out>/replays.jsonl.
#include <cstdio>
#include <thread>

#include "../engine/eval.hpp"
#include "agents.hpp"

using namespace k2048;

int main(int argc, char** argv) {
    return run_main([&] {
        Args a(argc, argv);
        EvalConfig cfg;
        cfg.games = int(a.num("games", 1000));
        cfg.seed0 = uint64_t(a.num("seed", 1'000'000));
        cfg.threads = int(a.num("threads", std::thread::hardware_concurrency()));
        cfg.record = a.has("record");
        LogMeta meta{a.str("run-id", "eval"), a.str("faz", "0"), a.str("ajan-id", a.str("agent", "random")),
                     int(a.num("nesil", 0))};
        std::string out = a.str("out", "runs/eval");
        auto agent = make_agent(a);
        a.check();

        EvalReport r = evaluate(agent, cfg);
        std::fputs(summary(r).c_str(), stdout);
        append_line(out + "/log.jsonl", log_line(meta, r));
        if (cfg.record)
            for (auto& g : r.results)
                append_line(out + "/replays.jsonl",
                            "{\"run_id\":\"" + meta.run_id + "\",\"seed\":" + std::to_string(g.seed) +
                                ",\"score\":" + std::to_string(g.score) + ",\"max_tile\":" +
                                std::to_string(1ULL << g.max_exp) + ",\"moves\":\"" + g.record + "\"}");
        std::printf("log: %s/log.jsonl\n", out.c_str());
        return 0;
    });
}
