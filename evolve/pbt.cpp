#include "pbt.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <thread>

#include "search/search.hpp"

namespace k2048 {

namespace {

struct Agent {
    int uid = 0, parent = -1;
    Genome g;
    int64_t games = 0;
    double fitness = 0;
    std::unique_ptr<NTupleNet> net;
};

std::vector<std::vector<int>> parse_stages(const std::string& s) {
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

std::string net_path(const PBTConfig& c, size_t slot) { return c.dir + "/agent_" + std::to_string(slot) + ".bin"; }

void save_state(const PBTConfig& c, int gen, int next_uid, const std::vector<Agent>& pop) {
    for (size_t i = 0; i < pop.size(); ++i) pop[i].net->save(net_path(c, i), "games=" + std::to_string(pop[i].games) + "\n");
    std::string tmp = c.dir + "/population.txt.tmp";
    {
        std::ofstream f(tmp);
        f << gen << ' ' << next_uid << '\n';
        for (auto& a : pop)
            f << a.uid << ' ' << a.parent << ' ' << a.g.alpha << ' ' << a.g.lambda << ' ' << a.g.oi << ' ' << a.games
              << ' ' << a.fitness << '\n';
    }
    std::filesystem::rename(tmp, c.dir + "/population.txt");
}

// Runs f(i) for i in [0, n) on up to `workers` threads.
template <class F>
void parallel_for(int n, int workers, F&& f) {
    std::atomic<int> next{0};
    std::vector<std::thread> pool;
    for (int t = 0; t < std::min(n, workers); ++t)
        pool.emplace_back([&] {
            for (int i; (i = next++) < n;) f(i);
        });
    for (auto& t : pool) t.join();
}

std::string genome_json(const Agent& a) {
    char buf[200];
    std::snprintf(buf, sizeof buf, "\"uid\":%d,\"ebeveyn\":%d,\"alpha\":%.5f,\"lambda\":%.4f,\"oi\":%.0f", a.uid,
                  a.parent, a.g.alpha, a.g.lambda, a.g.oi);
    return buf;
}

}  // namespace

Genome mutate(const Genome& g, Rng& rng) {
    Genome m = g;
    m.alpha *= rng.below(2) ? 1.25f : 0.8f;
    m.lambda = std::clamp(m.lambda + (rng.below(2) ? 0.1f : -0.1f), 0.0f, 0.9f);
    return m;
}

void run_pbt(const PBTConfig& c) {
    std::filesystem::create_directories(c.dir);
    std::vector<Agent> pop(size_t(c.population));
    int gen = 0, next_uid = 0;
    const std::string state = c.dir + "/population.txt";
    if (std::filesystem::exists(state)) {
        std::ifstream f(state);
        f >> gen >> next_uid;
        for (size_t i = 0; i < pop.size(); ++i) {
            auto& a = pop[i];
            f >> a.uid >> a.parent >> a.g.alpha >> a.g.lambda >> a.g.oi >> a.games >> a.fitness;
            a.net = NTupleNet::load(net_path(c, i));
        }
        if (!f) throw std::runtime_error("bad population state " + state);
        std::printf("devam: nesil %d\n", gen);
    } else {
        Rng rng(c.seed);
        for (auto& a : pop) {
            a.uid = next_uid++;
            a.g.alpha = c.alpha_lo * std::pow(c.alpha_hi / c.alpha_lo, float(rng.uniform()));
            a.g.lambda = c.lambda_hi * float(rng.uniform());
            a.g.oi = c.oi_hi * float(rng.uniform());
            a.net = std::make_unique<NTupleNet>(NTupleNet::preset(c.net), parse_stages(c.stages), c.tc, a.g.oi);
        }
    }
    const int per_agent = std::max(1, c.threads / c.population);
    const int concurrent = std::max(1, c.threads / per_agent);

    for (; gen < c.generations; ++gen) {
        // 1. Train.
        parallel_for(c.population, concurrent, [&](int i) {
            Agent& a = pop[size_t(i)];
            TrainOptions o;
            o.games = a.games + c.train_games;
            o.threads = per_agent;
            o.seed = (uint64_t(a.uid) + 1) << 40;  // lineage-specific training seeds
            o.log_every = c.train_games;
            o.log_path = c.dir + "/train.jsonl";
            o.meta = {c.run_id, "4", "a" + std::to_string(a.uid), gen};
            o.quiet = true;
            a.games = train(*a.net, {a.g.alpha, a.g.lambda}, o, a.games);
        });
        // 2. Tournament on common seeds.
        EvalConfig ec;
        ec.games = c.tourney_games;
        ec.seed0 = 2'000'000 + uint64_t(gen) * uint64_t(c.tourney_games);
        ec.threads = per_agent;
        std::vector<EvalReport> reps(pop.size());
        parallel_for(c.population, concurrent, [&](int i) {
            const NTupleNet* net = pop[size_t(i)].net.get();
            reps[size_t(i)] = evaluate(
                [net] {
                    ValueFn v = [net](Board x) { return net->value(x); };
                    return Policy([v](Board b) { return greedy_move(b, v); });
                },
                ec);
            pop[size_t(i)].fitness = reps[size_t(i)].mean_score();
        });
        for (size_t i = 0; i < pop.size(); ++i) {
            LogMeta m{c.run_id, "4", "a" + std::to_string(pop[i].uid), gen, pop[i].games};
            std::string line = log_line(m, reps[i]);
            line.back() = ',';
            append_line(c.dir + "/generations.jsonl", line + genome_json(pop[i]) + "}");
        }
        // 3. Exploit + explore.
        std::vector<size_t> order(pop.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](size_t x, size_t y) { return pop[x].fitness > pop[y].fitness; });
        std::printf("nesil %d: en iyi a%d %.0f (alpha %.4f lambda %.2f) | ortalama %.0f\n", gen, pop[order[0]].uid,
                    pop[order[0]].fitness, pop[order[0]].g.alpha, pop[order[0]].g.lambda,
                    [&] {
                        double s = 0;
                        for (auto& a : pop) s += a.fitness;
                        return s / double(pop.size());
                    }());
        std::fflush(stdout);
        if (gen + 1 < c.generations) {
            Rng rng(c.seed * 1000 + uint64_t(gen));
            size_t cut = std::max<size_t>(1, size_t(std::lround(0.25 * double(pop.size()))));
            for (size_t k = 0; k < cut; ++k) {
                Agent& loser = pop[order[order.size() - 1 - k]];
                const Agent& winner = pop[order[rng.below(uint32_t(cut))]];
                int old = loser.uid;
                loser.net->copy_from(*winner.net);
                loser.g = mutate(winner.g, rng);
                loser.games = winner.games;
                loser.parent = winner.uid;
                loser.uid = next_uid++;
                append_line(c.dir + "/lineage.jsonl", "{\"nesil\":" + std::to_string(gen) + ",\"yerine\":" +
                                                          std::to_string(old) + "," + genome_json(loser) + "}");
            }
        }
        save_state(c, gen + 1, next_uid, pop);
    }
}

}  // namespace k2048
