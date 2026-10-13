#include "agents.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <memory>
#include <sstream>

#include "council/council.hpp"
#include "ntuple/network.hpp"
#include "search/search.hpp"
#include "tablebase/tablebase.hpp"

namespace k2048 {

static Policy random_policy() {
    return [](Board b) {
        // Seeded from the position so a game is reproducible regardless of thread scheduling.
        Rng rng(b);
        int legal[4], n = 0;
        for (int d = 0; d < 4; ++d)
            if (move(b, d) != b) legal[n++] = d;
        return n ? legal[rng.below(uint32_t(n))] : -1;
    };
}

std::vector<std::shared_ptr<const NTupleNet>> load_nets(const std::string& list) {
    std::vector<std::shared_ptr<const NTupleNet>> nets;
    std::stringstream ss(list);
    for (std::string path; std::getline(ss, path, ',');) nets.push_back(NTupleNet::load(path));
    if (nets.empty()) throw std::runtime_error("--net required");
    return nets;
}

SearchConfig search_config(const Args& args, int default_depth) {
    SearchConfig sc;
    sc.depth = int(args.num("depth", default_depth));
    sc.time_ms = args.real("time-ms", 0);
    sc.prob_cut = float(args.real("prob-cut", 0));
    sc.tt_bits = int(args.num("tt-bits", 20));
    sc.downgrade = int(args.num("downgrade", 0));
    return sc;
}

std::function<Policy()> make_agent(const Args& args) {
    std::string kind = args.str("agent", "random");
    if (kind == "random") return random_policy;
    if (kind == "greedy" || kind == "expectimax") {
        SearchConfig sc = search_config(args, kind == "greedy" ? 1 : 2);
        auto nets = load_nets(args.str("net"));
        if (nets.size() != 1) throw std::runtime_error("use --agent council for several nets");
        auto net = nets[0];
        return [net, sc] {
            auto s = std::make_shared<Searcher>([net](Board a) { return net->value(a); }, sc);
            return Policy([s](Board b) { return s->choose(b); });
        };
    }
    if (kind == "council") {
        auto c = std::make_shared<Council>();
        c->nets = load_nets(args.str("net"));
        c->mode = parse_mode(args.str("combine", "mean"));
        std::stringstream ws(args.str("weights", ""));
        for (std::string w; std::getline(ws, w, ',');) c->weights.push_back(std::stof(w));
        if (!c->weights.empty() && c->weights.size() != c->nets.size())
            throw std::runtime_error("--weights needs one value per net");
        int depth = int(args.num("depth", 1));
        return [c, depth] { return Policy([c, depth](Board b) { return c->choose(b, depth); }); };
    }
    if (kind == "hybrid") {
        // Faz 6b: a table plays whenever its pattern applies and it knows the position (success
        // probability >= --table-min); otherwise the search with the network.
        SearchConfig sc = search_config(args, 2);
        auto nets = load_nets(args.str("net"));
        auto net = nets[0];
        std::vector<std::shared_ptr<const Tablebase>> tables;
        std::stringstream ts(args.str("tables"));
        for (std::string d; std::getline(ts, d, ',');) tables.push_back(std::make_shared<Tablebase>(d));
        if (tables.empty()) throw std::runtime_error("--tables required");
        float min_p = float(args.real("table-min", 0));
        // Switch point: the search's move stands unless its success probability is more than
        // `margin` below the table's best move (0 = the table always plays, 1 = never).
        float margin = float(args.real("table-margin", 0));
        struct Count {  // prints how often the tables played, when the agent goes away
            std::atomic<uint64_t> table{0}, total{0};
            ~Count() {
                std::fprintf(stderr, "tablo hamlesi: %llu / %llu (%.2f%%)\n", (unsigned long long)table.load(),
                             (unsigned long long)total.load(), total ? 100.0 * double(table) / double(total) : 0.0);
            }
        };
        auto count = std::make_shared<Count>();
        return [net, sc, tables, min_p, margin, count] {
            auto s = std::make_shared<Searcher>([net](Board a) { return net->value(a); }, sc);
            return Policy([s, tables, min_p, margin, count](Board b) {
                count->total++;
                int ds = s->choose(b);
                for (auto& t : tables) {
                    float pr[4];
                    if (!t->move_probs_any(b, pr)) continue;
                    int dt = int(std::max_element(pr, pr + 4) - pr);
                    if (pr[dt] < min_p || ds < 0) continue;
                    if (pr[ds] < pr[dt] - margin) {
                        count->table++;
                        return dt;
                    }
                    return ds;  // the table agrees closely enough with the search
                }
                return ds;
            });
        };
    }
    throw std::runtime_error("unknown --agent " + kind);
}

}  // namespace k2048
