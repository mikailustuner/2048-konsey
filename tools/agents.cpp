#include "agents.hpp"

#include <memory>
#include <sstream>

#include "council/council.hpp"
#include "ntuple/network.hpp"
#include "search/search.hpp"

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

std::function<Policy()> make_agent(const Args& args) {
    std::string kind = args.str("agent", "random");
    if (kind == "random") return random_policy;
    if (kind == "greedy" || kind == "expectimax") {
        int depth = kind == "greedy" ? 1 : int(args.num("depth", 2));
        auto nets = load_nets(args.str("net"));
        if (nets.size() != 1) throw std::runtime_error("use --agent council for several nets");
        auto net = nets[0];
        return [net, depth] {
            ValueFn v = [net](Board a) { return net->value(a); };
            return Policy([v, depth](Board b) { return expectimax_move(b, depth, v); });
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
    throw std::runtime_error("unknown --agent " + kind);
}

}  // namespace k2048
