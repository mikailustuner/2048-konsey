#include "agents.hpp"

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

std::function<Policy()> make_agent(const Args& args) {
    std::string kind = args.str("agent", "random");
    if (kind == "random") return random_policy;
    throw std::runtime_error("unknown --agent " + kind);
}

}  // namespace k2048
