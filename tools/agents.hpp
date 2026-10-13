#pragma once
#include <functional>

#include "../engine/game.hpp"
#include "cli.hpp"
#include "search/search.hpp"

namespace k2048 {

// Builds a per-thread policy factory from command line options (--agent ...).
std::function<Policy()> make_agent(const Args& args);

// --depth, --time-ms, --prob-cut, --tt-bits, --downgrade (Faz 6a search options).
SearchConfig search_config(const Args& args, int default_depth);

}  // namespace k2048
