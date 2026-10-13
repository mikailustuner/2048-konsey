#pragma once
#include <functional>

#include "../engine/game.hpp"
#include "cli.hpp"

namespace k2048 {

// Builds a per-thread policy factory from command line options (--agent ...).
std::function<Policy()> make_agent(const Args& args);

}  // namespace k2048
