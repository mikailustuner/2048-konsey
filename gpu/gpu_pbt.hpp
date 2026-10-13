#pragma once
#include <memory>

#include "evolve/pbt.hpp"

namespace k2048 {

// PBT backend with the whole population resident on the GPU (training, tournaments and the
// exploit copy all happen on the device; host nets are refreshed before every save).
std::unique_ptr<PBTBackend> make_gpu_pbt_backend(int slots);

}  // namespace k2048
