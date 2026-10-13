#pragma once
#include <cstdint>

namespace k2048 {

// xoshiro256** seeded through splitmix64. Same seed => same sequence on every machine.
class Rng {
public:
    explicit Rng(uint64_t seed = 0) {
        for (auto& x : s_) {
            seed += 0x9e3779b97f4a7c15ULL;
            uint64_t z = seed;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            x = z ^ (z >> 31);
        }
    }

    uint64_t next() {
        const uint64_t result = rotl(s_[1] * 5, 7) * 9;
        const uint64_t t = s_[1] << 17;
        s_[2] ^= s_[0];
        s_[3] ^= s_[1];
        s_[1] ^= s_[2];
        s_[0] ^= s_[3];
        s_[2] ^= t;
        s_[3] = rotl(s_[3], 45);
        return result;
    }

    // Uniform integer in [0, n). Multiply-shift (Lemire); bias is < n / 2^64.
    uint32_t below(uint32_t n) { return uint32_t((unsigned __int128)next() * n >> 64); }

    double uniform() { return (next() >> 11) * 0x1.0p-53; }

private:
    static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    uint64_t s_[4];
};

}  // namespace k2048
