#include "board.hpp"

#include <algorithm>

namespace k2048 {
namespace {

struct Tables {
    uint16_t left[65536], right[65536];
    uint32_t score_left[65536], score_right[65536];

    static uint16_t reverse(uint16_t r) {
        return uint16_t((r >> 12) | ((r >> 4) & 0x00F0) | ((r << 4) & 0x0F00) | (r << 12));
    }

    Tables() {
        for (uint32_t r = 0; r < 65536; ++r) {
            int v[4], out[4] = {0, 0, 0, 0}, n = 0;
            for (int c = 0; c < 4; ++c) v[c] = (r >> (4 * c)) & 0xF;
            uint32_t score = 0;
            int prev = 0;  // last tile placed and still mergeable
            for (int c = 0; c < 4; ++c) {
                if (!v[c]) continue;
                if (prev && prev == v[c] && v[c] < kMaxFastExp) {
                    out[n - 1] = v[c] + 1;
                    score += 1u << (v[c] + 1);
                    prev = 0;
                } else {
                    out[n++] = v[c];
                    prev = v[c];
                }
            }
            uint16_t res = 0;
            for (int c = 0; c < 4; ++c) res |= uint16_t(out[c] << (4 * c));
            left[r] = res;
            score_left[r] = score;
        }
        for (uint32_t r = 0; r < 65536; ++r) {
            uint16_t rev = reverse(uint16_t(r));
            right[r] = reverse(left[rev]);
            score_right[r] = score_left[rev];
        }
    }
};

const Tables& tables() {
    static const Tables t;
    return t;
}

// Static init so hot paths never hit the function-local static guard.
const Tables& T = tables();

template <bool kLeft>
inline Board move_rows(Board b, uint32_t& score) {
    const uint16_t* tab = kLeft ? T.left : T.right;
    const uint32_t* sc = kLeft ? T.score_left : T.score_right;
    Board res = 0;
    for (int r = 0; r < 4; ++r) {
        uint16_t row = uint16_t(b >> (16 * r));
        res |= Board(tab[row]) << (16 * r);
        score += sc[row];
    }
    return res;
}

}  // namespace

Board transpose(Board x) {
    Board a1 = x & 0xF0F00F0FF0F00F0FULL;
    Board a2 = x & 0x0000F0F00000F0F0ULL;
    Board a3 = x & 0x0F0F00000F0F0000ULL;
    Board a = a1 | (a2 << 12) | (a3 >> 12);
    Board b1 = a & 0xFF00FF0000FF00FFULL;
    Board b2 = a & 0x00FF00FF00000000ULL;
    Board b3 = a & 0x00000000FF00FF00ULL;
    return b1 | (b2 >> 24) | (b3 << 24);
}

Board move(Board b, int dir, uint32_t& score) {
    switch (dir) {
        case LEFT: return move_rows<true>(b, score);
        case RIGHT: return move_rows<false>(b, score);
        case UP: return transpose(move_rows<true>(transpose(b), score));
        default: return transpose(move_rows<false>(transpose(b), score));
    }
}

int count_empty(Board b) {
    // A nibble is empty iff all four bits are zero.
    Board x = b | (b >> 1);
    x |= x >> 2;
    return 16 - __builtin_popcountll(x & 0x1111111111111111ULL);
}

int max_exp(Board b) {
    int m = 0;
    for (int i = 0; i < 16; ++i) m = std::max(m, cell(b, i));
    return m;
}

int count_exp(Board b, int e) {
    int n = 0;
    for (int i = 0; i < 16; ++i) n += cell(b, i) == e;
    return n;
}

bool has_exp(Board b, int e) { return count_exp(b, e) > 0; }

bool can_move(Board b) {
    for (int d = 0; d < 4; ++d)
        if (move(b, d) != b) return true;
    return false;
}

Board spawn(Board b, Rng& rng) {
    uint32_t k = rng.below(uint32_t(count_empty(b)));
    int v = rng.below(10) == 0 ? 2 : 1;
    for (int i = 0; i < 16; ++i) {
        if (cell(b, i)) continue;
        if (k-- == 0) return set_cell(b, i, v);
    }
    return b;  // unreachable when an empty cell exists
}

Board initial_board(Rng& rng) { return spawn(spawn(0, rng), rng); }

Grid to_grid(Board b) {
    Grid g{};
    for (int i = 0; i < 16; ++i) g[i] = uint8_t(cell(b, i));
    return g;
}

Board from_grid(const Grid& g) {
    Board b = 0;
    for (int i = 0; i < 16; ++i) b = set_cell(b, i, g[i]);
    return b;
}

namespace ref {

// Cells of line `l` for a direction, ordered from the edge tiles move towards.
static int line_cell(int dir, int l, int k) {
    switch (dir) {
        case UP: return l + 4 * k;
        case DOWN: return l + 4 * (3 - k);
        case LEFT: return 4 * l + k;
        default: return 4 * l + (3 - k);
    }
}

bool move(Grid& g, int dir, uint64_t& score) {
    bool moved = false;
    for (int l = 0; l < 4; ++l) {
        int in[4], out[4] = {0, 0, 0, 0}, n = 0;
        for (int k = 0; k < 4; ++k) in[k] = g[line_cell(dir, l, k)];
        bool merged_last = false;
        for (int k = 0; k < 4; ++k) {
            if (!in[k]) continue;
            if (n > 0 && !merged_last && out[n - 1] == in[k]) {
                out[n - 1]++;
                score += uint64_t(1) << out[n - 1];
                merged_last = true;
            } else {
                out[n++] = in[k];
                merged_last = false;
            }
        }
        for (int k = 0; k < 4; ++k) {
            int c = line_cell(dir, l, k);
            if (g[c] != out[k]) moved = true;
            g[c] = uint8_t(out[k]);
        }
    }
    return moved;
}

void spawn(Grid& g, Rng& rng) {
    uint32_t empty = 0;
    for (auto v : g) empty += v == 0;
    uint32_t k = rng.below(empty);
    uint8_t v = rng.below(10) == 0 ? 2 : 1;
    for (auto& c : g) {
        if (c) continue;
        if (k-- == 0) {
            c = v;
            return;
        }
    }
}

bool can_move(const Grid& g) {
    for (int d = 0; d < 4; ++d) {
        Grid t = g;
        uint64_t s = 0;
        if (move(t, d, s)) return true;
    }
    return false;
}

}  // namespace ref

Board downgrade(const Grid& g0) {
    Grid g = g0;
    for (;;) {
        int mx = *std::max_element(g.begin(), g.end());
        if (mx <= kMaxFastExp) break;
        bool present[32] = {};
        for (auto v : g) present[v] = true;
        int gap = 1;
        while (gap < mx && present[gap]) ++gap;
        if (gap >= mx) {  // no gap at all (needs 1..16 each exactly once): clamp
            for (auto& v : g) v = uint8_t(std::min<int>(v, kMaxFastExp));
            break;
        }
        for (auto& v : g)
            if (v > gap) --v;
    }
    return from_grid(g);
}

uint64_t MaskedSpace::size() const {
    uint64_t n = 1;
    for (int i = 0; i < free_cells(); ++i) n *= uint64_t(max_exp + 1);
    return n;
}

bool MaskedSpace::contains(Board b) const {
    for (int i = 0; i < 16; ++i) {
        if (fixed >> i & 1) {
            if (cell(b, i) != cell(fixed_vals, i)) return false;
        } else if (cell(b, i) > max_exp) {
            return false;
        }
    }
    return true;
}

uint64_t MaskedSpace::index(Board b) const {
    uint64_t idx = 0;
    for (int i = 0; i < 16; ++i)
        if (!(fixed >> i & 1)) idx = idx * uint64_t(max_exp + 1) + uint64_t(cell(b, i));
    return idx;
}

Board MaskedSpace::board(uint64_t idx) const {
    Board b = fixed_vals;
    for (int i = 15; i >= 0; --i) {
        if (fixed >> i & 1) continue;
        b = set_cell(b, i, int(idx % uint64_t(max_exp + 1)));
        idx /= uint64_t(max_exp + 1);
    }
    return b;
}

}  // namespace k2048
