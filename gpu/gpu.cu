#include "gpu.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace k2048 {

#define CK(x)                                                                                        \
    do {                                                                                             \
        cudaError_t e_ = (x);                                                                        \
        if (e_ != cudaSuccess)                                                                       \
            throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(e_) + " @ " #x);   \
    } while (0)

namespace {

constexpr unsigned FULL = 0xffffffffu;
constexpr int kMaxTuples = 16, kMaxLen = 6, kMaxStages = 8, kMaxMembers = 8, kMaxShapes = 16;

struct NetShape {
    int F, stride, stages;
    unsigned long long wps;  // weights per stage
    int len[kMaxTuples];
    unsigned long long off[kMaxTuples];
    unsigned char shift[kMaxTuples * 8][kMaxLen];
    unsigned stage_mask[kMaxStages];
};
__constant__ NetShape c_shapes[kMaxShapes];  // distinct network shapes of the population

struct SlotState {
    unsigned long long board, prev, rng[4], score;
    long long game;  // < 0: no game in progress
    unsigned moves;
    int has_prev;
};

struct Result {
    int agent;
    int max_exp;
    unsigned moves;
    unsigned long long score, seed;
};

struct Members {
    const float* w[kMaxMembers];
    float wt[kMaxMembers];
    int shape[kMaxMembers];
    int n;
};

// ---- engine (mirrors engine/board.cpp and engine/rng.hpp) ----

__device__ __forceinline__ unsigned long long d_transpose(unsigned long long x) {
    unsigned long long a1 = x & 0xF0F00F0FF0F00F0FULL, a2 = x & 0x0000F0F00000F0F0ULL, a3 = x & 0x0F0F00000F0F0000ULL;
    unsigned long long a = a1 | (a2 << 12) | (a3 >> 12);
    unsigned long long b1 = a & 0xFF00FF0000FF00FFULL, b2 = a & 0x00FF00FF00000000ULL, b3 = a & 0x00000000FF00FF00ULL;
    return b1 | (b2 >> 24) | (b3 << 24);
}

// One row towards column 0 (left) or column 3 (right); 15+15 does not merge, as on the CPU.
__device__ __forceinline__ unsigned d_row(unsigned row, bool left, unsigned& score) {
    int out[4] = {0, 0, 0, 0}, n = 0, prev = 0;
    for (int k = 0; k < 4; ++k) {
        int v = (row >> (4 * (left ? k : 3 - k))) & 0xF;
        if (!v) continue;
        if (prev && prev == v && v < 15) {
            out[n - 1] = v + 1;
            score += 1u << (v + 1);
            prev = 0;
        } else {
            out[n++] = v;
            prev = v;
        }
    }
    unsigned res = 0;
    for (int k = 0; k < 4; ++k) res |= unsigned(out[k]) << (4 * (left ? k : 3 - k));
    return res;
}

// dir: 0 up, 1 right, 2 down, 3 left
__device__ __forceinline__ unsigned long long d_move(unsigned long long b, int dir, unsigned& score) {
    bool vert = (dir & 1) == 0, left = dir == 0 || dir == 3;
    unsigned long long t = vert ? d_transpose(b) : b, res = 0;
    for (int r = 0; r < 4; ++r) res |= (unsigned long long)d_row(unsigned(t >> (16 * r)) & 0xFFFF, left, score) << (16 * r);
    return vert ? d_transpose(res) : res;
}

struct DRng {
    unsigned long long s[4];
    __device__ void seed(unsigned long long seed) {
        for (int i = 0; i < 4; ++i) {
            seed += 0x9e3779b97f4a7c15ULL;
            unsigned long long z = seed;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            s[i] = z ^ (z >> 31);
        }
    }
    __device__ static unsigned long long rotl(unsigned long long x, int k) { return (x << k) | (x >> (64 - k)); }
    __device__ unsigned long long next() {
        unsigned long long r = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 45);
        return r;
    }
    __device__ unsigned below(unsigned n) { return unsigned(__umul64hi(next(), n)); }
};

__device__ __forceinline__ int d_count_empty(unsigned long long b) {
    unsigned long long x = b | (b >> 1);
    x |= x >> 2;
    return 16 - __popcll(x & 0x1111111111111111ULL);
}

__device__ unsigned long long d_spawn(unsigned long long b, DRng& r) {
    unsigned k = r.below(unsigned(d_count_empty(b)));
    unsigned long long v = r.below(10) == 0 ? 2 : 1;
    for (int i = 0; i < 16; ++i) {
        if ((b >> (4 * i)) & 0xF) continue;
        if (k-- == 0) return b | (v << (4 * i));
    }
    return b;
}

__device__ int d_max_exp(unsigned long long b) {
    int m = 0;
    for (int i = 0; i < 16; ++i) m = max(m, int((b >> (4 * i)) & 0xF));
    return m;
}

__device__ int d_stage(const NetShape& S, unsigned long long b) {
    if (S.stages == 1) return 0;
    unsigned present = 0;
    for (int i = 0; i < 16; ++i) present |= 1u << ((b >> (4 * i)) & 0xF);
    for (int k = S.stages - 1; k > 0; --k)
        if ((S.stage_mask[k] & ~present) == 0) return k;
    return 0;
}

// ---- network ----

// Weight index (not multiplied by the TC stride) of feature f = tuple*8 + symmetry.
__device__ __forceinline__ unsigned long long d_feature(const NetShape& S, unsigned long long b, unsigned long long base, int f) {
    int t = f >> 3;
    unsigned long long idx = 0;
    for (int j = 0; j < S.len[t]; ++j) idx |= ((b >> S.shift[f][j]) & 0xF) << (4 * j);
    return base + S.off[t] + idx;
}

// Warp-cooperative value. Lanes fetch their features, then every lane sums all of them in
// feature order (f = tuple*8 + symmetry), the same order as NTupleNet::value on the CPU: the
// result is bit-identical to the CPU and to every other lane, so control flow stays uniform.
// Weights never change inside a kernel (training only accumulates, see apply_kernel), so the
// read-only cache path is safe.
__device__ float warp_value(const NetShape& S, const float* w, unsigned long long b, int lane) {
    unsigned long long base = (unsigned long long)d_stage(S, b) * S.wps;
    float x[kMaxTuples * 8 / 32];
#pragma unroll
    for (int j = 0; j < kMaxTuples * 8 / 32; ++j) {
        int f = j * 32 + lane;
        x[j] = f < S.F ? __ldg(w + d_feature(S, b, base, f) * (unsigned long long)S.stride) : 0.0f;
    }
    float v = 0;
#pragma unroll
    for (int j = 0; j < kMaxTuples * 8 / 32; ++j)
        for (int l = 0; l < 32 && j * 32 + l < S.F; ++l) v += __shfl_sync(FULL, x[j], l);
    return v;
}

// Batched TD: the TD error of every feature occurrence is summed (g) and counted (n); a weight
// touched for the first time in this launch is appended to `list`. The weights move once per
// launch in apply_kernel.
struct Accum {
    float* g;
    unsigned* n;
    unsigned long long* list;
    unsigned* list_count;
    unsigned long long list_cap;
};

__device__ void warp_accumulate(const NetShape& S, const Accum& acc, unsigned long long b, float delta, int lane) {
    unsigned long long base = (unsigned long long)d_stage(S, b) * S.wps;
    for (int f = lane; f < S.F; f += 32) {
        unsigned long long i = d_feature(S, b, base, f);
        atomicAdd(acc.g + i, delta);
        if (atomicAdd(acc.n + i, 1u) == 0) {
            unsigned pos = atomicAdd(acc.list_count, 1u);
            if (pos < acc.list_cap) acc.list[pos] = i;
        }
    }
}

// A weight hit c times in a launch takes min(c, K) sequential updates with the mean error
// (K = kappa / alpha). Few hits (rare positions): exactly sequential TD, TC included. Many hits
// (early-game features shared by every game in flight): bounded, which keeps thousands of
// concurrent games stable.
__device__ __forceinline__ void apply_one(float* w, float* g, unsigned* n, unsigned long long i, int stride, float step,
                                          unsigned K) {
    unsigned c = n[i];
    if (!c) return;
    float d = g[i] / float(c);
    unsigned reps = c < K ? c : K;
    g[i] = 0;
    n[i] = 0;
    if (stride == 3) {
        float* p = w + 3 * i;
        for (unsigned r = 0; r < reps; ++r) {
            float coherence = p[2] > 0 ? fabsf(p[1]) / p[2] : 1.0f;
            p[0] += step * coherence * d;
            p[1] += d;
            p[2] += fabsf(d);
        }
    } else {
        for (unsigned r = 0; r < reps; ++r) w[i] += step * d;
    }
}

__global__ void apply_list_kernel(float* w, float* g, unsigned* n, const unsigned long long* list, unsigned count, int stride,
                                  float step, unsigned K) {
    for (unsigned k = blockIdx.x * blockDim.x + threadIdx.x; k < count; k += gridDim.x * blockDim.x)
        apply_one(w, g, n, list[k], stride, step, K);
}

// Fallback when the touched list overflowed: sweep every weight.
__global__ void apply_all_kernel(float* w, float* g, unsigned* n, unsigned long long count, int stride, float step, unsigned K) {
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < count;
         i += (unsigned long long)gridDim.x * blockDim.x)
        apply_one(w, g, n, i, stride, step, K);
}

__device__ float members_value(const Members& m, unsigned long long a, int lane) {
    float v = 0, ws = 0;
    for (int k = 0; k < m.n; ++k) {
        v += m.wt[k] * warp_value(c_shapes[m.shape[k]], m.w[k], a, lane);
        ws += m.wt[k];
    }
    return v / ws;
}

template <int D>
__device__ float chance_node(const Members& m, unsigned long long a, int lane);

template <int D>
__device__ float max_node(const Members& m, unsigned long long b, int lane, int* best_dir) {
    float best = 0;
    int dir = -1;
    for (int d = 0; d < 4; ++d) {
        unsigned r = 0;
        unsigned long long a = d_move(b, d, r);
        if (a == b) continue;
        float val;
        if constexpr (D == 1) val = float(r) + members_value(m, a, lane);
        else val = float(r) + chance_node<D - 1>(m, a, lane);
        if (dir < 0 || val > best) dir = d, best = val;
    }
    if (best_dir) *best_dir = dir;
    return best;
}

template <int D>
__device__ float chance_node(const Members& m, unsigned long long a, int lane) {
    float sum = 0;
    int n = 0;
    for (int i = 0; i < 16; ++i) {
        if ((a >> (4 * i)) & 0xF) continue;
        sum += 0.9f * max_node<D>(m, a | (1ULL << (4 * i)), lane, nullptr) +
               0.1f * max_node<D>(m, a | (2ULL << (4 * i)), lane, nullptr);
        n++;
    }
    return n ? sum / float(n) : 0;
}

__device__ void start_game(SlotState& s, long long g, unsigned long long seed) {
    DRng r;
    r.seed(seed);
    s.board = d_spawn(d_spawn(0, r), r);
    for (int i = 0; i < 4; ++i) s.rng[i] = r.s[i];
    s.game = g;
    s.prev = 0;
    s.has_prev = 0;
    s.score = 0;
    s.moves = 0;
}

__device__ void advance(SlotState& s, unsigned long long after, unsigned r) {
    DRng rng;
    for (int i = 0; i < 4; ++i) rng.s[i] = s.rng[i];
    s.board = d_spawn(after, rng);
    for (int i = 0; i < 4; ++i) s.rng[i] = rng.s[i];
    s.score += r;
    s.moves++;
}

// ---- kernels ----

__global__ void train_kernel(SlotState* st, int nslots, const int* slot_agent, const int* agent_shape,
                             const float* const* wts, const Accum* accs,
                             unsigned long long* next_game, const long long* target, const unsigned long long* seed_base,
                             Result* res, unsigned* res_count, unsigned res_cap, int* stage_hit, int moves) {
    int warp = int((blockIdx.x * blockDim.x + threadIdx.x) >> 5), lane = int(threadIdx.x & 31);
    if (warp >= nslots) return;
    SlotState s = st[warp];
    const int a = slot_agent[warp];
    const NetShape& S = c_shapes[agent_shape[a]];
    const float* w = wts[a];
    const Accum acc = accs[a];
    for (int k = 0; k < moves; ++k) {
        if (s.game < 0) {
            unsigned long long g = 0;
            if (lane == 0) g = atomicAdd(&next_game[a], 1ULL);
            g = __shfl_sync(FULL, g, 0);
            if ((long long)g >= target[a]) break;
            start_game(s, (long long)g, seed_base[a] + g);
        }
        int best = -1;
        float best_v = 0, best_av = 0;
        unsigned best_r = 0;
        unsigned long long best_a = 0;
        for (int d = 0; d < 4; ++d) {
            unsigned r = 0;
            unsigned long long af = d_move(s.board, d, r);
            if (af == s.board) continue;
            float av = warp_value(S, w, af, lane);
            if (best < 0 || float(r) + av > best_v) best = d, best_v = float(r) + av, best_av = av, best_r = r, best_a = af;
        }
        if (best < 0) {
            if (s.has_prev) warp_accumulate(S, acc, s.prev, -warp_value(S, w, s.prev, lane), lane);
            if (lane == 0) {
                unsigned p = atomicAdd(res_count, 1u);
                if (p < res_cap) res[p] = {a, d_max_exp(s.board), s.moves, s.score, seed_base[a] + (unsigned long long)s.game};
            }
            s.game = -1;
            continue;
        }
        if (s.has_prev) warp_accumulate(S, acc, s.prev, float(best_r) + best_av - warp_value(S, w, s.prev, lane), lane);
        if (S.stages > 1 && lane == 0) {
            int sg = d_stage(S, best_a);
            if (sg > 0) stage_hit[a * kMaxStages + sg] = 1;
        }
        s.prev = best_a;
        s.has_prev = 1;
        advance(s, best_a, best_r);
    }
    if (lane == 0) st[warp] = s;
}

template <int D>
__global__ void eval_kernel(SlotState* st, int nslots, Members m, unsigned long long* next_game, long long n,
                            unsigned long long seed0, Result* res, unsigned* finished, int moves) {
    int warp = int((blockIdx.x * blockDim.x + threadIdx.x) >> 5), lane = int(threadIdx.x & 31);
    if (warp >= nslots) return;
    SlotState s = st[warp];
    for (int k = 0; k < moves; ++k) {
        if (s.game < 0) {
            unsigned long long g = 0;
            if (lane == 0) g = atomicAdd(next_game, 1ULL);
            g = __shfl_sync(FULL, g, 0);
            if ((long long)g >= n) break;
            start_game(s, (long long)g, seed0 + g);
        }
        int dir;
        max_node<D>(m, s.board, lane, &dir);
        if (dir < 0) {
            if (lane == 0) {
                res[s.game] = {0, d_max_exp(s.board), s.moves, s.score, seed0 + (unsigned long long)s.game};
                atomicAdd(finished, 1u);
            }
            s.game = -1;
            continue;
        }
        unsigned r = 0;
        advance(s, d_move(s.board, dir, r), r);
    }
    if (lane == 0) st[warp] = s;
}

constexpr int kThreads = 256;
int blocks_for(int slots) { return (slots * 32 + kThreads - 1) / kThreads; }

}  // namespace

std::string gpu_info() {
    int dev = 0;
    cudaDeviceProp p{};
    CK(cudaGetDevice(&dev));
    CK(cudaGetDeviceProperties(&p, dev));
    size_t free_b = 0, total_b = 0;
    CK(cudaMemGetInfo(&free_b, &total_b));
    char buf[1024];
    std::snprintf(buf, sizeof buf, "%s, sm_%d%d, %d SM, %.1f/%.1f GB bos", p.name, p.major, p.minor, p.multiProcessorCount,
                  double(free_b) / 1e9, double(total_b) / 1e9);
    return buf;
}

struct GpuPopulation::Impl {
    std::vector<NetShape> shapes;       // distinct shapes (uploaded to c_shapes)
    std::vector<int> agent_shape;       // agent -> shape index
    std::vector<size_t> floats;         // per agent
    int P = 0, slots = 0, moves_per_launch = 32;
    std::vector<float*> d_w, d_g;
    std::vector<unsigned*> d_n;
    float** d_wptrs = nullptr;
    int* d_agent_shape = nullptr;
    std::vector<std::vector<bool>> seen;
    SlotState* d_state = nullptr;

    ~Impl() {  // also runs when the constructor throws half way
        for (float* d : d_w) cudaFree(d);
        for (float* d : d_g) cudaFree(d);
        for (unsigned* d : d_n) cudaFree(d);
        cudaFree(d_wptrs);
        cudaFree(d_agent_shape);
        cudaFree(d_state);
    }
};

static NetShape make_shape(const NTupleNet& n) {
    if (n.patterns().size() > size_t(kMaxTuples) || n.stages() > kMaxStages)
        throw std::runtime_error("GpuPopulation: network too large for the GPU kernels");
    NetShape sh{};
    sh.F = n.features();
    sh.stride = n.tc() ? 3 : 1;
    sh.stages = n.stages();
    sh.wps = n.weights_per_stage();
    for (size_t t = 0; t < n.patterns().size(); ++t) {
        const auto& p = n.patterns()[t];
        if (p.size() > size_t(kMaxLen)) throw std::runtime_error("GpuPopulation: tuple longer than 6");
        sh.len[t] = int(p.size());
        sh.off[t] = n.tuple_offset(t);
        for (int s = 0; s < 8; ++s)
            for (size_t j = 0; j < p.size(); ++j) sh.shift[t * 8 + size_t(s)][j] = (unsigned char)(4 * NTupleNet::sym_cell(p[j], s));
    }
    for (int k = 0; k < sh.stages; ++k) sh.stage_mask[k] = n.stage_mask(k);
    return sh;
}

GpuPopulation::GpuPopulation(const std::vector<NTupleNet*>& nets, int slots)
    : impl_(std::make_unique<Impl>()) {
    Impl& m = *impl_;
    if (nets.empty()) throw std::runtime_error("GpuPopulation: no nets");
    m.P = int(nets.size());
    for (auto* n : nets) {
        NetShape sh = make_shape(*n);
        int idx = -1;
        for (size_t i = 0; i < m.shapes.size(); ++i)
            if (std::memcmp(&m.shapes[i], &sh, sizeof sh) == 0) idx = int(i);
        if (idx < 0) {
            if (m.shapes.size() == size_t(kMaxShapes)) throw std::runtime_error("GpuPopulation: too many distinct shapes");
            idx = int(m.shapes.size());
            m.shapes.push_back(sh);
        }
        m.agent_shape.push_back(idx);
        m.floats.push_back(n->data_size());
        float* d = nullptr;
        CK(cudaMalloc(&d, n->data_size() * sizeof(float)));
        CK(cudaMemcpy(d, n->data(), n->data_size() * sizeof(float), cudaMemcpyHostToDevice));
        m.d_w.push_back(d);
        size_t nw = n->data_size() / size_t(sh.stride);
        float* dg = nullptr;
        unsigned* dn = nullptr;
        CK(cudaMalloc(&dg, nw * sizeof(float)));
        CK(cudaMalloc(&dn, nw * sizeof(unsigned)));
        CK(cudaMemset(dg, 0, nw * sizeof(float)));
        CK(cudaMemset(dn, 0, nw * sizeof(unsigned)));
        m.d_g.push_back(dg);
        m.d_n.push_back(dn);
        std::vector<bool> sn;
        for (int k = 0; k < n->stages(); ++k) sn.push_back(n->seen(k));
        m.seen.push_back(sn);
    }
    CK(cudaMemcpyToSymbol(c_shapes, m.shapes.data(), sizeof(NetShape) * m.shapes.size()));
    CK(cudaMalloc(&m.d_wptrs, sizeof(float*) * size_t(m.P)));
    CK(cudaMemcpy(m.d_wptrs, m.d_w.data(), sizeof(float*) * size_t(m.P), cudaMemcpyHostToDevice));

    CK(cudaMalloc(&m.d_agent_shape, sizeof(int) * size_t(m.P)));
    CK(cudaMemcpy(m.d_agent_shape, m.agent_shape.data(), sizeof(int) * size_t(m.P), cudaMemcpyHostToDevice));

    if (slots <= 0) {
        cudaDeviceProp p{};
        CK(cudaGetDeviceProperties(&p, 0));
        slots = p.multiProcessorCount * (p.maxThreadsPerMultiProcessor / 32) * 2;
    }
    m.slots = slots;
    CK(cudaMalloc(&m.d_state, sizeof(SlotState) * size_t(slots)));
}

GpuPopulation::~GpuPopulation() = default;

int GpuPopulation::size() const { return impl_->P; }
int GpuPopulation::slots() const { return impl_->slots; }
void GpuPopulation::set_moves_per_launch(int m) { impl_->moves_per_launch = std::max(1, m); }

void GpuPopulation::download(int a, NTupleNet& net) const {
    const size_t n = impl_->floats[size_t(a)];
    if (net.data_size() != n) throw std::runtime_error("download: shape mismatch");
    CK(cudaMemcpy(net.data(), impl_->d_w[size_t(a)], n * sizeof(float), cudaMemcpyDeviceToHost));
    for (int k = 0; k < net.stages(); ++k) net.set_seen(k, impl_->seen[size_t(a)][size_t(k)]);
}

void GpuPopulation::copy_agent(int dst, int src) {
    if (impl_->agent_shape[size_t(dst)] != impl_->agent_shape[size_t(src)]) throw std::runtime_error("copy_agent: shapes differ");
    CK(cudaMemcpy(impl_->d_w[size_t(dst)], impl_->d_w[size_t(src)], impl_->floats[size_t(src)] * sizeof(float),
                  cudaMemcpyDeviceToDevice));
    impl_->seen[size_t(dst)] = impl_->seen[size_t(src)];
}

// Resets every slot to "no game in progress".
static void reset_slots(SlotState* d, int n) {
    std::vector<SlotState> h(static_cast<size_t>(n));
    for (auto& s : h) s.game = -1;
    CK(cudaMemcpy(d, h.data(), sizeof(SlotState) * size_t(n), cudaMemcpyHostToDevice));
}

template <class T>
struct DevBuf {
    T* p = nullptr;
    explicit DevBuf(size_t n) { CK(cudaMalloc(&p, sizeof(T) * std::max<size_t>(n, 1))); }
    ~DevBuf() { cudaFree(p); }
    void put(const std::vector<T>& v) { CK(cudaMemcpy(p, v.data(), sizeof(T) * v.size(), cudaMemcpyHostToDevice)); }
    void get(std::vector<T>& v) const { CK(cudaMemcpy(v.data(), p, sizeof(T) * v.size(), cudaMemcpyDeviceToHost)); }
};

void GpuPopulation::train(const std::vector<GpuAgentParams>& params, std::vector<int64_t>& done,
                          const std::vector<int64_t>& target, const std::vector<uint64_t>& seed_base, int64_t window,
                          const std::function<void(int, const EvalReport&, int64_t)>& on_window,
                          const std::function<void()>& on_tick) {
    Impl& m = *impl_;
    const size_t P = size_t(m.P);
    std::vector<int> active;
    for (size_t a = 0; a < P; ++a)
        if (done[a] < target[a]) active.push_back(int(a));
    if (active.empty()) return;

    // Slots are shared evenly by the networks still training. No concurrency cap is needed:
    // batched updates (apply_kernel) keep hot features stable however many games touch them.
    const int per_agent = std::max(1, m.slots / int(active.size()));
    std::vector<int> slot_agent;
    for (int a : active) slot_agent.insert(slot_agent.end(), size_t(per_agent), a);
    const int nslots = int(slot_agent.size());
    std::vector<unsigned long long> next(done.begin(), done.end());
    std::vector<long long> tgt(target.begin(), target.end());
    std::vector<unsigned long long> sb(seed_base.begin(), seed_base.end());

    DevBuf<int> d_slot_agent(slot_agent.size());
    DevBuf<unsigned long long> d_next(P), d_seed(P);
    DevBuf<long long> d_target(P);
    DevBuf<int> d_hits(P * kMaxStages);
    const int moves = m.moves_per_launch;
    const unsigned cap = unsigned(nslots) * 8 + 1024;  // a slot ends at most ~moves/16 games per launch
    DevBuf<Result> d_res(cap);
    DevBuf<unsigned> d_count(1);
    d_slot_agent.put(slot_agent);
    d_next.put(next);
    d_seed.put(sb);
    d_target.put(tgt);
    std::vector<int> hits(P * kMaxStages, 0);
    d_hits.put(hits);
    // Touched-weight lists: at most one update (F features) per slot and move.
    std::vector<unsigned long long> list_cap(P, 0);
    for (int a : active)
        list_cap[size_t(a)] = (unsigned long long)per_agent * unsigned(moves) * unsigned(m.shapes[size_t(m.agent_shape[size_t(a)])].F);
    std::vector<DevBuf<unsigned long long>*> lists;
    DevBuf<unsigned> d_list_count(P);
    std::vector<Accum> accs(P);
    for (size_t a = 0; a < P; ++a) {
        lists.push_back(new DevBuf<unsigned long long>(list_cap[a]));
        accs[a] = {m.d_g[a], m.d_n[a], lists.back()->p, d_list_count.p + a, list_cap[a]};
    }
    struct Free {
        std::vector<DevBuf<unsigned long long>*>& v;
        ~Free() { for (auto* p : v) delete p; }
    } free_lists{lists};
    DevBuf<Accum> d_accs(P);
    d_accs.put(accs);
    std::vector<unsigned> list_count(P, 0);
    d_list_count.put(list_count);
    reset_slots(m.d_state, nslots);

    std::vector<EvalReport> win(P);
    std::vector<Result> res(cap);
    std::vector<unsigned> count(1, 0);
    auto t_win = std::vector<std::chrono::steady_clock::time_point>(P, std::chrono::steady_clock::now());
    auto unfinished = [&] {
        for (int a : active)
            if (done[size_t(a)] < target[size_t(a)]) return true;
        return false;
    };
    while (unfinished()) {
        count[0] = 0;
        d_count.put(count);
        train_kernel<<<blocks_for(nslots), kThreads>>>(m.d_state, nslots, d_slot_agent.p, m.d_agent_shape, m.d_wptrs,
                                                       d_accs.p, d_next.p,
                                                        d_target.p, d_seed.p, d_res.p, d_count.p, cap, d_hits.p, moves);
        CK(cudaGetLastError());
        d_list_count.get(list_count);
        for (int a : active) {
            const size_t ai = size_t(a);
            const NetShape& sh = m.shapes[size_t(m.agent_shape[ai])];
            const GpuAgentParams& p = params[ai];
            const float step = p.alpha / float(sh.F);
            const double kd = std::max(1.0, double(p.kappa) / double(p.alpha));
            const unsigned K = kd > 4e9 ? 0xffffffffu : unsigned(kd);
            if (list_count[ai] <= list_cap[ai]) {
                if (list_count[ai])
                    apply_list_kernel<<<std::min(4096u, (list_count[ai] + kThreads - 1) / kThreads), kThreads>>>(
                        m.d_w[ai], m.d_g[ai], m.d_n[ai], lists[ai]->p, list_count[ai], sh.stride, step, K);
            } else {
                apply_all_kernel<<<4096, kThreads>>>(m.d_w[ai], m.d_g[ai], m.d_n[ai], m.floats[ai] / size_t(sh.stride),
                                                     sh.stride, step, K);
            }
            list_count[ai] = 0;
        }
        d_list_count.put(list_count);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        d_count.get(count);
        if (count[0] > cap) throw std::runtime_error("GPU result buffer overflow");
        if (count[0]) CK(cudaMemcpy(res.data(), d_res.p, sizeof(Result) * count[0], cudaMemcpyDeviceToHost));
        for (unsigned i = 0; i < count[0]; ++i) {
            const Result& r = res[i];
            size_t a = size_t(r.agent);
            GameResult g;
            g.seed = r.seed;
            g.score = r.score;
            g.max_exp = r.max_exp;
            g.moves = int(r.moves);
            done[a]++;
            win[a].results.push_back(g);
            if (window > 0 && int64_t(win[a].results.size()) >= window) {
                auto now = std::chrono::steady_clock::now();
                win[a].seconds = std::chrono::duration<double>(now - t_win[a]).count();
                t_win[a] = now;
                if (on_window) on_window(int(a), win[a], done[a]);
                win[a].results.clear();
            }
        }
        {  // weight promotion for newly reached stages
            d_hits.get(hits);
            for (size_t a = 0; a < P; ++a) {
                const NetShape& sh = m.shapes[size_t(m.agent_shape[a])];
                for (int k = 1; k < sh.stages; ++k) {
                    if (!hits[a * kMaxStages + size_t(k)] || m.seen[a][size_t(k)]) continue;
                    for (int j = 1; j <= k; ++j) {
                        if (m.seen[a][size_t(j)]) continue;
                        size_t n = sh.wps * size_t(sh.stride);
                        CK(cudaMemcpy(m.d_w[a] + n * size_t(j), m.d_w[a] + n * size_t(j - 1), n * sizeof(float),
                                      cudaMemcpyDeviceToDevice));
                        m.seen[a][size_t(j)] = true;
                    }
                }
            }
        }
        if (on_tick) on_tick();
    }
}

EvalReport GpuPopulation::evaluate(const std::vector<int>& members, const std::vector<float>& weights, int depth,
                                   const EvalConfig& cfg) const {
    const Impl& m = *impl_;
    if (members.empty() || members.size() > size_t(kMaxMembers)) throw std::runtime_error("evaluate: 1..8 members");
    Members mem{};
    mem.n = int(members.size());
    for (size_t i = 0; i < members.size(); ++i) {
        mem.w[i] = m.d_w[size_t(members[i])];
        mem.wt[i] = weights.empty() ? 1.0f : weights[i];
        mem.shape[i] = m.agent_shape[size_t(members[i])];
    }
    const int n = cfg.games, slots = std::min(m.slots, n);
    const int moves = depth == 1 ? 256 : depth == 2 ? 4 : 1;  // keep launches short
    DevBuf<unsigned long long> d_next(1);
    DevBuf<Result> d_res(static_cast<size_t>(n));
    DevBuf<unsigned> d_fin(1);
    d_next.put({0ULL});
    d_fin.put({0u});
    reset_slots(m.d_state, slots);
    auto t0 = std::chrono::steady_clock::now();
    std::vector<unsigned> fin(1, 0);
    while (int(fin[0]) < n) {
        switch (depth) {
            case 1: eval_kernel<1><<<blocks_for(slots), kThreads>>>(m.d_state, slots, mem, d_next.p, n, cfg.seed0, d_res.p, d_fin.p, moves); break;
            case 2: eval_kernel<2><<<blocks_for(slots), kThreads>>>(m.d_state, slots, mem, d_next.p, n, cfg.seed0, d_res.p, d_fin.p, moves); break;
            case 3: eval_kernel<3><<<blocks_for(slots), kThreads>>>(m.d_state, slots, mem, d_next.p, n, cfg.seed0, d_res.p, d_fin.p, moves); break;
            default: throw std::runtime_error("evaluate: depth 1..3");
        }
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        d_fin.get(fin);
    }
    std::vector<Result> res(static_cast<size_t>(n));
    d_res.get(res);
    EvalReport rep;
    rep.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    for (const Result& r : res) {
        GameResult g;
        g.seed = r.seed;
        g.score = r.score;
        g.max_exp = r.max_exp;
        g.moves = int(r.moves);
        rep.results.push_back(g);
    }
    return rep;
}

}  // namespace k2048
