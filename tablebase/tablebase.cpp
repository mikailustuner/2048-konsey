#include "tablebase.hpp"

#include "ntuple/network.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace k2048 {

// ---------------- pattern & rules ----------------

Pattern Pattern::parse(const std::string& spec) {
    size_t colon = spec.find(':');
    if (colon == std::string::npos) throw std::runtime_error("pattern must look like \"0,1,2,3:11\"");
    Pattern p;
    std::stringstream ss(spec.substr(0, colon));
    for (std::string c; std::getline(ss, c, ',');) {
        int i = std::stoi(c);
        if (i < 0 || i > 15) throw std::runtime_error("cell out of range in pattern");
        p.fixed |= uint16_t(1u << i);
    }
    p.target = std::stoi(spec.substr(colon + 1));
    if (p.target < 2 || p.target > 15) throw std::runtime_error("pattern target must be 2..15");
    if (16 - p.locked() < p.target)  // placeholders 15, 14, ... must stay >= target
        throw std::runtime_error("too many locked cells for this target (need locked <= 16 - target)");
    if (p.locked() > 14) throw std::runtime_error("pattern needs at least two free cells");
    return p;
}

std::string Pattern::str() const {
    std::string s;
    for (int i = 0; i < 16; ++i)
        if (fixed >> i & 1) s += (s.empty() ? "" : ",") + std::to_string(i);
    return s + ":" + std::to_string(target);
}

namespace tb {

static Board locked_bits(const Pattern& p) {
    Board m = 0;
    for (int i = 0; i < 16; ++i)
        if (p.fixed >> i & 1) m |= Board(0xF) << (4 * i);
    return m;
}

Board with_placeholders(const Pattern& p, Board b) {
    int v = 15;
    for (int i = 0; i < 16; ++i)
        if (p.fixed >> i & 1) b = set_cell(b, i, v--);
    return b;
}

int layer_of(const Pattern& p, Board t) {
    int s = 0;
    for (int i = 0; i < 16; ++i)
        if (!(p.fixed >> i & 1) && cell(t, i)) s += 1 << (cell(t, i) - 1);
    return s;
}

Board allowed_move(const Pattern& p, Board t, int dir) {
    Board a = move(t, dir);
    if (a == t) return t;
    Board m = locked_bits(p);
    return (a & m) == (t & m) ? a : t;
}

bool has_target(const Pattern& p, Board t) {
    for (int i = 0; i < 16; ++i)
        if (!(p.fixed >> i & 1) && cell(t, i) >= p.target) return true;
    return false;
}

}  // namespace tb

// ---------------- helpers ----------------

namespace {

template <class F>
void parallel(int threads, size_t n, F&& f) {  // f(begin, end, thread index)
    threads = std::max(1, std::min<int>(threads, int(std::max<size_t>(1, n / 4096))));
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t)
        pool.emplace_back([&, t] { f(n * size_t(t) / size_t(threads), n * size_t(t + 1) / size_t(threads), t); });
    for (auto& th : pool) th.join();
}

void sort_unique(std::vector<Board>& v, int threads) {
    // Sort chunks in parallel, then merge pairwise (also in parallel), then drop duplicates.
    int k = std::max(1, std::min<int>(threads, int(v.size() / 65536) + 1));
    std::vector<size_t> cut;
    for (int i = 0; i <= k; ++i) cut.push_back(v.size() * size_t(i) / size_t(k));
    std::vector<std::thread> pool;
    for (int i = 0; i < k; ++i) pool.emplace_back([&, i] { std::sort(v.begin() + long(cut[i]), v.begin() + long(cut[i + 1])); });
    for (auto& t : pool) t.join();
    while (cut.size() > 2) {
        std::vector<size_t> next;
        pool.clear();
        for (size_t i = 0; i + 2 < cut.size(); i += 2) {
            pool.emplace_back([&, i] {
                std::inplace_merge(v.begin() + long(cut[i]), v.begin() + long(cut[i + 1]), v.begin() + long(cut[i + 2]));
            });
            next.push_back(cut[i]);
        }
        for (auto& t : pool) t.join();
        if (cut.size() % 2 == 0) next.push_back(cut[cut.size() - 2]);
        next.push_back(cut.back());
        cut = next;
    }
    v.erase(std::unique(v.begin(), v.end()), v.end());
}

std::string states_path(const std::string& dir, int u) { return dir + "/L" + std::to_string(u) + ".states"; }
std::string probs_path(const std::string& dir, int u) { return dir + "/L" + std::to_string(u) + ".probs"; }

template <class T>
void write_file(const std::string& path, const std::vector<T>& v) {
    std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary);
        f.write(reinterpret_cast<const char*>(v.data()), std::streamsize(v.size() * sizeof(T)));
        if (!f) throw std::runtime_error("write failed: " + tmp);
    }
    std::filesystem::rename(tmp, path);
}

template <class T>
std::vector<T> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<T> v(size_t(f.tellg()) / sizeof(T));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(v.data()), std::streamsize(v.size() * sizeof(T)));
    return v;
}

struct LayerData {  // in-memory layer during the backward pass
    std::vector<Board> states;
    std::vector<float> probs;
    float at(Board t) const {
        auto it = std::lower_bound(states.begin(), states.end(), t);
        if (it == states.end() || *it != t) throw std::logic_error("tablebase: successor missing from its layer");
        return probs[size_t(it - states.begin())];
    }
};

// Value of a position given the next two layers (shared by build and lookup).
// afterstate = true: t is already the board after a move; returns the spawn expectation.
template <class Prob>
float position_value(const Pattern& p, Board t, const Prob& prob, int* best_dir, bool afterstate = false) {
    double best = 0;
    int dir = -1;
    for (int d = 0; d < 4; ++d) {
        Board a = afterstate ? t : tb::allowed_move(p, t, d);
        if (afterstate && d > 0) break;
        if (!afterstate && a == t) continue;
        double v;
        if (tb::has_target(p, a)) {
            v = 1;
        } else {
            double sum = 0;
            int n = 0;
            for (int i = 0; i < 16; ++i) {
                if (cell(a, i)) continue;
                double p2 = prob(set_cell(a, i, 1)), p4 = prob(set_cell(a, i, 2));
                if (p2 < 0 || p4 < 0) return -1;  // unknown successor (lookup only)
                sum += 0.9 * p2 + 0.1 * p4;
                n++;
            }
            v = sum / n;
        }
        if (dir < 0 || v > best) best = v, dir = d;
    }
    if (best_dir) *best_dir = dir;
    return float(best);
}

}  // namespace

// ---------------- build ----------------

BuildStats build_tablebase(const Pattern& p, const std::string& dir, int threads, bool quiet, uint64_t max_states) {
    auto t0 = std::chrono::steady_clock::now();
    std::filesystem::create_directories(dir);
    BuildStats st;
    std::vector<int> free_cells;
    for (int i = 0; i < 16; ++i)
        if (!(p.fixed >> i & 1)) free_cells.push_back(i);

    // Forward: every position reachable from two small tiles in the free cells.
    std::map<int, std::vector<Board>> pending;
    for (size_t i = 0; i < free_cells.size(); ++i)
        for (size_t j = i + 1; j < free_cells.size(); ++j)
            for (int vi : {1, 2})
                for (int vj : {1, 2}) {
                    Board b = tb::with_placeholders(p, set_cell(set_cell(0, free_cells[i], vi), free_cells[j], vj));
                    pending[tb::layer_of(p, b)].push_back(b);
                }
    int umin = pending.begin()->first, umax = umin;
    std::mutex mu;
    while (!pending.empty()) {
        int u = pending.begin()->first;
        std::vector<Board> states = std::move(pending.begin()->second);
        pending.erase(pending.begin());
        sort_unique(states, threads);
        write_file(states_path(dir, u), states);
        st.states += states.size();
        st.layers++;
        if (max_states && st.states > max_states)
            throw std::runtime_error("tablebase " + p.str() + ": more than " + std::to_string(max_states) +
                                     " positions (layer " + std::to_string(u) + "); pattern too large, delete " + dir);
        umax = u;
        std::vector<Board> next1, next2;
        parallel(threads, states.size(), [&](size_t b, size_t e, int) {
            std::vector<Board> n1, n2;
            for (size_t k = b; k < e; ++k)
                for (int d = 0; d < 4; ++d) {
                    Board a = tb::allowed_move(p, states[k], d);
                    if (a == states[k] || tb::has_target(p, a)) continue;
                    for (int i : free_cells)
                        if (!cell(a, i)) n1.push_back(set_cell(a, i, 1)), n2.push_back(set_cell(a, i, 2));
                }
            std::lock_guard<std::mutex> l(mu);
            next1.insert(next1.end(), n1.begin(), n1.end());
            next2.insert(next2.end(), n2.begin(), n2.end());
        });
        for (auto [k, add] : {std::pair{u + 1, &next1}, std::pair{u + 2, &next2}}) {
            if (add->empty()) continue;
            auto& v = pending[k];
            v.insert(v.end(), add->begin(), add->end());
            sort_unique(v, threads);  // keep the pending layers compact
        }
        if (!quiet && st.layers % 50 == 0) {
            std::printf("  ileri: katman %d, %llu durum\n", u, (unsigned long long)st.states);
            std::fflush(stdout);
        }
    }

    // Backward: success probabilities from the largest sums down.
    LayerData l1, l2;  // layers u+1 and u+2
    for (int u = umax; u >= umin; --u) {
        LayerData cur;
        if (std::filesystem::exists(states_path(dir, u))) cur.states = read_file<Board>(states_path(dir, u));
        cur.probs.resize(cur.states.size());
        auto prob = [&](Board t) -> double {
            int lu = tb::layer_of(p, t);
            return (lu == u + 1 ? l1 : l2).at(t);
        };
        parallel(threads, cur.states.size(), [&](size_t b, size_t e, int) {
            for (size_t k = b; k < e; ++k) cur.probs[k] = position_value(p, cur.states[k], prob, nullptr);
        });
        if (!cur.states.empty()) write_file(probs_path(dir, u), cur.probs);
        l2 = std::move(l1);
        l1 = std::move(cur);
    }
    std::ofstream meta(dir + "/meta.txt");
    meta << "pattern=" << p.str() << "\nmin=" << umin << "\nmax=" << umax << "\nstates=" << st.states << "\n";
    st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (!quiet)
        std::printf("tablo %s: %d katman, %llu durum, %.1f sn\n", p.str().c_str(), st.layers,
                    (unsigned long long)st.states, st.seconds);
    return st;
}

// ---------------- reader ----------------

struct Tablebase::Layer {
    const Board* states = nullptr;
    const float* probs = nullptr;
    size_t n = 0;
    void* map_s = nullptr;
    void* map_p = nullptr;
    size_t bytes_s = 0, bytes_p = 0;

    static void* map(const std::string& path, size_t& bytes) {
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("cannot open " + path);
        struct stat sb{};
        fstat(fd, &sb);
        bytes = size_t(sb.st_size);
        void* m = bytes ? mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fd, 0) : nullptr;
        ::close(fd);
        if (m == MAP_FAILED) throw std::runtime_error("mmap failed: " + path);
        return m;
    }
    ~Layer() {
        if (map_s) munmap(map_s, bytes_s);
        if (map_p) munmap(map_p, bytes_p);
    }
};

Tablebase::Tablebase(const std::string& dir) {
    std::ifstream meta(dir + "/meta.txt");
    if (!meta) throw std::runtime_error("not a tablebase (no meta.txt): " + dir);
    int umin = 0, umax = -1;
    for (std::string line; std::getline(meta, line);) {
        if (line.rfind("pattern=", 0) == 0) pattern_ = Pattern::parse(line.substr(8));
        if (line.rfind("min=", 0) == 0) umin = std::stoi(line.substr(4));
        if (line.rfind("max=", 0) == 0) umax = std::stoi(line.substr(4));
    }
    layers_.resize(size_t(std::max(0, umax + 1)));
    for (int u = umin; u <= umax; ++u) {
        if (!std::filesystem::exists(states_path(dir, u))) continue;
        auto l = std::make_unique<Layer>();
        l->map_s = Layer::map(states_path(dir, u), l->bytes_s);
        l->map_p = Layer::map(probs_path(dir, u), l->bytes_p);
        l->states = static_cast<const Board*>(l->map_s);
        l->probs = static_cast<const float*>(l->map_p);
        l->n = l->bytes_s / sizeof(Board);
        layers_[size_t(u)] = std::move(l);
    }
}

Tablebase::~Tablebase() = default;

uint64_t Tablebase::states() const {
    uint64_t n = 0;
    for (auto& l : layers_)
        if (l) n += l->n;
    return n;
}

float Tablebase::lookup(const Layer& l, Board t) const {
    const Board* it = std::lower_bound(l.states, l.states + l.n, t);
    return it != l.states + l.n && *it == t ? l.probs[it - l.states] : -1.0f;
}

bool Tablebase::applies(const Pattern& p, Board real) {
    uint32_t seen = 0;
    int smallest = 99;
    for (int i = 0; i < 16; ++i) {
        int v = cell(real, i);
        if (p.fixed >> i & 1) {
            if (v < p.target || (seen >> v & 1)) return false;
            seen |= 1u << v;
            smallest = std::min(smallest, v);
        } else if (v >= p.target) {
            return false;
        }
    }
    return smallest == p.target;
}

bool Tablebase::to_table(Board real, Board& t) const {
    if (!applies(pattern_, real)) return false;
    Board free_part = 0;
    for (int i = 0; i < 16; ++i)
        if (!(pattern_.fixed >> i & 1)) free_part = set_cell(free_part, i, cell(real, i));
    t = tb::with_placeholders(pattern_, free_part);
    return true;
}

float Tablebase::prob(Board t) const {
    size_t u = size_t(tb::layer_of(pattern_, t));
    if (u >= layers_.size() || !layers_[u]) return -1;
    return lookup(*layers_[u], t);
}

int Tablebase::best_move(Board real, float* p) const {
    Board t;
    if (!to_table(real, t) || prob(t) < 0) return -1;
    int dir;
    float v = position_value(pattern_, t, [&](Board x) { return double(prob(x)); }, &dir);
    if (v < 0) return -1;
    if (p) *p = v;
    return dir;
}

namespace {

Board sym_board(Board b, int s) {
    Board r = 0;
    for (int i = 0; i < 16; ++i) r = set_cell(r, NTupleNet::sym_cell(i, s), cell(b, i));
    return r;
}

// dir_map[s][d]: the real direction whose image under symmetry s is direction d.
struct DirMap {
    int m[8][4];
    DirMap() {
        Board b = 0x0000000012340121ULL;  // asymmetric: every direction gives a different board
        for (int s = 0; s < 8; ++s)
            for (int d = 0; d < 4; ++d)
                for (int r = 0; r < 4; ++r)
                    if (sym_board(move(b, r), s) == move(sym_board(b, s), d)) m[s][d] = r;
    }
};
// Built on first use: a namespace-scope static could run before the move tables exist.
const DirMap& dir_map() {
    static const DirMap m;
    return m;
}

}  // namespace

bool Tablebase::move_probs_any(Board real, float probs[4]) const {
    bool any = false;
    for (int d = 0; d < 4; ++d) probs[d] = -1;
    for (int s = 0; s < 8; ++s) {
        Board t;
        if (!to_table(sym_board(real, s), t) || prob(t) < 0) continue;
        for (int d = 0; d < 4; ++d) {
            Board a = tb::allowed_move(pattern_, t, d);
            if (a == t) continue;
            float v = position_value(pattern_, a, [&](Board x) { return double(prob(x)); }, nullptr, true);
            if (v < 0) continue;
            any = true;
            float& out = probs[dir_map().m[s][d]];
            out = std::max(out, v);
        }
    }
    return any;
}

int Tablebase::best_move_any(Board real, float* p) const {
    int best = -1;
    float bp = -1;
    for (int s = 0; s < 8; ++s) {
        float q;
        int d = best_move(sym_board(real, s), &q);
        if (d >= 0 && q > bp) bp = q, best = dir_map().m[s][d];
    }
    if (p && best >= 0) *p = bp;
    return best;
}

}  // namespace k2048
