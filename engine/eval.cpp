#include "eval.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <thread>

namespace k2048 {

Wilson wilson(uint64_t k, uint64_t n) {
    if (n == 0) return {0, 0, 1};
    const double z = 1.96, nn = double(n), p = double(k) / nn;
    const double den = 1 + z * z / nn;
    const double mid = (p + z * z / (2 * nn)) / den;
    const double half = z * std::sqrt(p * (1 - p) / nn + z * z / (4 * nn * nn)) / den;
    return {p, std::max(0.0, mid - half), std::min(1.0, mid + half)};
}

double EvalReport::mean_score() const {
    double s = 0;
    for (auto& g : results) s += double(g.score);
    return results.empty() ? 0 : s / double(results.size());
}

uint64_t EvalReport::max_score() const {
    uint64_t m = 0;
    for (auto& g : results) m = std::max(m, g.score);
    return m;
}

uint64_t EvalReport::total_moves() const {
    uint64_t m = 0;
    for (auto& g : results) m += uint64_t(g.moves);
    return m;
}

uint64_t EvalReport::reached(int e) const {
    uint64_t k = 0;
    for (auto& g : results) k += g.max_exp >= e;
    return k;
}

EvalReport evaluate(const std::function<Policy()>& make_policy, const EvalConfig& cfg) {
    EvalReport rep;
    rep.results.resize(size_t(cfg.games));
    std::atomic<int> next{0};
    auto t0 = std::chrono::steady_clock::now();
    auto worker = [&] {
        Policy p = make_policy();
        for (int i; (i = next++) < cfg.games;)
            rep.results[size_t(i)] = play_game(cfg.seed0 + uint64_t(i), p, cfg.record);
    };
    std::vector<std::thread> pool;
    for (int t = 1; t < cfg.threads; ++t) pool.emplace_back(worker);
    worker();
    for (auto& t : pool) t.join();
    rep.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return rep;
}

static std::string esc(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') o += '\\';
        o += c;
    }
    return o;
}

std::string log_line(const LogMeta& m, const EvalReport& r) {
    char buf[256];
    std::string s = "{\"run_id\":\"" + esc(m.run_id) + "\",\"faz\":\"" + esc(m.faz) +
                    "\",\"ajan_id\":\"" + esc(m.ajan_id) + "\",\"nesil\":" + std::to_string(m.nesil) +
                    ",\"oyun_sayisi\":" + std::to_string(r.games()) +
                    ",\"hamle_sayisi\":" + std::to_string(r.total_moves());
    if (m.egitim_oyunu >= 0) s += ",\"egitim_oyunu\":" + std::to_string(m.egitim_oyunu);
    std::string rates = ",\"oranlar\":{", ci = ",\"guven\":{";
    for (int e = 11; e <= 16; ++e) {
        Wilson w = wilson(r.reached(e), uint64_t(r.games()));
        const char* sep = e == 11 ? "" : ",";
        std::snprintf(buf, sizeof buf, "%s\"%d\":%.6f", sep, 1 << e, w.p);
        rates += buf;
        std::snprintf(buf, sizeof buf, "%s\"%d\":[%.6f,%.6f]", sep, 1 << e, w.lo, w.hi);
        ci += buf;
    }
    s += rates + "}" + ci + "}";
    std::snprintf(buf, sizeof buf, ",\"ort_skor\":%.1f,\"maks_skor\":%llu,\"sure\":%.3f}", r.mean_score(),
                  (unsigned long long)r.max_score(), r.seconds);
    return s + buf;
}

std::string summary(const EvalReport& r) {
    char buf[256];
    std::string s;
    std::snprintf(buf, sizeof buf, "oyun %d | ort. skor %.0f | maks. skor %llu | %.1f hamle/oyun | %.2f sn\n",
                  r.games(), r.mean_score(), (unsigned long long)r.max_score(),
                  r.games() ? double(r.total_moves()) / r.games() : 0.0, r.seconds);
    s += buf;
    for (int e = 7; e <= 17; ++e) {
        uint64_t k = r.reached(e);
        if (k == 0 && e > 11) break;
        Wilson w = wilson(k, uint64_t(r.games()));
        std::snprintf(buf, sizeof buf, "  %6d: %6.2f%%  [%6.2f, %6.2f]  (%llu)\n", 1 << e, 100 * w.p, 100 * w.lo,
                      100 * w.hi, (unsigned long long)k);
        s += buf;
    }
    return s;
}

void append_line(const std::string& path, const std::string& line) {
    std::filesystem::path p(path);
    if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
    std::ofstream f(path, std::ios::app);
    if (!f) throw std::runtime_error("cannot open " + path);
    f << line << '\n';
}

}  // namespace k2048
