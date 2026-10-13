#include "council.hpp"

#include <atomic>
#include <cstdio>
#include <mutex>
#include <thread>

#include "search/search.hpp"

namespace k2048 {

Council::Mode parse_mode(const std::string& s) {
    if (s == "mean") return Council::Mode::Mean;
    if (s == "weighted") return Council::Mode::Weighted;
    if (s == "vote") return Council::Mode::Vote;
    throw std::runtime_error("unknown --combine " + s);
}

static float weight(const Council& c, size_t i) { return c.weights.empty() ? 1.0f : c.weights[i]; }

float Council::value(Board a) const {
    float v = 0, w = 0;
    for (size_t i = 0; i < nets.size(); ++i) {
        float wi = mode == Mode::Mean ? 1.0f : weight(*this, i);
        v += wi * nets[i]->value(a);
        w += wi;
    }
    return v / w;
}

int Council::choose(Board b, int depth) const {
    if (mode != Mode::Vote) return expectimax_move(b, depth, [this](Board a) { return value(a); });
    float votes[4] = {0, 0, 0, 0};
    int count[4] = {0, 0, 0, 0};
    for (size_t i = 0; i < nets.size(); ++i) {
        const NTupleNet* n = nets[i].get();
        int d = expectimax_move(b, depth, [n](Board a) { return n->value(a); });
        if (d < 0) return -1;
        count[d]++;
        votes[d] += weight(*this, i);
    }
    int best = -1;
    for (int d = 0; d < 4; ++d)
        if (count[d] && (best < 0 || count[d] > count[best] || (count[d] == count[best] && votes[d] > votes[best])))
            best = d;
    return best;
}

double distill(NTupleNet& student, const Council& teacher, int64_t games, int threads, uint64_t seed, float alpha,
               const std::string& log_path, const LogMeta& meta) {
    const float step = alpha / float(student.features());
    std::atomic<int64_t> next{0};
    std::mutex mu;
    EvalReport window, all;
    auto worker = [&] {
        for (int64_t g; (g = next++) < games;) {
            GameResult res;
            res.seed = seed + uint64_t(g);
            Rng rng(res.seed);
            Board b = initial_board(rng);
            for (;;) {
                int best = -1;
                float bv = 0;
                Board ba = 0;
                uint32_t br = 0;
                for (int d = 0; d < 4; ++d) {
                    uint32_t r = 0;
                    Board a = move(b, d, r);
                    if (a == b) continue;
                    float tv = teacher.value(a);
                    if (student.stages() > 1) student.promote_if_new(student.stage(a));
                    student.update(a, tv - student.value(a), step);
                    if (best < 0 || float(r) + tv > bv) best = d, bv = float(r) + tv, ba = a, br = r;
                }
                if (best < 0) break;
                res.score += br;
                res.moves++;
                b = spawn(ba, rng);
            }
            res.max_exp = max_exp(b);
            std::lock_guard<std::mutex> lock(mu);
            window.results.push_back(res);
            all.results.push_back(res);
            if (window.games() >= 1000 || g + 1 == games) {
                LogMeta m = meta;
                m.egitim_oyunu = int64_t(all.results.size());
                if (!log_path.empty()) append_line(log_path, log_line(m, window));
                std::printf("[damitma %lld oyun] ogretmen ort %.0f\n", (long long)all.results.size(), window.mean_score());
                std::fflush(stdout);
                window.results.clear();
            }
        }
    };
    std::vector<std::thread> pool;
    for (int t = 1; t < threads; ++t) pool.emplace_back(worker);
    worker();
    for (auto& t : pool) t.join();
    return all.mean_score();
}

}  // namespace k2048
