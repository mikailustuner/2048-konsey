#include "td.hpp"

#include <chrono>
#include <cstdio>
#include <thread>

namespace k2048 {

GameResult td_episode(NTupleNet& net, const TDParams& p, uint64_t seed) {
    const float step = p.alpha / float(net.features());
    const bool multistage = net.stages() > 1;
    GameResult res;
    res.seed = seed;
    Rng rng(seed);
    Board b = initial_board(rng), prev = 0;
    bool has_prev = false;
    std::vector<Board> traj;
    std::vector<float> rew;

    for (;;) {
        int best = -1;
        float best_v = 0, best_after_v = 0;
        uint32_t best_r = 0;
        Board best_a = 0;
        for (int d = 0; d < 4; ++d) {
            uint32_t r = 0;
            Board a = move(b, d, r);
            if (a == b) continue;
            float av = net.value(a);
            if (best < 0 || float(r) + av > best_v) best = d, best_v = float(r) + av, best_after_v = av, best_r = r, best_a = a;
        }
        if (best < 0) break;
        if (multistage) net.promote_if_new(net.stage(best_a));
        if (p.lambda > 0) {
            traj.push_back(best_a);
            rew.push_back(float(best_r));
        } else if (has_prev) {
            net.update(prev, float(best_r) + best_after_v - net.value(prev), step);
        }
        prev = best_a;
        has_prev = true;
        res.score += best_r;
        res.moves++;
        b = spawn(best_a, rng);
    }

    if (p.lambda > 0 && !traj.empty()) {
        float g = 0;  // the last afterstate leads to a dead position: target 0
        size_t t = traj.size() - 1;
        net.update(traj[t], g - net.value(traj[t]), step);
        while (t-- > 0) {
            g = rew[t + 1] + (1 - p.lambda) * net.value(traj[t + 1]) + p.lambda * g;
            net.update(traj[t], g - net.value(traj[t]), step);
        }
    } else if (has_prev) {
        net.update(prev, -net.value(prev), step);
    }
    res.max_exp = max_exp(b);
    return res;
}

std::string ckpt_header(const TDParams& p, int64_t games) {
    return "games=" + std::to_string(games) + "\nalpha=" + std::to_string(p.alpha) +
           "\nlambda=" + std::to_string(p.lambda) + "\n";
}

int64_t ckpt_games(const std::string& h) {
    size_t i = h.find("games=");
    return i == std::string::npos ? 0 : std::stoll(h.substr(i + 6));
}

int64_t train(NTupleNet& net, const TDParams& p, const TrainOptions& o, int64_t done) {
    std::atomic<int64_t> next{done};
    std::mutex mu;
    EvalReport window;
    auto t0 = std::chrono::steady_clock::now();
    auto last_ckpt = t0, last_log = t0;

    auto worker = [&](int tid) {
        for (int64_t g; (g = next++) < o.games;) {
            GameResult r = td_episode(net, p, o.seed + uint64_t(g));
            std::lock_guard<std::mutex> lock(mu);
            window.results.push_back(r);
            auto now = std::chrono::steady_clock::now();
            if (int64_t(window.results.size()) >= o.log_every) {
                window.seconds = std::chrono::duration<double>(now - last_log).count();
                LogMeta m = o.meta;
                m.egitim_oyunu = g + 1;
                if (!o.log_path.empty()) append_line(o.log_path, log_line(m, window));
                if (!o.quiet)
                    std::printf("[%lld oyun, %.0f oyun/sn] ort %.0f | 2048 %.1f%% 8192 %.1f%% 16384 %.1f%% 32768 %.2f%%\n",
                                (long long)(g + 1), double(window.games()) / window.seconds, window.mean_score(),
                                100.0 * double(window.reached(11)) / window.games(),
                                100.0 * double(window.reached(13)) / window.games(),
                                100.0 * double(window.reached(14)) / window.games(),
                                100.0 * double(window.reached(15)) / window.games());
                std::fflush(stdout);
                window.results.clear();
                last_log = now;
            }
            if (tid == 0 && !o.ckpt.empty() &&
                std::chrono::duration<double>(now - last_ckpt).count() >= o.ckpt_sec) {
                net.save(o.ckpt, ckpt_header(p, g + 1));
                last_ckpt = now;
            }
        }
    };
    std::vector<std::thread> pool;
    for (int t = 1; t < o.threads; ++t) pool.emplace_back(worker, t);
    worker(0);
    for (auto& t : pool) t.join();
    int64_t total = std::max(done, o.games);
    if (!o.ckpt.empty()) net.save(o.ckpt, ckpt_header(p, total));
    return total;
}

}  // namespace k2048
