#include "gpu_pbt.hpp"

#include "gpu.hpp"

namespace k2048 {

namespace {

class GpuBackend : public PBTBackend {
public:
    explicit GpuBackend(int slots) : slots_(slots) {}

    void begin(std::vector<PBTAgent>& pop) override {
        std::vector<NTupleNet*> nets;
        for (auto& a : pop) nets.push_back(a.net.get());
        gp_ = std::make_unique<GpuPopulation>(nets, slots_);
    }

    void train(std::vector<PBTAgent>& pop, const PBTConfig& c, int gen) override {
        std::vector<GpuAgentParams> params;
        std::vector<int64_t> done, target;
        std::vector<uint64_t> seeds;
        for (auto& a : pop) {
            params.push_back({a.g.alpha});
            done.push_back(a.games);
            target.push_back(a.games + c.train_games);
            seeds.push_back((uint64_t(a.uid) + 1) << 40);  // lineage-specific training seeds
        }
        gp_->train(params, done, target, seeds, c.train_games, [&](int i, const EvalReport& w, int64_t games) {
            TrainOptions o;
            o.log_path = c.dir + "/train.jsonl";
            o.meta = {c.run_id, "4", "a" + std::to_string(pop[size_t(i)].uid), gen};
            o.quiet = true;
            report_window(o, games, w);
        });
        for (size_t i = 0; i < pop.size(); ++i) pop[i].games = done[i];
    }

    std::vector<EvalReport> tournament(std::vector<PBTAgent>& pop, const EvalConfig& ec) override {
        std::vector<EvalReport> reps;
        for (size_t i = 0; i < pop.size(); ++i) reps.push_back(gp_->evaluate({int(i)}, {}, 1, ec));
        return reps;
    }

    void exploit(std::vector<PBTAgent>&, size_t dst, size_t src) override { gp_->copy_agent(int(dst), int(src)); }

    void sync_to_host(std::vector<PBTAgent>& pop) override {
        for (size_t i = 0; i < pop.size(); ++i) gp_->download(int(i), *pop[i].net);
    }

private:
    int slots_;
    std::unique_ptr<GpuPopulation> gp_;
};

}  // namespace

std::unique_ptr<PBTBackend> make_gpu_pbt_backend(int slots) { return std::make_unique<GpuBackend>(slots); }

}  // namespace k2048
