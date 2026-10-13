// distill: trains one student network on a council's values (Faz 5).
//   distill --teachers a.bin,b.bin,c.bin [--combine mean|weighted --weights ..] --net 4x6
//           --games 200000 --alpha 0.1 --out runs/faz5/student.bin
#include <cstdio>
#include <filesystem>
#include <sstream>
#include <thread>

#include "council/council.hpp"
#include "tools/cli.hpp"

using namespace k2048;

int main(int argc, char** argv) {
    return run_main([&] {
        Args a(argc, argv);
        Council c;
        std::stringstream ts(a.str("teachers"));
        for (std::string p; std::getline(ts, p, ',');) c.nets.push_back(NTupleNet::load(p));
        c.mode = parse_mode(a.str("combine", "mean"));
        std::stringstream ws(a.str("weights", ""));
        for (std::string w; std::getline(ws, w, ',');) c.weights.push_back(std::stof(w));
        std::string out = a.str("out", "runs/faz5/student.bin");
        std::string preset = a.str("net", "4x6");
        int64_t games = a.num("games", 100'000);
        float alpha = float(a.real("alpha", 0.1));
        int threads = int(a.num("threads", std::thread::hardware_concurrency()));
        uint64_t seed = uint64_t(a.num("seed", 3LL << 40));
        LogMeta meta{a.str("run-id", "distill"), "5", "ogrenci", 0};
        a.check();
        if (c.nets.empty() || c.mode == Council::Mode::Vote) throw std::runtime_error("need --teachers, mean/weighted");

        NTupleNet student(NTupleNet::preset(preset), {{}}, false);
        std::filesystem::path op(out);
        std::filesystem::create_directories(op.parent_path());
        distill(student, c, games, threads, seed, alpha, (op.parent_path() / "distill.jsonl").string(), meta);
        student.save(out, "games=" + std::to_string(games) + "\n");
        std::printf("ogrenci: %s\n", out.c_str());
        return 0;
    });
}
