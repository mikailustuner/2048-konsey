#pragma once
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

namespace k2048 {

// Minimal "--key value" / "--flag" parser. Unknown keys are an error (caught by check()).
class Args {
public:
    Args(int argc, char** argv) {
        for (int i = 1; i < argc; ++i) {
            std::string k = argv[i];
            if (k.rfind("--", 0) != 0) throw std::runtime_error("unexpected argument: " + k);
            k = k.substr(2);
            if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0)
                kv_[k] = argv[++i];
            else
                kv_[k] = "1";
        }
    }

    bool has(const std::string& k) const {
        used_.insert(k);
        return kv_.count(k) > 0;
    }
    std::string str(const std::string& k, const std::string& def = "") const {
        return has(k) ? kv_.at(k) : def;
    }
    long long num(const std::string& k, long long def) const {
        return has(k) ? std::stoll(kv_.at(k)) : def;
    }
    double real(const std::string& k, double def) const { return has(k) ? std::stod(kv_.at(k)) : def; }

    // Call after reading all options: rejects typos.
    void check() const {
        for (auto& [k, v] : kv_)
            if (!used_.count(k)) throw std::runtime_error("unknown option --" + k);
    }

private:
    std::map<std::string, std::string> kv_;
    mutable std::set<std::string> used_;
};

// Runs a tool main, printing exceptions instead of aborting.
template <class F>
int run_main(F&& f) {
    try {
        return f();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "hata: %s\n", e.what());
        return 1;
    }
}

}  // namespace k2048
