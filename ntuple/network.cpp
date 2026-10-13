#include "network.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace k2048 {

namespace {

}  // namespace

// The 8 symmetries of the square: optional transpose, row flip, column flip.
int NTupleNet::sym_cell(int cell, int s) {
    int r = cell / 4, c = cell % 4;
    if (s & 4) std::swap(r, c);
    if (s & 1) r = 3 - r;
    if (s & 2) c = 3 - c;
    return 4 * r + c;
}

namespace {

std::string join(const std::vector<std::vector<int>>& v, char outer, char inner) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) s += outer;
        for (size_t j = 0; j < v[i].size(); ++j) {
            if (j) s += inner;
            s += std::to_string(v[i][j]);
        }
    }
    return s;
}

std::vector<std::vector<int>> split(const std::string& s, char outer, char inner) {
    std::vector<std::vector<int>> v(1);
    std::string num;
    auto flush = [&] {
        if (!num.empty()) v.back().push_back(std::stoi(num));
        num.clear();
    };
    for (char c : s) {
        if (c == outer) {
            flush();
            v.emplace_back();
        } else if (c == inner) {
            flush();
        } else {
            num += c;
        }
    }
    flush();
    return v;
}

}  // namespace

std::vector<std::vector<int>> NTupleNet::preset(const std::string& name) {
    // 4x6: the standard Yeh / Jaśkowski / moporgic shapes (row+2, 2x3 corner, 2x3 middle).
    std::vector<std::vector<int>> p = {{0, 1, 2, 3, 4, 5}, {4, 5, 6, 7, 8, 9}, {0, 1, 2, 4, 5, 6}, {4, 5, 6, 8, 9, 10}};
    if (name == "4x6") return p;
    if (name == "8x6") {
        // + four snake/hook shapes along the corner, not symmetric copies of the first four.
        p.insert(p.end(), {{0, 1, 5, 6, 7, 10}, {0, 1, 2, 5, 9, 10}, {0, 1, 5, 9, 13, 14}, {0, 1, 5, 8, 9, 13}});
        return p;
    }
    throw std::runtime_error("unknown network preset " + name);
}

NTupleNet::NTupleNet(std::vector<std::vector<int>> patterns, std::vector<std::vector<int>> stage_conds, bool tc,
                     float init_value)
    : patterns_(std::move(patterns)), stage_conds_(std::move(stage_conds)), stride_(tc ? 3 : 1) {
    if (stage_conds_.empty()) stage_conds_.push_back({});
    for (auto& cond : stage_conds_) {
        uint32_t m = 0;
        for (int e : cond) m |= 1u << e;
        stage_mask_.push_back(m);
    }
    for (auto& p : patterns_) {
        offset_.push_back(weights_per_stage_);
        weights_per_stage_ += size_t(1) << (4 * p.size());
        for (int s = 0; s < 8; ++s) {
            std::vector<uint8_t> sh;
            for (int c : p) sh.push_back(uint8_t(4 * sym_cell(c, s)));
            shifts_.push_back(sh);
        }
    }
    // Optimistic initialisation: every state starts at init_value.
    float w0 = init_value / float(features());
    data_.assign(weight_count() * size_t(stride_), 0.0f);
    for (size_t i = 0; i < weight_count(); ++i) data_[i * size_t(stride_)] = w0;
    seen_ = std::make_unique<std::atomic<bool>[]>(stage_conds_.size());
    seen_[0] = true;
}

int NTupleNet::stage(Board b) const {
    if (stage_mask_.size() == 1) return 0;
    uint32_t present = 0;
    for (int i = 0; i < 16; ++i) present |= 1u << cell(b, i);
    for (int k = int(stage_mask_.size()) - 1; k > 0; --k)
        if ((stage_mask_[size_t(k)] & ~present) == 0) return k;
    return 0;
}

template <class F>
inline void NTupleNet::for_each_feature(Board b, F&& f) const {
    size_t base = weights_per_stage_ * size_t(stage(b));
    for (size_t t = 0; t < patterns_.size(); ++t) {
        for (int s = 0; s < 8; ++s) {
            const auto& sh = shifts_[t * 8 + size_t(s)];
            size_t idx = 0;
            for (size_t j = 0; j < sh.size(); ++j) idx |= size_t((b >> sh[j]) & 0xF) << (4 * j);
            f((base + offset_[t] + idx) * size_t(stride_));
        }
    }
}

float NTupleNet::value(Board b) const {
    float v = 0;
    for_each_feature(b, [&](size_t i) { v += data_[i]; });
    return v;
}

void NTupleNet::update(Board b, float delta, float step) {
    if (stride_ == 1) {
        float d = step * delta;
        for_each_feature(b, [&](size_t i) { data_[i] += d; });
        return;
    }
    float ad = std::fabs(delta);
    for_each_feature(b, [&](size_t i) {
        float& e = data_[i + 1];
        float& a = data_[i + 2];
        float coherence = a > 0 ? std::fabs(e) / a : 1.0f;
        data_[i] += step * coherence * delta;
        e += delta;
        a += ad;
    });
}

void NTupleNet::promote_if_new(int k) {
    if (k == 0 || seen_[k].load(std::memory_order_acquire)) return;
    std::lock_guard<std::mutex> lock(promote_mu_);
    if (seen_[k]) return;
    promote_if_new(k - 1);
    size_t n = weights_per_stage_ * size_t(stride_);
    std::copy(data_.begin() + long(n) * (k - 1), data_.begin() + long(n) * k, data_.begin() + long(n) * k);
    seen_[k].store(true, std::memory_order_release);
}

void NTupleNet::copy_from(const NTupleNet& o) {
    if (o.patterns_ != patterns_ || o.stage_conds_ != stage_conds_ || o.stride_ != stride_)
        throw std::runtime_error("copy_from: network shapes differ");
    data_ = o.data_;
    for (size_t k = 0; k < stage_conds_.size(); ++k) seen_[k] = o.seen_[k].load();
}

void NTupleNet::save(const std::string& path, const std::string& extra) const {
    std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary);
        if (!f) throw std::runtime_error("cannot write " + tmp);
        f << "K2048NT1\n"
          << "patterns=" << join(patterns_, ';', ',') << "\n"
          << "stages=" << join(stage_conds_, ';', ',') << "\n"
          << "tc=" << (tc() ? 1 : 0) << "\n"
          << "seen=";
        for (size_t k = 0; k < stage_conds_.size(); ++k) f << (seen_[k] ? '1' : '0');
        f << "\n" << extra << "end\n";
        f.write(reinterpret_cast<const char*>(data_.data()), std::streamsize(data_.size() * sizeof(float)));
        if (!f) throw std::runtime_error("write failed: " + tmp);
    }
    std::filesystem::rename(tmp, path);  // atomic: a crash never leaves a half-written checkpoint
}

std::unique_ptr<NTupleNet> NTupleNet::load(const std::string& path, std::string* extra) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::string line, patterns, stages, seen;
    bool tc = false;
    std::getline(f, line);
    if (line != "K2048NT1") throw std::runtime_error("not a network file: " + path);
    while (std::getline(f, line) && line != "end") {
        if (line.rfind("patterns=", 0) == 0) patterns = line.substr(9);
        else if (line.rfind("stages=", 0) == 0) stages = line.substr(7);
        else if (line.rfind("tc=", 0) == 0) tc = line[3] == '1';
        else if (line.rfind("seen=", 0) == 0) seen = line.substr(5);
        else if (extra) *extra += line + "\n";
    }
    auto net = std::make_unique<NTupleNet>(split(patterns, ';', ','), split(stages, ';', ','), tc);
    f.read(reinterpret_cast<char*>(net->data_.data()), std::streamsize(net->data_.size() * sizeof(float)));
    if (!f) throw std::runtime_error("truncated network file: " + path);
    for (size_t k = 0; k < seen.size() && k < net->stage_conds_.size(); ++k) net->seen_[k] = seen[k] == '1';
    return net;
}

}  // namespace k2048
