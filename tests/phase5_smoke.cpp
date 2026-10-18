#include "stratakv/generation/generator.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
int checks = 0;
void check(bool ok) {
    ++checks;
    if (!ok) throw std::runtime_error("check failed: " + std::to_string(checks));
}
template<class F> void rejects(F operation) {
    bool threw = false;
    try { operation(); } catch (const std::exception&) { threw = true; }
    check(threw);
}
std::vector<unsigned char> fp32(const std::vector<float>& values) {
    std::vector<unsigned char> bytes;
    for (float v : values) {
        std::uint32_t bits;
        std::memcpy(&bits, &v, 4);
        for (unsigned i = 0; i < 4; ++i)
            bytes.push_back(static_cast<unsigned char>((bits >> (8 * i)) & 255u));
    }
    return bytes;
}
void write_file(const std::filesystem::path& file, std::string header,
                const std::vector<unsigned char>& payload, bool aligned = true) {
    if (aligned) while ((8 + header.size()) % 64 != 0) header += ' ';
    else if ((8 + header.size()) % 64 == 0) header += ' ';
    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    const auto length = static_cast<std::uint64_t>(header.size());
    for (unsigned i = 0; i < 8; ++i) out.put(static_cast<char>((length >> (8 * i)) & 255u));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    if (!payload.empty())
        out.write(reinterpret_cast<const char*>(payload.data()),
                  static_cast<std::streamsize>(payload.size()));
}

void make_model(const std::filesystem::path& path, bool bf16) {
    std::string header = "{";
    std::vector<unsigned char> payload;
    auto add = [&](const char* name, std::vector<std::size_t> shape,
                   const std::vector<float>& values) {
        if (header.size() > 1) header += ',';
        header += '"' + std::string(name) + "\":{\"dtype\":\"" +
                  (bf16 ? "BF16" : "F32") + "\",\"shape\":[";
        for (std::size_t i = 0; i < shape.size(); ++i) {
            if (i) header += ',';
            header += std::to_string(shape[i]);
        }
        const auto begin = payload.size();
        const auto bytes = fp32(values);
        if (bf16) {
            for (std::size_t i = 0; i < values.size(); ++i) {
                payload.push_back(bytes[4 * i + 2]);
                payload.push_back(bytes[4 * i + 3]);
            }
        } else payload.insert(payload.end(), bytes.begin(), bytes.end());
        header += "],\"data_offsets\":[" + std::to_string(begin) + ',' +
                  std::to_string(payload.size()) + "]}";
    };
    add("tok_embeddings.weight", {3, 2}, {1, 0, 0, 1, -1, 0});
    for (auto name : {"block.input_norm.weight", "block.post_attention_norm.weight", "norm.weight"})
        add(name, {2}, {1, 1});
    for (auto name : {"block.attn.q_proj.weight", "block.attn.k_proj.weight",
                      "block.attn.v_proj.weight", "block.attn.o_proj.weight"})
        add(name, {2, 2}, {1, 0, 0, 1});
    add("block.ffn.up_proj.weight", {4, 2}, std::vector<float>(8, 0));
    add("block.ffn.down_proj.weight", {2, 4}, std::vector<float>(8, 0));
    add("lm_head.weight", {3, 2}, {0, 0, 2, 0, 0, 2});
    write_file(path, header + '}', payload);
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2 || !std::filesystem::is_directory(argv[1]))
            throw std::runtime_error("supply an existing scratch directory");
        const auto root = std::filesystem::path(argv[1]);
        const auto file = root / "stratakv_phase5_tensor.safetensors";
        const std::string header =
            R"({"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})";
        write_file(file, header, fp32({1.5f, -2.0f}));
        {
            auto weights = stratakv::io::load_safetensors(file, true);
            check(weights.size() == 1);
            stratakv::Tensor retained = std::move(weights.at("x"));
            weights.clear();
            check(retained[0] == 1.5f && retained[1] == -2.0f);
            retained[0] = 99; // Private mapping must not modify the disk file.
            check(stratakv::io::load_safetensors(file, true).at("x")[0] == 1.5f);
        }
        write_file(file, header, fp32({1.5f, -2.0f}), false);
        check(stratakv::io::load_safetensors(file).at("x")[1] == -2);
        rejects([&] { (void)stratakv::io::load_safetensors(file, true); });
        write_file(file,
            R"({"x":{"data_offsets":[0,4],"shape":[2],"dtype":"BF16"}})",
            {0x80, 0x3F, 0x20, 0xC0});
        {
            // Native BF16 tensors retain the mmap owner, so release this
            // view before the next case replaces the scratch file on Windows.
            auto bf = stratakv::io::load_safetensors(file);
            check(bf.at("x")[0] == 1 && bf.at("x")[1] == -2.5f);
        }

        write_file(file, R"({"z":{"dtype":"F32","shape":[1],"data_offsets":[4,8]},"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})", fp32({7, 9}));
        {
            auto ordered = stratakv::io::load_safetensors(file);
            check(ordered.at("a")[0] == 7 && ordered.at("z")[0] == 9);
        }
        write_file(file, R"({"\uD83D\uDE00":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})", fp32({7}));
        check(stratakv::io::load_safetensors(file).count("\xF0\x9F\x98\x80") == 1);

        write_file(file, R"({"__metadata__":{"note":"ok"},"\u0078":{"dtype":"F32","shape":[],"data_offsets":[0,4]}})", fp32({3}));
        check(stratakv::io::load_safetensors(file).at("x").ndim() == 0);
        write_file(file, R"({"x":{"dtype":"F32","shape":[0,2],"data_offsets":[0,0]}})", {});
        check(stratakv::io::load_safetensors(file).at("x").num_elements() == 0);

        for (const std::string bad : {
            R"({"x":{"dtype":"F16","shape":[2],"data_offsets":[0,8]}})",
            R"({"x":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}})",
            R"({"x":{"dtype":"F32","dtype":"F32","shape":[2],"data_offsets":[0,8]}})",
            R"({"x":{"dtype":"F32","shape":[02],"data_offsets":[0,8]}})",
            R"({"x":{"dtype":"F32","shape":[-2],"data_offsets":[0,8]}})",
            R"({"x":{"dtype":"F32","shape":[18446744073709551616],"data_offsets":[0,8]}})",
            R"({"x":{"dtype":"F32","shape":[2],"data_offsets":[1,9]}})",
            R"({"x":{"dtype":"F32","shape":[2],"data_offsets":[8,0]}})",
            R"({"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"y":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})",
            R"({"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}} true)",
            R"({"\uD800":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})",
            R"({"__metadata__":{"bad":3},"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})"}) {
            write_file(file, bad, fp32({1, 2}));
            rejects([&] { (void)stratakv::io::load_safetensors(file); });
        }
        { std::ofstream short_file(file, std::ios::binary | std::ios::trunc); short_file << "123"; }
        rejects([&] { (void)stratakv::io::load_safetensors(file); });
        { std::ofstream bad_length(file, std::ios::binary | std::ios::trunc);
          for (unsigned i = 0; i < 8; ++i) bad_length.put(static_cast<char>(255)); }
        rejects([&] { (void)stratakv::io::load_safetensors(file); });
        write_file(file, "{}", fp32({1})); // Payload bytes cannot be unreferenced.
        rejects([&] { (void)stratakv::io::load_safetensors(file); });

        stratakv::generation::Sampler sampler;
        stratakv::Tensor logits({3});
        logits[0] = -2; logits[1] = 4; logits[2] = 4;
        check(sampler.sample(logits) == 1);
        logits[2] = (std::numeric_limits<float>::quiet_NaN)();
        rejects([&] { (void)sampler.sample(logits); });
        stratakv::Tensor empty({0}), batched({2, 3});
        rejects([&] { (void)sampler.sample(empty); });
        rejects([&] { (void)sampler.sample(batched); });

        stratakv::generation::GeneratorConfig cfg{2, 4, 1, 8, 2, 3, 2};
        for (bool use_bf16 : {false, true}) {
            make_model(file, use_bf16);
            stratakv::generation::Generator engine(cfg, stratakv::io::load_safetensors(file));
            for (int repeat = 0; repeat < 2; ++repeat) {
                check(engine.generate({0}, 4) == std::vector<int>({0, 1, 2}));
                check(engine.free_block_count() == 8);
            }
            check(engine.generate({0, 0, 0, 0, 0}, 4) ==
                  std::vector<int>({0, 0, 0, 0, 0, 1, 2}));
            check(engine.free_block_count() == 8);
            check(engine.generate({0}, 1) == std::vector<int>({0, 1}));
            check(engine.generate({0}, 0) == std::vector<int>({0}));
            check(engine.generate({2}, 4) == std::vector<int>({2}));
            rejects([&] { (void)engine.generate({}, 1); });
            rejects([&] { (void)engine.generate({3}, 1); });
            rejects([&] { (void)engine.generate({0}, -1); });
            rejects([&] { (void)engine.generate({0}, 20); });
            check(engine.free_block_count() == 8);
        }
        make_model(file, false);
        auto missing = stratakv::io::load_safetensors(file);
        missing.erase("lm_head.weight");
        rejects([&] { stratakv::generation::Generator bad(cfg, std::move(missing)); });
        std::cout << "Phase 5: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Phase 5 failure: " << e.what() << '\n';
        return 1;
    }
}
