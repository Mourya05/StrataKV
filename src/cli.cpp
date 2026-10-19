#include "stratakv/generation/generator.h"

#include <charconv>
#include <array>
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <chrono>
#include <system_error>
#include <utility>
#include <vector>

namespace {

using WeightMap = stratakv::io::WeightMap;
using Tensor = stratakv::Tensor;

const Tensor& find_weight(const WeightMap& weights,
                          std::initializer_list<const char*> names) {
    for (const char* name : names) {
        const auto it = weights.find(name);
        if (it != weights.end())
            return it->second;
    }
    throw std::invalid_argument("checkpoint is missing a required architecture weight");
}

bool has_prefix_and_suffix(const std::string& value,
                           const std::string& prefix,
                           const std::string& suffix,
                           std::size_t& index) {
    if (value.rfind(prefix, 0) != 0 ||
        value.size() <= prefix.size() + suffix.size() ||
        value.compare(value.size() - suffix.size(), suffix.size(), suffix) != 0) {
        return false;
    }
    const char* first = value.data() + prefix.size();
    const char* last = value.data() + value.size() - suffix.size();
    unsigned long long parsed = 0;
    const auto result = std::from_chars(first, last, parsed, 10);
    if (result.ec != std::errc{} || result.ptr != last ||
        parsed > (std::numeric_limits<std::size_t>::max)()) {
        return false;
    }
    index = static_cast<std::size_t>(parsed);
    return true;
}

std::size_t infer_head_dim(std::size_t hidden_dim, std::size_t kv_dim) {
    // Safetensors stores q/k/v as two-dimensional matrices, so the number of
    // heads is not independently encoded: hidden_dim = heads * head_dim.  Use
    // the standard Llama head-size candidates, preferring 64 (the size used by
    // TinyLlama), and derive both head counts from the actual matrix shapes.
    constexpr std::array<std::size_t, 10> candidates = {
        64, 128, 80, 96, 32, 16, 8, 4, 2, 1};
    for (const std::size_t candidate : candidates) {
        if (hidden_dim % candidate == 0 && kv_dim % candidate == 0)
            return candidate;
    }
    throw std::invalid_argument(
        "checkpoint attention shapes do not expose a supported head dimension");
}

struct ModelDimensions {
    std::size_t hidden_dim = 0;
    std::size_t intermediate_dim = 0;
    std::size_t num_heads = 0;
    std::size_t num_kv_heads = 0;
    std::size_t num_layers = 0;
    std::size_t vocab_size = 0;
};

ModelDimensions infer_model_dimensions(const WeightMap& weights) {
    const Tensor& embeddings = find_weight(
        weights, {"model.embed_tokens.weight", "tok_embeddings.weight"});
    if (embeddings.ndim() != 2 || embeddings.shape()[0] == 0 ||
        embeddings.shape()[1] == 0) {
        throw std::invalid_argument(
            "checkpoint embedding weight must have shape [vocab, hidden]");
    }

    const bool llama_schema = weights.find(
        "model.layers.0.input_layernorm.weight") != weights.end();
    ModelDimensions dimensions;
    dimensions.vocab_size = embeddings.shape()[0];
    dimensions.hidden_dim = embeddings.shape()[1];

    if (llama_schema) {
        std::size_t highest_layer = 0;
        bool found_layer = false;
        for (const auto& entry : weights) {
            std::size_t layer = 0;
            if (has_prefix_and_suffix(entry.first, "model.layers.",
                                      ".input_layernorm.weight", layer)) {
                highest_layer = (std::max)(highest_layer, layer);
                found_layer = true;
            }
        }
        if (!found_layer)
            throw std::invalid_argument("checkpoint has no transformer layers");
        dimensions.num_layers = highest_layer + 1;
    } else {
        dimensions.num_layers = 1;
    }

    const Tensor& gate = llama_schema
        ? find_weight(weights, {"model.layers.0.mlp.gate_proj.weight"})
        : find_weight(weights, {"block.ffn.up_proj.weight"});
    const Tensor& query = llama_schema
        ? find_weight(weights, {"model.layers.0.self_attn.q_proj.weight"})
        : find_weight(weights, {"block.attn.q_proj.weight"});
    const Tensor& key = llama_schema
        ? find_weight(weights, {"model.layers.0.self_attn.k_proj.weight"})
        : find_weight(weights, {"block.attn.k_proj.weight"});

    if (gate.ndim() != 2 || gate.shape()[1] != dimensions.hidden_dim ||
        gate.shape()[0] == 0 || query.ndim() != 2 || key.ndim() != 2 ||
        query.shape()[1] != dimensions.hidden_dim ||
        query.shape()[0] != dimensions.hidden_dim ||
        key.shape()[1] != dimensions.hidden_dim || key.shape()[0] == 0) {
        throw std::invalid_argument(
            "checkpoint projection shapes are inconsistent with embeddings");
    }
    dimensions.intermediate_dim = gate.shape()[0];
    const std::size_t head_dim =
        infer_head_dim(dimensions.hidden_dim, key.shape()[0]);
    dimensions.num_heads = query.shape()[0] / head_dim;
    dimensions.num_kv_heads = key.shape()[0] / head_dim;
    if (dimensions.num_heads == 0 || dimensions.num_kv_heads == 0 ||
        dimensions.num_heads % dimensions.num_kv_heads != 0) {
        throw std::invalid_argument(
            "checkpoint attention projections have incompatible head counts");
    }
    return dimensions;
}

int parse_int(const char* text, const char* argument_name) {
    if (text == nullptr || *text == '\0') {
        throw std::invalid_argument(std::string("empty ") + argument_name);
    }

    const char* first = text;
    const char* last = text + std::char_traits<char>::length(text);
    int value = 0;
    const auto parsed = std::from_chars(first, last, value, 10);
    if (parsed.ec == std::errc::result_out_of_range) {
        throw std::out_of_range(std::string(argument_name) + " is outside int range");
    }
    if (parsed.ec != std::errc{} || parsed.ptr != last) {
        throw std::invalid_argument(std::string("invalid ") + argument_name +
                                    ": " + text);
    }
    return value;
}

float parse_float(const char* text, const char* argument_name) {
    if (text == nullptr || *text == '\0') {
        throw std::invalid_argument(std::string("empty ") + argument_name);
    }
    char* parsed_end = nullptr;
    errno = 0;
    const float value = std::strtof(text, &parsed_end);
    if (errno == ERANGE || parsed_end == text || *parsed_end != '\0' ||
        !std::isfinite(value)) {
        throw std::invalid_argument(std::string("invalid ") + argument_name +
                                    ": " + text);
    }
    return value;
}

void print_usage(const char* executable) {
    std::cerr << "Usage: " << executable
              << " [--temp T] [--top-k K] [--top-p P]"
                 " <model.safetensors> <max_new_tokens> <prompt_id>...\n"
              << "       options may also appear between max_new_tokens and"
                 " the prompt tokens\n";
}

struct ParsedArguments {
    std::string model_path;
    int max_new_tokens = 0;
    std::vector<int> prompt;
    stratakv::generation::SamplerConfig sampler{0.0f, 0, 1.0f};
};

ParsedArguments parse_arguments(int argc, char** argv) {
    std::vector<std::string> positional;
    positional.reserve(argc > 1 ? static_cast<std::size_t>(argc - 1) : 0u);
    ParsedArguments result;

    for (int index = 1; index < argc;) {
        const std::string argument = argv[index];
        if (argument == "--temp") {
            if (++index >= argc)
                throw std::invalid_argument("--temp requires a value");
            result.sampler.temperature = parse_float(argv[index], "--temp");
            ++index;
            continue;
        }
        if (argument == "--top-k") {
            if (++index >= argc)
                throw std::invalid_argument("--top-k requires a value");
            result.sampler.top_k = parse_int(argv[index], "--top-k");
            ++index;
            continue;
        }
        if (argument == "--top-p") {
            if (++index >= argc)
                throw std::invalid_argument("--top-p requires a value");
            result.sampler.top_p = parse_float(argv[index], "--top-p");
            ++index;
            continue;
        }
        if (argument.rfind("--", 0) == 0)
            throw std::invalid_argument("unknown option: " + argument);
        positional.push_back(argument);
        ++index;
    }

    if (positional.size() < 3)
        throw std::invalid_argument(
            "expected a model path, max_new_tokens, and at least one prompt token");

    result.model_path = positional[0];
    result.max_new_tokens = parse_int(positional[1].c_str(), "max_new_tokens");
    if (result.max_new_tokens < 0)
        throw std::invalid_argument("max_new_tokens must be nonnegative");

    result.prompt.reserve(positional.size() - 2);
    for (std::size_t index = 2; index < positional.size(); ++index) {
        const int token = parse_int(positional[index].c_str(), "prompt token");
        if (token < 0)
            throw std::invalid_argument("prompt token must be nonnegative");
        result.prompt.push_back(token);
    }
    return result;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argc > 0 ? argv[0] : "stratakv_cli");
        return 2;
    }

    try {
        const ParsedArguments arguments = parse_arguments(argc, argv);

        const auto load_start = std::chrono::steady_clock::now();
        auto weights = stratakv::io::load_safetensors(
            std::filesystem::path(arguments.model_path));
        const ModelDimensions dimensions = infer_model_dimensions(weights);

        stratakv::generation::GeneratorConfig config;
#if defined(STRATAKV_HAS_CUDA)
        // Safetensors remains host-resident as loader staging; Generator moves
        // its persistent weights and decode workspaces to the GPU explicitly.
        config.device = stratakv::Device::GPU;
#else
        config.device = stratakv::Device::CPU;
#endif
        config.hidden_dim = dimensions.hidden_dim;
        config.intermediate_dim = dimensions.intermediate_dim;
        config.num_heads = dimensions.num_heads;
        config.num_kv_heads = dimensions.num_kv_heads;
        config.num_layers = dimensions.num_layers;
        config.total_blocks = 1024;
        config.block_size = 16;
        config.vocab_size = dimensions.vocab_size;
        config.eos_token_id = 2;
        config.sampler = arguments.sampler;

        stratakv::generation::Generator generator(config, std::move(weights));
        const auto load_end = std::chrono::steady_clock::now();
        const auto gen_start = load_end;
        const std::vector<int> output =
            generator.generate(arguments.prompt, arguments.max_new_tokens);
        const auto gen_end = std::chrono::steady_clock::now();
        const double load_ms = std::chrono::duration<double, std::milli>(
            load_end - load_start).count();
        const double measured_gen_ms = std::chrono::duration<double, std::milli>(
            gen_end - gen_start).count();
        const double gen_ms = generator.last_generation_ms() > 0.0
            ? generator.last_generation_ms() : measured_gen_ms;
        const std::size_t generated_tokens =
            output.size() > arguments.prompt.size()
                ? output.size() - arguments.prompt.size() : 0;
        const double pure_decode_tps =
            gen_ms > 0.0 ? static_cast<double>(generated_tokens) /
                (gen_ms / 1000.0) : 0.0;
        const double load_time = load_ms;
        const double gen_time = gen_ms;
        const int tokens_generated = static_cast<int>(generated_tokens);
        const double tps = pure_decode_tps;
        fprintf(stderr, "[TIMING] load_ms = %f, gen_ms = %f, tokens = %d, pure_decode_tps = %f\n", load_time, gen_time, tokens_generated, tps);
        const auto load_breakdown = stratakv::io::last_load_breakdown();
        std::cerr << "[LOAD_BREAKDOWN] mmap_open_ms = "
                  << load_breakdown.mmap_open_ms
                  << ", header_parse_ms = " << load_breakdown.header_parse_ms
                  << ", tensor_decode_ms = " << load_breakdown.tensor_decode_ms
                  << ", generator_init_ms = "
                  << generator.initialization_ms() << '\n';
        for (std::size_t index = 0; index < output.size(); ++index) {
            if (index != 0) std::cout << ' ';
            std::cout << output[index];
        }
        std::cout << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "stratakv_cli: " << error.what() << '\n';
        return 1;
    }
}
