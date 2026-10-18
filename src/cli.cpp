#include "stratakv/generation/generator.h"

#include <charconv>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <chrono>
#include <system_error>
#include <utility>
#include <vector>

namespace {

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

        stratakv::generation::GeneratorConfig config;
        config.hidden_dim = 2048;
        config.intermediate_dim = 5632;
        config.num_heads = 32;
        config.num_kv_heads = 4;
        config.num_layers = 22;
        config.total_blocks = 1024;
        config.block_size = 16;
        config.vocab_size = 32000;
        config.eos_token_id = 2;
        config.sampler = arguments.sampler;

        const auto load_start = std::chrono::steady_clock::now();
        auto weights = stratakv::io::load_safetensors(
            std::filesystem::path(arguments.model_path));
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
        std::cerr << "[TIMING] load_ms = " << load_ms
                  << ", gen_ms = " << gen_ms
                  << ", tokens = " << generated_tokens
                  << ", pure_decode_tps = " << pure_decode_tps << '\n';

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
