#include "stratakv/generation/generator.h"

#include <charconv>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
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

void print_usage(const char* executable) {
    std::cerr << "Usage: " << executable
              << " <model.safetensors> <max_new_tokens> <prompt_id>...\n";
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        print_usage(argc > 0 ? argv[0] : "stratakv_cli");
        return 2;
    }

    try {
        const int max_new_tokens = parse_int(argv[2], "max_new_tokens");
        if (max_new_tokens < 0) {
            throw std::invalid_argument("max_new_tokens must be nonnegative");
        }

        std::vector<int> prompt;
        prompt.reserve(static_cast<std::size_t>(argc - 3));
        for (int index = 3; index < argc; ++index) {
            const int token = parse_int(argv[index], "prompt token");
            if (token < 0) {
                throw std::invalid_argument("prompt token must be nonnegative");
            }
            prompt.push_back(token);
        }

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

        auto weights = stratakv::io::load_safetensors(
            std::filesystem::path(argv[1]));
        stratakv::generation::Generator generator(config, std::move(weights));
        const std::vector<int> output =
            generator.generate(prompt, max_new_tokens);

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
