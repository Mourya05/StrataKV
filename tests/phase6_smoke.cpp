#include "stratakv/generation/generator.h"
#include "stratakv/layers/attention.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

stratakv::Tensor tensor(std::initializer_list<std::size_t> shape,
                        float value = 0.0f) {
    stratakv::Tensor result{std::vector<std::size_t>(shape)};
    for (std::size_t i = 0; i < result.num_elements(); ++i) result[i] = value;
    return result;
}

void add(stratakv::io::WeightMap& weights,
         const std::string& key,
         std::initializer_list<std::size_t> shape,
         float value = 0.0f) {
    weights.emplace(key, tensor(shape, value));
}

void add_identity(stratakv::io::WeightMap& weights,
                  const std::string& key,
                  std::size_t dimension) {
    stratakv::Tensor value({dimension, dimension});
    for (std::size_t row = 0; row < dimension; ++row)
        value[row * dimension + row] = 1.0f;
    weights.emplace(key, std::move(value));
}

void add_rectangular_identity(stratakv::io::WeightMap& weights,
                              const std::string& key,
                              std::size_t output,
                              std::size_t input) {
    stratakv::Tensor value({output, input});
    for (std::size_t index = 0; index < (std::min)(output, input); ++index)
        value[index * input + index] = 1.0f;
    weights.emplace(key, std::move(value));
}

void set_identity(stratakv::layers::Linear& layer) {
    check(layer.input_dim() == layer.output_dim(),
          "test identity requires a square layer");
    const std::size_t dimension = layer.input_dim();
    for (std::size_t row = 0; row < dimension; ++row)
        for (std::size_t col = 0; col < dimension; ++col)
            layer.weight()[row * dimension + col] = row == col ? 1.0f : 0.0f;
}

void check_rope() {
    constexpr std::size_t hidden = 4;
    stratakv::layers::PagedAttention attention(hidden, 2, 2, 2);
    set_identity(attention.query_projection());
    set_identity(attention.key_projection());
    set_identity(attention.value_projection());
    set_identity(attention.output_projection());

    stratakv::memory::BlockPool pool(2, 2);
    stratakv::memory::PageTable pages;
    pages.allocate_sequence(0, pool);

    stratakv::Tensor first({1, hidden});
    first[0] = 1.0f;
    first[2] = 1.0f;
    const stratakv::Tensor first_output =
        attention.forward(first, 0, pages, pool);
    check(std::fabs(first_output[0] - 1.0f) < 1.0e-5f,
          "RoPE first token output mismatch");

    stratakv::Tensor second({1, hidden});
    second[1] = 1.0f;
    second[3] = 1.0f;
    const stratakv::Tensor second_output =
        attention.forward(second, 0, pages, pool);
    const float sine = std::sin(1.0f);
    const float negative_score = -sine / std::sqrt(2.0f);
    const float current_score = 1.0f / std::sqrt(2.0f);
    const float current_weight = std::exp(current_score) /
        (std::exp(negative_score) + std::exp(current_score));
    check(std::fabs(second_output[0] - (1.0f - current_weight)) < 1.0e-5f,
          "RoPE rotation was not applied to the key");
    check(std::fabs(second_output[1] - current_weight) < 1.0e-5f,
          "RoPE rotation was not applied to the query");
    check(pages.get_state(0).sequence_length == 2,
          "RoPE test did not append exactly once");
}

void check_gqa_attention() {
    constexpr std::size_t hidden = 4;
    stratakv::layers::PagedAttention attention(hidden, 2, 1, 2, 2);
    check(attention.key_projection().output_dim() == 2,
          "GQA key projection dimension mismatch");
    check(attention.value_projection().output_dim() == 2,
          "GQA value projection dimension mismatch");
    set_identity(attention.query_projection());
    set_identity(attention.output_projection());
    for (std::size_t input = 0; input < hidden; ++input) {
        for (std::size_t output = 0; output < 2; ++output) {
            const float value = input == output ? 1.0f : 0.0f;
            attention.key_projection().weight()[input * 2 + output] = value;
            attention.value_projection().weight()[input * 2 + output] = value;
        }
    }

    stratakv::memory::BlockPool pool(2, 2);
    stratakv::memory::PageTable pages;
    pages.allocate_sequence(0, pool);
    stratakv::Tensor input({2, hidden});
    input[0] = 1.0f;
    input[5] = 1.0f;
    const stratakv::Tensor output = attention.forward(input, 0, pages, pool);
    check(output.shape() == std::vector<std::size_t>({2, hidden}),
          "GQA attention output shape mismatch");
    check(pages.get_state(0).sequence_length == 2,
          "GQA attention appended an incorrect number of tokens");
}

void check_swiglu() {
    constexpr std::size_t hidden = 4;
    stratakv::models::TransformerBlock block(hidden, hidden, 2, 2, 2);
    set_identity(block.gate_projection());
    set_identity(block.up_projection());
    set_identity(block.down_projection());

    stratakv::memory::BlockPool pool(2, 2);
    stratakv::memory::PageTable pages;
    pages.allocate_sequence(0, pool);
    stratakv::Tensor input({1, hidden});
    for (std::size_t i = 0; i < hidden; ++i) input[i] = 1.0f;
    const stratakv::Tensor output = block.forward(input, 0, pages, pool);

    const float normalized = 1.0f / std::sqrt(1.0f + 1.0e-5f);
    const float expected = 1.0f + normalized *
        (1.0f / (1.0f + std::exp(-normalized))) * normalized;
    check(std::fabs(output[0] - expected) < 1.0e-5f,
          "SwiGLU activation or projection mismatch");
}

stratakv::io::WeightMap make_llama_weights() {
    constexpr std::size_t hidden = 4;
    constexpr std::size_t intermediate = 4;
    constexpr std::size_t vocab = 3;
    constexpr std::size_t layers = 2;
    constexpr std::size_t kv_heads = 1;
    constexpr std::size_t kv_dim = hidden / 2 * kv_heads;
    stratakv::io::WeightMap weights;
    add(weights, "model.embed_tokens.weight", {vocab, hidden}, 0.0f);
    weights.at("model.embed_tokens.weight")[0] = 1.0f;
    add(weights, "model.norm.weight", {hidden}, 1.0f);
    add(weights, "lm_head.weight", {vocab, hidden}, 0.0f);
    for (std::size_t layer = 0; layer < layers; ++layer) {
        const std::string prefix = "model.layers." +
            std::to_string(layer) + ".";
        add(weights, prefix + "input_layernorm.weight", {hidden}, 1.0f);
        add(weights, prefix + "post_attention_layernorm.weight", {hidden}, 1.0f);
        add_identity(weights, prefix + "self_attn.q_proj.weight", hidden);
        add_rectangular_identity(weights, prefix + "self_attn.k_proj.weight",
                                 kv_dim, hidden);
        add_rectangular_identity(weights, prefix + "self_attn.v_proj.weight",
                                 kv_dim, hidden);
        add_identity(weights, prefix + "self_attn.o_proj.weight", hidden);
        add(weights, prefix + "mlp.gate_proj.weight",
            {intermediate, hidden}, 0.0f);
        add(weights, prefix + "mlp.up_proj.weight",
            {intermediate, hidden}, 0.0f);
        add(weights, prefix + "mlp.down_proj.weight",
            {hidden, intermediate}, 0.0f);
    }
    return weights;
}

void check_multi_layer_generator() {
    stratakv::generation::GeneratorConfig config;
    config.hidden_dim = 4;
    config.intermediate_dim = 4;
    config.num_heads = 2;
    config.num_kv_heads = 1;
    config.num_layers = 2;
    config.total_blocks = 4;
    config.block_size = 2;
    config.vocab_size = 3;
    config.eos_token_id = -1;

    stratakv::generation::Generator engine(
        config, make_llama_weights());
    const std::vector<int> result = engine.generate({0}, 2);
    check(result == std::vector<int>({0, 0, 0}),
          "multi-layer Llama generation mismatch");
    check(engine.free_block_count() == 4,
          "multi-layer generation did not reclaim KV pages");
}

} // namespace

int main() {
    try {
        check_rope();
        check_gqa_attention();
        check_swiglu();
        check_multi_layer_generator();
        std::cout << "Phase 6: smoke checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Phase 6 failure: " << error.what() << '\n';
        return 1;
    }
}
