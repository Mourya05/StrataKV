// =============================================================================
//  block_pool.cpp  --  StrataKV Phase 3: Physical Block Allocator (impl)
// =============================================================================

#include "stratakv/memory/block_pool.h"

#if defined(STRATAKV_HAS_CUDA)
#include <cuda_runtime_api.h>
#endif

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace stratakv {
namespace memory  {

namespace {

std::size_t checked_element_count(int total, int block_size,
                                  int num_kv_heads, int head_dim) {
    if (total <= 0 || block_size <= 0 || num_kv_heads <= 0 || head_dim <= 0)
        throw std::invalid_argument(
            "BlockPool: GPU storage dimensions must be positive");
    const std::size_t result = static_cast<std::size_t>(total) *
        static_cast<std::size_t>(block_size) *
        static_cast<std::size_t>(num_kv_heads) *
        static_cast<std::size_t>(head_dim);
    if (result > std::numeric_limits<std::size_t>::max() / sizeof(float))
        throw std::overflow_error("BlockPool: KV storage size overflow");
    return result;
}

#if defined(STRATAKV_HAS_CUDA)
void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("BlockPool: ") + operation +
                                 " failed: " + cudaGetErrorString(status));
}
#endif

} // namespace

// ---------------------------------------------------------------------------
//  Constructor
//  Pre-populate the free list with block IDs [0, total-1].
//  Block 0 is pushed last so it sits on top of the stack — it will be the
//  first ID handed out, giving predictable output for the smoke test.
// ---------------------------------------------------------------------------
BlockPool::BlockPool(int total, int blk_sz)
    : BlockPool(total, blk_sz, 0, 0, Device::CPU)
{
}

BlockPool::BlockPool(int total, int blk_sz, int num_kv_heads, int head_dim,
                     Device device)
    : total_blocks_(total)
    , block_size_(blk_sz)
    , num_kv_heads_(num_kv_heads)
    , head_dim_(head_dim)
    , device_(device)
{
    if (total  <= 0) throw std::invalid_argument("BlockPool: total must be > 0");
    if (blk_sz <= 0) throw std::invalid_argument("BlockPool: block_size must be > 0");
    if (device_ == Device::GPU &&
        (num_kv_heads_ <= 0 || head_dim_ <= 0)) {
        throw std::invalid_argument(
            "BlockPool: GPU mode requires positive KV dimensions");
    }

    free_list_.reserve(static_cast<std::size_t>(total));
    // Push in reverse order so that block 0 is on top (back of vector).
    for (int id = total - 1; id >= 0; --id)
        free_list_.push_back(id);
    initialize_storage();
}

BlockPool::BlockPool(int total, int blk_sz, Device device, int num_kv_heads,
                     int head_dim)
    : BlockPool(total, blk_sz, num_kv_heads, head_dim, device)
{
}

void BlockPool::initialize_storage() {
    if (device_ != Device::GPU)
        return;
#if defined(STRATAKV_HAS_CUDA)
    const std::size_t bytes = checked_element_count(
        total_blocks_, block_size_, num_kv_heads_, head_dim_) * sizeof(float);
    try {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&key_cache_), bytes),
                   "cudaMalloc(key_cache)");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&value_cache_), bytes),
                   "cudaMalloc(value_cache)");
    } catch (...) {
        cudaFree(key_cache_);
        key_cache_ = nullptr;
        throw;
    }
#else
    throw std::runtime_error(
        "BlockPool: GPU mode requires a CUDA-enabled build");
#endif
}

void BlockPool::release_storage() noexcept {
#if defined(STRATAKV_HAS_CUDA)
    if (key_cache_ != nullptr)
        (void)cudaFree(key_cache_);
    if (value_cache_ != nullptr)
        (void)cudaFree(value_cache_);
#endif
    key_cache_ = nullptr;
    value_cache_ = nullptr;
}

BlockPool::BlockPool(const BlockPool& other)
    : total_blocks_(other.total_blocks_)
    , block_size_(other.block_size_)
    , num_kv_heads_(other.num_kv_heads_)
    , head_dim_(other.head_dim_)
    , device_(other.device_)
    , free_list_(other.free_list_)
{
    initialize_storage();
#if defined(STRATAKV_HAS_CUDA)
    if (device_ == Device::GPU) {
        const std::size_t bytes = checked_element_count(
            total_blocks_, block_size_, num_kv_heads_, head_dim_) *
            sizeof(float);
        try {
            check_cuda(cudaMemcpy(key_cache_, other.key_cache_, bytes,
                                  cudaMemcpyDeviceToDevice),
                       "cudaMemcpy(key_cache)");
            check_cuda(cudaMemcpy(value_cache_, other.value_cache_, bytes,
                                  cudaMemcpyDeviceToDevice),
                       "cudaMemcpy(value_cache)");
        } catch (...) {
            release_storage();
            throw;
        }
    }
#endif
}

BlockPool& BlockPool::operator=(const BlockPool& other) {
    if (this == &other)
        return *this;
    BlockPool copy(other);
    *this = std::move(copy);
    return *this;
}

BlockPool::BlockPool(BlockPool&& other) noexcept
    : total_blocks_(other.total_blocks_)
    , block_size_(other.block_size_)
    , num_kv_heads_(other.num_kv_heads_)
    , head_dim_(other.head_dim_)
    , device_(other.device_)
    , key_cache_(other.key_cache_)
    , value_cache_(other.value_cache_)
    , free_list_(std::move(other.free_list_))
{
    other.key_cache_ = nullptr;
    other.value_cache_ = nullptr;
}

BlockPool& BlockPool::operator=(BlockPool&& other) noexcept {
    if (this == &other)
        return *this;
    release_storage();
    total_blocks_ = other.total_blocks_;
    block_size_ = other.block_size_;
    num_kv_heads_ = other.num_kv_heads_;
    head_dim_ = other.head_dim_;
    device_ = other.device_;
    key_cache_ = other.key_cache_;
    value_cache_ = other.value_cache_;
    free_list_ = std::move(other.free_list_);
    other.key_cache_ = nullptr;
    other.value_cache_ = nullptr;
    return *this;
}

BlockPool::~BlockPool() noexcept {
    release_storage();
}

// ---------------------------------------------------------------------------
//  allocate_block
// ---------------------------------------------------------------------------
int BlockPool::allocate_block() {
    if (free_list_.empty())
        throw std::runtime_error("OOM");

    const int id = free_list_.back();
    free_list_.pop_back();
    return id;
}

// ---------------------------------------------------------------------------
//  free_block
// ---------------------------------------------------------------------------
void BlockPool::free_block(int block_id) {
    free_list_.push_back(block_id);
}

} // namespace memory
} // namespace stratakv
