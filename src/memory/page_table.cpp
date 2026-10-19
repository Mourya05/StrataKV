// =============================================================================
//  page_table.cpp  --  StrataKV Phase 3: Sequence → Physical Block Mapper (impl)
// =============================================================================

#include "stratakv/memory/page_table.h"

#if defined(STRATAKV_HAS_CUDA)
#include <cuda_runtime_api.h>
#endif

#include <stdexcept>
#include <string>
#include <limits>
#include <utility>

namespace stratakv {
namespace memory  {

namespace {
#if defined(STRATAKV_HAS_CUDA)
void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("PageTable: ") + operation +
                                 " failed: " + cudaGetErrorString(status));
}
#endif
} // namespace

// ---------------------------------------------------------------------------
//  helpers
// ---------------------------------------------------------------------------
static SequenceState& checked_lookup(std::unordered_map<int, SequenceState>& table,
                                     int seq_id,
                                     const char* caller) {
    auto it = table.find(seq_id);
    if (it == table.end())
        throw std::runtime_error(std::string(caller) +
                                 ": unknown seq_id=" + std::to_string(seq_id));
    return it->second;
}

PageTable::PageTable(const PageTable& other)
    : table_(other.table_)
{
}

PageTable& PageTable::operator=(const PageTable& other) {
    if (this != &other) {
        release_device_block_table();
        table_ = other.table_;
        staged_seq_id_ = 0;
        staged_block_count_ = 0;
        device_table_dirty_ = true;
    }
    return *this;
}

PageTable::PageTable(PageTable&& other) noexcept
    : table_(std::move(other.table_))
    , d_block_table_(other.d_block_table_)
    , d_block_table_capacity_(other.d_block_table_capacity_)
    , staged_seq_id_(other.staged_seq_id_)
    , staged_block_count_(other.staged_block_count_)
    , device_table_dirty_(other.device_table_dirty_)
{
    other.d_block_table_ = nullptr;
    other.d_block_table_capacity_ = 0;
    other.staged_block_count_ = 0;
    other.device_table_dirty_ = true;
}

PageTable& PageTable::operator=(PageTable&& other) noexcept {
    if (this != &other) {
        release_device_block_table();
        table_ = std::move(other.table_);
        d_block_table_ = other.d_block_table_;
        d_block_table_capacity_ = other.d_block_table_capacity_;
        staged_seq_id_ = other.staged_seq_id_;
        staged_block_count_ = other.staged_block_count_;
        device_table_dirty_ = other.device_table_dirty_;
        other.d_block_table_ = nullptr;
        other.d_block_table_capacity_ = 0;
        other.staged_block_count_ = 0;
        other.device_table_dirty_ = true;
    }
    return *this;
}

PageTable::~PageTable() noexcept {
    release_device_block_table();
}

void PageTable::release_device_block_table() noexcept {
#if defined(STRATAKV_HAS_CUDA)
    if (d_block_table_ != nullptr)
        (void)cudaFree(d_block_table_);
#endif
    d_block_table_ = nullptr;
    d_block_table_capacity_ = 0;
    staged_block_count_ = 0;
    device_table_dirty_ = true;
}

// ---------------------------------------------------------------------------
//  allocate_sequence
// ---------------------------------------------------------------------------
void PageTable::allocate_sequence(int seq_id, BlockPool& pool) {
    if (table_.count(seq_id))
        throw std::runtime_error("allocate_sequence: seq_id=" +
                                 std::to_string(seq_id) + " already exists");

    SequenceState state;
    state.sequence_length = 0;
    state.physical_blocks.push_back(pool.allocate_block());  // first block
    table_.emplace(seq_id, std::move(state));
    device_table_dirty_ = true;
}

// ---------------------------------------------------------------------------
//  append_token
//  Invariant: after N tokens, the sequence needs ceil(N / block_size) blocks.
//  We allocate the (N+1)-th token's block lazily: if the previous token was
//  the last slot in its block (sequence_length % block_size == 0 after the
//  increment), a new block is needed.
// ---------------------------------------------------------------------------
void PageTable::append_token(int seq_id, BlockPool& pool) {
    SequenceState& state = checked_lookup(table_, seq_id, "append_token");

    if (state.sequence_length == std::numeric_limits<int>::max()) {
        throw std::overflow_error(
            "append_token: sequence length exceeds int range");
    }

    const std::size_t allocated_capacity =
        state.physical_blocks.size() *
        static_cast<std::size_t>(pool.block_size());

    // Allocate before writing a token only when all assigned pages are full.
    // Thus N tokens require ceil(N / block_size) pages.
    if (static_cast<std::size_t>(state.sequence_length) >=
        allocated_capacity) {
        state.physical_blocks.push_back(pool.allocate_block());
    }

    ++state.sequence_length;
    device_table_dirty_ = true;
}

// ---------------------------------------------------------------------------
//  free_sequence
// ---------------------------------------------------------------------------
void PageTable::free_sequence(int seq_id, BlockPool& pool) {
    SequenceState& state = checked_lookup(table_, seq_id, "free_sequence");

    for (int blk_id : state.physical_blocks)
        pool.free_block(blk_id);

    table_.erase(seq_id);
    device_table_dirty_ = true;
}

// ---------------------------------------------------------------------------
//  get_state
// ---------------------------------------------------------------------------
const SequenceState& PageTable::get_state(int seq_id) const {
    auto it = table_.find(seq_id);
    if (it == table_.end())
        throw std::runtime_error("get_state: unknown seq_id=" +
                                 std::to_string(seq_id));
    return it->second;
}

// ---------------------------------------------------------------------------
//  has_sequence
// ---------------------------------------------------------------------------
bool PageTable::has_sequence(int seq_id) const noexcept {
    return table_.count(seq_id) != 0;
}

const int* PageTable::get_device_block_table(int seq_id) {
    const SequenceState& state = get_state(seq_id);
#if defined(STRATAKV_HAS_CUDA)
    const std::size_t required = state.physical_blocks.size();
    if (required == 0)
        throw std::runtime_error(
            "get_device_block_table: sequence has no physical blocks");
    const std::size_t bytes = required * sizeof(int);
    if (required > d_block_table_capacity_) {
        int* replacement = nullptr;
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&replacement),
                              bytes),
                   "cudaMalloc(block_table)");
        if (d_block_table_ != nullptr)
            (void)cudaFree(d_block_table_);
        d_block_table_ = replacement;
        d_block_table_capacity_ = required;
        device_table_dirty_ = true;
    }
    if (device_table_dirty_ || staged_seq_id_ != seq_id ||
        staged_block_count_ != required) {
        check_cuda(cudaMemcpy(d_block_table_, state.physical_blocks.data(),
                              bytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy(block_table)");
        staged_seq_id_ = seq_id;
        staged_block_count_ = required;
        device_table_dirty_ = false;
    }
    return d_block_table_;
#else
    (void)state;
    throw std::runtime_error(
        "get_device_block_table: CUDA support is not enabled");
#endif
}

void PageTable::reserve_device_block_table(std::size_t capacity) {
#if defined(STRATAKV_HAS_CUDA)
    if (capacity <= d_block_table_capacity_)
        return;
    if (capacity > (std::numeric_limits<std::size_t>::max)() /
            sizeof(int)) {
        throw std::overflow_error(
            "reserve_device_block_table: capacity is too large");
    }

    int* replacement = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&replacement),
                          capacity * sizeof(int)),
               "cudaMalloc(block_table reserve)");
    if (d_block_table_ != nullptr)
        (void)cudaFree(d_block_table_);
    d_block_table_ = replacement;
    d_block_table_capacity_ = capacity;
    device_table_dirty_ = true;
#else
    (void)capacity;
#endif
}

} // namespace memory
} // namespace stratakv
