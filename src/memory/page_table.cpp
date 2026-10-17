// =============================================================================
//  page_table.cpp  --  StrataKV Phase 3: Sequence → Physical Block Mapper (impl)
// =============================================================================

#include "stratakv/memory/page_table.h"

#include <stdexcept>
#include <string>

namespace stratakv {
namespace memory  {

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

    state.sequence_length += 1;

    // If we've just filled a full block, allocate the next one.
    if (state.sequence_length % pool.block_size() == 0)
        state.physical_blocks.push_back(pool.allocate_block());
}

// ---------------------------------------------------------------------------
//  free_sequence
// ---------------------------------------------------------------------------
void PageTable::free_sequence(int seq_id, BlockPool& pool) {
    SequenceState& state = checked_lookup(table_, seq_id, "free_sequence");

    for (int blk_id : state.physical_blocks)
        pool.free_block(blk_id);

    table_.erase(seq_id);
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

} // namespace memory
} // namespace stratakv
