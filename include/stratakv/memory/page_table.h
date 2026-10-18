// =============================================================================
//  page_table.h  --  StrataKV Phase 3: Sequence → Physical Block Mapper
//
//  Implements the "page table" layer of the virtual memory analogy:
//    logical token index  →  (physical block ID, slot within block)
//
//  Each sequence is tracked by a SequenceState that stores:
//    • sequence_length  : how many tokens have been appended so far
//    • physical_blocks  : ordered list of block IDs assigned to this sequence
//
//  A new block is acquired from the BlockPool exactly when the current token
//  would overflow the last allocated block (i.e. on every block_size-th
//  append after the first).
//
//  Methods
//  -------
//   allocate_sequence(seq_id, pool) : register seq_id, allocate first block.
//   append_token     (seq_id, pool) : advance length; allocate new block if
//                                      the current block boundary is crossed.
//   free_sequence    (seq_id, pool) : return all blocks to pool, erase entry.
//   get_state        (seq_id)       : read-only view of a SequenceState.
// =============================================================================

#pragma once

#include "stratakv/memory/block_pool.h"

#include <cstddef>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace stratakv {
namespace memory  {

// ---------------------------------------------------------------------------
//  SequenceState
//  Plain data holder: the page-table entry for one live sequence.
// ---------------------------------------------------------------------------
struct SequenceState {
    int              sequence_length = 0;  // tokens appended so far
    std::vector<int> physical_blocks;      // ordered list of block IDs
};

// ---------------------------------------------------------------------------
//  PageTable
// ---------------------------------------------------------------------------
class PageTable {
public:
    PageTable() = default;
    PageTable(const PageTable& other);
    PageTable& operator=(const PageTable& other);
    PageTable(PageTable&& other) noexcept;
    PageTable& operator=(PageTable&& other) noexcept;
    ~PageTable() noexcept;

    // -------------------------------------------------------------------------
    //  allocate_sequence
    //  Register a new sequence with seq_id and allocate its first block from
    //  pool. Throws std::runtime_error if seq_id is already registered.
    // -------------------------------------------------------------------------
    void allocate_sequence(int seq_id, BlockPool& pool);

    // -------------------------------------------------------------------------
    //  append_token
    //  Advance the sequence length by one. If that token is the first token
    //  of a new block boundary (sequence_length % block_size == 0) a new
    //  block is allocated from pool and appended to the block list.
    //  Throws std::runtime_error if seq_id is not registered.
    // -------------------------------------------------------------------------
    void append_token(int seq_id, BlockPool& pool);

    // -------------------------------------------------------------------------
    //  free_sequence
    //  Return every physical block owned by seq_id back to pool, then remove
    //  the entry. Throws std::runtime_error if seq_id is not registered.
    // -------------------------------------------------------------------------
    void free_sequence(int seq_id, BlockPool& pool);

    // -------------------------------------------------------------------------
    //  get_state  --  read-only view; throws if seq_id is not found.
    // -------------------------------------------------------------------------
    const SequenceState& get_state(int seq_id) const;

    // -------------------------------------------------------------------------
    //  has_sequence  --  non-throwing existence check.
    // -------------------------------------------------------------------------
    bool has_sequence(int seq_id) const noexcept;

    // Copy the sequence's physical block IDs into a reusable GPU buffer.
    const int* get_device_block_table(int seq_id);

private:
    void release_device_block_table() noexcept;

    std::unordered_map<int, SequenceState> table_;
    int* d_block_table_ = nullptr;
    std::size_t d_block_table_capacity_ = 0;
    int staged_seq_id_ = 0;
    bool device_table_dirty_ = true;
};

} // namespace memory
} // namespace stratakv
