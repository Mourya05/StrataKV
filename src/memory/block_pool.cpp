// =============================================================================
//  block_pool.cpp  --  StrataKV Phase 3: Physical Block Allocator (impl)
// =============================================================================

#include "stratakv/memory/block_pool.h"

#include <stdexcept>

namespace stratakv {
namespace memory  {

// ---------------------------------------------------------------------------
//  Constructor
//  Pre-populate the free list with block IDs [0, total-1].
//  Block 0 is pushed last so it sits on top of the stack — it will be the
//  first ID handed out, giving predictable output for the smoke test.
// ---------------------------------------------------------------------------
BlockPool::BlockPool(int total, int blk_sz)
    : total_blocks_(total), block_size_(blk_sz)
{
    if (total  <= 0) throw std::invalid_argument("BlockPool: total must be > 0");
    if (blk_sz <= 0) throw std::invalid_argument("BlockPool: block_size must be > 0");

    free_list_.reserve(static_cast<std::size_t>(total));
    // Push in reverse order so that block 0 is on top (back of vector).
    for (int id = total - 1; id >= 0; --id)
        free_list_.push_back(id);
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
