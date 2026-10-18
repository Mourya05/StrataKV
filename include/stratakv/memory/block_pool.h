// =============================================================================
//  block_pool.h  --  StrataKV Phase 3: Physical Block Allocator
//
//  Manages a fixed-size pool of memory "blocks". Each block holds the KV
//  cache for exactly block_size tokens. The pool tracks which block IDs are
//  currently free using a LIFO free-list (std::vector used as a stack so that
//  recently freed blocks are reused first, improving cache locality).
//
//  Interface contract
//  ------------------
//   allocate_block() -> int   : pop and return a free block ID.
//                               Throws std::runtime_error("OOM") if empty.
//   free_block(int block_id)  : push block_id back onto the free list.
//   free_count() -> size_t    : number of unallocated blocks (diagnostic).
//   block_size()  -> int      : tokens per block (read-only config).
//   total_blocks() -> int     : total blocks in the pool (read-only config).
// =============================================================================

#pragma once

#include <stdexcept>
#include <vector>
#include "stratakv/tensor.h"

namespace stratakv {
namespace memory  {

class BlockPool {
public:
    // -------------------------------------------------------------------------
    //  Constructor
    //  total  : total number of physical blocks in the pool
    //  blk_sz : number of tokens that fit inside one block
    // -------------------------------------------------------------------------
    BlockPool(int total, int blk_sz);
    BlockPool(int total, int blk_sz, int num_kv_heads, int head_dim,
              Device device = Device::CPU);
    BlockPool(int total, int blk_sz, Device device, int num_kv_heads,
              int head_dim);

    BlockPool(const BlockPool& other);
    BlockPool& operator=(const BlockPool& other);
    BlockPool(BlockPool&& other) noexcept;
    BlockPool& operator=(BlockPool&& other) noexcept;
    ~BlockPool() noexcept;

    // -------------------------------------------------------------------------
    //  allocate_block
    //  Pop one block ID from the free list and return it.
    //  Throws std::runtime_error("OOM") when the pool is exhausted.
    // -------------------------------------------------------------------------
    int allocate_block();

    // -------------------------------------------------------------------------
    //  free_block
    //  Return a previously allocated block back to the free list.
    //  Behaviour is undefined if block_id is out of range or already free.
    // -------------------------------------------------------------------------
    void free_block(int block_id);

    // -------------------------------------------------------------------------
    //  Diagnostics / read-only accessors
    // -------------------------------------------------------------------------
    std::size_t free_count()   const noexcept { return free_list_.size(); }
    int         block_size()   const noexcept { return block_size_; }
    int         total_blocks() const noexcept { return total_blocks_; }
    int         num_kv_heads() const noexcept { return num_kv_heads_; }
    int         head_dim()     const noexcept { return head_dim_; }
    float*      key_cache()          noexcept { return key_cache_; }
    float*      value_cache()        noexcept { return value_cache_; }
    Device      device()       const noexcept { return device_; }
    const float* key_cache()   const noexcept { return key_cache_; }
    const float* value_cache() const noexcept { return value_cache_; }

private:
    void initialize_storage();
    void release_storage() noexcept;

    int              total_blocks_;
    int              block_size_;
    int              num_kv_heads_ = 0;
    int              head_dim_ = 0;
    Device           device_ = Device::CPU;
    float*           key_cache_ = nullptr;
    float*           value_cache_ = nullptr;
    std::vector<int> free_list_;   // stack: back() is the next block to hand out
};

} // namespace memory
} // namespace stratakv
