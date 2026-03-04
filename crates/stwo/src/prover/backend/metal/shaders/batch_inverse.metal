// ============================================================================
// Batch Inverse Kernels (Montgomery's trick, GPU-parallel)
// ============================================================================
//
// Uses a binary tree approach within each threadgroup:
// 1. Forward pass: compute cumulative products up the tree
// 2. Invert the root (single element inversion)
// 3. Backward pass: propagate inverses down the tree
//
// Each threadgroup handles BLOCK_SIZE elements.

#define BATCH_INV_BLOCK_SIZE 512
#define BATCH_INV_LOG_BLOCK_SIZE 9

/// GPU-parallel batch inverse for M31 base field elements.
/// Each threadgroup handles BATCH_INV_BLOCK_SIZE elements using shared memory.
kernel void batch_inverse_m31(
    device const uint32_t* input [[buffer(0)]],
    device uint32_t* output [[buffer(1)]],
    constant uint32_t& total_size [[buffer(2)]],
    uint tid_in_group [[thread_index_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint group_size [[threads_per_threadgroup]]
) {
    // Each threadgroup handles BATCH_INV_BLOCK_SIZE elements
    // Each thread loads 2 elements
    uint block_offset = group_id * BATCH_INV_BLOCK_SIZE;
    uint idx = tid_in_group;

    // Shared memory: leaves + inner tree nodes
    threadgroup uint32_t s_leaves[BATCH_INV_BLOCK_SIZE];
    threadgroup uint32_t s_tree[BATCH_INV_BLOCK_SIZE]; // inner nodes (at most n-1 needed)

    // Load 2 elements per thread into shared memory
    if (block_offset + idx < total_size) {
        s_leaves[idx] = input[block_offset + idx];
    } else {
        s_leaves[idx] = 1; // pad with identity
    }
    if (block_offset + idx + group_size < total_size) {
        s_leaves[idx + group_size] = input[block_offset + idx + group_size];
    } else {
        s_leaves[idx + group_size] = 1; // pad with identity
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Forward pass: build cumulative product tree
    // Level 0: pairs of leaves -> first level of inner tree
    uint size = BATCH_INV_BLOCK_SIZE >> 1;
    if (idx < size) {
        s_tree[idx] = m31_mul(s_leaves[idx << 1], s_leaves[(idx << 1) + 1]);
    }

    uint from_offset = 0;
    uint dst_offset = size;
    size >>= 1;

    // Build tree levels, stopping when we have <= 32 elements (warp-size sync)
    int step = 1;
    while (step + 5 < BATCH_INV_LOG_BLOCK_SIZE) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (idx < size) {
            s_tree[dst_offset + idx] = m31_mul(
                s_tree[from_offset + (idx << 1)],
                s_tree[from_offset + (idx << 1) + 1]
            );
        }
        from_offset = dst_offset;
        dst_offset = dst_offset + size;
        size >>= 1;
        step++;
    }

    // Invert the remaining top elements (small number, ~32)
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (idx < (size << 1)) {
        s_tree[from_offset + idx] = m31_inverse(s_tree[from_offset + idx]);
    }

    // Backward pass: propagate inverses down the tree
    step = 5;
    size = 32;
    dst_offset = from_offset - (size << 1);
    while (step < BATCH_INV_LOG_BLOCK_SIZE - 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (idx < size) {
            // inv_left  = inv_parent * right_child
            // inv_right = inv_parent * left_child
            uint32_t temp = s_tree[dst_offset + (idx << 1)];
            s_tree[dst_offset + (idx << 1)] = m31_mul(
                s_tree[from_offset + idx],
                s_tree[dst_offset + (idx << 1) + 1]
            );
            s_tree[dst_offset + (idx << 1) + 1] = m31_mul(
                s_tree[from_offset + idx],
                temp
            );
        }
        size <<= 1;
        from_offset = dst_offset;
        dst_offset = from_offset - (size << 1);
        step++;
    }

    // Final level: compute leaf inverses from first level of inner tree
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (idx < (BATCH_INV_BLOCK_SIZE >> 1)) {
        uint32_t inv_parent = s_tree[idx];
        uint32_t left = s_leaves[idx << 1];
        uint32_t right = s_leaves[(idx << 1) + 1];
        // inv_left = inv_parent * right
        // inv_right = inv_parent * left
        uint32_t inv_left = m31_mul(inv_parent, right);
        uint32_t inv_right = m31_mul(inv_parent, left);

        if (block_offset + (idx << 1) < total_size) {
            output[block_offset + (idx << 1)] = inv_left;
        }
        if (block_offset + (idx << 1) + 1 < total_size) {
            output[block_offset + (idx << 1) + 1] = inv_right;
        }
    }
}
