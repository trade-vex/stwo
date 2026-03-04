// ============================================================================
// Merkle Kernels
// ============================================================================

/// Merkle tree internal node hashing kernel
/// Hashes two child hashes together: parent = BLAKE2s(left || right)
/// Each hash is 32 bytes (8 x uint32_t)
kernel void merkle_blake2s(
    device const uint32_t* children [[buffer(0)]],  // Input: child hashes (size * 2 * 8 u32s)
    device uint32_t* parents [[buffer(1)]],          // Output: parent hashes (size * 8 u32s)
    constant bool& is_m31_output [[buffer(2)]],     // Whether to reduce modulo M31
    constant uint32_t& size [[buffer(3)]],          // Number of parent nodes
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= size) return;

    // Each parent node hashes two 32-byte children
    // Message layout: [left_hash (32 bytes), right_hash (32 bytes)] = 64 bytes = 16 u32s
    uint32_t message[16];

    // Load left child (32 bytes = 8 u32s)
    for (int i = 0; i < 8; i++) {
        message[i] = children[(tid * 2) * 8 + i];
    }

    // Load right child (32 bytes = 8 u32s)
    for (int i = 0; i < 8; i++) {
        message[8 + i] = children[(tid * 2 + 1) * 8 + i];
    }

    // Initialize BLAKE2s state (standard init, no prefix)
    // Matches Rust: Blake2sHasherGeneric::default() → Blake2s256::new()
    uint32_t state[8];
    blake2s_init(state);

    // Compress the 64-byte message (two child hashes)
    // Counter is 64 (single block of 64 bytes, final block)
    blake2s_compress(state, message, 64, 0, true);

    // Optionally reduce output modulo M31
    if (is_m31_output) {
        blake2s_reduce_m31(state);
    }

    // Write output hash (32 bytes = 8 u32s)
    for (int i = 0; i < 8; i++) {
        parents[tid * 8 + i] = state[i];
    }
}

/// Merkle tree leaf hashing kernel with lifting (variable-size columns).
///
/// Each thread computes one leaf hash by iterating through all columns in order,
/// feeding M31 values as uint32 into Blake2s. This matches the CPU build_leaves
/// algorithm: Blake2s256::new() → update(value.to_le_bytes()) for each value → finalize().
///
/// The lifting formula maps output position gid (in 2^lifting_log_size domain)
/// to a column index: lifted_idx = (gid >> (log_ratio + 1) << 1) | (gid & 1)
/// where log_ratio = lifting_log_size - col_log_size.
///
/// Blake2s block handling:
/// - Accumulate uint32 values into 16-element message blocks (64 bytes)
/// - Compress all full blocks except the last with is_last=false
/// - Compress the final block (full or partial, zero-padded) with is_last=true
/// - This matches the blake2 crate's update/finalize semantics
kernel void merkle_blake2s_leaf_lifted(
    device const uint32_t* columns_data [[buffer(0)]],   // All column data packed sequentially
    device const uint32_t* col_offsets [[buffer(1)]],     // Element offset per column into columns_data
    device const uint32_t* col_log_sizes [[buffer(2)]],   // Log2(size) per column
    device uint32_t* output [[buffer(3)]],                // Output leaf hashes (n_leaves * 8 u32s)
    constant uint32_t& n_columns [[buffer(4)]],
    constant uint32_t& lifting_log_size [[buffer(5)]],
    constant bool& is_m31_output [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint32_t n_leaves = 1u << lifting_log_size;
    if (gid >= n_leaves) return;

    // Initialize standard Blake2s state (matches Blake2s256::new())
    uint32_t state[8];
    blake2s_init(state);

    // Process all columns, feeding values into Blake2s buffer
    // Blake2s compresses every 16 u32 values (64 bytes)
    uint32_t message[16];
    for (int i = 0; i < 16; i++) message[i] = 0;
    uint32_t buf_pos = 0;
    uint32_t byte_count = 0;

    for (uint32_t col = 0; col < n_columns; col++) {
        uint32_t col_log_size = col_log_sizes[col];
        uint32_t log_ratio = lifting_log_size - col_log_size;

        // Compute lifted index into this column
        uint32_t lifted_idx;
        if (log_ratio == 0) {
            lifted_idx = gid;
        } else {
            lifted_idx = ((gid >> (log_ratio + 1)) << 1) | (gid & 1u);
        }

        message[buf_pos] = columns_data[col_offsets[col] + lifted_idx];
        buf_pos++;

        // Compress when buffer is full AND this is NOT the last column.
        // The last block (whether full or partial) is handled in finalization.
        if (buf_pos == 16 && col < n_columns - 1) {
            byte_count += 64;
            blake2s_compress(state, message, byte_count, 0, false);
            buf_pos = 0;
            for (int i = 0; i < 16; i++) message[i] = 0;
        }
    }

    // Finalize: compress the remaining data as the final block
    if (n_columns > 0) {
        byte_count += buf_pos * 4;
        blake2s_compress(state, message, byte_count, 0, true);
    } else {
        // Empty message: hash of empty string
        blake2s_compress(state, message, 0, 0, true);
    }

    // Optionally reduce output modulo M31
    if (is_m31_output) {
        blake2s_reduce_m31(state);
    }

    // Write output hash
    for (int i = 0; i < 8; i++) {
        output[gid * 8 + i] = state[i];
    }
}

