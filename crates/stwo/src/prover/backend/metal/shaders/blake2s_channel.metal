// ============================================================================
// GPU Channel State Kernels
// ============================================================================

/// GPU Blake2s channel mix kernel
/// Computes: new_digest = BLAKE2s(old_digest || data)
/// This is used to mix Merkle roots into the Fiat-Shamir channel state
///
/// Single-thread execution: Channel operations are inherently serial
kernel void blake2s_channel_mix(
    device const uint32_t* old_digest [[buffer(0)]],  // 8 u32s (32 bytes)
    device const uint32_t* data [[buffer(1)]],         // 8 u32s (32 bytes) - root hash
    constant bool& is_m31_output [[buffer(2)]],
    device uint32_t* new_digest [[buffer(3)]],         // Output: 8 u32s
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;  // Single thread only

    // Prepare 64-byte message: old_digest || data
    uint32_t message[16];
    for (int i = 0; i < 8; i++) {
        message[i] = old_digest[i];
        message[8 + i] = data[i];
    }

    // Initialize BLAKE2s state
    uint32_t state[8];
    blake2s_init(state);

    // Compress the 64-byte message (final block)
    blake2s_compress(state, message, 64, 0, true);

    // Reduce to M31 if required
    if (is_m31_output) {
        blake2s_reduce_m31(state);
    }

    // Write output
    for (int i = 0; i < 8; i++) {
        new_digest[i] = state[i];
    }
}

/// GPU Blake2s channel draw kernel
/// Computes: output = BLAKE2s(digest || counter || domain_separator)
/// This is used to draw random field elements from the channel
///
/// Single-thread execution: Channel operations are inherently serial
kernel void blake2s_channel_draw(
    device const uint32_t* digest [[buffer(0)]],       // 8 u32s (32 bytes)
    constant uint32_t& counter [[buffer(1)]],          // Draw counter
    constant uint8_t& domain_sep [[buffer(2)]],        // Domain separator (typically 0)
    constant bool& is_m31_output [[buffer(3)]],
    device uint32_t* output [[buffer(4)]],             // Output: 8 u32s
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;  // Single thread only

    // Prepare 37-byte message: digest || counter || domain_sep
    // BLAKE2s processes in 64-byte blocks, so we need to pad
    uint32_t message[16];  // 64 bytes total

    // Copy digest (32 bytes = 8 u32s)
    for (int i = 0; i < 8; i++) {
        message[i] = digest[i];
    }

    // Add counter as little-endian u32 (4 bytes)
    message[8] = counter;

    // Add domain separator (1 byte) + padding
    // Pack domain_sep into message[9] as first byte
    message[9] = (uint32_t)domain_sep;

    // Zero remaining bytes (28 bytes padding)
    for (int i = 10; i < 16; i++) {
        message[i] = 0;
    }

    // Initialize BLAKE2s state
    uint32_t state[8];
    blake2s_init(state);

    // Compress the 37-byte message (final block, length = 37)
    blake2s_compress(state, message, 37, 0, true);

    // Reduce to M31 if required
    if (is_m31_output) {
        blake2s_reduce_m31(state);
    }

    // Write output
    for (int i = 0; i < 8; i++) {
        output[i] = state[i];
    }
}

/// GPU Blake2s channel mix_felts kernel
/// Computes: new_digest = BLAKE2s(old_digest || felts_bytes)
/// This is used to mix SecureField (QM31) elements into the Fiat-Shamir channel state
///
/// Each QM31 = 4 M31 values, each M31 = u32 (4 bytes) = 16 bytes per QM31
/// We serialize QM31 elements to bytes in little-endian format and mix into digest
///
/// Single-thread execution: Channel operations are inherently serial
/// For large arrays, we process in 64-byte chunks (limited by BLAKE2s block size)
kernel void blake2s_channel_mix_felts(
    device const uint32_t* old_digest [[buffer(0)]],  // 8 u32s (32 bytes)
    device const uint32_t* felts [[buffer(1)]],        // QM31 elements (4 u32s each)
    constant uint32_t& num_felts [[buffer(2)]],        // Number of QM31 elements
    constant bool& is_m31_output [[buffer(3)]],
    device uint32_t* new_digest [[buffer(4)]],         // Output: 8 u32s
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;  // Single thread only

    // Initialize BLAKE2s state
    uint32_t state[8];
    blake2s_init(state);

    // Total bytes to hash: 32 (old_digest) + num_felts * 16 (QM31 serialized)
    uint32_t total_bytes = 32 + num_felts * 16;
    uint32_t bytes_processed = 0;

    // Message buffer for BLAKE2s compression (64 bytes = 16 u32s)
    uint32_t message[16];

    // First block: old_digest (32 bytes) + up to 32 bytes of felts
    for (int i = 0; i < 8; i++) {
        message[i] = old_digest[i];
    }

    // Add felts to first block (up to 8 u32s = 32 bytes)
    uint32_t felts_in_first_block = (num_felts * 4 <= 8) ? num_felts * 4 : 8;
    for (uint32_t i = 0; i < felts_in_first_block; i++) {
        message[8 + i] = felts[i];
    }

    // If first block is complete (64 bytes), compress it
    if (total_bytes >= 64) {
        blake2s_compress(state, message, 64, 0, false);
        bytes_processed = 64;

        // Process remaining full 64-byte blocks
        uint32_t felt_offset = 8;  // Already processed first 8 u32s of felts
        while (bytes_processed + 64 <= total_bytes) {
            // Copy 16 u32s (64 bytes) from felts to message
            for (int i = 0; i < 16; i++) {
                message[i] = felts[felt_offset + i];
            }
            blake2s_compress(state, message, 64, 0, false);
            bytes_processed += 64;
            felt_offset += 16;
        }

        // Process final partial block if any
        uint32_t remaining_bytes = total_bytes - bytes_processed;
        if (remaining_bytes > 0) {
            // Copy remaining felts to message buffer
            uint32_t remaining_u32s = (remaining_bytes + 3) / 4;  // Round up
            for (uint32_t i = 0; i < remaining_u32s; i++) {
                message[i] = felts[felt_offset + i];
            }
            // Zero padding for remaining message slots
            for (uint32_t i = remaining_u32s; i < 16; i++) {
                message[i] = 0;
            }
            blake2s_compress(state, message, remaining_bytes, 0, true);
        } else {
            // No remaining bytes, mark last compression as final
            // Re-compress last block with final flag
            felt_offset -= 16;
            for (int i = 0; i < 16; i++) {
                message[i] = felts[felt_offset + i];
            }
            blake2s_compress(state, message, 64, 0, true);
        }
    } else {
        // Total message fits in one block (<= 64 bytes)
        // Zero padding for remaining message slots
        for (uint32_t i = 8 + felts_in_first_block; i < 16; i++) {
            message[i] = 0;
        }
        blake2s_compress(state, message, total_bytes, 0, true);
    }

    // Reduce to M31 if required
    if (is_m31_output) {
        blake2s_reduce_m31(state);
    }

    // Write output
    for (int i = 0; i < 8; i++) {
        new_digest[i] = state[i];
    }
}

