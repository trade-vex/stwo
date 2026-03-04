// ============================================================================
// Proof-of-Work Grinding Kernels
// ============================================================================

/// Maximum number of candidate nonces the kernel can collect per dispatch.
/// Host reads these and picks the valid minimum.
#define MAX_GRIND_CANDIDATES 64

/// Proof-of-work grinding kernel
/// Searches for a nonce that produces a hash with at least pow_bits trailing zeros.
/// Each thread tries a range of nonces.
///
/// Found nonces are written to a candidates buffer using an atomic counter.
/// The host picks the minimum valid nonce from the candidates.
kernel void grind_pow(
    device const uint32_t* digest [[buffer(0)]],        // Prefix digest (8 u32s)
    device uint64_t* candidates [[buffer(1)]],           // Output: candidate nonces buffer
    device atomic_uint* candidate_count [[buffer(2)]],   // Atomic count of candidates found
    constant uint32_t& pow_bits [[buffer(3)]],           // Required trailing zero bits
    constant uint64_t& start_nonce [[buffer(4)]],        // Starting nonce for this batch
    constant uint32_t& batch_size [[buffer(5)]],         // Nonces to try per thread
    constant bool& is_m31_output [[buffer(6)]],          // Whether to reduce modulo M31
    uint tid [[thread_position_in_grid]]
) {
    uint64_t base_nonce = start_nonce + tid * batch_size;

    for (uint32_t i = 0; i < batch_size; i++) {
        uint64_t nonce = base_nonce + i;

        // Prepare message: digest (32 bytes) || nonce (8 bytes)
        uint32_t message[16];

        // Load digest (8 u32s = 32 bytes)
        for (int j = 0; j < 8; j++) {
            message[j] = digest[j];
        }

        // Add nonce (8 bytes = 2 u32s) in little-endian
        message[8] = uint32_t(nonce);
        message[9] = uint32_t(nonce >> 32);

        // Zero-pad remaining message words
        for (int j = 10; j < 16; j++) {
            message[j] = 0;
        }

        // Initialize BLAKE2s state
        uint32_t state[8];
        blake2s_init(state);

        // Compress: message is 40 bytes (digest + nonce)
        blake2s_compress(state, message, 40, 0, true);

        // Optionally reduce modulo M31
        uint32_t hash0 = state[0];
        if (is_m31_output) {
            hash0 = m31_reduce(hash0);
        }

        // Count trailing zeros in first word
        uint32_t trailing_zeros = ctz(hash0);

        // Check if we found a solution
        if (trailing_zeros >= pow_bits) {
            // Atomically claim a slot in the candidates buffer
            uint slot = atomic_fetch_add_explicit(candidate_count, 1, memory_order_relaxed);
            if (slot < MAX_GRIND_CANDIDATES) {
                candidates[slot] = nonce;
            }
            return;
        }
    }
}

