// ============================================================================
// Bit Reversal Kernels
// ============================================================================

/// Reverse the bits of a 32-bit integer, then shift right to get log_size bits.
inline uint32_t bit_reverse_index(uint32_t idx, uint32_t log_size) {
    uint32_t rev = reverse_bits(idx);
    return rev >> (32u - log_size);
}

/// In-place bit-reversal permutation for M31 (uint32_t) elements.
/// Each thread handles one swap (idx < rev only, to avoid double-swaps).
kernel void bit_reverse_m31(
    device uint32_t* data [[buffer(0)]],
    constant uint32_t& log_size [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    uint32_t n = 1u << log_size;
    if (gid >= n) return;

    uint32_t rev = bit_reverse_index(gid, log_size);

    // Only swap when gid < rev to avoid double-swapping
    if (gid < rev) {
        uint32_t tmp = data[gid];
        data[gid] = data[rev];
        data[rev] = tmp;
    }
}

/// In-place bit-reversal permutation for QM31 (SecureField) elements.
/// Each QM31 element is 4 contiguous uint32_t values.
kernel void bit_reverse_qm31(
    device uint32_t* data [[buffer(0)]],
    constant uint32_t& log_size [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    uint32_t n = 1u << log_size;
    if (gid >= n) return;

    uint32_t rev = bit_reverse_index(gid, log_size);

    // Only swap when gid < rev to avoid double-swapping
    if (gid < rev) {
        uint32_t base_gid = gid * 4u;
        uint32_t base_rev = rev * 4u;

        uint32_t t0 = data[base_gid];
        uint32_t t1 = data[base_gid + 1u];
        uint32_t t2 = data[base_gid + 2u];
        uint32_t t3 = data[base_gid + 3u];

        data[base_gid]      = data[base_rev];
        data[base_gid + 1u] = data[base_rev + 1u];
        data[base_gid + 2u] = data[base_rev + 2u];
        data[base_gid + 3u] = data[base_rev + 3u];

        data[base_rev]      = t0;
        data[base_rev + 1u] = t1;
        data[base_rev + 2u] = t2;
        data[base_rev + 3u] = t3;
    }
}
