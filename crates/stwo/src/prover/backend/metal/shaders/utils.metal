/// IFFT normalization kernel: multiply all elements by 1/N.
///
/// After inverse FFT, we need to normalize by dividing by domain size.
/// Each thread processes one M31 element.
kernel void ifft_normalize_m31(
    device uint32_t* data [[buffer(0)]],
    constant uint32_t& n_inv [[buffer(1)]],
    constant uint32_t& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    uint32_t val = data[gid];
    data[gid] = m31_mul(val, n_inv);
}

/// Accumulate M31 columns: dst += src (elementwise)
///
/// This kernel adds src to dst elementwise for M31 values.
/// Used for accumulating coordinate columns in SecureColumnByCoords.
/// Each thread processes one M31 element.
kernel void accumulate_m31(
    device uint32_t* dst [[buffer(0)]],        // Destination column (count elements)
    device const uint32_t* src [[buffer(1)]],  // Source column (count elements)
    constant uint32_t& count [[buffer(2)]],    // Number of M31 elements
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    dst[gid] = m31_add(dst[gid], src[gid]);
}

/// Pack 4 coordinate columns into interleaved QM31 layout.
/// Input: 4 separate M31 coordinate buffers (col0, col1, col2, col3)
/// Output: Interleaved u32 buffer [a0,b0,c0,d0, a1,b1,c1,d1, ...]
/// where QM31 = (CM31(a, b), CM31(c, d))
kernel void pack_coords_to_qm31(
    device const uint32_t* col0 [[buffer(0)]],  // First coordinate (a)
    device const uint32_t* col1 [[buffer(1)]],  // Second coordinate (b)
    device const uint32_t* col2 [[buffer(2)]],  // Third coordinate (c)
    device const uint32_t* col3 [[buffer(3)]],  // Fourth coordinate (d)
    device uint32_t* out_qm31 [[buffer(4)]],    // Interleaved output
    constant uint32_t& len [[buffer(5)]],       // Number of QM31 elements
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;

    // Write interleaved: [a, b, c, d] for this element
    uint32_t out_idx = gid * 4;
    out_qm31[out_idx + 0] = col0[gid];
    out_qm31[out_idx + 1] = col1[gid];
    out_qm31[out_idx + 2] = col2[gid];
    out_qm31[out_idx + 3] = col3[gid];
}

/// Unpack interleaved QM31 layout into 4 coordinate columns.
/// Input: Interleaved u32 buffer [a0,b0,c0,d0, a1,b1,c1,d1, ...]
/// Output: 4 separate M31 coordinate buffers (col0, col1, col2, col3)
kernel void unpack_qm31_to_coords(
    device const uint32_t* in_qm31 [[buffer(0)]],  // Interleaved input
    device uint32_t* col0 [[buffer(1)]],           // First coordinate (a)
    device uint32_t* col1 [[buffer(2)]],           // Second coordinate (b)
    device uint32_t* col2 [[buffer(3)]],           // Third coordinate (c)
    device uint32_t* col3 [[buffer(4)]],           // Fourth coordinate (d)
    constant uint32_t& len [[buffer(5)]],          // Number of QM31 elements
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;

    // Read interleaved: [a, b, c, d] for this element
    uint32_t in_idx = gid * 4;
    col0[gid] = in_qm31[in_idx + 0];
    col1[gid] = in_qm31[in_idx + 1];
    col2[gid] = in_qm31[in_idx + 2];
    col3[gid] = in_qm31[in_idx + 3];
}

