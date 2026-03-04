// ============================================================================
// FFT Butterfly Operations
// ============================================================================

/// Forward FFT butterfly operation.
/// Computes: (v0 + t*v1, v0 - t*v1)
inline void fft_butterfly(
    thread uint32_t& v0,
    thread uint32_t& v1,
    uint32_t twiddle_dbl
) {
    uint32_t prod = m31_mul_twiddle_dbl(v1, twiddle_dbl);
    uint32_t sum = m31_add(v0, prod);
    uint32_t diff = m31_sub(v0, prod);
    v0 = sum;
    v1 = diff;
}

/// Inverse FFT butterfly operation.
/// Computes: (v0 + v1, (v0 - v1) * t)
inline void ifft_butterfly(
    thread uint32_t& v0,
    thread uint32_t& v1,
    uint32_t twiddle_dbl
) {
    uint32_t sum = m31_add(v0, v1);
    uint32_t diff = m31_sub(v0, v1);
    uint32_t prod = m31_mul_twiddle_dbl(diff, twiddle_dbl);
    v0 = sum;
    v1 = prod;
}

/// QM31 inverse butterfly for FRI folding
/// Computes: (v0 + v1, (v0 - v1) * t) for SecureField elements
inline void qm31_ifft_butterfly(
    thread QM31& v0,
    thread QM31& v1,
    uint32_t twiddle_dbl
) {
    QM31 sum = qm31_add(v0, v1);
    QM31 diff = qm31_sub(v0, v1);

    // Multiply diff by M31 twiddle applied to all 4 components
    // diff = (c0.a + c0.b*i) + (c1.a + c1.b*i)*u
    // Each M31 component is multiplied by the same M31 twiddle
    QM31 prod;
    prod.c0.a = m31_mul_twiddle_dbl(diff.c0.a, twiddle_dbl);
    prod.c0.b = m31_mul_twiddle_dbl(diff.c0.b, twiddle_dbl);
    prod.c1.a = m31_mul_twiddle_dbl(diff.c1.a, twiddle_dbl);
    prod.c1.b = m31_mul_twiddle_dbl(diff.c1.b, twiddle_dbl);

    v0 = sum;
    v1 = prod;
}

/// Fused vecwise FFT kernel for layers 1-4.
///
/// Processes 4 consecutive radix-2 layers in a single kernel using threadgroup memory.
/// This eliminates the overhead of multiple kernel launches and memory round-trips.
/// Each threadgroup processes a contiguous block of data through all 4 layers.
/// IMPORTANT: Layers are processed in REVERSE order (4→3→2→1) to match FFT algorithm.
kernel void circle_fft_vecwise_fused(
    device uint32_t* data [[buffer(0)]],
    device const uint32_t* twiddles_layer1 [[buffer(1)]],
    device const uint32_t* twiddles_layer2 [[buffer(2)]],
    device const uint32_t* twiddles_layer3 [[buffer(3)]],
    device const uint32_t* twiddles_layer4 [[buffer(4)]],
    constant uint32_t& log_size [[buffer(5)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]]
) {
    // Process layers in REVERSE order (4, 3, 2, 1) as required by FFT algorithm
    // Each layer has different stride and butterfly patterns

    // Layer 4: stride = 16
    {
        uint32_t stride = 16;
        uint32_t pair_count = 1u << (log_size - 1);

        if (gid < pair_count) {
            uint32_t block_size = stride * 2;
            uint32_t block_idx = gid / stride;
            uint32_t in_block_idx = gid % stride;

            uint32_t idx0 = block_idx * block_size + in_block_idx;
            uint32_t idx1 = idx0 + stride;

            uint32_t v0 = data[idx0];
            uint32_t v1 = data[idx1];

            uint32_t twiddle_dbl = twiddles_layer4[block_idx];
            fft_butterfly(v0, v1, twiddle_dbl);

            data[idx0] = v0;
            data[idx1] = v1;
        }
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Layer 3: stride = 8
    {
        uint32_t stride = 8;
        uint32_t pair_count = 1u << (log_size - 1);

        if (gid < pair_count) {
            uint32_t block_size = stride * 2;
            uint32_t block_idx = gid / stride;
            uint32_t in_block_idx = gid % stride;

            uint32_t idx0 = block_idx * block_size + in_block_idx;
            uint32_t idx1 = idx0 + stride;

            uint32_t v0 = data[idx0];
            uint32_t v1 = data[idx1];

            uint32_t twiddle_dbl = twiddles_layer3[block_idx];
            fft_butterfly(v0, v1, twiddle_dbl);

            data[idx0] = v0;
            data[idx1] = v1;
        }
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Layer 2: stride = 4
    {
        uint32_t stride = 4;
        uint32_t pair_count = 1u << (log_size - 1);

        if (gid < pair_count) {
            uint32_t block_size = stride * 2;
            uint32_t block_idx = gid / stride;
            uint32_t in_block_idx = gid % stride;

            uint32_t idx0 = block_idx * block_size + in_block_idx;
            uint32_t idx1 = idx0 + stride;

            uint32_t v0 = data[idx0];
            uint32_t v1 = data[idx1];

            uint32_t twiddle_dbl = twiddles_layer2[block_idx];
            fft_butterfly(v0, v1, twiddle_dbl);

            data[idx0] = v0;
            data[idx1] = v1;
        }
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Layer 1: stride = 2
    {
        uint32_t stride = 2;
        uint32_t pair_count = 1u << (log_size - 1);

        if (gid < pair_count) {
            uint32_t block_size = stride * 2;
            uint32_t block_idx = gid / stride;
            uint32_t in_block_idx = gid % stride;

            uint32_t idx0 = block_idx * block_size + in_block_idx;
            uint32_t idx1 = idx0 + stride;

            uint32_t v0 = data[idx0];
            uint32_t v1 = data[idx1];

            uint32_t twiddle_dbl = twiddles_layer1[block_idx];
            fft_butterfly(v0, v1, twiddle_dbl);

            data[idx0] = v0;
            data[idx1] = v1;
        }
    }
}

// ============================================================================
// FFT Kernels
// ============================================================================

/// Circle FFT Radix-8 kernel.
///
/// Each thread processes 8 elements through 3 butterfly layers (radix-8).
/// This implements a Cooley-Tukey decimation-in-frequency FFT adapted for
/// the circle group using bit-reversed twiddle factors.
///
/// Parameters:
/// - data: Input/output buffer in natural order (input) / bit-reversed order (output)
/// - twiddles_layer0/1/2: Doubled twiddle factors for each layer (bit-reversed)
/// - log_size: log2(FFT size)
/// - log_step: Current layer offset
/// - layer: Starting layer index (layer, layer+1, layer+2 are processed)
kernel void circle_fft_radix8(
    device uint32_t* data [[buffer(0)]],
    device const uint32_t* twiddles_layer0 [[buffer(1)]],
    device const uint32_t* twiddles_layer1 [[buffer(2)]],
    device const uint32_t* twiddles_layer2 [[buffer(3)]],
    constant uint32_t& log_size [[buffer(4)]],
    constant uint32_t& layer [[buffer(5)]],
    constant uint32_t& tw0_len [[buffer(6)]],
    constant uint32_t& tw1_len [[buffer(7)]],
    constant uint32_t& tw2_len [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    // Each thread processes 8 elements with stride 2^layer
    // Data is divided into blocks of size 2^(layer+3)
    // Thread gid is decomposed into: block_idx (which block) and in_block_idx (position within block)
    uint32_t stride = 1u << layer;
    uint32_t block_idx = gid >> layer;  // Which block of size 2^(layer+3)
    uint32_t in_block_idx = gid & ((1u << layer) - 1);  // Position within the block
    uint32_t offset = (block_idx << (layer + 3)) + in_block_idx;

    // Twiddle index is just the block index
    uint32_t index = block_idx;

    // Load 8 elements with stride (matching SIMD fft3 implementation)
    uint32_t v0 = data[offset + (0 << layer)];
    uint32_t v1 = data[offset + (1 << layer)];
    uint32_t v2 = data[offset + (2 << layer)];
    uint32_t v3 = data[offset + (3 << layer)];
    uint32_t v4 = data[offset + (4 << layer)];
    uint32_t v5 = data[offset + (5 << layer)];
    uint32_t v6 = data[offset + (6 << layer)];
    uint32_t v7 = data[offset + (7 << layer)];

    // Layer 2: 4 butterflies (coarsest, stride = 4)
    // SIMD uses: twiddle_dbl[2][(index + i) & (len - 1)]
    // Use actual buffer length for mask (not layer-based formula)
    uint32_t tw2_mask = tw2_len - 1;
    uint32_t tw2 = twiddles_layer2[index & tw2_mask];
    fft_butterfly(v0, v4, tw2);
    fft_butterfly(v1, v5, tw2);
    fft_butterfly(v2, v6, tw2);
    fft_butterfly(v3, v7, tw2);

    // Layer 1: 4 butterflies (middle, stride = 2)
    // SIMD uses: twiddle_dbl[1][(index * 2 + i) & (len - 1)]
    uint32_t tw1_mask = tw1_len - 1;
    uint32_t tw1_0 = twiddles_layer1[(index * 2 + 0) & tw1_mask];
    uint32_t tw1_1 = twiddles_layer1[(index * 2 + 1) & tw1_mask];
    fft_butterfly(v0, v2, tw1_0);
    fft_butterfly(v1, v3, tw1_0);
    fft_butterfly(v4, v6, tw1_1);
    fft_butterfly(v5, v7, tw1_1);

    // Layer 0: 4 butterflies (finest, stride = 1)
    // SIMD uses: twiddle_dbl[0][(index * 4 + i) & (len - 1)]
    uint32_t tw0_mask = tw0_len - 1;
    uint32_t tw0_0 = twiddles_layer0[(index * 4 + 0) & tw0_mask];
    uint32_t tw0_1 = twiddles_layer0[(index * 4 + 1) & tw0_mask];
    uint32_t tw0_2 = twiddles_layer0[(index * 4 + 2) & tw0_mask];
    uint32_t tw0_3 = twiddles_layer0[(index * 4 + 3) & tw0_mask];
    fft_butterfly(v0, v1, tw0_0);
    fft_butterfly(v2, v3, tw0_1);
    fft_butterfly(v4, v5, tw0_2);
    fft_butterfly(v6, v7, tw0_3);

    // Store results with stride
    data[offset + (0 << layer)] = v0;
    data[offset + (1 << layer)] = v1;
    data[offset + (2 << layer)] = v2;
    data[offset + (3 << layer)] = v3;
    data[offset + (4 << layer)] = v4;
    data[offset + (5 << layer)] = v5;
    data[offset + (6 << layer)] = v6;
    data[offset + (7 << layer)] = v7;
}

/// Circle IFFT Radix-8 kernel.
///
/// Each thread processes 8 elements through 3 inverse butterfly layers.
/// This implements decimation-in-time inverse FFT for the circle group.
kernel void circle_ifft_radix8(
    device uint32_t* data [[buffer(0)]],
    device const uint32_t* itwiddles_layer0 [[buffer(1)]],
    device const uint32_t* itwiddles_layer1 [[buffer(2)]],
    device const uint32_t* itwiddles_layer2 [[buffer(3)]],
    constant uint32_t& log_size [[buffer(4)]],
    constant uint32_t& layer [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    // Each thread processes 8 elements with stride 2^layer
    // Data is divided into blocks of size 2^(layer+3)
    // Thread gid is decomposed into: block_idx (which block) and in_block_idx (position within block)
    uint32_t stride = 1u << layer;
    uint32_t block_idx = gid >> layer;  // Which block of size 2^(layer+3)
    uint32_t in_block_idx = gid & ((1u << layer) - 1);  // Position within the block
    uint32_t offset = (block_idx << (layer + 3)) + in_block_idx;

    // Twiddle index is just the block index
    uint32_t index = block_idx;

    // Load 8 elements with stride (matching SIMD ifft3 implementation)
    uint32_t v0 = data[offset + (0 << layer)];
    uint32_t v1 = data[offset + (1 << layer)];
    uint32_t v2 = data[offset + (2 << layer)];
    uint32_t v3 = data[offset + (3 << layer)];
    uint32_t v4 = data[offset + (4 << layer)];
    uint32_t v5 = data[offset + (5 << layer)];
    uint32_t v6 = data[offset + (6 << layer)];
    uint32_t v7 = data[offset + (7 << layer)];

    // Layer 0: 4 inverse butterflies (finest, stride = 1)
    // SIMD uses: twiddle_dbl[0][(index * 4 + i) & mask]
    // Twiddle array has length 2^layer, mask is 2^layer - 1
    uint32_t itw0_0 = itwiddles_layer0[(index * 4 + 0) & ((1u << layer) - 1)];
    uint32_t itw0_1 = itwiddles_layer0[(index * 4 + 1) & ((1u << layer) - 1)];
    uint32_t itw0_2 = itwiddles_layer0[(index * 4 + 2) & ((1u << layer) - 1)];
    uint32_t itw0_3 = itwiddles_layer0[(index * 4 + 3) & ((1u << layer) - 1)];
    ifft_butterfly(v0, v1, itw0_0);
    ifft_butterfly(v2, v3, itw0_1);
    ifft_butterfly(v4, v5, itw0_2);
    ifft_butterfly(v6, v7, itw0_3);

    // Layer 1: 4 inverse butterflies (middle, stride = 2)
    // SIMD uses: twiddle_dbl[1][(index * 2 + i) & mask]
    // Twiddle array has length 2^(layer+1), mask is 2^(layer+1) - 1
    uint32_t itw1_0 = itwiddles_layer1[(index * 2 + 0) & ((1u << (layer + 1)) - 1)];
    uint32_t itw1_1 = itwiddles_layer1[(index * 2 + 1) & ((1u << (layer + 1)) - 1)];
    ifft_butterfly(v0, v2, itw1_0);
    ifft_butterfly(v1, v3, itw1_0);
    ifft_butterfly(v4, v6, itw1_1);
    ifft_butterfly(v5, v7, itw1_1);

    // Layer 2: 4 inverse butterflies (coarsest, stride = 4)
    // SIMD uses: twiddle_dbl[2][(index + i) & mask]
    // Twiddle array has length 2^(layer+2), mask is 2^(layer+2) - 1
    uint32_t itw2 = itwiddles_layer2[index & ((1u << (layer + 2)) - 1)];
    ifft_butterfly(v0, v4, itw2);
    ifft_butterfly(v1, v5, itw2);
    ifft_butterfly(v2, v6, itw2);
    ifft_butterfly(v3, v7, itw2);

    // Store results with stride
    data[offset + (0 << layer)] = v0;
    data[offset + (1 << layer)] = v1;
    data[offset + (2 << layer)] = v2;
    data[offset + (3 << layer)] = v3;
    data[offset + (4 << layer)] = v4;
    data[offset + (5 << layer)] = v5;
    data[offset + (6 << layer)] = v6;
    data[offset + (7 << layer)] = v7;
}

// ============================================================================
// Vecwise FFT Kernels (Radix-2)
// ============================================================================

/// Circle FFT Radix-2 kernel for vecwise layers (0-4).
///
/// Each thread processes a single butterfly (2 elements).
/// This is used for the bottom 5 layers where elements are close together.
///
/// Parameters:
/// - data: Input/output buffer
/// - twiddles: Doubled twiddle factors for this layer
/// - log_size: log2(FFT size)
/// - layer: Current layer index (0-4 for vecwise)
kernel void circle_fft_radix2(
    device uint32_t* data [[buffer(0)]],
    device const uint32_t* twiddles [[buffer(1)]],
    constant uint32_t& log_size [[buffer(2)]],
    constant uint32_t& layer [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    // Each thread processes one butterfly
    // Stride for this layer
    uint32_t stride = 1u << layer;
    uint32_t pair_count = 1u << (log_size - 1);

    if (gid >= pair_count) return;

    // Calculate which butterfly within the block
    uint32_t block_size = stride * 2;
    uint32_t block_idx = gid / stride;
    uint32_t in_block_idx = gid % stride;

    // Indices of the two elements in this butterfly
    uint32_t idx0 = block_idx * block_size + in_block_idx;
    uint32_t idx1 = idx0 + stride;

    // Load values
    uint32_t v0 = data[idx0];
    uint32_t v1 = data[idx1];

    // Get twiddle factor for this butterfly
    // All butterflies in the same superblock use the same twiddle
    // The superblock index (block_idx) determines which twiddle to use
    uint32_t twiddle_idx = block_idx;
    uint32_t twiddle_dbl = twiddles[twiddle_idx];

    // Apply butterfly
    fft_butterfly(v0, v1, twiddle_dbl);

    // Store results
    data[idx0] = v0;
    data[idx1] = v1;
}

/// Circle IFFT Radix-2 kernel for vecwise layers (0-4).
///
/// Each thread processes a single inverse butterfly (2 elements).
kernel void circle_ifft_radix2(
    device uint32_t* data [[buffer(0)]],
    device const uint32_t* itwiddles [[buffer(1)]],
    constant uint32_t& log_size [[buffer(2)]],
    constant uint32_t& layer [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    // Each thread processes one butterfly
    uint32_t stride = 1u << layer;
    uint32_t pair_count = 1u << (log_size - 1);

    if (gid >= pair_count) return;

    // Calculate which butterfly within the block
    uint32_t block_size = stride * 2;
    uint32_t block_idx = gid / stride;
    uint32_t in_block_idx = gid % stride;

    // Indices of the two elements in this butterfly
    uint32_t idx0 = block_idx * block_size + in_block_idx;
    uint32_t idx1 = idx0 + stride;

    // Load values
    uint32_t v0 = data[idx0];
    uint32_t v1 = data[idx1];

    // Get inverse twiddle factor
    // All butterflies in the same superblock use the same twiddle
    // The superblock index (block_idx) determines which twiddle to use
    uint32_t twiddle_idx = block_idx;
    uint32_t itwiddle_dbl = itwiddles[twiddle_idx];

    // Apply inverse butterfly
    ifft_butterfly(v0, v1, itwiddle_dbl);

    // Store results
    data[idx0] = v0;
    data[idx1] = v1;
}

