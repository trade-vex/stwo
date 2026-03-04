// ============================================================================
// FRI Folding Kernels
// ============================================================================

/// FRI fold_line kernel
/// Folds a line evaluation in half using inverse butterflies and alpha combination
/// Output: val0 + alpha * val1
///
/// Parameters:
/// - input: Input SecureField values (length N)
/// - output: Output SecureField values (length N/2)
/// - itwiddles: Inverse twiddles for the line domain (length N/2, doubled)
/// - alpha: SecureField folding parameter
/// - log_size: log2(N)
kernel void fri_fold_line(
    device const QM31* input [[buffer(0)]],
    device QM31* output [[buffer(1)]],
    device const uint32_t* itwiddles [[buffer(2)]],
    constant QM31& alpha [[buffer(3)]],
    constant uint32_t& log_size [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    uint32_t n_half = 1u << (log_size - 1);
    if (tid >= n_half) return;

    // Load pair of values
    QM31 val0 = input[tid * 2];
    QM31 val1 = input[tid * 2 + 1];

    // Apply inverse butterfly with M31 twiddle (doubled)
    uint32_t twiddle_dbl = itwiddles[tid];
    qm31_ifft_butterfly(val0, val1, twiddle_dbl);

    // Combine: val0 + alpha * val1
    QM31 result = qm31_add(val0, qm31_mul(alpha, val1));

    output[tid] = result;
}

/// FRI fold_circle_into_line kernel
/// Folds circle domain values into line domain and accumulates
///
/// Parameters:
/// - src: Source circle evaluation (SecureField, length N)
/// - dst: Destination line evaluation (SecureField, length N/2) - accumulated into
/// - itwiddles: Inverse twiddles for line domain (layer 1 twiddles, doubled)
/// - alpha: SecureField folding parameter
/// - alpha_sq: alpha^2 (precomputed)
/// - log_size: log2(N)
kernel void fri_fold_circle_into_line(
    device const QM31* src [[buffer(0)]],
    device QM31* dst [[buffer(1)]],
    device const uint32_t* itwiddles [[buffer(2)]],
    constant QM31& alpha [[buffer(3)]],
    constant QM31& alpha_sq [[buffer(4)]],
    constant uint32_t& log_size [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    uint32_t n_half = 1u << (log_size - 1);
    if (tid >= n_half) return;

    // Load pair of values from circle domain
    QM31 val0 = src[tid * 2];
    QM31 val1 = src[tid * 2 + 1];

    // Compute layer 0 twiddle from layer 1 twiddles (same as compute_first_twiddles)
    // Layer 0 pattern: [y, -y, -x, x] from layer 1 [x, y]
    // For thread tid:
    //   k = tid / 4 (group index)
    //   offset = tid % 4 (position in group)
    //   Layer 1 twiddles at [2k, 2k+1] are [x, y]
    //   Layer 0 twiddle depends on offset:
    //     0 -> y, 1 -> -y, 2 -> -x, 3 -> x
    uint32_t k = tid / 4;
    uint32_t offset = tid % 4;

    // Doubled prime for negation (P*2 = (2^31 - 1) * 2 = 2^32 - 2)
    const uint32_t P2 = 0xFFFFFFFE;
    uint32_t twiddle_dbl;

    if (offset == 0) {
        // y (no negation)
        twiddle_dbl = itwiddles[2 * k + 1];
    } else if (offset == 1) {
        // -y (negate)
        twiddle_dbl = itwiddles[2 * k + 1] ^ P2;
    } else if (offset == 2) {
        // -x (negate)
        twiddle_dbl = itwiddles[2 * k] ^ P2;
    } else {
        // x (no negation)
        twiddle_dbl = itwiddles[2 * k];
    }

    // Apply inverse butterfly with layer 0 twiddle (doubled)
    qm31_ifft_butterfly(val0, val1, twiddle_dbl);

    // Combine: val0 + alpha * val1
    QM31 folded = qm31_add(val0, qm31_mul(alpha, val1));

    // Accumulate into dst: dst[tid] = dst[tid] * alpha^2 + folded
    QM31 prev = dst[tid];
    QM31 scaled = qm31_mul(prev, alpha_sq);
    dst[tid] = qm31_add(scaled, folded);
}

/// Fused FRI fold_line kernel (coords → coords)
/// Combines pack + fold + unpack into a single kernel to eliminate intermediate buffers
///
/// Parameters:
/// - in_c0/c1/c2/c3: Input coordinate buffers (M31, length 2*N each)
/// - out_c0/c1/c2/c3: Output coordinate buffers (M31, length N each)
/// - itwiddles: Inverse twiddles (M31 doubled)
/// - alpha: SecureField folding parameter (QM31)
/// - log_size: log2(2*N)
kernel void fri_fold_line_coords(
    device const uint32_t* in_c0 [[buffer(0)]],
    device const uint32_t* in_c1 [[buffer(1)]],
    device const uint32_t* in_c2 [[buffer(2)]],
    device const uint32_t* in_c3 [[buffer(3)]],
    device uint32_t* out_c0 [[buffer(4)]],
    device uint32_t* out_c1 [[buffer(5)]],
    device uint32_t* out_c2 [[buffer(6)]],
    device uint32_t* out_c3 [[buffer(7)]],
    device const uint32_t* itwiddles [[buffer(8)]],
    constant QM31& alpha [[buffer(9)]],
    constant uint32_t& log_size [[buffer(10)]],
    uint tid [[thread_position_in_grid]]
) {
    uint32_t n_half = 1u << (log_size - 1);
    if (tid >= n_half) return;

    // Pack input pairs into QM31
    QM31 val0 = {CM31{in_c0[tid * 2], in_c1[tid * 2]}, CM31{in_c2[tid * 2], in_c3[tid * 2]}};
    QM31 val1 = {CM31{in_c0[tid * 2 + 1], in_c1[tid * 2 + 1]}, CM31{in_c2[tid * 2 + 1], in_c3[tid * 2 + 1]}};

    // Apply inverse butterfly with M31 twiddle (doubled)
    uint32_t twiddle_dbl = itwiddles[tid];
    qm31_ifft_butterfly(val0, val1, twiddle_dbl);

    // Combine: val0 + alpha * val1
    QM31 result = qm31_add(val0, qm31_mul(alpha, val1));

    // Unpack result to output coords
    out_c0[tid] = result.c0.a;
    out_c1[tid] = result.c0.b;
    out_c2[tid] = result.c1.a;
    out_c3[tid] = result.c1.b;
}

/// Fused FRI fold_circle_into_line kernel (coords → coords)
/// Combines pack + fold + unpack into a single kernel
///
/// Parameters:
/// - src_c0/c1/c2/c3: Source coordinate buffers (M31, length N each)
/// - dst_c0/c1/c2/c3: Destination coordinate buffers (M31, length N/2 each) - accumulated into
/// - itwiddles: Inverse twiddles for line domain (layer 1 twiddles, doubled)
/// - alpha: SecureField folding parameter
/// - alpha_sq: alpha^2 (precomputed)
/// - log_size: log2(N)
kernel void fri_fold_circle_into_line_coords(
    device const uint32_t* src_c0 [[buffer(0)]],
    device const uint32_t* src_c1 [[buffer(1)]],
    device const uint32_t* src_c2 [[buffer(2)]],
    device const uint32_t* src_c3 [[buffer(3)]],
    device uint32_t* dst_c0 [[buffer(4)]],
    device uint32_t* dst_c1 [[buffer(5)]],
    device uint32_t* dst_c2 [[buffer(6)]],
    device uint32_t* dst_c3 [[buffer(7)]],
    device const uint32_t* itwiddles [[buffer(8)]],
    constant QM31& alpha [[buffer(9)]],
    constant QM31& alpha_sq [[buffer(10)]],
    constant uint32_t& log_size [[buffer(11)]],
    uint tid [[thread_position_in_grid]]
) {
    uint32_t n_half = 1u << (log_size - 1);
    if (tid >= n_half) return;

    // Pack source pairs into QM31
    QM31 val0 = {CM31{src_c0[tid * 2], src_c1[tid * 2]}, CM31{src_c2[tid * 2], src_c3[tid * 2]}};
    QM31 val1 = {CM31{src_c0[tid * 2 + 1], src_c1[tid * 2 + 1]}, CM31{src_c2[tid * 2 + 1], src_c3[tid * 2 + 1]}};

    // Compute layer 0 twiddle from layer 1 twiddles
    uint32_t k = tid / 4;
    uint32_t offset = tid % 4;
    const uint32_t P2 = 0xFFFFFFFE;
    uint32_t twiddle_dbl;

    if (offset == 0) {
        twiddle_dbl = itwiddles[2 * k + 1];
    } else if (offset == 1) {
        twiddle_dbl = itwiddles[2 * k + 1] ^ P2;
    } else if (offset == 2) {
        twiddle_dbl = itwiddles[2 * k] ^ P2;
    } else {
        twiddle_dbl = itwiddles[2 * k];
    }

    // Apply inverse butterfly
    qm31_ifft_butterfly(val0, val1, twiddle_dbl);

    // Combine: val0 + alpha * val1
    QM31 folded = qm31_add(val0, qm31_mul(alpha, val1));

    // Pack previous destination value and accumulate
    QM31 prev = {CM31{dst_c0[tid], dst_c1[tid]}, CM31{dst_c2[tid], dst_c3[tid]}};
    QM31 scaled = qm31_mul(prev, alpha_sq);
    QM31 result = qm31_add(scaled, folded);

    // Unpack result to output coords
    dst_c0[tid] = result.c0.a;
    dst_c1[tid] = result.c0.b;
    dst_c2[tid] = result.c1.a;
    dst_c3[tid] = result.c1.b;
}

/// FRI decompose kernel: applies g[i] = eval[i] ± lambda
/// First half: g[i] = eval[i] - lambda
/// Second half: g[i] = eval[i] + lambda
kernel void fri_decompose(
    device const QM31* input_qm31 [[buffer(0)]],   // Input values (QM31)
    device QM31* output_qm31 [[buffer(1)]],        // Output values (QM31)
    constant QM31& lambda [[buffer(2)]],           // Lambda value (QM31)
    constant uint32_t& half_size [[buffer(3)]],    // Half of domain size
    uint gid [[thread_position_in_grid]]
) {
    // Read input value (QM31)
    QM31 input_val = input_qm31[gid];

    // Compute output: subtract for first half, add for second half
    QM31 output_val;
    if (gid < half_size) {
        output_val = qm31_sub(input_val, lambda);
    } else {
        output_val = qm31_add(input_val, lambda);
    }

    // Write output (QM31)
    output_qm31[gid] = output_val;
}

/// Compute two partial sums for FRI decompose (first half and second half).
/// Uses parallel reduction with threadgroup_size threads per group.
/// Each threadgroup produces one partial sum for first half and one for second half.
/// Output: array of QM31 partial sums [a_partial_0, b_partial_0, a_partial_1, b_partial_1, ...]
kernel void fri_decompose_sum(
    device const QM31* input [[buffer(0)]],         // Input QM31 array
    device QM31* partial_sums [[buffer(1)]],        // Output partial sums (2 per threadgroup)
    constant uint32_t& half_size [[buffer(2)]],     // Half of domain size
    constant uint32_t& grid_size [[buffer(3)]],     // Total threads in grid
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]]
) {
    // Shared memory for reduction within threadgroup
    threadgroup QM31 shared_a[256];
    threadgroup QM31 shared_b[256];

    QM31 zero = {0, 0, 0, 0};

    // Each thread sums elements assigned to it
    QM31 local_a = zero;
    QM31 local_b = zero;

    // Grid-stride loop to handle arrays larger than grid size
    uint domain_size = half_size * 2;
    for (uint i = gid; i < domain_size; i += grid_size) {
        QM31 val = input[i];
        if (i < half_size) {
            local_a = qm31_add(local_a, val);
        } else {
            local_b = qm31_add(local_b, val);
        }
    }

    // Store local sums to shared memory
    shared_a[tid] = local_a;
    shared_b[tid] = local_b;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction within threadgroup (fixed size 256)
    if (tid < 128) {
        shared_a[tid] = qm31_add(shared_a[tid], shared_a[tid + 128]);
        shared_b[tid] = qm31_add(shared_b[tid], shared_b[tid + 128]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 64) {
        shared_a[tid] = qm31_add(shared_a[tid], shared_a[tid + 64]);
        shared_b[tid] = qm31_add(shared_b[tid], shared_b[tid + 64]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 32) {
        shared_a[tid] = qm31_add(shared_a[tid], shared_a[tid + 32]);
        shared_b[tid] = qm31_add(shared_b[tid], shared_b[tid + 32]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 16) {
        shared_a[tid] = qm31_add(shared_a[tid], shared_a[tid + 16]);
        shared_b[tid] = qm31_add(shared_b[tid], shared_b[tid + 16]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 8) {
        shared_a[tid] = qm31_add(shared_a[tid], shared_a[tid + 8]);
        shared_b[tid] = qm31_add(shared_b[tid], shared_b[tid + 8]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 4) {
        shared_a[tid] = qm31_add(shared_a[tid], shared_a[tid + 4]);
        shared_b[tid] = qm31_add(shared_b[tid], shared_b[tid + 4]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 2) {
        shared_a[tid] = qm31_add(shared_a[tid], shared_a[tid + 2]);
        shared_b[tid] = qm31_add(shared_b[tid], shared_b[tid + 2]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 1) {
        shared_a[tid] = qm31_add(shared_a[tid], shared_a[tid + 1]);
        shared_b[tid] = qm31_add(shared_b[tid], shared_b[tid + 1]);
    }

    // Thread 0 writes threadgroup's partial sum
    if (tid == 0) {
        partial_sums[tg_id * 2] = shared_a[0];
        partial_sums[tg_id * 2 + 1] = shared_b[0];
    }
}

