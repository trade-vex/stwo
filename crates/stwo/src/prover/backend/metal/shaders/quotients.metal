// ============================================================================
// Quotient Accumulation Kernels
// ============================================================================

/// Montgomery batch inversion for CM31 (device function).
/// Inverts an array of CM31 elements in-place using Montgomery's trick.
/// Reduces num_batches individual inversions to 1 + constant overhead.
///
/// Algorithm:
/// 1. Forward pass: Compute cumulative products
/// 2. Invert the final product (single CM31 inversion)
/// 3. Backward pass: Compute individual inverses
///
/// This is GPU-native - no CPU round-trip required.
void cm31_batch_inverse_inplace(thread CM31* values, uint count) {
    if (count == 0) return;
    if (count == 1) {
        values[0] = cm31_inverse(values[0]);
        return;
    }

    // Allocate temporary storage for cumulative products
    // Max num_batches is typically ~20, so stack allocation is fine
    CM31 products[32];  // Support up to 32 batches

    // Forward pass: Compute cumulative products
    products[0] = values[0];
    for (uint i = 1; i < count; i++) {
        products[i] = cm31_mul(products[i - 1], values[i]);
    }

    // Invert the final cumulative product (single inversion)
    CM31 inverse = cm31_inverse(products[count - 1]);

    // Backward pass: Compute individual inverses
    // Must save original value before overwriting, since we need it to update `inverse`.
    for (uint i = count - 1; i > 0; i--) {
        CM31 original_i = values[i];
        values[i] = cm31_mul(products[i - 1], inverse);
        inverse = cm31_mul(inverse, original_i);
    }
    values[0] = inverse;
}

/// Partial numerator accumulation kernel for QuotientOps::accumulate_numerators.
///
/// For one sample batch, computes partial numerator per domain point:
///   partial_num(p) = Σ_i (c_i * col_i(p) - b_i)
///
/// One dispatch per sample batch. All columns have the same domain size.
/// Output is 4 separate M31 coordinate buffers matching SecureColumnByCoords layout.
kernel void quotient_partial_numerator(
    device const uint32_t* columns_packed [[buffer(0)]],  // All column data packed contiguously
    device const uint32_t* col_indices [[buffer(1)]],     // Column index per coefficient
    device const QM31* bc_coeffs [[buffer(2)]],           // Flattened (b, c) pairs as QM31
    constant uint32_t& domain_size [[buffer(3)]],
    constant uint32_t& batch_size [[buffer(4)]],          // Number of columns in this batch
    device uint32_t* out_coord0 [[buffer(5)]],            // Output M31 coord 0 (c0.a)
    device uint32_t* out_coord1 [[buffer(6)]],            // Output M31 coord 1 (c0.b)
    device uint32_t* out_coord2 [[buffer(7)]],            // Output M31 coord 2 (c1.a)
    device uint32_t* out_coord3 [[buffer(8)]],            // Output M31 coord 3 (c1.b)
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= domain_size) return;

    QM31 numerator = qm31_zero();

    for (uint i = 0; i < batch_size; i++) {
        QM31 b = bc_coeffs[i * 2];
        QM31 c = bc_coeffs[i * 2 + 1];
        uint col_idx = col_indices[i];
        uint32_t col_value = columns_packed[col_idx * domain_size + tid];

        // numerator += c * col_value - b
        QM31 term = qm31_sub(qm31_mul_m31(c, col_value), b);
        numerator = qm31_add(numerator, term);
    }

    // Write output as 4 separate M31 coordinates
    out_coord0[tid] = numerator.c0.a;
    out_coord1[tid] = numerator.c0.b;
    out_coord2[tid] = numerator.c1.a;
    out_coord3[tid] = numerator.c1.b;
}

/// Quotient combination kernel for QuotientOps::compute_quotients_and_combine.
///
/// For each domain point in the lifting domain:
///   quotient(p) = Σ_i (lifted_partial_num_i(p) - a_i * p.y) / denom_i(p)
///
/// Where denom_i = (sample_x_i - p.x).real * sample_y_i.imag
///              - (sample_y_i - p.y).real * sample_x_i.imag
///
/// Uses cm31_batch_inverse_inplace for per-thread batch denominator inversion.
/// Lifting formula (scalar): lifted_idx = (tid >> (log_ratio + 1) << 1) | (tid & 1)
///
/// Accumulator coords are packed as: for each accumulation i of size S_i,
///   coords stored at acc_data + acc_offsets[i], stride = S_i between coords.
///   Layout: [coord0_0..coord0_{S-1}, coord1_0..coord1_{S-1}, coord2_0..coord2_{S-1}, coord3_0..coord3_{S-1}]
kernel void quotient_combine(
    device const uint32_t* domain_x [[buffer(0)]],          // Lifting domain X coords (M31)
    device const uint32_t* domain_y [[buffer(1)]],          // Lifting domain Y coords (M31)
    device const uint32_t* acc_data [[buffer(2)]],          // All acc numerator coords packed
    device const uint32_t* acc_offsets [[buffer(3)]],       // Element offset per accumulation
    device const uint32_t* acc_log_sizes [[buffer(4)]],     // Log size per accumulation
    device const QM31* first_linear_terms [[buffer(5)]],    // 'a' coefficient per accumulation
    device const QM31* sample_points_x [[buffer(6)]],       // Sample point X coords (QM31)
    device const QM31* sample_points_y [[buffer(7)]],       // Sample point Y coords (QM31)
    constant uint32_t& lifting_log_size [[buffer(8)]],
    constant uint32_t& num_accumulations [[buffer(9)]],
    device uint32_t* out_coord0 [[buffer(10)]],
    device uint32_t* out_coord1 [[buffer(11)]],
    device uint32_t* out_coord2 [[buffer(12)]],
    device uint32_t* out_coord3 [[buffer(13)]],
    uint tid [[thread_position_in_grid]]
) {
    uint lifting_size = 1u << lifting_log_size;
    if (tid >= lifting_size) return;

    uint32_t dx_val = domain_x[tid];
    uint32_t dy_val = domain_y[tid];

    // Phase 1: Compute denominators for all accumulations
    CM31 denominators[32];
    for (uint i = 0; i < num_accumulations; i++) {
        QM31 sx = sample_points_x[i];
        QM31 sy = sample_points_y[i];

        CM31 dx = cm31_sub(sx.c0, cm31_from_m31(dx_val));
        CM31 dy = cm31_sub(sy.c0, cm31_from_m31(dy_val));

        denominators[i] = cm31_sub(cm31_mul(dx, sy.c1), cm31_mul(dy, sx.c1));
    }

    // Phase 2: Batch invert all denominators
    cm31_batch_inverse_inplace(denominators, num_accumulations);

    // Phase 3: Accumulate quotient
    QM31 quotient = qm31_zero();

    for (uint i = 0; i < num_accumulations; i++) {
        uint acc_log_size = acc_log_sizes[i];
        uint log_ratio = lifting_log_size - acc_log_size;
        uint acc_size = 1u << acc_log_size;

        // Compute lifted index: (tid >> (log_ratio + 1) << 1) | (tid & 1)
        uint lifted_idx;
        if (log_ratio == 0) {
            lifted_idx = tid;
        } else {
            lifted_idx = ((tid >> (log_ratio + 1)) << 1) | (tid & 1);
        }

        // Read partial numerator from packed coords
        uint base = acc_offsets[i];
        QM31 partial_num;
        partial_num.c0.a = acc_data[base + lifted_idx];
        partial_num.c0.b = acc_data[base + acc_size + lifted_idx];
        partial_num.c1.a = acc_data[base + 2 * acc_size + lifted_idx];
        partial_num.c1.b = acc_data[base + 3 * acc_size + lifted_idx];

        // full_numerator = partial_num - first_linear_term * domain_y
        QM31 a_times_y = qm31_mul_m31(first_linear_terms[i], dy_val);
        QM31 full_num = qm31_sub(partial_num, a_times_y);

        // quotient += full_num * denom_inv
        QM31 contribution = qm31_mul_cm31(full_num, denominators[i]);
        quotient = qm31_add(quotient, contribution);
    }

    // Write output as 4 separate M31 coordinates
    out_coord0[tid] = quotient.c0.a;
    out_coord1[tid] = quotient.c0.b;
    out_coord2[tid] = quotient.c1.a;
    out_coord3[tid] = quotient.c1.b;
}

/// Legacy one-pass quotient accumulation kernel (kept for pipeline compatibility).
/// The two-step kernels above (quotient_partial_numerator + quotient_combine)
/// are the active implementation matching the QuotientOps trait interface.
kernel void quotient_accumulate(
    device const uint32_t* domain_points_x [[buffer(0)]],
    device const uint32_t* domain_points_y [[buffer(1)]],
    device const uint32_t* columns [[buffer(2)]],
    constant uint32_t& num_columns [[buffer(3)]],
    constant uint32_t& domain_size [[buffer(4)]],
    device const uint32_t* column_indices [[buffer(5)]],
    device const QM31* line_coeffs [[buffer(6)]],
    device const QM31* sample_points_x [[buffer(7)]],
    device const QM31* sample_points_y [[buffer(8)]],
    device const uint32_t* batch_sizes [[buffer(9)]],
    constant uint32_t& num_batches [[buffer(10)]],
    device QM31* output [[buffer(11)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= domain_size) return;

    uint32_t domain_x = domain_points_x[tid];
    uint32_t domain_y = domain_points_y[tid];

    CM31 denominators[32];
    for (uint batch_idx = 0; batch_idx < num_batches; batch_idx++) {
        QM31 sample_x = sample_points_x[batch_idx];
        QM31 sample_y = sample_points_y[batch_idx];
        CM31 dx = cm31_sub(sample_x.c0, cm31_from_m31(domain_x));
        CM31 dy = cm31_sub(sample_y.c0, cm31_from_m31(domain_y));
        denominators[batch_idx] = cm31_sub(cm31_mul(dx, sample_y.c1), cm31_mul(dy, sample_x.c1));
    }
    cm31_batch_inverse_inplace(denominators, num_batches);

    QM31 accumulator = qm31_zero();
    uint line_coeff_offset = 0;
    for (uint batch_idx = 0; batch_idx < num_batches; batch_idx++) {
        uint batch_size = batch_sizes[batch_idx];
        CM31 denominator_inv = denominators[batch_idx];
        QM31 numerator = qm31_zero();
        for (uint col_offset = 0; col_offset < batch_size; col_offset++) {
            uint coeff_idx = line_coeff_offset + col_offset * 3;
            QM31 a = line_coeffs[coeff_idx + 0];
            QM31 b = line_coeffs[coeff_idx + 1];
            QM31 c = line_coeffs[coeff_idx + 2];
            uint col_idx = column_indices[line_coeff_offset / 3 + col_offset];
            uint32_t col_value = columns[col_idx * domain_size + tid];
            QM31 term = qm31_sub(qm31_mul_m31(c, col_value), qm31_add(qm31_mul_m31(a, domain_y), b));
            numerator = qm31_add(numerator, term);
        }
        accumulator = qm31_add(accumulator, qm31_mul_cm31(numerator, denominator_inv));
        line_coeff_offset += batch_size * 3;
    }
    output[tid] = accumulator;
}

