// ============================================================================
// Lookup (MLE) Kernels
// ============================================================================

/// MLE fix_first_variable for BaseField → SecureField.
/// Computes: result[i] = assignment * (input[i + midpoint] - input[i]) + input[i]
/// Input: M31 array of length 2*N
/// Output: QM31 array of length N
kernel void mle_fold_m31_to_qm31(
    device const uint32_t* input [[buffer(0)]],     // M31 values (length 2*N)
    device QM31* output [[buffer(1)]],               // QM31 values (length N)
    constant QM31& assignment [[buffer(2)]],         // QM31 assignment value
    constant uint32_t& log_size [[buffer(3)]],       // log2(2*N) = log2 of input length
    uint tid [[thread_position_in_grid]]
) {
    uint32_t n_half = 1u << (log_size - 1);
    if (tid >= n_half) return;

    // Read M31 values at positions i and i + n_half
    uint32_t eval0_m31 = input[tid];
    uint32_t eval1_m31 = input[tid + n_half];

    // Convert M31 to QM31 (real part only)
    QM31 eval0 = {{eval0_m31, 0}, {0, 0}};
    QM31 eval1 = {{eval1_m31, 0}, {0, 0}};

    // Compute: assignment * (eval1 - eval0) + eval0
    QM31 diff = qm31_sub(eval1, eval0);
    QM31 prod = qm31_mul(assignment, diff);
    QM31 result = qm31_add(prod, eval0);

    output[tid] = result;
}

/// MLE fix_first_variable for SecureField → SecureField.
/// Computes: result[i] = assignment * (input[i + midpoint] - input[i]) + input[i]
/// Input: QM31 array of length 2*N
/// Output: QM31 array of length N
kernel void mle_fold_qm31_to_qm31(
    device const QM31* input [[buffer(0)]],          // QM31 values (length 2*N)
    device QM31* output [[buffer(1)]],               // QM31 values (length N)
    constant QM31& assignment [[buffer(2)]],         // QM31 assignment value
    constant uint32_t& log_size [[buffer(3)]],       // log2(2*N) = log2 of input length
    uint tid [[thread_position_in_grid]]
) {
    uint32_t n_half = 1u << (log_size - 1);
    if (tid >= n_half) return;

    // Read QM31 values at positions i and i + n_half
    QM31 eval0 = input[tid];
    QM31 eval1 = input[tid + n_half];

    // Compute: assignment * (eval1 - eval0) + eval0
    QM31 diff = qm31_sub(eval1, eval0);
    QM31 prod = qm31_mul(assignment, diff);
    QM31 result = qm31_add(prod, eval0);

    output[tid] = result;
}

