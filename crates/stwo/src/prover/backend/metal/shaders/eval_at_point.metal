// ============================================================================
// Circle Polynomial Evaluation at Points
// ============================================================================

/// CirclePoint in GPU memory: (x, y) where both are QM31
struct CirclePointQM31 {
    QM31 x;
    QM31 y;
};

/// Double the x-coordinate of a circle point: pi(x, y) = (2*x^2 - 1, y)
inline QM31 circle_double_x(QM31 x) {
    // Compute 2*x^2 - 1
    QM31 x_squared = qm31_mul(x, x);

    // Multiply by 2: we'll add x_squared to itself
    QM31 two_x_squared = qm31_add(x_squared, x_squared);

    // Subtract 1: QM31{1, 0, 0, 0}
    QM31 one = {CM31{1, 0}, CM31{0, 0}};
    return qm31_sub(two_x_squared, one);
}

/// Evaluate polynomial at a point using binary tree folding (Horner's method).
/// For a polynomial of size 2^n, we recursively split and fold:
///   eval(coeffs) = eval(left_half) + eval(right_half) * folding_factor
///
/// Folding factors are computed from the circle point as: [y, x, pi(x), pi^2(x), ...]
/// where pi(x) = 2*x^2 - 1 (doubling formula for circle points).
///
/// This kernel evaluates multiple polynomials at different points in a single dispatch.
/// Each thread processes one evaluation.
///
/// Parameters:
///   coeffs_base: Base pointer to coefficient arrays (M31 elements)
///   coeff_offsets: Offset in coeffs_base for each polynomial (in M31 elements)
///   log_sizes: log2(size) of each polynomial
///   points: Array of CirclePointQM31 evaluation points
///   results: Output array of QM31 values (one per point)
///   num_evals: Total number of evaluations to perform
kernel void circle_eval_at_point(
    device const uint32_t* coeffs_base [[buffer(0)]],
    device const uint32_t* coeff_offsets [[buffer(1)]],
    device const uint32_t* log_sizes [[buffer(2)]],
    device const CirclePointQM31* points [[buffer(3)]],
    device QM31* results [[buffer(4)]],
    device const uint32_t* num_evals [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint eval_idx = gid;
    if (eval_idx >= *num_evals) {
        return;
    }

    // Get parameters for this evaluation
    uint coeff_offset = coeff_offsets[eval_idx];
    uint log_size = log_sizes[eval_idx];
    CirclePointQM31 point = points[eval_idx];
    uint size = 1u << log_size;

    device const uint32_t* coeffs = coeffs_base + coeff_offset;

    // Compute folding factors in the order expected by the iterative fold:
    // [pi^(n-2)(x), ..., pi(x), x, y]
    // where pi(x) = 2*x^2 - 1 is the circle doubling map.
    //
    // Factor at index 0 splits the top level, index n-1 splits pairs.
    // This matches the reference fold() in poly/utils.rs.
    QM31 folding_factors[32];  // Max log_size we support
    if (log_size > 32) {
        results[eval_idx] = qm31_zero();
        return;
    }

    if (log_size >= 1) {
        // Build [y, x, pi(x), pi^2(x), ..., pi^(n-2)(x)]
        // then reverse to get [pi^(n-2)(x), ..., pi(x), x, y]
        QM31 mappings[32];
        mappings[0] = point.y;
        if (log_size > 1) {
            mappings[1] = point.x;
            QM31 x_power = point.x;
            for (uint i = 2; i < log_size; i++) {
                x_power = circle_double_x(x_power);
                mappings[i] = x_power;
            }
        }
        // Reverse: mappings was [y, x, pi(x), ...], we want [pi^(n-2)(x), ..., x, y]
        for (uint i = 0; i < log_size; i++) {
            folding_factors[i] = mappings[log_size - 1 - i];
        }
    }

    // Recursive folding using binary tree
    // We'll use a workspace to avoid recursion (GPU doesn't support recursion well)
    // Allocate workspace for intermediate results
    QM31 workspace[1024];  // Max 2^10 = 1024 elements per level

    if (size > 1024) {
        // For very large polynomials, we need a different approach
        // For now, just handle up to 2^10 = 1024
        results[eval_idx] = qm31_zero();
        return;
    }

    // Base case: copy coeffs to workspace, converting M31 to QM31
    for (uint i = 0; i < size; i++) {
        workspace[i] = QM31{CM31{coeffs[i], 0}, CM31{0, 0}};
    }

    // Iterative folding: at each level, fold pairs
    uint current_size = size;
    for (uint level = 0; level < log_size; level++) {
        uint half_size = current_size / 2;
        QM31 folding_factor = folding_factors[level];

        for (uint i = 0; i < half_size; i++) {
            QM31 left = workspace[i];
            QM31 right = workspace[half_size + i];
            // fold: left + right * folding_factor
            workspace[i] = qm31_add(left, qm31_mul(right, folding_factor));
        }

        current_size = half_size;
    }

    results[eval_idx] = workspace[0];
}

