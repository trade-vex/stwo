//! Metal compute kernels for Stwo prover operations.
//!
//! All kernels operate on the Mersenne-31 prime field (p = 2^31 - 1).
//! Field elements are represented as uint32_t in range [0, 2^31 - 1].

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Constants and Field Arithmetic
// ============================================================================

constant uint32_t M31_PRIME = 0x7FFFFFFF;  // 2^31 - 1

/// Reduce a uint32_t to the range [0, M31_PRIME).
inline uint32_t m31_reduce(uint32_t x) {
    // Since 2^31 ≡ 1 (mod 2^31 - 1), we can reduce by:
    // x = (x & M31_PRIME) + (x >> 31)
    uint32_t reduced = (x & M31_PRIME) + (x >> 31);
    // May need one more reduction if reduced >= M31_PRIME
    // This can happen when x >> 31 == 1 and (x & M31_PRIME) == M31_PRIME
    if (reduced >= M31_PRIME) {
        reduced -= M31_PRIME;
    }
    return reduced;
}

/// Add two M31 field elements.
inline uint32_t m31_add(uint32_t a, uint32_t b) {
    return m31_reduce(a + b);
}

/// Subtract two M31 field elements.
inline uint32_t m31_sub(uint32_t a, uint32_t b) {
    // Note: M31_PRIME + a - b is always non-negative
    return m31_reduce(M31_PRIME + a - b);
}

/// Multiply two M31 field elements.
inline uint32_t m31_mul(uint32_t a, uint32_t b) {
    uint64_t product = uint64_t(a) * uint64_t(b);
    // Reduce: product mod (2^31 - 1)
    uint32_t low = uint32_t(product) & M31_PRIME;
    uint32_t high = uint32_t(product >> 31);
    return m31_reduce(low + high);
}

/// Multiply M31 element by a doubled twiddle factor.
/// Twiddles are stored as 2*value for optimization.
inline uint32_t m31_mul_twiddle_dbl(uint32_t a, uint32_t b_dbl) {
    uint64_t product = uint64_t(a) * uint64_t(b_dbl);
    // Since b_dbl = 2*b, product = 2*a*b
    // Shift right by 1 to get a*b, then reduce
    uint64_t shifted = product >> 1;
    // Extract low 31 bits and high bits properly from the 64-bit value
    uint32_t low = uint32_t(shifted & M31_PRIME);
    uint32_t high = uint32_t(shifted >> 31);
    return m31_reduce(low + high);
}

/// Square M31 element: v^2
inline uint32_t m31_square(uint32_t v) {
    return m31_mul(v, v);
}

/// Compute v^(2^n) by repeated squaring
inline uint32_t m31_sqn(uint32_t v, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        v = m31_square(v);
    }
    return v;
}

/// Compute M31 multiplicative inverse using optimized exponentiation chain.
/// Computes v^(p-2) = v^(2^31 - 3) using only 37 multiplications.
/// Algorithm from addchain optimization for Mersenne-31 field.
inline uint32_t m31_inverse(uint32_t v) {
    // Optimized addition chain for computing v^(2^31 - 3)
    uint32_t t0 = m31_mul(m31_sqn(v, 2), v);           // v^(2^2) * v = v^5
    uint32_t t1 = m31_mul(m31_sqn(t0, 1), t0);         // v^10 * v^5 = v^15
    uint32_t t2 = m31_mul(m31_sqn(t1, 3), t0);         // v^120 * v^5 = v^125
    uint32_t t3 = m31_mul(m31_sqn(t2, 1), t0);         // v^250 * v^5 = v^255
    uint32_t t4 = m31_mul(m31_sqn(t3, 8), t3);         // v^(255*256) * v^255 = v^65535
    uint32_t t5 = m31_mul(m31_sqn(t4, 8), t3);         // v^(65535*256) * v^255 = v^16777215
    return m31_mul(m31_sqn(t5, 7), t2);                // v^(16777215*128) * v^125 = v^(2^31-3)
}

// ============================================================================
// CM31 and QM31 Field Arithmetic (Complex and Secure Fields)
// ============================================================================

/// CM31: Complex extension of M31, represented as a + bi where i^2 = -1
struct CM31 {
    uint32_t a;  // Real part
    uint32_t b;  // Imaginary part
};

/// QM31: Quadratic extension of CM31, the secure field
/// Represented as (a+bi) + (c+di)u where u^2 = 2+i
struct QM31 {
    CM31 c0;  // First CM31 component (a+bi)
    CM31 c1;  // Second CM31 component (c+di)
};

/// Add two CM31 elements
inline CM31 cm31_add(CM31 x, CM31 y) {
    return CM31{m31_add(x.a, y.a), m31_add(x.b, y.b)};
}

/// Subtract two CM31 elements
inline CM31 cm31_sub(CM31 x, CM31 y) {
    return CM31{m31_sub(x.a, y.a), m31_sub(x.b, y.b)};
}

/// Multiply two CM31 elements: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
inline CM31 cm31_mul(CM31 x, CM31 y) {
    uint32_t real = m31_sub(m31_mul(x.a, y.a), m31_mul(x.b, y.b));
    uint32_t imag = m31_add(m31_mul(x.a, y.b), m31_mul(x.b, y.a));
    return CM31{real, imag};
}

/// Multiply CM31 by a doubled twiddle (for optimization)
inline CM31 cm31_mul_twiddle_dbl(CM31 x, CM31 y_dbl) {
    uint32_t real = m31_sub(m31_mul_twiddle_dbl(x.a, y_dbl.a), m31_mul_twiddle_dbl(x.b, y_dbl.b));
    uint32_t imag = m31_add(m31_mul_twiddle_dbl(x.a, y_dbl.b), m31_mul_twiddle_dbl(x.b, y_dbl.a));
    return CM31{real, imag};
}

/// Add two QM31 elements
inline QM31 qm31_add(QM31 x, QM31 y) {
    return QM31{cm31_add(x.c0, y.c0), cm31_add(x.c1, y.c1)};
}

/// Subtract two QM31 elements
inline QM31 qm31_sub(QM31 x, QM31 y) {
    return QM31{cm31_sub(x.c0, y.c0), cm31_sub(x.c1, y.c1)};
}

/// Multiply two QM31 elements
/// (a + bu)(c + du) = (ac + R·bd) + (ad + bc)u where R = 2+i
inline QM31 qm31_mul(QM31 x, QM31 y) {
    // R = 2 + i
    const CM31 R = {2, 1};

    // ac + R·bd
    CM31 ac = cm31_mul(x.c0, y.c0);
    CM31 bd = cm31_mul(x.c1, y.c1);
    CM31 R_bd = cm31_mul(R, bd);
    CM31 c0 = cm31_add(ac, R_bd);

    // ad + bc
    CM31 ad = cm31_mul(x.c0, y.c1);
    CM31 bc = cm31_mul(x.c1, y.c0);
    CM31 c1 = cm31_add(ad, bc);

    return QM31{c0, c1};
}

/// Multiply QM31 by a scalar (broadcast to all components)
inline QM31 qm31_mul_scalar(QM31 x, QM31 alpha) {
    return qm31_mul(x, alpha);
}

/// Create CM31 from BaseField (M31) value
inline CM31 cm31_from_m31(uint32_t x) {
    return CM31{x, 0};
}

/// Create zero QM31
inline QM31 qm31_zero() {
    return QM31{CM31{0, 0}, CM31{0, 0}};
}

/// Multiply QM31 by BaseField (M31) scalar
inline QM31 qm31_mul_m31(QM31 x, uint32_t scalar) {
    return QM31{
        CM31{m31_mul(x.c0.a, scalar), m31_mul(x.c0.b, scalar)},
        CM31{m31_mul(x.c1.a, scalar), m31_mul(x.c1.b, scalar)}
    };
}

/// Multiply QM31 by CM31
/// (c0 + c1*u) * cm31 = (c0 * cm31) + (c1 * cm31)*u
inline QM31 qm31_mul_cm31(QM31 x, CM31 y) {
    return QM31{cm31_mul(x.c0, y), cm31_mul(x.c1, y)};
}

/// Compute CM31 inverse using extended Euclidean algorithm
/// For a + bi, inverse is (a - bi) / (a^2 + b^2)
inline CM31 cm31_inverse(CM31 x) {
    // Compute norm: a^2 + b^2
    uint32_t a_sq = m31_mul(x.a, x.a);
    uint32_t b_sq = m31_mul(x.b, x.b);
    uint32_t norm = m31_add(a_sq, b_sq);

    // Compute inverse of norm
    uint32_t norm_inv = m31_inverse(norm);

    // Compute conjugate: (a - bi)
    uint32_t neg_b = m31_sub(0, x.b);

    // Multiply conjugate by inverse of norm
    return CM31{
        m31_mul(x.a, norm_inv),
        m31_mul(neg_b, norm_inv)
    };
}

