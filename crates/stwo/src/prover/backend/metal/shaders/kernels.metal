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

// ============================================================================
// BLAKE2s Hashing
// ============================================================================

// BLAKE2s initialization vectors
constant uint32_t BLAKE2S_IV[8] = {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
};

// Precomputed initial state for Merkle tree nodes
// This is the state after compressing the NODE_PREFIX ("node" + zeros)
constant uint32_t BLAKE2S_NODE_INITIAL_STATE[8] = {
    0xe5cf8926, 0x841cea30, 0x7b4acada, 0xfc5d8d28,
    0xfc6ef857, 0xb29da528, 0xc0d319c7, 0x8ae795c8
};

// Precomputed initial state for Merkle tree leaves
// This is the state after compressing the LEAF_PREFIX ("leaf" + zeros)
constant uint32_t BLAKE2S_LEAF_INITIAL_STATE[8] = {
    0x3fa5003d, 0x8ff3be4a, 0x2b843d58, 0xe0766c2d,
    0x5ca9b993, 0xa5c8cc74, 0x12f184e0, 0xd86f6c9e
};

// BLAKE2s message permutation schedule (SIGMA)
constant uint8_t BLAKE2S_SIGMA[10][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
    {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
    {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
    {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
    {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
    {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
    {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
    {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
    {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0}
};

/// Rotate right (bitwise rotation)
inline uint32_t rotate_right(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

/// BLAKE2s mixing function G
/// Mixes four words in the state using two message words
inline void blake2s_g(
    thread uint32_t& a, thread uint32_t& b,
    thread uint32_t& c, thread uint32_t& d,
    uint32_t x, uint32_t y
) {
    a = a + b + x;
    d = rotate_right(d ^ a, 16u);
    c = c + d;
    b = rotate_right(b ^ c, 12u);
    a = a + b + y;
    d = rotate_right(d ^ a, 8u);
    c = c + d;
    b = rotate_right(b ^ c, 7u);
}

/// BLAKE2s compression function
/// Compresses a 64-byte block into the 32-byte state
inline void blake2s_compress(
    thread uint32_t state[8],
    thread const uint32_t* message,
    uint32_t t0,  // Counter low word
    uint32_t t1,  // Counter high word
    bool is_last  // Last block flag
) {
    uint32_t v[16];

    // Initialize working variables
    for (int i = 0; i < 8; i++) {
        v[i] = state[i];
        v[i + 8] = BLAKE2S_IV[i];
    }

    // Mix counter into v[12:13]
    v[12] ^= t0;
    v[13] ^= t1;

    // Invert v[14] if last block
    if (is_last) {
        v[14] = ~v[14];
    }

    // 10 rounds of mixing
    for (int round = 0; round < 10; round++) {
        // Column step
        blake2s_g(v[0], v[4], v[8],  v[12], message[BLAKE2S_SIGMA[round][0]], message[BLAKE2S_SIGMA[round][1]]);
        blake2s_g(v[1], v[5], v[9],  v[13], message[BLAKE2S_SIGMA[round][2]], message[BLAKE2S_SIGMA[round][3]]);
        blake2s_g(v[2], v[6], v[10], v[14], message[BLAKE2S_SIGMA[round][4]], message[BLAKE2S_SIGMA[round][5]]);
        blake2s_g(v[3], v[7], v[11], v[15], message[BLAKE2S_SIGMA[round][6]], message[BLAKE2S_SIGMA[round][7]]);

        // Diagonal step
        blake2s_g(v[0], v[5], v[10], v[15], message[BLAKE2S_SIGMA[round][8]],  message[BLAKE2S_SIGMA[round][9]]);
        blake2s_g(v[1], v[6], v[11], v[12], message[BLAKE2S_SIGMA[round][10]], message[BLAKE2S_SIGMA[round][11]]);
        blake2s_g(v[2], v[7], v[8],  v[13], message[BLAKE2S_SIGMA[round][12]], message[BLAKE2S_SIGMA[round][13]]);
        blake2s_g(v[3], v[4], v[9],  v[14], message[BLAKE2S_SIGMA[round][14]], message[BLAKE2S_SIGMA[round][15]]);
    }

    // Finalization: XOR the two halves
    for (int i = 0; i < 8; i++) {
        state[i] ^= v[i] ^ v[i + 8];
    }
}

/// Initialize BLAKE2s state for standard hashing (32-byte output)
inline void blake2s_init(thread uint32_t state[8]) {
    for (int i = 0; i < 8; i++) {
        state[i] = BLAKE2S_IV[i];
    }
    // XOR first word with parameter block: hash_length=32, key_length=0
    state[0] ^= 0x01010020;
}

/// Reduce BLAKE2s hash output modulo M31 for each u32
inline void blake2s_reduce_m31(thread uint32_t hash[8]) {
    for (int i = 0; i < 8; i++) {
        hash[i] = m31_reduce(hash[i]);
    }
}

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
    for (uint i = count - 1; i > 0; i--) {
        values[i] = cm31_mul(products[i - 1], inverse);
        inverse = cm31_mul(inverse, values[i]);
    }
    values[0] = inverse;
}

/// Quotient accumulation kernel with GPU-native batch inversion.
///
/// Computes Q(x) = Σ_i (random_coeff^i * (P_i(x) - y_i) / (x - x_i))
///
/// For each domain point, accumulates quotient terms from all sample batches.
/// Uses Montgomery's trick to batch-invert denominators (GPU-native, no CPU round-trip).
///
/// Algorithmic improvement: O(num_batches) inversions per thread → O(1) inversion per thread
///
/// Layout:
/// - domain_points_x/y: Domain point coordinates (BaseField), bit-reversed order
/// - columns: Flattened column data (BaseField values)
/// - column_indices: Which column each line_coeff corresponds to
/// - line_coeffs: Flattened (a, b, c) coefficients as QM31 values
/// - sample_points_x/y: Sample point coordinates (QM31)
/// - batch_sizes: Number of columns in each sample batch
/// - output: Accumulated quotient (QM31)
kernel void quotient_accumulate(
    device const uint32_t* domain_points_x [[buffer(0)]],  // Domain X coords (M31)
    device const uint32_t* domain_points_y [[buffer(1)]],  // Domain Y coords (M31)
    device const uint32_t* columns [[buffer(2)]],           // All column data (M31)
    constant uint32_t& num_columns [[buffer(3)]],           // Total number of columns
    constant uint32_t& domain_size [[buffer(4)]],           // Size of domain
    device const uint32_t* column_indices [[buffer(5)]],    // Column index for each coeff
    device const QM31* line_coeffs [[buffer(6)]],           // Flattened (a,b,c) triplets
    device const QM31* sample_points_x [[buffer(7)]],       // Sample X coords (QM31)
    device const QM31* sample_points_y [[buffer(8)]],       // Sample Y coords (QM31)
    device const uint32_t* batch_sizes [[buffer(9)]],       // Columns per batch
    constant uint32_t& num_batches [[buffer(10)]],          // Number of sample batches
    device QM31* output [[buffer(11)]],                     // Output (QM31)
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= domain_size) {
        return;
    }

    // Get domain point coordinates (BaseField)
    uint32_t domain_x = domain_points_x[tid];
    uint32_t domain_y = domain_points_y[tid];

    // PHASE 1: Compute all denominators for this domain point
    // Stack-allocate array for denominators (num_batches is typically ~5-20)
    CM31 denominators[32];  // Support up to 32 batches

    for (uint batch_idx = 0; batch_idx < num_batches; batch_idx++) {
        // Get sample point (QM31)
        QM31 sample_x = sample_points_x[batch_idx];
        QM31 sample_y = sample_points_y[batch_idx];

        // Compute denominator
        // denominator = (sample.x - domain.x) * sample.y.1 - (sample.y - domain.y) * sample.x.1
        // Where sample.x = (sample.x.0, sample.x.1) in CM31
        CM31 sample_xr = sample_x.c0;  // Real part
        CM31 sample_xi = sample_x.c1;  // Imaginary part
        CM31 sample_yr = sample_y.c0;
        CM31 sample_yi = sample_y.c1;

        CM31 dx = cm31_sub(sample_xr, cm31_from_m31(domain_x));
        CM31 dy = cm31_sub(sample_yr, cm31_from_m31(domain_y));

        denominators[batch_idx] = cm31_sub(cm31_mul(dx, sample_yi), cm31_mul(dy, sample_xi));
    }

    // PHASE 2: Batch invert all denominators using Montgomery's trick
    // Reduces O(num_batches) individual inversions to O(1) + overhead
    cm31_batch_inverse_inplace(denominators, num_batches);

    // PHASE 3: Accumulate quotient using pre-computed inverses
    QM31 accumulator = qm31_zero();
    uint line_coeff_offset = 0;

    for (uint batch_idx = 0; batch_idx < num_batches; batch_idx++) {
        uint batch_size = batch_sizes[batch_idx];

        // Get pre-computed denominator inverse
        CM31 denominator_inv = denominators[batch_idx];

        // Accumulate numerator for this batch
        QM31 numerator = qm31_zero();

        for (uint col_offset = 0; col_offset < batch_size; col_offset++) {
            uint coeff_idx = line_coeff_offset + col_offset * 3;
            QM31 a = line_coeffs[coeff_idx + 0];
            QM31 b = line_coeffs[coeff_idx + 1];
            QM31 c = line_coeffs[coeff_idx + 2];

            // Get column index and value
            uint col_idx = column_indices[line_coeff_offset / 3 + col_offset];
            uint32_t col_value = columns[col_idx * domain_size + tid];

            // Compute: c * value - (a * domain_y + b)
            QM31 c_times_value = qm31_mul_m31(c, col_value);
            QM31 a_times_y = qm31_mul_m31(a, domain_y);
            QM31 linear_term = qm31_add(a_times_y, b);
            QM31 term = qm31_sub(c_times_value, linear_term);

            numerator = qm31_add(numerator, term);
        }

        // Multiply numerator by denominator inverse
        QM31 batch_contribution = qm31_mul_cm31(numerator, denominator_inv);
        accumulator = qm31_add(accumulator, batch_contribution);

        line_coeff_offset += batch_size * 3;
    }

    // Write output
    output[tid] = accumulator;
}

// ============================================================================
// Merkle Kernels
// ============================================================================

/// Merkle tree internal node hashing kernel
/// Hashes two child hashes together: parent = BLAKE2s(left || right)
/// Each hash is 32 bytes (8 x uint32_t)
kernel void merkle_blake2s(
    device const uint32_t* children [[buffer(0)]],  // Input: child hashes (size * 2 * 8 u32s)
    device uint32_t* parents [[buffer(1)]],          // Output: parent hashes (size * 8 u32s)
    constant bool& is_m31_output [[buffer(2)]],     // Whether to reduce modulo M31
    constant uint32_t& size [[buffer(3)]],          // Number of parent nodes
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= size) return;

    // Each parent node hashes two 32-byte children
    // Message layout: [left_hash (32 bytes), right_hash (32 bytes)] = 64 bytes = 16 u32s
    uint32_t message[16];

    // Load left child (32 bytes = 8 u32s)
    for (int i = 0; i < 8; i++) {
        message[i] = children[(tid * 2) * 8 + i];
    }

    // Load right child (32 bytes = 8 u32s)
    for (int i = 0; i < 8; i++) {
        message[8 + i] = children[(tid * 2 + 1) * 8 + i];
    }

    // Initialize BLAKE2s state with precomputed NODE_INITIAL_STATE
    // This state is already the result of compressing the NODE_PREFIX
    uint32_t state[8];
    for (int i = 0; i < 8; i++) {
        state[i] = BLAKE2S_NODE_INITIAL_STATE[i];
    }

    // Compress the 64-byte message (two child hashes)
    // Counter is 128 because we've already processed 64 bytes (NODE_PREFIX)
    blake2s_compress(state, message, 128, 0, true);

    // Optionally reduce output modulo M31
    if (is_m31_output) {
        blake2s_reduce_m31(state);
    }

    // Write output hash (32 bytes = 8 u32s)
    for (int i = 0; i < 8; i++) {
        parents[tid * 8 + i] = state[i];
    }
}

/// Merkle tree leaf hashing kernel
/// Hashes column values: leaf_hash = BLAKE2s(LEAF_PREFIX || column_values)
/// Supports multi-column hashing where each leaf contains multiple M31 values
kernel void merkle_blake2s_leaf(
    device const uint32_t* columns [[buffer(0)]],   // Flattened column data (M31 values)
    constant uint32_t& num_columns [[buffer(1)]],   // Number of columns
    constant uint32_t& domain_size [[buffer(2)]],   // Domain size
    constant bool& is_m31_output [[buffer(3)]],     // Whether to reduce modulo M31
    device uint32_t* output [[buffer(4)]],          // Output hashes (domain_size * 8 u32s)
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= domain_size) return;

    // Initialize with LEAF_INITIAL_STATE
    uint32_t state[8];
    for (int i = 0; i < 8; i++) {
        state[i] = BLAKE2S_LEAF_INITIAL_STATE[i];
    }

    // Collect column values for this domain index
    // Message is: column0[tid], column1[tid], ..., columnN[tid]
    // Each column value is 4 bytes (uint32_t)
    uint32_t message[16];
    uint32_t num_values = num_columns;

    // Load column values into message buffer
    for (uint32_t col_idx = 0; col_idx < num_columns && col_idx < 16; col_idx++) {
        message[col_idx] = columns[col_idx * domain_size + tid];
    }

    // Zero-pad remaining message slots if needed
    for (uint32_t i = num_columns; i < 16; i++) {
        message[i] = 0;
    }

    // Compress the message
    // Counter is 64 (LEAF_PREFIX) + num_columns * 4 (column bytes)
    uint32_t counter = 64 + num_columns * 4;
    blake2s_compress(state, message, counter, 0, true);

    // Optionally reduce output modulo M31
    if (is_m31_output) {
        blake2s_reduce_m31(state);
    }

    // Write output hash
    for (int i = 0; i < 8; i++) {
        output[tid * 8 + i] = state[i];
    }
}

// ============================================================================
// Proof-of-Work Grinding Kernels
// ============================================================================

/// Proof-of-work grinding kernel
/// Searches for a nonce that produces a hash with at least pow_bits trailing zeros
/// Each thread tries a range of nonces
kernel void grind_pow(
    device const uint32_t* digest [[buffer(0)]],     // Prefix digest (8 u32s)
    device uint64_t* result [[buffer(1)]],           // Output: found nonce (or UINT64_MAX if not found)
    constant uint32_t& pow_bits [[buffer(2)]],       // Required trailing zero bits
    constant uint64_t& start_nonce [[buffer(3)]],    // Starting nonce for this batch
    constant uint32_t& batch_size [[buffer(4)]],     // Nonces to try per thread
    constant bool& is_m31_output [[buffer(5)]],      // Whether to reduce modulo M31
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
            // TODO(correctness): This has a race condition! Two separate atomic_fetch_min
            // operations on low/high words are not atomic as a whole. Two threads could
            // write inconsistent halves (e.g., low from thread A, high from thread B).
            //
            // Mitigation: The result is verified by verify_pow_nonce on the host, so
            // invalid nonces from races will be rejected.
            //
            // Proper fix: Either use atomic_ulong (64-bit atomic) or collect multiple
            // candidate nonces in a buffer and reduce on CPU.
            atomic_fetch_min_explicit(
                (device atomic_uint*)result,
                uint(nonce),
                memory_order_relaxed
            );
            // Also store high word
            atomic_fetch_min_explicit(
                (device atomic_uint*)(result) + 1,
                uint(nonce >> 32),
                memory_order_relaxed
            );
            return;  // Found a solution, stop searching
        }
    }
}

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

    // Compute folding factors: [y, x, pi(x), pi^2(x), ...]
    // We need log_size factors total
    QM31 folding_factors[32];  // Max log_size we support
    if (log_size > 32) {
        // Error: polynomial too large
        results[eval_idx] = qm31_zero();
        return;
    }

    // Reverse order: folding_factors[0] = pi^(n-1)(x), ..., folding_factors[n-1] = y
    if (log_size >= 1) {
        QM31 x_power = point.x;

        // Compute pi^(n-2)(x), pi^(n-3)(x), ..., pi(x), x in reverse
        for (uint i = 0; i < log_size - 1; i++) {
            folding_factors[i] = x_power;
            x_power = circle_double_x(x_power);
        }

        // Last factor is y
        folding_factors[log_size - 1] = point.y;
    }

    // Reverse the folding_factors array to get [y, x, pi(x), ...]
    for (uint i = 0; i < log_size / 2; i++) {
        QM31 tmp = folding_factors[i];
        folding_factors[i] = folding_factors[log_size - 1 - i];
        folding_factors[log_size - 1 - i] = tmp;
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
        workspace[i] = qm31_mul_m31(qm31_zero(), 1);  // Zero
        workspace[i].c0.a = coeffs[i];  // Set real part of first CM31
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
