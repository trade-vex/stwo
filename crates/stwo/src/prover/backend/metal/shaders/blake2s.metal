// ============================================================================
// BLAKE2s Hashing
// ============================================================================

// BLAKE2s initialization vectors
constant uint32_t BLAKE2S_IV[8] = {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
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

