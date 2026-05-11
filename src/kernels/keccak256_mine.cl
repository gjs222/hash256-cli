/*
 * keccak256_mine.cl
 *
 * OpenCL kernel: each work-item tries one nonce candidate.
 * Input layout (64 bytes):
 *   bytes[ 0.. 31] = challenge (32 bytes, from getChallenge(address))
 *   bytes[32.. 55] = nonce_prefix (24 bytes, fixed per worker launch)
 *   bytes[56.. 63] = base_counter + get_global_id(0)  (big-endian uint64)
 *
 * Success condition: keccak256(input) < difficulty  (big-endian uint256 compare)
 *
 * On hit: found_flag is set to 1 and found_counter holds the winning value.
 * Multiple simultaneous hits are harmless — any valid nonce is acceptable.
 */

/* ── Keccak-f[1600] round constants ─────────────────────────────────────── */
__constant ulong RC[24] = {
    0x0000000000000001UL, 0x0000000000008082UL, 0x800000000000808aUL,
    0x8000000080008000UL, 0x000000000000808bUL, 0x0000000080000001UL,
    0x8000000080008081UL, 0x8000000000008009UL, 0x000000000000008aUL,
    0x0000000000000088UL, 0x0000000080008009UL, 0x000000008000000aUL,
    0x000000008000808bUL, 0x800000000000008bUL, 0x8000000000008089UL,
    0x8000000000008003UL, 0x8000000000008002UL, 0x8000000000000080UL,
    0x000000000000800aUL, 0x800000008000000aUL, 0x8000000080008081UL,
    0x8000000000008080UL, 0x0000000080000001UL, 0x8000000080008008UL,
};

#define ROTL64(x, n) (((x) << (n)) | ((x) >> (64u - (n))))

/* ── Keccak-f[1600] permutation ─────────────────────────────────────────── */
void keccak_f1600(ulong *st) {
    ulong C0, C1, C2, C3, C4;
    ulong D0, D1, D2, D3, D4;
    ulong B00, B01, B02, B03, B04;
    ulong B05, B06, B07, B08, B09;
    ulong B10, B11, B12, B13, B14;
    ulong B15, B16, B17, B18, B19;
    ulong B20, B21, B22, B23, B24;

    for (int r = 0; r < 24; r++) {
        /* θ */
        C0 = st[ 0] ^ st[ 5] ^ st[10] ^ st[15] ^ st[20];
        C1 = st[ 1] ^ st[ 6] ^ st[11] ^ st[16] ^ st[21];
        C2 = st[ 2] ^ st[ 7] ^ st[12] ^ st[17] ^ st[22];
        C3 = st[ 3] ^ st[ 8] ^ st[13] ^ st[18] ^ st[23];
        C4 = st[ 4] ^ st[ 9] ^ st[14] ^ st[19] ^ st[24];

        D0 = C4 ^ ROTL64(C1, 1);
        D1 = C0 ^ ROTL64(C2, 1);
        D2 = C1 ^ ROTL64(C3, 1);
        D3 = C2 ^ ROTL64(C4, 1);
        D4 = C3 ^ ROTL64(C0, 1);

        st[ 0] ^= D0; st[ 5] ^= D0; st[10] ^= D0; st[15] ^= D0; st[20] ^= D0;
        st[ 1] ^= D1; st[ 6] ^= D1; st[11] ^= D1; st[16] ^= D1; st[21] ^= D1;
        st[ 2] ^= D2; st[ 7] ^= D2; st[12] ^= D2; st[17] ^= D2; st[22] ^= D2;
        st[ 3] ^= D3; st[ 8] ^= D3; st[13] ^= D3; st[18] ^= D3; st[23] ^= D3;
        st[ 4] ^= D4; st[ 9] ^= D4; st[14] ^= D4; st[19] ^= D4; st[24] ^= D4;

        /* ρ + π */
        B00 =            st[ 0];
        B10 = ROTL64(st[ 1],  1); B20 = ROTL64(st[ 2], 62);
        B05 = ROTL64(st[ 3], 28); B15 = ROTL64(st[ 4], 27);
        B16 = ROTL64(st[ 5], 36); B01 = ROTL64(st[ 6], 44);
        B11 = ROTL64(st[ 7],  6); B21 = ROTL64(st[ 8], 55);
        B06 = ROTL64(st[ 9], 20); B07 = ROTL64(st[10],  3);
        B17 = ROTL64(st[11], 10); B02 = ROTL64(st[12], 43);
        B12 = ROTL64(st[13], 25); B22 = ROTL64(st[14], 39);
        B23 = ROTL64(st[15], 41); B08 = ROTL64(st[16], 45);
        B18 = ROTL64(st[17], 15); B03 = ROTL64(st[18], 21);
        B13 = ROTL64(st[19],  8); B14 = ROTL64(st[20], 18);
        B24 = ROTL64(st[21],  2); B09 = ROTL64(st[22], 61);
        B19 = ROTL64(st[23], 56); B04 = ROTL64(st[24], 14);

        /* χ */
        st[ 0] = B00 ^ (~B01 & B02); st[ 1] = B01 ^ (~B02 & B03);
        st[ 2] = B02 ^ (~B03 & B04); st[ 3] = B03 ^ (~B04 & B00);
        st[ 4] = B04 ^ (~B00 & B01);
        st[ 5] = B05 ^ (~B06 & B07); st[ 6] = B06 ^ (~B07 & B08);
        st[ 7] = B07 ^ (~B08 & B09); st[ 8] = B08 ^ (~B09 & B05);
        st[ 9] = B09 ^ (~B05 & B06);
        st[10] = B10 ^ (~B11 & B12); st[11] = B11 ^ (~B12 & B13);
        st[12] = B12 ^ (~B13 & B14); st[13] = B13 ^ (~B14 & B10);
        st[14] = B14 ^ (~B10 & B11);
        st[15] = B15 ^ (~B16 & B17); st[16] = B16 ^ (~B17 & B18);
        st[17] = B17 ^ (~B18 & B19); st[18] = B18 ^ (~B19 & B15);
        st[19] = B19 ^ (~B15 & B16);
        st[20] = B20 ^ (~B21 & B22); st[21] = B21 ^ (~B22 & B23);
        st[22] = B22 ^ (~B23 & B24); st[23] = B23 ^ (~B24 & B20);
        st[24] = B24 ^ (~B20 & B21);

        /* ι */
        st[0] ^= RC[r];
    }
}

/*
 * keccak256 of exactly 64 bytes.
 * Output: 32 bytes in hash[].
 *
 * Rate = 136 bytes (1088 bits), capacity = 512 bits.
 * Padding for original keccak256: append 0x01, zero-fill, OR 0x80 to last byte.
 */
void keccak256_64(const uchar *in, uchar *hash) {
    ulong st[25];

    /* Zero the state */
    for (int i = 0; i < 25; i++) st[i] = 0UL;

    /* Absorb 64 bytes as 8 little-endian uint64 lanes */
    for (int i = 0; i < 8; i++) {
        ulong v = 0UL;
        for (int j = 0; j < 8; j++)
            v |= ((ulong)in[i * 8 + j]) << (j * 8);
        st[i] = v;
    }

    /*
     * Padding:
     *   64 bytes of message already absorbed into st[0..7].
     *   Rate = 136 bytes = 17 lanes.
     *   First pad byte at position 64 → lane 8 byte 0 → st[8] |= 0x01
     *   Last byte of rate at position 135 → lane 16 byte 7 → st[16] |= 0x80<<56
     */
    st[ 8] ^= 0x0000000000000001UL;
    st[16] ^= 0x8000000000000000UL;

    keccak_f1600(st);

    /* Squeeze first 32 bytes (4 lanes) */
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++)
            hash[i * 8 + j] = (uchar)((st[i] >> (j * 8)) & 0xFFu);
    }
}

/*
 * Big-endian uint256 comparison: hash < difficulty ?
 * hash[0] is the most-significant byte (matches Ethereum convention).
 */
int hash_lt(const uchar *hash, __constant const uchar *diff) {
    for (int i = 0; i < 32; i++) {
        if (hash[i] < diff[i]) return 1;
        if (hash[i] > diff[i]) return 0;
    }
    return 0; /* equal → not less than */
}

/* ── Main mining kernel ─────────────────────────────────────────────────── */
__kernel void mine(
    __constant uchar *challenge,    /* 32 bytes */
    __constant uchar *difficulty,   /* 32 bytes, big-endian uint256 target */
    __constant uchar *nonce_prefix, /* 24 bytes */
    ulong             base_counter, /* starting counter for this batch */
    __global   int   *found_flag,   /* output: set to 1 on hit */
    __global   ulong *found_counter /* output: winning counter value */
) {
    ulong tid        = (ulong)get_global_id(0);
    ulong my_counter = base_counter + tid;

    /* Build 64-byte keccak input: challenge(32) || nonce_prefix(24) || counter_be(8) */
    uchar input[64];

    for (int i = 0; i < 32; i++) input[i]      = challenge[i];
    for (int i = 0; i < 24; i++) input[32 + i]  = nonce_prefix[i];

    /* Counter as big-endian uint64 */
    input[56] = (uchar)(my_counter >> 56);
    input[57] = (uchar)(my_counter >> 48);
    input[58] = (uchar)(my_counter >> 40);
    input[59] = (uchar)(my_counter >> 32);
    input[60] = (uchar)(my_counter >> 24);
    input[61] = (uchar)(my_counter >> 16);
    input[62] = (uchar)(my_counter >>  8);
    input[63] = (uchar)(my_counter      );

    uchar hash[32];
    keccak256_64(input, hash);

    if (hash_lt(hash, difficulty)) {
        /* Mark hit — last writer wins; any valid nonce is fine */
        *found_flag    = 1;
        *found_counter = my_counter;
    }
}
