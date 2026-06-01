// =============================================================================
// cosmos-vanity — shared GPU kernel for SHA-256 + RIPEMD-160 hashing
// =============================================================================
//
// This kernel takes compressed public key bytes and computes the Cosmos
// address hash (SHA256 → RIPEMD160). The actual secp256k1 point multiplication
// is done on CPU; GPU handles the expensive hash pipeline in bulk.
//
// Each work item processes one candidate public key.
//
// Input:  pubkeys[]  — array of 33-byte compressed public keys (packed)
// Output: hashes[]   — array of 20-byte address hashes (packed)
//         matches[]  — flags indicating pattern match (1 = match, 0 = no match)
//
// Pattern matching is done on the raw address bytes (pre-bech32) for speed.
// Matched candidates are verified on CPU with full bech32 encoding.

// ---- SHA-256 Constants ----
__constant uint K_SHA256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// ---- RIPEMD-160 Constants ----
__constant uint KL_RIPEMD[5] = { 0x00000000, 0x5a827999, 0x6ed9eba1, 0x8f1bbcdc, 0xa953fd4e };
__constant uint KR_RIPEMD[5] = { 0x50a28be6, 0x5c4dd124, 0x6d703ef3, 0x7a6d76e9, 0x00000000 };

__constant uchar RL[80] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    7, 4, 13, 1, 10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8,
    3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12,
    1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2,
    4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13
};

__constant uchar RR[80] = {
    5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12,
    6, 11, 3, 7, 0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2,
    15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13,
    8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14,
    12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11
};

__constant uchar SL[80] = {
    11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8,
    7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12,
    11, 13, 6, 7, 14, 9, 13, 15, 14, 8, 13, 6, 5, 12, 7, 5,
    11, 12, 14, 15, 14, 15, 9, 8, 9, 14, 5, 6, 8, 6, 5, 12,
    9, 15, 5, 11, 6, 8, 13, 12, 5, 12, 13, 14, 11, 8, 5, 6
};

__constant uchar SR[80] = {
    8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6,
    9, 13, 15, 7, 12, 8, 9, 11, 7, 7, 12, 7, 6, 15, 13, 11,
    9, 7, 15, 11, 8, 6, 6, 14, 12, 13, 5, 14, 13, 13, 7, 5,
    15, 5, 8, 11, 14, 14, 6, 14, 6, 9, 12, 9, 12, 5, 15, 8,
    8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6, 5, 15, 13, 11, 11
};

uint rotate_left(uint x, uint n) {
    return (x << n) | (x >> (32 - n));
}

// SHA-256 for exactly 33 bytes (compressed pubkey)
void sha256_33bytes(__global const uchar *input, uint *hash) {
    uint W[64];
    uint a, b, c, d, e, f, g, h;
    uint T1, T2;

    // Pad the 33-byte message into a single 512-bit block
    // Message: 33 bytes | 0x80 | zeros | length (big-endian 64-bit)
    for (int i = 0; i < 16; i++) W[i] = 0;

    for (int i = 0; i < 33; i++) {
        W[i / 4] |= ((uint)input[i]) << (24 - (i % 4) * 8);
    }
    W[33 / 4] |= 0x80 << (24 - (33 % 4) * 8);
    W[15] = 33 * 8; // bit length

    // Expand (σ0/σ1 use right-rotates and right-shifts)
    for (int i = 16; i < 64; i++) {
        uint s0 = (W[i-15] >> 7 | W[i-15] << 25) ^ (W[i-15] >> 18 | W[i-15] << 14) ^ (W[i-15] >> 3);
        uint s1 = (W[i-2] >> 17 | W[i-2] << 15) ^ (W[i-2] >> 19 | W[i-2] << 13) ^ (W[i-2] >> 10);
        W[i] = W[i-16] + s0 + W[i-7] + s1;
    }

    // Initialize
    a = 0x6a09e667; b = 0xbb67ae85; c = 0x3c6ef372; d = 0xa54ff53a;
    e = 0x510e527f; f = 0x9b05688c; g = 0x1f83d9ab; h = 0x5be0cd19;

    // Compress
    for (int i = 0; i < 64; i++) {
        uint S1 = (e >> 6 | e << 26) ^ (e >> 11 | e << 21) ^ (e >> 25 | e << 7);
        uint ch = (e & f) ^ (~e & g);
        T1 = h + S1 + ch + K_SHA256[i] + W[i];
        uint S0 = (a >> 2 | a << 30) ^ (a >> 13 | a << 19) ^ (a >> 22 | a << 10);
        uint maj = (a & b) ^ (a & c) ^ (b & c);
        T2 = S0 + maj;

        h = g; g = f; f = e; e = d + T1;
        d = c; c = b; b = a; a = T1 + T2;
    }

    hash[0] = a + 0x6a09e667;
    hash[1] = b + 0xbb67ae85;
    hash[2] = c + 0x3c6ef372;
    hash[3] = d + 0xa54ff53a;
    hash[4] = e + 0x510e527f;
    hash[5] = f + 0x9b05688c;
    hash[6] = g + 0x1f83d9ab;
    hash[7] = h + 0x5be0cd19;
}

// RIPEMD-160 for exactly 32 bytes (SHA-256 output)
void ripemd160_32bytes(uint *sha_hash, uint *rmd_hash) {
    uint W[16];

    // Copy SHA-256 output as little-endian words
    for (int i = 0; i < 8; i++) {
        uint v = sha_hash[i];
        W[i] = ((v >> 24) & 0xFF) | ((v >> 8) & 0xFF00) |
               ((v << 8) & 0xFF0000) | ((v << 24) & 0xFF000000);
    }
    // Padding for 32-byte message
    W[8] = 0x00000080;
    for (int i = 9; i < 14; i++) W[i] = 0;
    W[14] = 32 * 8; // bit length, little-endian
    W[15] = 0;

    uint al = 0x67452301, bl = 0xefcdab89, cl = 0x98badcfe, dl = 0x10325476, el = 0xc3d2e1f0;
    uint ar = al, br = bl, cr = cl, dr = dl, er = el;

    for (int j = 0; j < 80; j++) {
        uint fl, fr, tl, tr;
        int round = j / 16;

        // Left line
        if (round == 0)      fl = bl ^ cl ^ dl;
        else if (round == 1) fl = (bl & cl) | (~bl & dl);
        else if (round == 2) fl = (bl | ~cl) ^ dl;
        else if (round == 3) fl = (bl & dl) | (cl & ~dl);
        else                 fl = bl ^ (cl | ~dl);

        tl = al + fl + W[RL[j]] + KL_RIPEMD[round];
        tl = rotate_left(tl, (uint)SL[j]) + el;
        al = el; el = dl; dl = rotate_left(cl, 10); cl = bl; bl = tl;

        // Right line
        if (round == 0)      fr = br ^ (cr | ~dr);
        else if (round == 1) fr = (br & dr) | (cr & ~dr);
        else if (round == 2) fr = (br | ~cr) ^ dr;
        else if (round == 3) fr = (br & cr) | (~br & dr);
        else                 fr = br ^ cr ^ dr;

        tr = ar + fr + W[RR[j]] + KR_RIPEMD[round];
        tr = rotate_left(tr, (uint)SR[j]) + er;
        ar = er; er = dr; dr = rotate_left(cr, 10); cr = br; br = tr;
    }

    // Finalization: h0' = h1+cl+dr, h1' = h2+dl+er, h2' = h3+el+ar, h3' = h4+al+br, h4' = h0+bl+cr
    uint t = 0xefcdab89 + cl + dr;
    rmd_hash[0] = t;
    rmd_hash[1] = 0x98badcfe + dl + er;
    rmd_hash[2] = 0x10325476 + el + ar;
    rmd_hash[3] = 0xc3d2e1f0 + al + br;
    rmd_hash[4] = 0x67452301 + bl + cr;
}

// Main kernel: compute address hashes from compressed public keys
__kernel void compute_address_hashes(
    __global const uchar *pubkeys,    // N * 33 bytes
    __global uchar *hashes,           // N * 20 bytes
    __global const uchar *prefix,     // prefix bytes to match (in address hash space)
    uint prefix_len,                  // length of prefix to match (0 = no matching, just hash)
    __global uint *matches,           // N uints: 1 if match, 0 if not
    uint count                        // number of keys
) {
    uint gid = get_global_id(0);
    if (gid >= count) return;

    // Read compressed pubkey
    __global const uchar *pk = pubkeys + gid * 33;

    // SHA-256
    uint sha_out[8];
    sha256_33bytes(pk, sha_out);

    // RIPEMD-160
    uint rmd_out[5];
    ripemd160_32bytes(sha_out, rmd_out);

    // Write 20-byte hash output (little-endian to bytes)
    __global uchar *out = hashes + gid * 20;
    for (int i = 0; i < 5; i++) {
        out[i*4 + 0] = (rmd_out[i]) & 0xFF;
        out[i*4 + 1] = (rmd_out[i] >> 8) & 0xFF;
        out[i*4 + 2] = (rmd_out[i] >> 16) & 0xFF;
        out[i*4 + 3] = (rmd_out[i] >> 24) & 0xFF;
    }

    // Simple prefix matching on raw hash bytes
    if (prefix_len > 0) {
        uint match = 1;
        for (uint i = 0; i < prefix_len && i < 20; i++) {
            if (out[i] != prefix[i]) {
                match = 0;
                break;
            }
        }
        matches[gid] = match;
    } else {
        matches[gid] = 0;
    }
}
