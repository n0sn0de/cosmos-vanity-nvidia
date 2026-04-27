// =============================================================================
// cosmos-vanity — full GPU mnemonic pipeline
// SHA-512 → HMAC-SHA512 → PBKDF2 → BIP-32 → secp256k1 → SHA-256 → RIPEMD-160
// =============================================================================
//
// Each work item takes a mnemonic string (UTF-8 bytes) and produces:
// - The derived private key (32 bytes)
// - The address hash (20 bytes)
// - A match flag
//
// The secp256k1, SHA-256, and RIPEMD-160 code is included from secp256k1.cl
// via source concatenation at compile time.

// ============================= SHA-512 =====================================

__constant ulong K_SHA512[80] = {
    0x428a2f98d728ae22UL, 0x7137449123ef65cdUL, 0xb5c0fbcfec4d3b2fUL, 0xe9b5dba58189dbbcUL,
    0x3956c25bf348b538UL, 0x59f111f1b605d019UL, 0x923f82a4af194f9bUL, 0xab1c5ed5da6d8118UL,
    0xd807aa98a3030242UL, 0x12835b0145706fbeUL, 0x243185be4ee4b28cUL, 0x550c7dc3d5ffb4e2UL,
    0x72be5d74f27b896fUL, 0x80deb1fe3b1696b1UL, 0x9bdc06a725c71235UL, 0xc19bf174cf692694UL,
    0xe49b69c19ef14ad2UL, 0xefbe4786384f25e3UL, 0x0fc19dc68b8cd5b5UL, 0x240ca1cc77ac9c65UL,
    0x2de92c6f592b0275UL, 0x4a7484aa6ea6e483UL, 0x5cb0a9dcbd41fbd4UL, 0x76f988da831153b5UL,
    0x983e5152ee66dfabUL, 0xa831c66d2db43210UL, 0xb00327c898fb213fUL, 0xbf597fc7beef0ee4UL,
    0xc6e00bf33da88fc2UL, 0xd5a79147930aa725UL, 0x06ca6351e003826fUL, 0x142929670a0e6e70UL,
    0x27b70a8546d22ffcUL, 0x2e1b21385c26c926UL, 0x4d2c6dfc5ac42aedUL, 0x53380d139d95b3dfUL,
    0x650a73548baf63deUL, 0x766a0abb3c77b2a8UL, 0x81c2c92e47edaee6UL, 0x92722c851482353bUL,
    0xa2bfe8a14cf10364UL, 0xa81a664bbc423001UL, 0xc24b8b70d0f89791UL, 0xc76c51a30654be30UL,
    0xd192e819d6ef5218UL, 0xd69906245565a910UL, 0xf40e35855771202aUL, 0x106aa07032bbd1b8UL,
    0x19a4c116b8d2d0c8UL, 0x1e376c085141ab53UL, 0x2748774cdf8eeb99UL, 0x34b0bcb5e19b48a8UL,
    0x391c0cb3c5c95a63UL, 0x4ed8aa4ae3418acbUL, 0x5b9cca4f7763e373UL, 0x682e6ff3d6b2b8a3UL,
    0x748f82ee5defb2fcUL, 0x78a5636f43172f60UL, 0x84c87814a1f0ab72UL, 0x8cc702081a6439ecUL,
    0x90befffa23631e28UL, 0xa4506cebde82bde9UL, 0xbef9a3f7b2c67915UL, 0xc67178f2e372532bUL,
    0xca273eceea26619cUL, 0xd186b8c721c0c207UL, 0xeada7dd6cde0eb1eUL, 0xf57d4f7fee6ed178UL,
    0x06f067aa72176fbaUL, 0x0a637dc5a2c898a6UL, 0x113f9804bef90daeUL, 0x1b710b35131c471bUL,
    0x28db77f523047d84UL, 0x32caab7b40c72493UL, 0x3c9ebe0a15c9bebcUL, 0x431d67c49c100d4cUL,
    0x4cc5d4becb3e42b6UL, 0x597f299cfc657e2aUL, 0x5fcb6fab3ad6faecUL, 0x6c44198c4a475817UL
};

#define SHA512_BLOCK_SIZE 128
#define SHA512_DIGEST_SIZE 64

#define ROR64(x, n) (((x) >> (n)) | ((x) << (64 - (n))))
#define CH64(x, y, z)  (((x) & (y)) ^ (~(x) & (z)))
#define MAJ64(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define SIGMA0_512(x) (ROR64(x, 28) ^ ROR64(x, 34) ^ ROR64(x, 39))
#define SIGMA1_512(x) (ROR64(x, 14) ^ ROR64(x, 18) ^ ROR64(x, 41))
#define sigma0_512(x) (ROR64(x, 1) ^ ROR64(x, 8) ^ ((x) >> 7))
#define sigma1_512(x) (ROR64(x, 19) ^ ROR64(x, 61) ^ ((x) >> 6))

// Read big-endian ulong from byte array
ulong read_be64(const uchar *p) {
    return ((ulong)p[0] << 56) | ((ulong)p[1] << 48) | ((ulong)p[2] << 40) | ((ulong)p[3] << 32) |
           ((ulong)p[4] << 24) | ((ulong)p[5] << 16) | ((ulong)p[6] << 8)  | (ulong)p[7];
}

void write_be64(uchar *p, ulong v) {
    p[0] = (uchar)(v >> 56); p[1] = (uchar)(v >> 48); p[2] = (uchar)(v >> 40); p[3] = (uchar)(v >> 32);
    p[4] = (uchar)(v >> 24); p[5] = (uchar)(v >> 16); p[6] = (uchar)(v >> 8);  p[7] = (uchar)v;
}

// SHA-512 state
typedef struct {
    ulong h[8];
    uchar buf[SHA512_BLOCK_SIZE];
    uint buflen;
    ulong total;
} sha512_ctx;

void sha512_init(sha512_ctx *ctx) {
    ctx->h[0] = 0x6a09e667f3bcc908UL;
    ctx->h[1] = 0xbb67ae8584caa73bUL;
    ctx->h[2] = 0x3c6ef372fe94f82bUL;
    ctx->h[3] = 0xa54ff53a5f1d36f1UL;
    ctx->h[4] = 0x510e527fade682d1UL;
    ctx->h[5] = 0x9b05688c2b3e6c1fUL;
    ctx->h[6] = 0x1f83d9abfb41bd6bUL;
    ctx->h[7] = 0x5be0cd19137e2179UL;
    ctx->buflen = 0;
    ctx->total = 0;
}

void sha512_compress(sha512_ctx *ctx) {
    ulong W[80];
    ulong a, b, c, d, e, f, g, h;

    // Load message block as big-endian words
    for (int i = 0; i < 16; i++) {
        W[i] = read_be64(ctx->buf + i * 8);
    }

    // Expand
    for (int i = 16; i < 80; i++) {
        W[i] = sigma1_512(W[i-2]) + W[i-7] + sigma0_512(W[i-15]) + W[i-16];
    }

    a = ctx->h[0]; b = ctx->h[1]; c = ctx->h[2]; d = ctx->h[3];
    e = ctx->h[4]; f = ctx->h[5]; g = ctx->h[6]; h = ctx->h[7];

    for (int i = 0; i < 80; i++) {
        ulong T1 = h + SIGMA1_512(e) + CH64(e, f, g) + K_SHA512[i] + W[i];
        ulong T2 = SIGMA0_512(a) + MAJ64(a, b, c);
        h = g; g = f; f = e; e = d + T1;
        d = c; c = b; b = a; a = T1 + T2;
    }

    ctx->h[0] += a; ctx->h[1] += b; ctx->h[2] += c; ctx->h[3] += d;
    ctx->h[4] += e; ctx->h[5] += f; ctx->h[6] += g; ctx->h[7] += h;
}

void sha512_update(sha512_ctx *ctx, const uchar *data, uint len) {
    ctx->total += len;
    uint offset = 0;

    // Fill buffer
    if (ctx->buflen > 0) {
        uint space = SHA512_BLOCK_SIZE - ctx->buflen;
        uint copy = (len < space) ? len : space;
        for (uint i = 0; i < copy; i++)
            ctx->buf[ctx->buflen + i] = data[i];
        ctx->buflen += copy;
        offset += copy;
        len -= copy;
        if (ctx->buflen == SHA512_BLOCK_SIZE) {
            sha512_compress(ctx);
            ctx->buflen = 0;
        }
    }

    // Process full blocks
    while (len >= SHA512_BLOCK_SIZE) {
        for (int i = 0; i < SHA512_BLOCK_SIZE; i++)
            ctx->buf[i] = data[offset + i];
        sha512_compress(ctx);
        offset += SHA512_BLOCK_SIZE;
        len -= SHA512_BLOCK_SIZE;
    }

    // Buffer remaining
    for (uint i = 0; i < len; i++)
        ctx->buf[ctx->buflen + i] = data[offset + i];
    ctx->buflen += len;
}

// Update from private (non-global) memory
void sha512_update_priv(sha512_ctx *ctx, const uchar *data, uint len) {
    sha512_update(ctx, data, len);
}

void sha512_final(sha512_ctx *ctx, uchar *digest) {
    // Pad
    ctx->buf[ctx->buflen++] = 0x80;
    if (ctx->buflen > 112) {
        // Need two blocks
        while (ctx->buflen < SHA512_BLOCK_SIZE)
            ctx->buf[ctx->buflen++] = 0;
        sha512_compress(ctx);
        ctx->buflen = 0;
    }
    while (ctx->buflen < 112)
        ctx->buf[ctx->buflen++] = 0;

    // Length in bits (big-endian, 128-bit — we only use lower 64 bits)
    ulong bitlen = ctx->total * 8;
    for (int i = 0; i < 8; i++)
        ctx->buf[112 + i] = 0;
    write_be64(ctx->buf + 120, bitlen);

    sha512_compress(ctx);

    // Output
    for (int i = 0; i < 8; i++)
        write_be64(digest + i * 8, ctx->h[i]);
}

// Convenience: SHA-512 of a single message
void sha512(const uchar *msg, uint len, uchar *digest) {
    sha512_ctx ctx;
    sha512_init(&ctx);
    sha512_update(&ctx, msg, len);
    sha512_final(&ctx, digest);
}

// ============================= HMAC-SHA512 =================================

void hmac_sha512(const uchar *key, uint key_len,
                 const uchar *msg, uint msg_len,
                 uchar *out) {
    uchar k_pad[SHA512_BLOCK_SIZE];
    uchar tmp[SHA512_DIGEST_SIZE];
    sha512_ctx ctx;

    // If key > block size, hash it first
    // Buffer must be large enough for keys up to SHA512_BLOCK_SIZE (128 bytes)
    uchar real_key[SHA512_BLOCK_SIZE];
    uint real_key_len;
    if (key_len > SHA512_BLOCK_SIZE) {
        sha512(key, key_len, real_key);
        real_key_len = SHA512_DIGEST_SIZE;
    } else {
        for (uint i = 0; i < key_len; i++) real_key[i] = key[i];
        real_key_len = key_len;
    }

    // Inner: SHA512((K ^ ipad) || message)
    for (uint i = 0; i < SHA512_BLOCK_SIZE; i++)
        k_pad[i] = (i < real_key_len) ? (real_key[i] ^ 0x36) : 0x36;

    sha512_init(&ctx);
    sha512_update_priv(&ctx, k_pad, SHA512_BLOCK_SIZE);
    sha512_update(&ctx, msg, msg_len);
    sha512_final(&ctx, tmp);

    // Outer: SHA512((K ^ opad) || inner_hash)
    for (uint i = 0; i < SHA512_BLOCK_SIZE; i++)
        k_pad[i] = (i < real_key_len) ? (real_key[i] ^ 0x5c) : 0x5c;

    sha512_init(&ctx);
    sha512_update_priv(&ctx, k_pad, SHA512_BLOCK_SIZE);
    sha512_update_priv(&ctx, tmp, SHA512_DIGEST_SIZE);
    sha512_final(&ctx, out);
}

// HMAC with key from private memory and msg from private memory
void hmac_sha512_priv(const uchar *key, uint key_len,
                      const uchar *msg, uint msg_len,
                      uchar *out) {
    uchar k_pad[SHA512_BLOCK_SIZE];
    uchar tmp[SHA512_DIGEST_SIZE];
    sha512_ctx ctx;

    // Buffer must be large enough for keys up to SHA512_BLOCK_SIZE (128 bytes)
    uchar real_key[SHA512_BLOCK_SIZE];
    uint real_key_len;
    if (key_len > SHA512_BLOCK_SIZE) {
        sha512_ctx hctx;
        sha512_init(&hctx);
        sha512_update_priv(&hctx, key, key_len);
        sha512_final(&hctx, real_key);
        real_key_len = SHA512_DIGEST_SIZE;
    } else {
        for (uint i = 0; i < key_len; i++) real_key[i] = key[i];
        real_key_len = key_len;
    }

    // Inner
    for (uint i = 0; i < SHA512_BLOCK_SIZE; i++)
        k_pad[i] = (i < real_key_len) ? (real_key[i] ^ 0x36) : 0x36;
    sha512_init(&ctx);
    sha512_update_priv(&ctx, k_pad, SHA512_BLOCK_SIZE);
    sha512_update_priv(&ctx, msg, msg_len);
    sha512_final(&ctx, tmp);

    // Outer
    for (uint i = 0; i < SHA512_BLOCK_SIZE; i++)
        k_pad[i] = (i < real_key_len) ? (real_key[i] ^ 0x5c) : 0x5c;
    sha512_init(&ctx);
    sha512_update_priv(&ctx, k_pad, SHA512_BLOCK_SIZE);
    sha512_update_priv(&ctx, tmp, SHA512_DIGEST_SIZE);
    sha512_final(&ctx, out);
}

// ============================= PBKDF2-HMAC-SHA512 ==========================

// PBKDF2 with 2048 iterations, single block (dkLen ≤ 64)
// password = mnemonic UTF-8 bytes
// salt = "mnemonic" (8 bytes, standard BIP-39 with no passphrase)
void pbkdf2_sha512(const uchar *password, uint password_len,
                   const uchar *salt, uint salt_len,
                   uint iterations, uchar *out) {
    // U1 = HMAC(password, salt || INT32BE(1))
    uchar salt_ext[80]; // salt + 4 bytes for block index (max salt ~8-16 bytes)
    for (uint i = 0; i < salt_len; i++) salt_ext[i] = salt[i];
    salt_ext[salt_len] = 0;
    salt_ext[salt_len + 1] = 0;
    salt_ext[salt_len + 2] = 0;
    salt_ext[salt_len + 3] = 1; // Block index 1

    uchar U[SHA512_DIGEST_SIZE];
    uchar T[SHA512_DIGEST_SIZE];

    // U1 = HMAC(password, salt || 0x00000001)
    hmac_sha512(password, password_len, salt_ext, salt_len + 4, U);
    for (int i = 0; i < SHA512_DIGEST_SIZE; i++) T[i] = U[i];

    // U2..U2048
    for (uint iter = 1; iter < iterations; iter++) {
        uchar U_new[SHA512_DIGEST_SIZE];
        hmac_sha512_priv(password, password_len, U, SHA512_DIGEST_SIZE, U_new);
        for (int i = 0; i < SHA512_DIGEST_SIZE; i++) {
            U[i] = U_new[i];
            T[i] ^= U_new[i];
        }
    }

    for (int i = 0; i < SHA512_DIGEST_SIZE; i++) out[i] = T[i];
}

// ============================= BIP-32 ======================================

// secp256k1 curve order n (for modular addition of child keys)
__constant uint SECP256K1_N[8] = {
    0xD0364141, 0xBFD25E8C, 0xAF48A03B, 0xBAAEDCE6,
    0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
};

// Add two 256-bit numbers mod n (for BIP-32 child key derivation)
// a and b are big-endian 32-byte arrays, result in out (big-endian)
void add_privkeys_mod_n(const uchar *a, const uchar *b, uchar *out) {
    // Convert to little-endian limbs for easier arithmetic
    uint al[8], bl[8], rl[8];
    for (int i = 0; i < 8; i++) {
        al[i] = ((uint)a[31 - i*4] | ((uint)a[30 - i*4] << 8) |
                 ((uint)a[29 - i*4] << 16) | ((uint)a[28 - i*4] << 24));
        bl[i] = ((uint)b[31 - i*4] | ((uint)b[30 - i*4] << 8) |
                 ((uint)b[29 - i*4] << 16) | ((uint)b[28 - i*4] << 24));
    }

    // Add with carry
    ulong carry = 0;
    for (int i = 0; i < 8; i++) {
        ulong sum = (ulong)al[i] + (ulong)bl[i] + carry;
        rl[i] = (uint)sum;
        carry = sum >> 32;
    }

    // Reduce mod n: if result >= n, subtract n
    int ge_n = (carry > 0) ? 1 : 0;
    if (!ge_n) {
        // Compare rl >= SECP256K1_N
        for (int i = 7; i >= 0; i--) {
            if (rl[i] > SECP256K1_N[i]) { ge_n = 1; break; }
            if (rl[i] < SECP256K1_N[i]) { break; }
        }
    }

    if (ge_n) {
        ulong borrow = 0;
        for (int i = 0; i < 8; i++) {
            ulong diff = (ulong)rl[i] - (ulong)SECP256K1_N[i] - borrow;
            rl[i] = (uint)diff;
            borrow = (diff >> 63) & 1;
        }
    }

    // Convert back to big-endian
    for (int i = 0; i < 8; i++) {
        out[31 - i*4]     = (uchar)(rl[i]);
        out[30 - i*4] = (uchar)(rl[i] >> 8);
        out[29 - i*4] = (uchar)(rl[i] >> 16);
        out[28 - i*4] = (uchar)(rl[i] >> 24);
    }
}

// BIP-32: Derive child key from parent key + chain code
// For hardened (index >= 0x80000000): data = 0x00 || parent_key || index_be32
// For normal: data = compressed_pubkey || index_be32
// Returns: left 32 bytes = IL (for key addition), right 32 bytes = new chain code
void bip32_derive_child(const uchar *parent_key, const uchar *chain_code,
                        uint index, int is_hardened,
                        uchar *child_key, uchar *child_chain) {
    uchar data[37]; // max: 1 + 32 + 4 = 37 for hardened, or 33 + 4 = 37 for normal
    uint data_len;

    if (is_hardened) {
        data[0] = 0x00;
        for (int i = 0; i < 32; i++) data[1 + i] = parent_key[i];
        data[33] = (uchar)(index >> 24);
        data[34] = (uchar)(index >> 16);
        data[35] = (uchar)(index >> 8);
        data[36] = (uchar)(index);
        data_len = 37;
    } else {
        // Need compressed pubkey — caller must provide it in parent_key slot
        // For normal derivation, parent_key is actually the 33-byte compressed pubkey
        for (int i = 0; i < 33; i++) data[i] = parent_key[i];
        data[33] = (uchar)(index >> 24);
        data[34] = (uchar)(index >> 16);
        data[35] = (uchar)(index >> 8);
        data[36] = (uchar)(index);
        data_len = 37;
    }

    uchar hmac_out[64];
    hmac_sha512_priv(chain_code, 32, data, data_len, hmac_out);

    // IL = left 32 bytes, IR = right 32 bytes (new chain code)
    // child_key = (IL + parent_key) mod n
    // For hardened, parent_key is the 32-byte private key
    // For normal, we still need the original private key for addition
    // The caller handles this distinction

    for (int i = 0; i < 32; i++) {
        child_key[i] = hmac_out[i];       // IL (caller adds to parent)
        child_chain[i] = hmac_out[32 + i]; // IR = new chain code
    }
}

// Full BIP-32 derivation: seed → m/44'/118'/0'/0/0
// seed: 64 bytes, out_privkey: 32 bytes
void bip32_derive_cosmos(const uchar *seed, uchar *out_privkey) {
    uchar key[32], chain[32], child_il[32], child_chain[32];

    // Master key: HMAC-SHA512(key="Bitcoin seed", data=seed)
    uchar btc_seed[12] = {'B','i','t','c','o','i','n',' ','s','e','e','d'};
    uchar master[64];
    hmac_sha512_priv(btc_seed, 12, seed, 64, master);

    for (int i = 0; i < 32; i++) {
        key[i] = master[i];        // private key
        chain[i] = master[32 + i]; // chain code
    }

    // Derive hardened children: 44', 118', 0'
    uint hardened_indices[3] = { 0x8000002C, 0x80000076, 0x80000000 };
    for (int level = 0; level < 3; level++) {
        bip32_derive_child(key, chain, hardened_indices[level], 1, child_il, child_chain);
        // child_key = (IL + parent_key) mod n
        add_privkeys_mod_n(child_il, key, key);
        for (int i = 0; i < 32; i++) chain[i] = child_chain[i];
    }

    // Derive normal children: 0, 0
    // Path indices for the two normal derivation levels (both are 0 for m/44'/118'/0'/0/0)
    uint normal_indices[2] = { 0, 0 };
    // For normal derivation, we need the compressed pubkey of the current key
    for (int level = 0; level < 2; level++) {
        // Compute compressed pubkey from current private key using secp256k1
        // We need to call scalar_mul_G and compress — these are defined in secp256k1.cl
        // which is concatenated with this source

        // Convert key from big-endian bytes to uint256_t
        uint256_t privkey_val;
        for (int i = 0; i < 8; i++) {
            privkey_val.d[i] = ((uint)key[31 - i*4]) |
                               ((uint)key[30 - i*4] << 8) |
                               ((uint)key[29 - i*4] << 16) |
                               ((uint)key[28 - i*4] << 24);
        }

        point_jacobian pub_jac = scalar_mul_G(privkey_val);

        // Convert to affine and compress
        uint256_t z_inv = field_inv(pub_jac.z);
        uint256_t z_inv2 = field_sqr(z_inv);
        uint256_t z_inv3 = field_mul(z_inv2, z_inv);
        uint256_t pub_x = field_mul(pub_jac.x, z_inv2);
        uint256_t pub_y = field_mul(pub_jac.y, z_inv3);

        uchar compressed_pubkey[33];
        compressed_pubkey[0] = (pub_y.d[0] & 1) ? 0x03 : 0x02;
        for (int i = 7; i >= 0; i--) {
            compressed_pubkey[1 + (7-i)*4]     = (uchar)(pub_x.d[i] >> 24);
            compressed_pubkey[1 + (7-i)*4 + 1] = (uchar)(pub_x.d[i] >> 16);
            compressed_pubkey[1 + (7-i)*4 + 2] = (uchar)(pub_x.d[i] >> 8);
            compressed_pubkey[1 + (7-i)*4 + 3] = (uchar)(pub_x.d[i]);
        }

        // Normal derivation uses compressed pubkey as "key" in HMAC data
        bip32_derive_child(compressed_pubkey, chain, normal_indices[level], 0, child_il, child_chain);
        add_privkeys_mod_n(child_il, key, key);
        for (int i = 0; i < 32; i++) chain[i] = child_chain[i];
    }

    for (int i = 0; i < 32; i++) out_privkey[i] = key[i];
}

// ============================= MAIN KERNEL =================================

__kernel void mnemonic_to_address(
    __global const uchar *mnemonics,       // N × 256 bytes (zero-padded UTF-8)
    __global const uint *mnemonic_lens,    // N × uint (actual byte lengths)
    __global uchar *derived_privkeys,      // N × 32 bytes output
    __global uchar *hashes,                // N × 20 bytes output
    __global const uchar *prefix,          // prefix bytes to match
    uint prefix_len,
    __global uint *matches,                // N × uint match flags
    uint count
) {
    uint gid = get_global_id(0);
    if (gid >= count) return;

    // 1. Read mnemonic bytes
    __global const uchar *mnemonic_ptr = mnemonics + gid * 256;
    uint mnemonic_len = mnemonic_lens[gid];

    // Copy to private memory for faster access
    uchar mnemonic_local[256];
    for (uint i = 0; i < mnemonic_len; i++)
        mnemonic_local[i] = mnemonic_ptr[i];

    // 2. PBKDF2-HMAC-SHA512: mnemonic → seed (64 bytes)
    uchar salt[8] = {'m','n','e','m','o','n','i','c'};
    uchar seed[64];
    pbkdf2_sha512(mnemonic_local, mnemonic_len, salt, 8, 2048, seed);

    // 3. BIP-32: seed → private key (m/44'/118'/0'/0/0)
    uchar privkey[32];
    bip32_derive_cosmos(seed, privkey);

    // Store derived private key for CPU verification
    __global uchar *pk_out = derived_privkeys + gid * 32;
    for (int i = 0; i < 32; i++) pk_out[i] = privkey[i];

    // 4. secp256k1: privkey → compressed pubkey
    uint256_t pk_val;
    for (int i = 0; i < 8; i++) {
        pk_val.d[i] = ((uint)privkey[31 - i*4]) |
                      ((uint)privkey[30 - i*4] << 8) |
                      ((uint)privkey[29 - i*4] << 16) |
                      ((uint)privkey[28 - i*4] << 24);
    }

    point_jacobian pub_jac = scalar_mul_G(pk_val);

    // Affine conversion + compression
    uint256_t z_inv = field_inv(pub_jac.z);
    uint256_t z_inv2 = field_sqr(z_inv);
    uint256_t z_inv3 = field_mul(z_inv2, z_inv);
    uint256_t pub_x = field_mul(pub_jac.x, z_inv2);
    uint256_t pub_y = field_mul(pub_jac.y, z_inv3);

    uchar pubkey_compressed[33];
    pubkey_compressed[0] = (pub_y.d[0] & 1) ? 0x03 : 0x02;
    for (int i = 7; i >= 0; i--) {
        pubkey_compressed[1 + (7-i)*4]     = (uchar)(pub_x.d[i] >> 24);
        pubkey_compressed[1 + (7-i)*4 + 1] = (uchar)(pub_x.d[i] >> 16);
        pubkey_compressed[1 + (7-i)*4 + 2] = (uchar)(pub_x.d[i] >> 8);
        pubkey_compressed[1 + (7-i)*4 + 3] = (uchar)(pub_x.d[i]);
    }

    // 5. SHA-256(pubkey) → RIPEMD-160 → 20-byte hash
    uint sha_out[8];
    sha256_33(pubkey_compressed, sha_out);

    uint rmd_out[5];
    ripemd160_32(sha_out, rmd_out);

    // Write hash output
    __global uchar *hash_out = hashes + gid * 20;
    for (int i = 0; i < 5; i++) {
        hash_out[i*4 + 0] = (rmd_out[i]) & 0xFF;
        hash_out[i*4 + 1] = (rmd_out[i] >> 8) & 0xFF;
        hash_out[i*4 + 2] = (rmd_out[i] >> 16) & 0xFF;
        hash_out[i*4 + 3] = (rmd_out[i] >> 24) & 0xFF;
    }

    // 6. Prefix matching
    if (prefix_len > 0) {
        uint match = 1;
        for (uint i = 0; i < prefix_len && i < 20; i++) {
            if (hash_out[i] != prefix[i]) {
                match = 0;
                break;
            }
        }
        matches[gid] = match;
    } else {
        matches[gid] = 0;
    }
}

// ============================= DIAGNOSTIC KERNELS ==========================

// Diagnostic: test SHA-512
__kernel void test_sha512_kernel(
    __global const uchar *input,
    uint input_len,
    __global uchar *output  // 64 bytes
) {
    uchar local_input[256];
    for (uint i = 0; i < input_len; i++) local_input[i] = input[i];
    
    uchar digest[64];
    sha512(local_input, input_len, digest);
    for (int i = 0; i < 64; i++) output[i] = digest[i];
}

// Diagnostic: test HMAC-SHA512
__kernel void test_hmac_sha512_kernel(
    __global const uchar *key_in,
    uint key_len,
    __global const uchar *msg_in,
    uint msg_len,
    __global uchar *output  // 64 bytes
) {
    uchar local_key[256];
    for (uint i = 0; i < key_len; i++) local_key[i] = key_in[i];
    uchar local_msg[256];
    for (uint i = 0; i < msg_len; i++) local_msg[i] = msg_in[i];
    
    uchar out[64];
    hmac_sha512_priv(local_key, key_len, local_msg, msg_len, out);
    for (int i = 0; i < 64; i++) output[i] = out[i];
}

// Diagnostic: test PBKDF2 with given iterations
__kernel void test_pbkdf2_kernel(
    __global const uchar *password,
    uint password_len,
    __global const uchar *salt,
    uint salt_len,
    uint iterations,
    __global uchar *output  // 64 bytes
) {
    uchar local_pw[256];
    for (uint i = 0; i < password_len; i++) local_pw[i] = password[i];
    uchar local_salt[80];
    for (uint i = 0; i < salt_len; i++) local_salt[i] = salt[i];
    
    uchar out[64];
    pbkdf2_sha512(local_pw, password_len, local_salt, salt_len, iterations, out);
    for (int i = 0; i < 64; i++) output[i] = out[i];
}

// Diagnostic: test BIP-32 derivation from known seed
__kernel void test_bip32_kernel(
    __global const uchar *seed_in,  // 64 bytes
    __global uchar *output          // 32 bytes (privkey) + 5*32 bytes (intermediate keys) + 5*32 bytes (intermediate chains) = 352 bytes
) {
    uchar seed[64];
    for (int i = 0; i < 64; i++) seed[i] = seed_in[i];
    
    uchar key[32], chain[32], child_il[32], child_chain[32];
    
    // Master key
    uchar btc_seed[12] = {'B','i','t','c','o','i','n',' ','s','e','e','d'};
    uchar master[64];
    hmac_sha512_priv(btc_seed, 12, seed, 64, master);
    
    for (int i = 0; i < 32; i++) {
        key[i] = master[i];
        chain[i] = master[32 + i];
    }
    
    // Output master key
    for (int i = 0; i < 32; i++) output[i] = key[i];
    
    // Hardened: 44', 118', 0'
    uint hardened_indices[3] = { 0x8000002C, 0x80000076, 0x80000000 };
    for (int level = 0; level < 3; level++) {
        bip32_derive_child(key, chain, hardened_indices[level], 1, child_il, child_chain);
        add_privkeys_mod_n(child_il, key, key);
        for (int i = 0; i < 32; i++) chain[i] = child_chain[i];
        // Output this level's key
        for (int i = 0; i < 32; i++) output[32 + level*32 + i] = key[i];
    }
    
    // Normal: 0, 0
    uint normal_indices[2] = { 0, 0 };
    for (int level = 0; level < 2; level++) {
        uint256_t privkey_val;
        for (int i = 0; i < 8; i++) {
            privkey_val.d[i] = ((uint)key[31 - i*4]) |
                               ((uint)key[30 - i*4] << 8) |
                               ((uint)key[29 - i*4] << 16) |
                               ((uint)key[28 - i*4] << 24);
        }
        point_jacobian pub_jac = scalar_mul_G(privkey_val);
        uint256_t z_inv = field_inv(pub_jac.z);
        uint256_t z_inv2 = field_sqr(z_inv);
        uint256_t z_inv3 = field_mul(z_inv2, z_inv);
        uint256_t pub_x = field_mul(pub_jac.x, z_inv2);
        uint256_t pub_y = field_mul(pub_jac.y, z_inv3);
        
        uchar compressed_pubkey[33];
        compressed_pubkey[0] = (pub_y.d[0] & 1) ? 0x03 : 0x02;
        for (int i = 7; i >= 0; i--) {
            compressed_pubkey[1 + (7-i)*4]     = (uchar)(pub_x.d[i] >> 24);
            compressed_pubkey[1 + (7-i)*4 + 1] = (uchar)(pub_x.d[i] >> 16);
            compressed_pubkey[1 + (7-i)*4 + 2] = (uchar)(pub_x.d[i] >> 8);
            compressed_pubkey[1 + (7-i)*4 + 3] = (uchar)(pub_x.d[i]);
        }
        
        bip32_derive_child(compressed_pubkey, chain, normal_indices[level], 0, child_il, child_chain);
        add_privkeys_mod_n(child_il, key, key);
        for (int i = 0; i < 32; i++) chain[i] = child_chain[i];
        // Output this level's key (levels 3 and 4)
        for (int i = 0; i < 32; i++) output[32 + (3+level)*32 + i] = key[i];
    }
}
