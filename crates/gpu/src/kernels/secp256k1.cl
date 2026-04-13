// =============================================================================
// cosmos-vanity — shared secp256k1 GPU kernel
// =============================================================================
//
// Full secp256k1 EC point multiplication on GPU:
//   privkey (32 bytes) → pubkey (33 bytes compressed) → SHA-256 → RIPEMD-160
//
// All arithmetic is mod p = 2^256 - 2^32 - 977 (secp256k1 field prime).
// Points use Jacobian coordinates for efficiency.
// Scalar multiplication uses double-and-add.
//
// Target: AMD RX 9070 XT (RDNA 4, gfx1201), ROCm 7.2 OpenCL

// ---- 256-bit unsigned integer (8 × 32-bit limbs, little-endian) ----
typedef struct { uint d[8]; } uint256_t;

// ---- Jacobian point (X, Y, Z) where affine = (X/Z², Y/Z³) ----
typedef struct { uint256_t x, y, z; } point_jacobian;

// ---- Affine point (x, y) ----
typedef struct { uint256_t x, y; } point_affine;

// Windows CUDA uses 32-bit `long`, while OpenCL `long` is 64-bit.
// Keep signed 64-bit arithmetic explicit so borrow/carry math is stable across toolchains.
#ifdef __CUDACC__
typedef long long slong;
#else
typedef long slong;
#endif

// ---- secp256k1 field prime p ----
// p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
// In little-endian limbs: d[0] = least significant
__constant uint P[8] = {
    0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
};

// 2*P for subtraction borrow handling
__constant uint P2[8] = {
    0xFFFFF85E, 0xFFFFFFFD, 0xFFFFFFFF, 0xFFFFFFFF,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
};

// ---- Generator point G (affine, big-endian hex converted to LE limbs) ----
// Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
__constant uint GX[8] = {
    0x16F81798, 0x59F2815B, 0x2DCE28D9, 0x029BFCDB,
    0xCE870B07, 0x55A06295, 0xF9DCBBAC, 0x79BE667E
};
// Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
__constant uint GY[8] = {
    0xFB10D4B8, 0x9C47D08F, 0xA6855419, 0xFD17B448,
    0x0E1108A8, 0x5DA4FBFC, 0x26A3C465, 0x483ADA77
};

// ---- uint256 helpers ----

uint256_t uint256_zero() {
    uint256_t r;
    for (int i = 0; i < 8; i++) r.d[i] = 0;
    return r;
}

uint256_t uint256_one() {
    uint256_t r;
    r.d[0] = 1;
    for (int i = 1; i < 8; i++) r.d[i] = 0;
    return r;
}

int uint256_is_zero(uint256_t a) {
    for (int i = 0; i < 8; i++)
        if (a.d[i] != 0) return 0;
    return 1;
}

// Compare a >= b
int uint256_gte(uint256_t a, uint256_t b) {
    for (int i = 7; i >= 0; i--) {
        if (a.d[i] > b.d[i]) return 1;
        if (a.d[i] < b.d[i]) return 0;
    }
    return 1; // equal
}

// Compare a >= P (constant)
int uint256_gte_p(uint256_t a) {
    for (int i = 7; i >= 0; i--) {
        if (a.d[i] > P[i]) return 1;
        if (a.d[i] < P[i]) return 0;
    }
    return 1; // equal
}

// ---- Field arithmetic mod p ----

// a + b mod p
uint256_t field_add(uint256_t a, uint256_t b) {
    uint256_t r;
    ulong carry = 0;
    for (int i = 0; i < 8; i++) {
        ulong s = (ulong)a.d[i] + (ulong)b.d[i] + carry;
        r.d[i] = (uint)s;
        carry = s >> 32;
    }
    // If carry or r >= p, subtract p
    if (carry || uint256_gte_p(r)) {
        ulong borrow = 0;
        for (int i = 0; i < 8; i++) {
            ulong s = (ulong)r.d[i] - (ulong)P[i] - borrow;
            r.d[i] = (uint)s;
            borrow = (s >> 63) & 1;
        }
    }
    return r;
}

// a - b mod p
uint256_t field_sub(uint256_t a, uint256_t b) {
    uint256_t r;
    slong borrow = 0;
    for (int i = 0; i < 8; i++) {
        slong s = (slong)(ulong)a.d[i] - (slong)(ulong)b.d[i] - borrow;
        r.d[i] = (uint)s;
        borrow = (s < 0) ? 1 : 0;
    }
    if (borrow) {
        // Add p back
        ulong carry = 0;
        for (int i = 0; i < 8; i++) {
            ulong s = (ulong)r.d[i] + (ulong)P[i] + carry;
            r.d[i] = (uint)s;
            carry = s >> 32;
        }
    }
    return r;
}

// Reduce a 512-bit product modulo p using secp256k1's special form:
//   p = 2^256 - C where C = 2^32 + 977 = 0x1000003D1
// So: 2^256 ≡ C (mod p), meaning high * C + low (mod p)
// We do this iteratively since the result of high * C may itself overflow.
uint256_t field_reduce(uint lo[8], uint hi[8]) {
    // result = lo + hi * (2^32 + 977)
    // hi * 2^32 = shift hi left by one limb
    // hi * 977 = hi * 0x3D1
    uint256_t r;
    ulong carry = 0;

    // First pass: result = lo + hi * (2^32 + 977)
    // hi * 977: we need to multiply each hi limb by 977 and add
    // hi * 2^32: effectively hi[i] goes to position i+1

    // We compute: acc = lo + hi_shifted + hi * 977
    // hi_shifted[0] = 0, hi_shifted[i] = hi[i-1] for i=1..8, hi_shifted[9] = hi[7] (overflow)

    uint t[10]; // temporary 10-limb result
    for (int i = 0; i < 10; i++) t[i] = 0;

    // Start with lo
    for (int i = 0; i < 8; i++) t[i] = lo[i];

    // Add hi << 32 (shifted by one limb position)
    carry = 0;
    for (int i = 0; i < 8; i++) {
        ulong s = (ulong)t[i + 1] + (ulong)hi[i] + carry;
        t[i + 1] = (uint)s;
        carry = s >> 32;
    }
    t[9] = (uint)carry;

    // Add hi * 977
    carry = 0;
    for (int i = 0; i < 8; i++) {
        ulong s = (ulong)t[i] + (ulong)hi[i] * 977UL + carry;
        t[i] = (uint)s;
        carry = s >> 32;
    }
    // Propagate remaining carry
    for (int i = 8; i < 10; i++) {
        ulong s = (ulong)t[i] + carry;
        t[i] = (uint)s;
        carry = s >> 32;
    }

    // Now t is at most ~258 bits. The overflow (t[8], t[9]) needs another reduction.
    // overflow = t[8..9], new result = t[0..7] + overflow * (2^32 + 977)
    uint hi2[2] = { t[8], t[9] };

    // Second pass reduction (overflow is small, at most ~34 bits)
    carry = 0;
    // Add hi2 * 977
    for (int i = 0; i < 2; i++) {
        ulong s = (ulong)t[i] + (ulong)hi2[i] * 977UL + carry;
        t[i] = (uint)s;
        carry = s >> 32;
    }
    for (int i = 2; i < 8; i++) {
        ulong s = (ulong)t[i] + carry;
        t[i] = (uint)s;
        carry = s >> 32;
    }

    // Add hi2 << 32 (shift by 1 limb)
    ulong carry2 = 0;
    for (int i = 0; i < 2; i++) {
        ulong s = (ulong)t[i + 1] + (ulong)hi2[i] + carry2;
        t[i + 1] = (uint)s;
        carry2 = s >> 32;
    }
    for (int i = 3; i < 8; i++) {
        ulong s = (ulong)t[i] + carry2;
        t[i] = (uint)s;
        carry2 = s >> 32;
    }

    // Handle any final carry: carry2 * (2^32 + 977)
    if (carry2) {
        ulong c = 0;
        ulong s = (ulong)t[0] + carry2 * 977UL + c;
        t[0] = (uint)s;
        c = s >> 32;
        s = (ulong)t[1] + carry2 + c;
        t[1] = (uint)s;
        c = s >> 32;
        for (int i = 2; i < 8 && c; i++) {
            s = (ulong)t[i] + c;
            t[i] = (uint)s;
            c = s >> 32;
        }
    }

    for (int i = 0; i < 8; i++) r.d[i] = t[i];

    // Final conditional subtraction
    if (uint256_gte_p(r)) {
        ulong borrow = 0;
        for (int i = 0; i < 8; i++) {
            ulong s = (ulong)r.d[i] - (ulong)P[i] - borrow;
            r.d[i] = (uint)s;
            borrow = (s >> 63) & 1;
        }
    }
    return r;
}

// a * b mod p — schoolbook multiplication with secp256k1 fast reduction
uint256_t field_mul(uint256_t a, uint256_t b) {
    uint lo[8], hi[8];
    for (int i = 0; i < 8; i++) { lo[i] = 0; hi[i] = 0; }

    // 8×8 schoolbook multiplication → 16 limbs
    uint prod[16];
    for (int i = 0; i < 16; i++) prod[i] = 0;

    for (int i = 0; i < 8; i++) {
        ulong carry = 0;
        for (int j = 0; j < 8; j++) {
            ulong p = (ulong)a.d[i] * (ulong)b.d[j] + (ulong)prod[i + j] + carry;
            prod[i + j] = (uint)p;
            carry = p >> 32;
        }
        prod[i + 8] = (uint)carry;
    }

    for (int i = 0; i < 8; i++) { lo[i] = prod[i]; hi[i] = prod[i + 8]; }
    return field_reduce(lo, hi);
}

// a² mod p — optimized squaring
uint256_t field_sqr(uint256_t a) {
    uint prod[16];
    for (int i = 0; i < 16; i++) prod[i] = 0;

    // Off-diagonal terms (doubled)
    for (int i = 0; i < 8; i++) {
        ulong carry = 0;
        for (int j = i + 1; j < 8; j++) {
            ulong p = (ulong)a.d[i] * (ulong)a.d[j] + (ulong)prod[i + j] + carry;
            prod[i + j] = (uint)p;
            carry = p >> 32;
        }
        prod[i + 8] = (uint)carry;
    }

    // Double
    ulong carry = 0;
    for (int i = 0; i < 16; i++) {
        ulong v = ((ulong)prod[i] << 1) | carry;
        prod[i] = (uint)v;
        carry = v >> 32;
    }

    // Add diagonal terms (a[i]*a[i])
    carry = 0;
    for (int i = 0; i < 8; i++) {
        ulong p = (ulong)a.d[i] * (ulong)a.d[i] + (ulong)prod[2 * i] + carry;
        prod[2 * i] = (uint)p;
        carry = p >> 32;
        ulong s = (ulong)prod[2 * i + 1] + carry;
        prod[2 * i + 1] = (uint)s;
        carry = s >> 32;
    }

    uint lo[8], hi[8];
    for (int i = 0; i < 8; i++) { lo[i] = prod[i]; hi[i] = prod[i + 8]; }
    return field_reduce(lo, hi);
}

// Field negation: -a mod p = p - a (if a != 0)
uint256_t field_neg(uint256_t a) {
    if (uint256_is_zero(a)) return a;
    uint256_t r;
    ulong borrow = 0;
    for (int i = 0; i < 8; i++) {
        ulong s = (ulong)P[i] - (ulong)a.d[i] - borrow;
        r.d[i] = (uint)s;
        borrow = (s >> 63) & 1;
    }
    return r;
}

// Modular inverse via Fermat's little theorem: a^(p-2) mod p
// Using an addition chain for secp256k1's p-2
uint256_t field_inv(uint256_t a) {
    // p - 2 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2D
    //
    // We use a standard addition chain for secp256k1 field inversion.
    // This requires ~255 squarings and ~15 multiplications.

    uint256_t x2, x3, x6, x9, x11, x22, x44, x88, x176, x220, x223, t;

    // x2 = a^(2^2 - 1) = a^3
    x2 = field_sqr(a);
    x2 = field_mul(x2, a);

    // x3 = a^(2^3 - 1) = a^7
    x3 = field_sqr(x2);
    x3 = field_mul(x3, a);

    // x6 = a^(2^6 - 1)
    x6 = x3;
    for (int i = 0; i < 3; i++) x6 = field_sqr(x6);
    x6 = field_mul(x6, x3);

    // x9 = a^(2^9 - 1)
    x9 = x6;
    for (int i = 0; i < 3; i++) x9 = field_sqr(x9);
    x9 = field_mul(x9, x3);

    // x11 = a^(2^11 - 1)
    x11 = x9;
    for (int i = 0; i < 2; i++) x11 = field_sqr(x11);
    x11 = field_mul(x11, x2);

    // x22 = a^(2^22 - 1)
    x22 = x11;
    for (int i = 0; i < 11; i++) x22 = field_sqr(x22);
    x22 = field_mul(x22, x11);

    // x44 = a^(2^44 - 1)
    x44 = x22;
    for (int i = 0; i < 22; i++) x44 = field_sqr(x44);
    x44 = field_mul(x44, x22);

    // x88 = a^(2^88 - 1)
    x88 = x44;
    for (int i = 0; i < 44; i++) x88 = field_sqr(x88);
    x88 = field_mul(x88, x44);

    // x176 = a^(2^176 - 1)
    x176 = x88;
    for (int i = 0; i < 88; i++) x176 = field_sqr(x176);
    x176 = field_mul(x176, x88);

    // x220 = a^(2^220 - 1)
    x220 = x176;
    for (int i = 0; i < 44; i++) x220 = field_sqr(x220);
    x220 = field_mul(x220, x44);

    // x223 = a^(2^223 - 1)
    x223 = x220;
    for (int i = 0; i < 3; i++) x223 = field_sqr(x223);
    x223 = field_mul(x223, x3);

    // Now compute a^(p-2):
    // p-2 = 2^256 - 2^32 - 979
    // = 2^256 - 2^32 - 0x3D3
    // The exponent in binary from MSB:
    // 223 ones, then specific pattern for the low 33 bits
    //
    // t = x223 << 23 (23 squarings)
    t = x223;
    for (int i = 0; i < 23; i++) t = field_sqr(t);
    // Now t = a^((2^223-1) * 2^23) = a^(2^246 - 2^23)
    // Multiply by x22 = a^(2^22-1)
    t = field_mul(t, x22);
    // t = a^(2^246 - 2^23 + 2^22 - 1) = a^(2^246 - 2^22 - 1)
    // Hmm, let me use the standard secp256k1 inversion chain more carefully.

    // Standard chain for p-2:
    // p-2 has the bit pattern:
    // bits 255..32: all 1s except bit 32 is 0 (since p = 2^256 - 2^32 - 977)
    // Actually: p-2 = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFFC2D
    // bits[255:33] = all 1s (223 bits)
    // bit 32 = 0
    // bits[31:0] = 0xFFFFFC2D = 11111111111111111111110000101101

    // Restart with cleaner approach:
    // t = x223
    t = x223;

    // Square 23 times to shift up
    for (int i = 0; i < 23; i++) t = field_sqr(t);
    t = field_mul(t, x22);
    // 5 more squarings
    for (int i = 0; i < 5; i++) t = field_sqr(t);
    t = field_mul(t, a);
    // 3 squarings
    for (int i = 0; i < 3; i++) t = field_sqr(t);
    t = field_mul(t, x2);
    // 2 squarings
    for (int i = 0; i < 2; i++) t = field_sqr(t);
    t = field_mul(t, a);

    return t;
}

// ---- Point operations in Jacobian coordinates ----

// Check if point is at infinity (Z == 0)
int point_is_infinity(point_jacobian P) {
    return uint256_is_zero(P.z);
}

// Point doubling: R = 2*P (Jacobian)
// Using standard formulas for a=0 (secp256k1):
//   A = Y1^2
//   B = X1 * A
//   C = A^2
//   D = 3 * X1^2  (since a=0, simplified from 3*X1^2 + a*Z1^4)
//   X3 = D^2 - 8*B  (using S = 2*B: X3 = D^2 - 2*S)
//   Y3 = D*(S - X3) - 8*C (using M = D: Y3 = M*(S-X3) - 8*C)
//   wait, let me use the standard doubling formula directly:
//
// For secp256k1 (a=0):
//   M = 3 * X1^2
//   S = 4 * X1 * Y1^2
//   T = M^2 - 2*S
//   X3 = T
//   Y3 = M * (S - T) - 8 * Y1^4
//   Z3 = 2 * Y1 * Z1
point_jacobian point_double(point_jacobian P) {
    point_jacobian R;

    if (point_is_infinity(P)) return P;

    uint256_t Y2 = field_sqr(P.y);       // Y1^2
    uint256_t S = field_mul(P.x, Y2);    // X1 * Y1^2
    S = field_add(S, S);
    S = field_add(S, S);                 // S = 4 * X1 * Y1^2

    uint256_t X2 = field_sqr(P.x);       // X1^2
    uint256_t M = field_add(X2, field_add(X2, X2)); // M = 3 * X1^2

    uint256_t T = field_sqr(M);           // M^2
    uint256_t S2 = field_add(S, S);       // 2*S
    T = field_sub(T, S2);                 // T = M^2 - 2*S
    R.x = T;

    uint256_t Y4 = field_sqr(Y2);        // Y1^4
    uint256_t Y4_8 = field_add(Y4, Y4);
    Y4_8 = field_add(Y4_8, Y4_8);
    Y4_8 = field_add(Y4_8, Y4_8);         // 8 * Y1^4

    R.y = field_sub(S, T);               // S - T
    R.y = field_mul(M, R.y);             // M * (S - T)
    R.y = field_sub(R.y, Y4_8);          // M * (S - T) - 8 * Y1^4

    R.z = field_mul(P.y, P.z);           // Y1 * Z1
    R.z = field_add(R.z, R.z);           // 2 * Y1 * Z1

    return R;
}

// Point addition: R = P + Q (Jacobian, P != Q)
// Standard Jacobian addition formulas:
//   U1 = X1*Z2^2, U2 = X2*Z1^2
//   S1 = Y1*Z2^3, S2 = Y2*Z1^3
//   H = U2 - U1, R = S2 - S1
//   X3 = R^2 - H^3 - 2*U1*H^2
//   Y3 = R*(U1*H^2 - X3) - S1*H^3
//   Z3 = H * Z1 * Z2
point_jacobian point_add(point_jacobian P, point_jacobian Q) {
    if (point_is_infinity(P)) return Q;
    if (point_is_infinity(Q)) return P;

    uint256_t Z1sq = field_sqr(P.z);
    uint256_t Z2sq = field_sqr(Q.z);

    uint256_t U1 = field_mul(P.x, Z2sq);          // X1 * Z2^2
    uint256_t U2 = field_mul(Q.x, Z1sq);          // X2 * Z1^2

    uint256_t Z1cu = field_mul(Z1sq, P.z);
    uint256_t Z2cu = field_mul(Z2sq, Q.z);

    uint256_t S1 = field_mul(P.y, Z2cu);           // Y1 * Z2^3
    uint256_t S2 = field_mul(Q.y, Z1cu);           // Y2 * Z1^3

    uint256_t H = field_sub(U2, U1);               // H = U2 - U1
    uint256_t R = field_sub(S2, S1);               // R = S2 - S1

    // If H == 0, points have same x-coordinate
    if (uint256_is_zero(H)) {
        if (uint256_is_zero(R)) {
            // Same point: use doubling
            return point_double(P);
        } else {
            // Point at infinity (P = -Q)
            point_jacobian inf;
            inf.x = uint256_one();
            inf.y = uint256_one();
            inf.z = uint256_zero();
            return inf;
        }
    }

    uint256_t H2 = field_sqr(H);                   // H^2
    uint256_t H3 = field_mul(H2, H);               // H^3
    uint256_t U1H2 = field_mul(U1, H2);            // U1 * H^2

    point_jacobian Res;
    uint256_t R2 = field_sqr(R);                    // R^2
    Res.x = field_sub(R2, H3);                     // R^2 - H^3
    Res.x = field_sub(Res.x, field_add(U1H2, U1H2)); // R^2 - H^3 - 2*U1*H^2

    Res.y = field_sub(U1H2, Res.x);               // U1*H^2 - X3
    Res.y = field_mul(R, Res.y);                   // R * (U1*H^2 - X3)
    Res.y = field_sub(Res.y, field_mul(S1, H3));   // - S1*H^3

    Res.z = field_mul(H, P.z);
    Res.z = field_mul(Res.z, Q.z);                 // H * Z1 * Z2

    return Res;
}

// Mixed addition: R = P (Jacobian) + Q (Affine, Z=1)
// Saves multiplications since Z2 = 1
point_jacobian point_add_mixed(point_jacobian P, point_affine Q) {
    if (point_is_infinity(P)) {
        point_jacobian R;
        R.x = Q.x;
        R.y = Q.y;
        R.z = uint256_one();
        return R;
    }

    uint256_t Z1sq = field_sqr(P.z);
    uint256_t Z1cu = field_mul(Z1sq, P.z);

    uint256_t U1 = P.x;                            // X1 (since Z2 = 1)
    uint256_t U2 = field_mul(Q.x, Z1sq);           // X2 * Z1^2

    uint256_t S1 = P.y;                            // Y1 (since Z2 = 1)
    uint256_t S2 = field_mul(Q.y, Z1cu);           // Y2 * Z1^3

    uint256_t H = field_sub(U2, U1);
    uint256_t R = field_sub(S2, S1);

    if (uint256_is_zero(H)) {
        if (uint256_is_zero(R)) {
            return point_double(P);
        } else {
            point_jacobian inf;
            inf.x = uint256_one();
            inf.y = uint256_one();
            inf.z = uint256_zero();
            return inf;
        }
    }

    uint256_t H2 = field_sqr(H);
    uint256_t H3 = field_mul(H2, H);
    uint256_t U1H2 = field_mul(U1, H2);

    point_jacobian Res;
    uint256_t R2 = field_sqr(R);
    Res.x = field_sub(R2, H3);
    Res.x = field_sub(Res.x, field_add(U1H2, U1H2));

    Res.y = field_sub(U1H2, Res.x);
    Res.y = field_mul(R, Res.y);
    Res.y = field_sub(Res.y, field_mul(S1, H3));

    Res.z = field_mul(H, P.z);                     // H * Z1 (since Z2 = 1)

    return Res;
}

// ---- Scalar multiplication: k * G using double-and-add ----

point_jacobian scalar_mul_G(uint256_t k) {
    point_affine G;
    for (int i = 0; i < 8; i++) {
        G.x.d[i] = GX[i];
        G.y.d[i] = GY[i];
    }

    point_jacobian R;
    R.x = uint256_one();
    R.y = uint256_one();
    R.z = uint256_zero(); // Point at infinity

    // Find highest set bit
    int start_bit = -1;
    for (int w = 7; w >= 0; w--) {
        if (k.d[w] != 0) {
            // Find highest bit in this word
            uint v = k.d[w];
            int bit = 31;
            while (bit >= 0 && !(v & (1u << bit))) bit--;
            start_bit = w * 32 + bit;
            break;
        }
    }

    if (start_bit < 0) return R; // k == 0

    // Double-and-add from MSB to LSB
    for (int i = start_bit; i >= 0; i--) {
        R = point_double(R);
        uint word = k.d[i / 32];
        uint bit = (word >> (i % 32)) & 1;
        if (bit) {
            R = point_add_mixed(R, G);
        }
    }

    return R;
}

// ---- Jacobian → Affine conversion ----

// Convert Jacobian point to compressed public key (33 bytes, big-endian)
void point_to_compressed(point_jacobian P, uchar *out) {
    if (point_is_infinity(P)) {
        // This shouldn't happen for valid private keys
        for (int i = 0; i < 33; i++) out[i] = 0;
        return;
    }

    uint256_t z_inv = field_inv(P.z);
    uint256_t z_inv2 = field_sqr(z_inv);
    uint256_t z_inv3 = field_mul(z_inv2, z_inv);

    uint256_t x = field_mul(P.x, z_inv2);
    uint256_t y = field_mul(P.y, z_inv3);

    // Prefix: 0x02 if y even, 0x03 if y odd
    out[0] = (y.d[0] & 1) ? 0x03 : 0x02;

    // X coordinate in big-endian
    for (int i = 7; i >= 0; i--) {
        int offset = 1 + (7 - i) * 4;
        out[offset + 0] = (x.d[i] >> 24) & 0xFF;
        out[offset + 1] = (x.d[i] >> 16) & 0xFF;
        out[offset + 2] = (x.d[i] >> 8) & 0xFF;
        out[offset + 3] = x.d[i] & 0xFF;
    }
}


// =============================================================================
// SHA-256 and RIPEMD-160 (self-contained, same as vanity_search.cl)
// =============================================================================

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

__constant uint KL_RMD[5] = { 0x00000000, 0x5a827999, 0x6ed9eba1, 0x8f1bbcdc, 0xa953fd4e };
__constant uint KR_RMD[5] = { 0x50a28be6, 0x5c4dd124, 0x6d703ef3, 0x7a6d76e9, 0x00000000 };

__constant uchar RL_T[80] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    7, 4, 13, 1, 10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8,
    3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12,
    1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2,
    4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13
};

__constant uchar RR_T[80] = {
    5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12,
    6, 11, 3, 7, 0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2,
    15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13,
    8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14,
    12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11
};

__constant uchar SL_T[80] = {
    11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8,
    7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12,
    11, 13, 6, 7, 14, 9, 13, 15, 14, 8, 13, 6, 5, 12, 7, 5,
    11, 12, 14, 15, 14, 15, 9, 8, 9, 14, 5, 6, 8, 6, 5, 12,
    9, 15, 5, 11, 6, 8, 13, 12, 5, 12, 13, 14, 11, 8, 5, 6
};

__constant uchar SR_T[80] = {
    8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6,
    9, 13, 15, 7, 12, 8, 9, 11, 7, 7, 12, 7, 6, 15, 13, 11,
    9, 7, 15, 11, 8, 6, 6, 14, 12, 13, 5, 14, 13, 13, 7, 5,
    15, 5, 8, 11, 14, 14, 6, 14, 6, 9, 12, 9, 12, 5, 15, 8,
    8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6, 5, 15, 13, 11, 11
};

uint rotl32(uint x, uint n) {
    return (x << n) | (x >> (32 - n));
}

// SHA-256 of exactly 33 bytes
void sha256_33(const uchar *input, uint *hash) {
    uint W[64];
    for (int i = 0; i < 16; i++) W[i] = 0;
    for (int i = 0; i < 33; i++) {
        W[i / 4] |= ((uint)input[i]) << (24 - (i % 4) * 8);
    }
    W[33 / 4] |= 0x80 << (24 - (33 % 4) * 8);
    W[15] = 33 * 8;

    for (int i = 16; i < 64; i++) {
        uint s0 = (W[i-15] >> 7 | W[i-15] << 25) ^ (W[i-15] >> 18 | W[i-15] << 14) ^ (W[i-15] >> 3);
        uint s1 = (W[i-2] >> 17 | W[i-2] << 15) ^ (W[i-2] >> 19 | W[i-2] << 13) ^ (W[i-2] >> 10);
        W[i] = W[i-16] + s0 + W[i-7] + s1;
    }

    uint a = 0x6a09e667, b = 0xbb67ae85, c = 0x3c6ef372, d = 0xa54ff53a;
    uint e = 0x510e527f, f = 0x9b05688c, g = 0x1f83d9ab, h = 0x5be0cd19;

    for (int i = 0; i < 64; i++) {
        uint S1 = (e >> 6 | e << 26) ^ (e >> 11 | e << 21) ^ (e >> 25 | e << 7);
        uint ch = (e & f) ^ (~e & g);
        uint T1 = h + S1 + ch + K_SHA256[i] + W[i];
        uint S0 = (a >> 2 | a << 30) ^ (a >> 13 | a << 19) ^ (a >> 22 | a << 10);
        uint maj = (a & b) ^ (a & c) ^ (b & c);
        uint T2 = S0 + maj;
        h = g; g = f; f = e; e = d + T1;
        d = c; c = b; b = a; a = T1 + T2;
    }

    hash[0] = a + 0x6a09e667; hash[1] = b + 0xbb67ae85;
    hash[2] = c + 0x3c6ef372; hash[3] = d + 0xa54ff53a;
    hash[4] = e + 0x510e527f; hash[5] = f + 0x9b05688c;
    hash[6] = g + 0x1f83d9ab; hash[7] = h + 0x5be0cd19;
}

// RIPEMD-160 of exactly 32 bytes
void ripemd160_32(uint *sha_hash, uint *rmd_hash) {
    uint W[16];
    for (int i = 0; i < 8; i++) {
        uint v = sha_hash[i];
        W[i] = ((v >> 24) & 0xFF) | ((v >> 8) & 0xFF00) |
               ((v << 8) & 0xFF0000) | ((v << 24) & 0xFF000000);
    }
    W[8] = 0x00000080;
    for (int i = 9; i < 14; i++) W[i] = 0;
    W[14] = 32 * 8;
    W[15] = 0;

    uint al = 0x67452301, bl = 0xefcdab89, cl = 0x98badcfe, dl = 0x10325476, el = 0xc3d2e1f0;
    uint ar = al, br = bl, cr = cl, dr = dl, er = el;

    for (int j = 0; j < 80; j++) {
        uint fl, fr, tl, tr;
        int round = j / 16;

        if (round == 0)      fl = bl ^ cl ^ dl;
        else if (round == 1) fl = (bl & cl) | (~bl & dl);
        else if (round == 2) fl = (bl | ~cl) ^ dl;
        else if (round == 3) fl = (bl & dl) | (cl & ~dl);
        else                 fl = bl ^ (cl | ~dl);

        tl = al + fl + W[RL_T[j]] + KL_RMD[round];
        tl = rotl32(tl, (uint)SL_T[j]) + el;
        al = el; el = dl; dl = rotl32(cl, 10); cl = bl; bl = tl;

        if (round == 0)      fr = br ^ (cr | ~dr);
        else if (round == 1) fr = (br & dr) | (cr & ~dr);
        else if (round == 2) fr = (br | ~cr) ^ dr;
        else if (round == 3) fr = (br & cr) | (~br & dr);
        else                 fr = br ^ cr ^ dr;

        tr = ar + fr + W[RR_T[j]] + KR_RMD[round];
        tr = rotl32(tr, (uint)SR_T[j]) + er;
        ar = er; er = dr; dr = rotl32(cr, 10); cr = br; br = tr;
    }

    uint t = 0xefcdab89 + cl + dr;
    rmd_hash[0] = t;
    rmd_hash[1] = 0x98badcfe + dl + er;
    rmd_hash[2] = 0x10325476 + el + ar;
    rmd_hash[3] = 0xc3d2e1f0 + al + br;
    rmd_hash[4] = 0x67452301 + bl + cr;
}


// =============================================================================
// Main kernel: privkey → pubkey → address hash
// =============================================================================

__kernel void generate_addresses(
    __global const uchar *privkeys,    // N × 32 bytes (big-endian private keys)
    __global uchar *pubkeys,           // N × 33 bytes (compressed pubkeys)
    __global uchar *hashes,            // N × 20 bytes (address hashes)
    __global const uchar *prefix,      // prefix bytes to match
    uint prefix_len,                   // length of prefix (0 = no matching)
    __global uint *matches,            // N × uint: 1 if match, 0 if not
    uint count                         // number of keys
) {
    uint gid = get_global_id(0);
    if (gid >= count) return;

    // 1. Read private key (big-endian bytes → little-endian limbs)
    __global const uchar *pk_bytes = privkeys + gid * 32;
    uint256_t k;
    for (int i = 0; i < 8; i++) {
        int byte_offset = (7 - i) * 4; // big-endian: MSB first
        k.d[i] = ((uint)pk_bytes[byte_offset] << 24) |
                 ((uint)pk_bytes[byte_offset + 1] << 16) |
                 ((uint)pk_bytes[byte_offset + 2] << 8) |
                 ((uint)pk_bytes[byte_offset + 3]);
    }

    // 2. Scalar multiplication: pubkey = k * G
    point_jacobian pub_point = scalar_mul_G(k);

    // 3. Compress: Jacobian → 33-byte compressed pubkey
    uchar compressed[33];
    point_to_compressed(pub_point, compressed);

    // Write compressed pubkey to output
    __global uchar *pk_out = pubkeys + gid * 33;
    for (int i = 0; i < 33; i++) pk_out[i] = compressed[i];

    // 4. SHA-256(compressed_pubkey)
    uint sha_out[8];
    sha256_33(compressed, sha_out);

    // 5. RIPEMD-160(SHA-256(pubkey))
    uint rmd_out[5];
    ripemd160_32(sha_out, rmd_out);

    // Write 20-byte hash output (little-endian words to bytes)
    __global uchar *hash_out = hashes + gid * 20;
    for (int i = 0; i < 5; i++) {
        hash_out[i*4 + 0] = (rmd_out[i]) & 0xFF;
        hash_out[i*4 + 1] = (rmd_out[i] >> 8) & 0xFF;
        hash_out[i*4 + 2] = (rmd_out[i] >> 16) & 0xFF;
        hash_out[i*4 + 3] = (rmd_out[i] >> 24) & 0xFF;
    }

    // 6. Prefix matching on raw hash bytes
    if (prefix_len > 0) {
        uint match_flag = 1;
        for (uint i = 0; i < prefix_len && i < 20; i++) {
            if (hash_out[i] != prefix[i]) {
                match_flag = 0;
                break;
            }
        }
        matches[gid] = match_flag;
    } else {
        matches[gid] = 0;
    }
}
