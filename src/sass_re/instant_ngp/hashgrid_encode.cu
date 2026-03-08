/*
 * Instant-NGP Hash Grid Encoding — SASS-level inline PTX
 *
 * This is the hottest kernel in instant-NGP. For each 3D query point,
 * it encodes position into a feature vector by:
 *   1. For each of L resolution levels:
 *      a. Scale position to grid coordinates
 *      b. Find the 8 corners of the enclosing voxel
 *      c. Hash each corner → index into feature table
 *      d. Load 2 features per corner (16 loads per level)
 *      e. Trilinear interpolation → 2 output features
 *   2. Concatenate all 2L features as network input
 *
 * Target: SM 8.9 (Ada Lovelace)
 * Every critical instruction is hand-written in PTX to control SASS output.
 *
 * SASS instruction budget per level (target):
 *   - 6× FFMA  (scale + floor + fract)
 *   - 3× F2I   (float grid coords → int)
 *   - 24× IMAD (8 corners × 3 primes for hashing)
 *   - 8× LOP3  (8 corners XOR reduction)
 *   - 8× IMAD  (hash modulo via multiply-shift trick)
 *   - 16× LDG  (8 corners × 2 features)
 *   - 14× FFMA (trilinear interpolation: 7 lerps × 2 features)
 *   Total: ~79 instructions per level, ~948 for 12 levels
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

/* ════════════════════════════════════════════════════════════════
 * Configuration — matches the engine's NeRFConfig
 * ════════════════════════════════════════════════════════════════ */

#define NGP_NUM_LEVELS      12
#define NGP_FEATURES_PER    2       /* features per hash entry */
#define NGP_HASHMAP_SIZE    131072  /* 2^17 entries per level */
#define NGP_LOG2_HASHMAP    17
#define NGP_BASE_RES        16
#define NGP_PER_LEVEL_SCALE 1.5f
#define NGP_TOTAL_FEATURES  (NGP_NUM_LEVELS * NGP_FEATURES_PER)  /* 24 */

/* Spatial hash primes (same as engine's ysu_hash_ijk) */
#define PRIME_X 73856093u
#define PRIME_Y 19349663u
#define PRIME_Z 83492791u


/* ════════════════════════════════════════════════════════════════
 * KERNEL 1: Hash Grid Encoding — Full inline PTX
 *
 * Each thread encodes ONE 3D point → 24 output features.
 * Grid: <<<(N+255)/256, 256>>>
 *
 * Args:
 *   positions:  [N][3] float — normalized 3D positions in [0,1]
 *   hash_table: [L][HASHMAP_SIZE][2] float — feature table
 *   features:   [N][24] float — output encoded features
 *   N:          number of points
 * ════════════════════════════════════════════════════════════════ */

extern "C" __global__ void __launch_bounds__(256)
hashgrid_encode_ptx(
    const float * __restrict__ positions,
    const float * __restrict__ hash_table,
    float       * __restrict__ features,
    int N
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    /* ── Load position (3 coalesced LDG.E.32) ── */
    float px, py, pz;
    {
        const float *pos_ptr = positions + tid * 3;
        asm volatile("ld.global.f32 %0, [%1];"     : "=f"(px) : "l"(pos_ptr));
        asm volatile("ld.global.f32 %0, [%1+4];"   : "=f"(py) : "l"(pos_ptr));
        asm volatile("ld.global.f32 %0, [%1+8];"   : "=f"(pz) : "l"(pos_ptr));
    }

    /* ── Per-level encoding ── */
    float *feat_out = features + tid * NGP_TOTAL_FEATURES;

    /* Precompute resolution table (compile-time unroll) */
    float scale = (float)NGP_BASE_RES;

    #pragma unroll
    for (int level = 0; level < NGP_NUM_LEVELS; level++) {
        /* ── Step 1: Scale position to grid resolution ──
         * SASS: 3× FFMA (px*scale, py*scale, pz*scale) */
        float gx, gy, gz;
        asm volatile("mul.f32 %0, %1, %2;" : "=f"(gx) : "f"(px), "f"(scale));
        asm volatile("mul.f32 %0, %1, %2;" : "=f"(gy) : "f"(py), "f"(scale));
        asm volatile("mul.f32 %0, %1, %2;" : "=f"(gz) : "f"(pz), "f"(scale));

        /* ── Step 2: Floor and fract ──
         * SASS: 3× CVT.RMI (floor), 3× FSUB (fract) */
        float fx, fy, fz;  /* floor */
        float wx, wy, wz;  /* fractional weights for trilinear interp */

        asm volatile("cvt.rmi.f32.f32 %0, %1;" : "=f"(fx) : "f"(gx));
        asm volatile("cvt.rmi.f32.f32 %0, %1;" : "=f"(fy) : "f"(gy));
        asm volatile("cvt.rmi.f32.f32 %0, %1;" : "=f"(fz) : "f"(gz));

        asm volatile("sub.f32 %0, %1, %2;" : "=f"(wx) : "f"(gx), "f"(fx));
        asm volatile("sub.f32 %0, %1, %2;" : "=f"(wy) : "f"(gy), "f"(fy));
        asm volatile("sub.f32 %0, %1, %2;" : "=f"(wz) : "f"(gz), "f"(fz));

        /* Convert floor to integer grid coords
         * SASS: 3× F2I.S32 */
        int ix, iy, iz;
        asm volatile("cvt.rni.s32.f32 %0, %1;" : "=r"(ix) : "f"(fx));
        asm volatile("cvt.rni.s32.f32 %0, %1;" : "=r"(iy) : "f"(fy));
        asm volatile("cvt.rni.s32.f32 %0, %1;" : "=r"(iz) : "f"(fz));

        /* ── Step 3: Hash 8 voxel corners ──
         * Corner (dx,dy,dz) for dx,dy,dz ∈ {0,1}:
         *   hash = ((ix+dx)*PRIME_X) ^ ((iy+dy)*PRIME_Y) ^ ((iz+dz)*PRIME_Z)
         *   index = hash & (HASHMAP_SIZE - 1)   // power-of-2 mask
         *
         * Each corner: 3 IMAD + 2 LOP3.LUT (XOR,XOR) + 1 LOP3 (AND mask)
         * = 6 integer ops per corner × 8 corners = 48 integer ops
         *
         * Optimization: precompute base products, then IADD3 for +1 offsets
         */

        /* Base products (reused across corners) */
        unsigned hx0, hy0, hz0;  /* hash components for (ix, iy, iz) */
        unsigned hx1, hy1, hz1;  /* hash components for (ix+1, iy+1, iz+1) */
        {
            unsigned uix = (unsigned)ix;
            unsigned uiy = (unsigned)iy;
            unsigned uiz = (unsigned)iz;

            /* IMAD: hx0 = ix * PRIME_X
             * SASS: IMAD.U32 */
            asm volatile("mul.lo.u32 %0, %1, %2;"
                : "=r"(hx0) : "r"(uix), "r"(PRIME_X));
            asm volatile("mul.lo.u32 %0, %1, %2;"
                : "=r"(hy0) : "r"(uiy), "r"(PRIME_Y));
            asm volatile("mul.lo.u32 %0, %1, %2;"
                : "=r"(hz0) : "r"(uiz), "r"(PRIME_Z));

            /* +1 offsets: hx1 = (ix+1)*PRIME_X = hx0 + PRIME_X
             * SASS: IADD3 (single instruction, no multiply needed) */
            asm volatile("add.u32 %0, %1, %2;"
                : "=r"(hx1) : "r"(hx0), "r"(PRIME_X));
            asm volatile("add.u32 %0, %1, %2;"
                : "=r"(hy1) : "r"(hy0), "r"(PRIME_Y));
            asm volatile("add.u32 %0, %1, %2;"
                : "=r"(hz1) : "r"(hz0), "r"(PRIME_Z));
        }

        /* Hash + mask for all 8 corners
         * LOP3.LUT with truth table 0x96 = XOR(a,b,c) — 3-input XOR in ONE instruction!
         * This is the key Ada advantage: Pascal needs 2 separate XOR ops.
         *
         * Then AND with (HASHMAP_SIZE-1) for power-of-2 modulo */
        unsigned corner_idx[8];
        const unsigned hash_mask = NGP_HASHMAP_SIZE - 1;

        /* Corner ordering: (dx,dy,dz) = 000,001,010,011,100,101,110,111 */
        #define HASH_CORNER(n, HX, HY, HZ) \
            asm volatile("lop3.b32 %0, %1, %2, %3, 0x96;" \
                : "=r"(corner_idx[n]) : "r"(HX), "r"(HY), "r"(HZ)); \
            asm volatile("and.b32 %0, %0, %1;" \
                : "+r"(corner_idx[n]) : "r"(hash_mask));

        HASH_CORNER(0, hx0, hy0, hz0)  /* (0,0,0) */
        HASH_CORNER(1, hx0, hy0, hz1)  /* (0,0,1) */
        HASH_CORNER(2, hx0, hy1, hz0)  /* (0,1,0) */
        HASH_CORNER(3, hx0, hy1, hz1)  /* (0,1,1) */
        HASH_CORNER(4, hx1, hy0, hz0)  /* (1,0,0) */
        HASH_CORNER(5, hx1, hy0, hz1)  /* (1,0,1) */
        HASH_CORNER(6, hx1, hy1, hz0)  /* (1,1,0) */
        HASH_CORNER(7, hx1, hy1, hz1)  /* (1,1,1) */

        #undef HASH_CORNER

        /* ── Step 4: Load features from hash table ──
         * Each entry has 2 floats. Load as LDG.E.64 (float2) for
         * 2× bandwidth vs two LDG.E.32.
         *
         * Table layout: hash_table[level * HASHMAP_SIZE * 2 + idx * 2]
         */
        float f0[8], f1[8]; /* feature 0 and 1 for each corner */
        {
            unsigned long long level_base =
                (unsigned long long)level * NGP_HASHMAP_SIZE * NGP_FEATURES_PER;
            const float *level_ptr = hash_table + level_base;

            #pragma unroll
            for (int c = 0; c < 8; c++) {
                const float *entry = level_ptr + corner_idx[c] * 2;
                /* LDG.E.64: load 2 floats in one transaction
                 * SASS: LDG.E.64 Rd, [Ra] */
                asm volatile(
                    "ld.global.v2.f32 {%0, %1}, [%2];"
                    : "=f"(f0[c]), "=f"(f1[c])
                    : "l"(entry)
                );
            }
        }

        /* ── Step 5: Trilinear interpolation ──
         * 7 lerps per feature = 14 FFMA total for 2 features.
         *
         * lerp(a, b, t) = a + t*(b-a) = fma(t, b-a, a)
         *   → 1 FSUB + 1 FFMA = 2 ops per lerp
         *   → OR: fma(t, b, fma(-t, a, a)) = 2 FFMA (no FSUB)
         *
         * We use the FMA form: lerp(a,b,t) = fma(t, b-a, a)
         *
         * Interpolation order:
         *   X-axis: 4 lerps (corners 0↔4, 1↔5, 2↔6, 3↔7)
         *   Y-axis: 2 lerps (results 0↔2, 1↔3)
         *   Z-axis: 1 lerp  (results 0↔1)
         */

        /* X-axis interpolation (feature 0) */
        float x00_f0, x01_f0, x10_f0, x11_f0;
        float d0;
        /* lerp(f0[0], f0[4], wx) */
        asm volatile("sub.f32 %0, %1, %2;" : "=f"(d0) : "f"(f0[4]), "f"(f0[0]));
        asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(x00_f0) : "f"(wx), "f"(d0), "f"(f0[0]));
        asm volatile("sub.f32 %0, %1, %2;" : "=f"(d0) : "f"(f0[5]), "f"(f0[1]));
        asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(x01_f0) : "f"(wx), "f"(d0), "f"(f0[1]));
        asm volatile("sub.f32 %0, %1, %2;" : "=f"(d0) : "f"(f0[6]), "f"(f0[2]));
        asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(x10_f0) : "f"(wx), "f"(d0), "f"(f0[2]));
        asm volatile("sub.f32 %0, %1, %2;" : "=f"(d0) : "f"(f0[7]), "f"(f0[3]));
        asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(x11_f0) : "f"(wx), "f"(d0), "f"(f0[3]));

        /* Y-axis interpolation (feature 0) */
        float y0_f0, y1_f0;
        asm volatile("sub.f32 %0, %1, %2;" : "=f"(d0) : "f"(x10_f0), "f"(x00_f0));
        asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y0_f0) : "f"(wy), "f"(d0), "f"(x00_f0));
        asm volatile("sub.f32 %0, %1, %2;" : "=f"(d0) : "f"(x11_f0), "f"(x01_f0));
        asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y1_f0) : "f"(wy), "f"(d0), "f"(x01_f0));

        /* Z-axis interpolation (feature 0) → final */
        float result_f0;
        asm volatile("sub.f32 %0, %1, %2;" : "=f"(d0) : "f"(y1_f0), "f"(y0_f0));
        asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(result_f0) : "f"(wz), "f"(d0), "f"(y0_f0));

        /* Repeat for feature 1 */
        float x00_f1, x01_f1, x10_f1, x11_f1;
        asm volatile("sub.f32 %0, %1, %2;" : "=f"(d0) : "f"(f1[4]), "f"(f1[0]));
        asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(x00_f1) : "f"(wx), "f"(d0), "f"(f1[0]));
        asm volatile("sub.f32 %0, %1, %2;" : "=f"(d0) : "f"(f1[5]), "f"(f1[1]));
        asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(x01_f1) : "f"(wx), "f"(d0), "f"(f1[1]));
        asm volatile("sub.f32 %0, %1, %2;" : "=f"(d0) : "f"(f1[6]), "f"(f1[2]));
        asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(x10_f1) : "f"(wx), "f"(d0), "f"(f1[2]));
        asm volatile("sub.f32 %0, %1, %2;" : "=f"(d0) : "f"(f1[7]), "f"(f1[3]));
        asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(x11_f1) : "f"(wx), "f"(d0), "f"(f1[3]));

        float y0_f1, y1_f1;
        asm volatile("sub.f32 %0, %1, %2;" : "=f"(d0) : "f"(x10_f1), "f"(x00_f1));
        asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y0_f1) : "f"(wy), "f"(d0), "f"(x00_f1));
        asm volatile("sub.f32 %0, %1, %2;" : "=f"(d0) : "f"(x11_f1), "f"(x01_f1));
        asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y1_f1) : "f"(wy), "f"(d0), "f"(x01_f1));

        float result_f1;
        asm volatile("sub.f32 %0, %1, %2;" : "=f"(d0) : "f"(y1_f1), "f"(y0_f1));
        asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(result_f1) : "f"(wz), "f"(d0), "f"(y0_f1));

        /* ── Step 6: Store 2 features ──
         * STG.E.64 for coalesced write */
        {
            float *out = feat_out + level * NGP_FEATURES_PER;
            asm volatile(
                "st.global.v2.f32 [%0], {%1, %2};"
                : : "l"(out), "f"(result_f0), "f"(result_f1)
            );
        }

        /* Scale for next level: scale *= PER_LEVEL_SCALE
         * SASS: FMUL */
        asm volatile("mul.f32 %0, %0, %1;" : "+f"(scale) : "f"(NGP_PER_LEVEL_SCALE));
    }
}


/* ════════════════════════════════════════════════════════════════
 * Reference CUDA implementation (for validation)
 * ════════════════════════════════════════════════════════════════ */

__device__ __forceinline__
uint32_t hash_corner_ref(int ix, int iy, int iz) {
    uint32_t h = ((uint32_t)ix * PRIME_X) ^ ((uint32_t)iy * PRIME_Y)
               ^ ((uint32_t)iz * PRIME_Z);
    return h & (NGP_HASHMAP_SIZE - 1);
}

__device__ __forceinline__
float lerp_ref(float a, float b, float t) {
    return a + t * (b - a);
}

extern "C" __global__ void __launch_bounds__(256)
hashgrid_encode_ref(
    const float * __restrict__ positions,
    const float * __restrict__ hash_table,
    float       * __restrict__ features,
    int N
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    float px = positions[tid * 3 + 0];
    float py = positions[tid * 3 + 1];
    float pz = positions[tid * 3 + 2];

    float *feat_out = features + tid * NGP_TOTAL_FEATURES;
    float scale = (float)NGP_BASE_RES;

    for (int level = 0; level < NGP_NUM_LEVELS; level++) {
        float gx = px * scale;
        float gy = py * scale;
        float gz = pz * scale;

        int ix = (int)floorf(gx);
        int iy = (int)floorf(gy);
        int iz = (int)floorf(gz);

        float wx = gx - floorf(gx);
        float wy = gy - floorf(gy);
        float wz = gz - floorf(gz);

        /* 8 corner lookups */
        float f0[8], f1[8];
        for (int dz = 0; dz < 2; dz++)
        for (int dy = 0; dy < 2; dy++)
        for (int dx = 0; dx < 2; dx++) {
            int c = dx * 4 + dy * 2 + dz;
            uint32_t idx = hash_corner_ref(ix + dx, iy + dy, iz + dz);
            const float *entry = hash_table
                + (size_t)level * NGP_HASHMAP_SIZE * NGP_FEATURES_PER
                + idx * NGP_FEATURES_PER;
            f0[c] = entry[0];
            f1[c] = entry[1];
        }

        /* Trilinear interpolation */
        float x00 = lerp_ref(f0[0], f0[4], wx);
        float x01 = lerp_ref(f0[1], f0[5], wx);
        float x10 = lerp_ref(f0[2], f0[6], wx);
        float x11 = lerp_ref(f0[3], f0[7], wx);
        float y0  = lerp_ref(x00, x10, wy);
        float y1  = lerp_ref(x01, x11, wy);
        feat_out[level * 2 + 0] = lerp_ref(y0, y1, wz);

        x00 = lerp_ref(f1[0], f1[4], wx);
        x01 = lerp_ref(f1[1], f1[5], wx);
        x10 = lerp_ref(f1[2], f1[6], wx);
        x11 = lerp_ref(f1[3], f1[7], wx);
        y0  = lerp_ref(x00, x10, wy);
        y1  = lerp_ref(x01, x11, wy);
        feat_out[level * 2 + 1] = lerp_ref(y0, y1, wz);

        scale *= NGP_PER_LEVEL_SCALE;
    }
}
