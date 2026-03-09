#include "nerf_simd.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <cpuid.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* Portable prefetch macro for reducing cache misses on random hashgrid lookups */
#if defined(__GNUC__) || defined(__clang__)
#define YSU_PREFETCH(addr) __builtin_prefetch((const void*)(addr), 0, 0)
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#define YSU_PREFETCH(addr) _mm_prefetch((const char*)(addr), _MM_HINT_T0)
#else
#define YSU_PREFETCH(addr) ((void)0)
#endif

/* Detect AVX2 and AVX-512 at runtime */
static void cpuid(uint32_t leaf, uint32_t subleaf, uint32_t *eax, uint32_t *ebx, uint32_t *ecx, uint32_t *edx) {
    #ifdef _MSC_VER
        int regs[4];
        __cpuidex(regs, leaf, subleaf);
        *eax = regs[0];
        *ebx = regs[1];
        *ecx = regs[2];
        *edx = regs[3];
    #else
        __cpuid_count(leaf, subleaf, *eax, *ebx, *ecx, *edx);
    #endif
}

CPUFeatures ysu_detect_cpu_features(void) {
    CPUFeatures features = {0};
    
    uint32_t eax, ebx, ecx, edx;
    
    /* Check leaf 1 for AVX (ECX bit 28) */
    cpuid(1, 0, &eax, &ebx, &ecx, &edx);
    bool has_avx = (ecx & (1 << 28)) != 0;
    
    /* Check leaf 7 for AVX2 (EBX bit 5) and AVX-512 (EBX bits 16-17) */
    cpuid(7, 0, &eax, &ebx, &ecx, &edx);
    features.has_avx2 = has_avx && ((ebx & (1 << 5)) != 0);
    features.has_avx512f = (ebx & (1 << 16)) != 0;
    
    if (features.has_avx2) {
        fprintf(stderr, "✓ CPU supports AVX2\n");
    } else {
        fprintf(stderr, "ℹ CPU does NOT support AVX2 (will use scalar fallback)\n");
    }
    
    if (features.has_avx512f) {
        fprintf(stderr, "✓ CPU supports AVX-512F\n");
    }
    
    return features;
}

/* Half-float utilities (for fp16 exported weights/tables) */
static float ysu_half_to_float(uint16_t h) {
    uint16_t exp = h & 0x7C00u;
    uint16_t mant = h & 0x03FFu;
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t f_exp;
    uint32_t f_mant;
    if (exp == 0x7C00u) {
        /* NaN or Inf */
        f_exp = 0xFFu;
        f_mant = (uint32_t)mant << 13;
    } else if (exp != 0) {
        /* Normalized */
        f_exp = (exp >> 10) + (127u - 15u);
        f_mant = (uint32_t)mant << 13;
    } else if (mant != 0) {
        /* Subnormal */
        uint32_t sig = mant;
        int shift = -1;
        do {
            shift++;
            sig <<= 1;
        } while ((sig & 0x0400u) == 0u);
        sig &= 0x03FFu;
        f_exp = (uint32_t)(127 - 15 - shift);
        f_mant = sig << 13;
    } else {
        /* Zero */
        f_exp = 0;
        f_mant = 0;
    }

    uint32_t bits = sign | (f_exp << 23) | f_mant;
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static bool ysu_read_half_block(FILE *f, float *dst, uint32_t count) {
    size_t bytes = (size_t)count * sizeof(uint16_t);
    uint16_t *tmp = (uint16_t*)malloc(bytes);
    if (!tmp) {
        return false;
    }
    if (fread(tmp, sizeof(uint16_t), count, f) != count) {
        free(tmp);
        return false;
    }
    for (uint32_t i = 0; i < count; i++) {
        dst[i] = ysu_half_to_float(tmp[i]);
    }
    free(tmp);
    return true;
}

/* ===== SIMD Utilities ===== */

#ifdef __AVX2__
#include <immintrin.h>
#include <x86intrin.h>

static inline uint64_t ysu_rdtsc(void) {
    return __builtin_ia32_rdtsc();
}

static inline __m256 ysu_sigmoid_avx2(__m256 x) {
    /* sigmoid(x) = 1 / (1 + exp(-x)) - Fast approximation */
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 neg_one = _mm256_set1_ps(-1.0f);
    
    /* Fast sigmoid approximation: clamp(0.5 + 0.2 * x, 0, 1) */
    __m256 sigmoid_approx = _mm256_mul_ps(x, _mm256_set1_ps(0.2f));
    sigmoid_approx = _mm256_add_ps(sigmoid_approx, _mm256_set1_ps(0.5f));
    sigmoid_approx = _mm256_max_ps(sigmoid_approx, _mm256_set1_ps(0.0f));
    sigmoid_approx = _mm256_min_ps(sigmoid_approx, one);
    
    return sigmoid_approx;
}

static inline __m256 ysu_relu_avx2(__m256 x) {
    return _mm256_max_ps(x, _mm256_set1_ps(0.0f));
}

#else  /* Scalar fallback for CPUs without AVX2 */

static inline uint64_t ysu_rdtsc(void) {
    /* Fallback: not available on older CPUs */
    return 0;
}

static inline float ysu_sigmoid_scalar(float x) {
    /* sigmoid(x) ≈ clamp(0.5 + 0.2 * x, 0, 1) */
    float result = 0.5f + 0.2f * x;
    if (result < 0.0f) return 0.0f;
    if (result > 1.0f) return 1.0f;
    return result;
}

static inline float ysu_relu_scalar(float x) {
    return x > 0.0f ? x : 0.0f;
}

#endif

/* ===== Hash Function ===== */

static inline uint32_t ysu_hash_ijk(
    int32_t i, int32_t j, int32_t k,
    uint32_t hash_size
) {
    /* Cast to unsigned for hash */
    uint32_t ui = (uint32_t)i;
    uint32_t uj = (uint32_t)j;
    uint32_t uk = (uint32_t)k;
    
    /* Spatial hash with prime multipliers */
    uint32_t hash = (ui * 73856093u) ^ (uj * 19349663u) ^ (uk * 83492791u);
    return hash % hash_size;
}

static inline uint32_t ysu_hash_position(
    const Vec3 *pos,
    float level_scale,
    uint32_t hash_size
) {
    /* Scale position and convert to integer coordinates */
    int32_t x = (int32_t)floorf(pos->x * level_scale);
    int32_t y = (int32_t)floorf(pos->y * level_scale);
    int32_t z = (int32_t)floorf(pos->z * level_scale);
    
    return ysu_hash_ijk(x, y, z, hash_size);
}

/* ===== NeRF Data Loading ===== */

NeRFData* ysu_nerf_data_load(const char *hashgrid_path, const char *occ_path) {
    FILE *f_hash = fopen(hashgrid_path, "rb");
    if (!f_hash) {
        fprintf(stderr, "ERROR: Cannot open hashgrid file: %s\n", hashgrid_path);
        return NULL;
    }

    FILE *f_occ = fopen(occ_path, "rb");
    if (!f_occ) {
        fprintf(stderr, "ERROR: Cannot open occupancy file: %s\n", occ_path);
        fclose(f_hash);
        return NULL;
    }

    NeRFData *data = (NeRFData*)malloc(sizeof(NeRFData));
    if (!data) {
        fclose(f_hash);
        fclose(f_occ);
        return NULL;
    }
    memset(data, 0, sizeof(NeRFData));

    /* Read header (15 u32 = 60 bytes) */
    uint32_t header[15];
    if (fread(header, sizeof(uint32_t), 15, f_hash) != 15) {
        fprintf(stderr, "ERROR: Failed to read NeRF header\n");
        fclose(f_hash);
        fclose(f_occ);
        free(data);
        return NULL;
    }

    data->config.magic = header[0];
    data->config.version = header[1];
    data->config.num_levels = header[2];
    data->config.features_per_entry = header[3];
    data->config.hashmap_size = header[4];
    data->config.base_res = header[5];
    data->config.per_level_scale = *(float*)&header[6];
    data->config.mlp_in_dim = header[7];
    data->config.mlp_hidden_dim = header[8];
    data->config.mlp_num_layers = header[9];
    data->config.mlp_out_dim = header[10];
    data->config.scale = *(float*)&header[11];
    data->config.center.x = *(float*)&header[12];
    data->config.center.y = *(float*)&header[13];
    data->config.center.z = *(float*)&header[14];

    bool fp16_format = data->config.version >= 2;

    printf("[NeRF] Loaded config: levels=%u, hash_size=%u, mlp=%u->%u->%u (v%u, %s weights)\n",
           data->config.num_levels, data->config.hashmap_size,
           data->config.mlp_in_dim, data->config.mlp_hidden_dim, data->config.mlp_out_dim,
           data->config.version, fp16_format ? "fp16" : "fp32");

    /* Hashgrid: stored as fp16 tables in version >= 2 */
    uint32_t grid_elems = data->config.num_levels * data->config.hashmap_size *
                          data->config.features_per_entry;
    size_t grid_bytes = (size_t)grid_elems * sizeof(uint16_t);
    uint16_t *grid_raw = (uint16_t*)malloc(grid_bytes);
    if (!grid_raw) {
        fclose(f_hash);
        fclose(f_occ);
        free(data);
        return NULL;
    }
    if (fread(grid_raw, sizeof(uint16_t), grid_elems, f_hash) != grid_elems) {
        fprintf(stderr, "ERROR: Failed to read hashgrid data\n");
        free(grid_raw);
        fclose(f_hash);
        fclose(f_occ);
        free(data);
        return NULL;
    }
    data->hashgrid_data = (float*)malloc((size_t)grid_elems * sizeof(float));
    if (!data->hashgrid_data) {
        free(grid_raw);
        fclose(f_hash);
        fclose(f_occ);
        free(data);
        return NULL;
    }
    for (uint32_t i = 0; i < grid_elems; i++) {
        if (fp16_format) {
            data->hashgrid_data[i] = ysu_half_to_float(grid_raw[i]);
        } else {
            float v = (float)grid_raw[i] / 32767.5f - 1.0f;
            if (v > 1.0f) v = 1.0f;
            if (v < -1.0f) v = -1.0f;
            data->hashgrid_data[i] = v;
        }
    }
    free(grid_raw);

    /* MLP sizes
     * WEIGHT LAYOUT NOTE: this CPU inference path stores W0 as
     * [in_dim][hidden_dim] = [27][64] (row-major, outer index = input feature).
     * The GPU kernel in src/sass_re/instant_ngp/mlp_forward.cu stores W0 as
     * [hidden_dim][in_dim] = [64][27] (outer index = neuron).
     * These are TRANSPOSED from each other — GPU-trained checkpoints cannot
     * be loaded here without transposing W0 and W1 first.
     * Set env var YSU_NERF_GPU_WEIGHTS=1 to emit a reminder at load time. */
    if (getenv("YSU_NERF_GPU_WEIGHTS")) {
        fprintf(stderr,
            "[NeRF] WARNING: YSU_NERF_GPU_WEIGHTS set. GPU checkpoints store W0 as\n"
            "  [hidden][in]=[64][27]. This CPU code expects [in][hidden]=[27][64].\n"
            "  Transpose W0 and W1 before loading, or results will be wrong.\n");
    }
    uint32_t w0_size = data->config.mlp_hidden_dim * data->config.mlp_in_dim;
    uint32_t b0_size = data->config.mlp_hidden_dim;
    uint32_t w1_size = data->config.mlp_hidden_dim * data->config.mlp_hidden_dim;
    uint32_t b1_size = data->config.mlp_hidden_dim;
    uint32_t w_out_size = data->config.mlp_out_dim * data->config.mlp_hidden_dim;
    uint32_t b_out_size = data->config.mlp_out_dim;

    uint32_t total_weight_elems = w0_size + w1_size + w_out_size;
    uint32_t total_bias_elems = b0_size + b1_size + b_out_size;

    data->mlp_weights = (float*)malloc((size_t)total_weight_elems * sizeof(float));
    data->mlp_biases = (float*)malloc((size_t)total_bias_elems * sizeof(float));
    if (!data->mlp_weights || !data->mlp_biases) {
        free(data->hashgrid_data);
        free(data->mlp_weights);
        free(data->mlp_biases);
        fclose(f_hash);
        fclose(f_occ);
        free(data);
        return NULL;
    }

    if (fp16_format) {
        uint32_t w_off = 0;
        uint32_t b_off = 0;
        if (!ysu_read_half_block(f_hash, data->mlp_weights + w_off, w0_size)) goto load_fail;
        w_off += w0_size;
        if (!ysu_read_half_block(f_hash, data->mlp_biases + b_off, b0_size)) goto load_fail;
        b_off += b0_size;
        if (!ysu_read_half_block(f_hash, data->mlp_weights + w_off, w1_size)) goto load_fail;
        w_off += w1_size;
        if (!ysu_read_half_block(f_hash, data->mlp_biases + b_off, b1_size)) goto load_fail;
        b_off += b1_size;
        if (!ysu_read_half_block(f_hash, data->mlp_weights + w_off, w_out_size)) goto load_fail;
        w_off += w_out_size;
        if (!ysu_read_half_block(f_hash, data->mlp_biases + b_off, b_out_size)) goto load_fail;
    } else {
        size_t weights_bytes = (size_t)total_weight_elems * sizeof(float);
        size_t biases_bytes = (size_t)total_bias_elems * sizeof(float);
        if (fread(data->mlp_weights, 1, weights_bytes, f_hash) != weights_bytes) goto load_fail;
        if (fread(data->mlp_biases, 1, biases_bytes, f_hash) != biases_bytes) goto load_fail;
    }

    /* Load occupancy grid (version >=2 includes a 16-byte header) */
    uint32_t occ_dim = 64;
    float occ_threshold = 0.0f;
    if (fp16_format) {
        uint32_t occ_magic = 0;
        if (fread(&occ_magic, sizeof(uint32_t), 1, f_occ) != 1) goto load_fail;
        if (occ_magic != 0x31474F4Eu) {
            fprintf(stderr, "ERROR: Occupancy grid magic mismatch (0x%08X)\n", occ_magic);
            goto load_fail;
        }
        if (fread(&occ_dim, sizeof(uint32_t), 1, f_occ) != 1) goto load_fail;
        float voxel_size = 0.0f;
        if (fread(&voxel_size, sizeof(float), 1, f_occ) != 1) goto load_fail;
        if (fread(&occ_threshold, sizeof(float), 1, f_occ) != 1) goto load_fail;
        (void)voxel_size;  /* Currently unused */
    }

    size_t occ_count = (size_t)occ_dim * occ_dim * occ_dim;
    data->occupancy_grid = (uint8_t*)malloc(occ_count);
    if (!data->occupancy_grid) goto load_fail;
    if (fread(data->occupancy_grid, 1, occ_count, f_occ) != occ_count) {
        fprintf(stderr, "ERROR: Failed to read occupancy grid\n");
        goto load_fail;
    }

    fclose(f_hash);
    fclose(f_occ);

    printf("[NeRF] Loaded %.2f KB hashgrid, %u weights, %u biases, occupancy %u^3 (thr=%.4f)\n",
           (double)(grid_elems * sizeof(float)) / 1024.0,
           total_weight_elems, total_bias_elems, occ_dim, occ_threshold);

    return data;

load_fail:
    fclose(f_hash);
    fclose(f_occ);
    ysu_nerf_data_free(data);
    return NULL;
}

void ysu_nerf_data_free(NeRFData *data) {
    if (!data) return;
    free(data->hashgrid_data);
    free(data->mlp_weights);
    free(data->mlp_biases);
    free(data->occupancy_grid);
    free(data);
}

/* ===== Batched Hashgrid Lookup with Trilinear Interpolation ===== */

void ysu_hashgrid_lookup_batch(
    const Vec3 positions[SIMD_BATCH_SIZE],
    const NeRFConfig *config,
    const float *hashgrid_data,
    float features_out[SIMD_BATCH_SIZE][24]
) {
    /* For each level, look up features per position with trilinear interpolation.
     * Output array is features_out[SIMD_BATCH_SIZE][24], so we must ensure
     * batch_levels * features_per_entry <= 24 to avoid out-of-bounds writes. */
    uint32_t fpe = config->features_per_entry > 0 ? config->features_per_entry : 2;
    uint32_t max_levels = 24 / fpe;   /* max levels that fit in the [24] output */
    uint32_t batch_levels = config->num_levels < max_levels ? config->num_levels : max_levels;
    for (uint32_t level = 0; level < batch_levels; level++) {
        /* Match Python: res = int(base_res * (per_level_scale ** l)) */
        float res = (float)(int)(config->base_res * powf(config->per_level_scale, (float)level));
        uint32_t level_offset = level * config->hashmap_size * config->features_per_entry;
        
        for (uint32_t ray = 0; ray < SIMD_BATCH_SIZE; ray++) {
            Vec3 p = positions[ray];
            
            /* Scale position to grid coordinates */
            float gx = p.x * res;
            float gy = p.y * res;
            float gz = p.z * res;
            
            /* Get integer corner and fractional weights */
            float fx = floorf(gx);
            float fy = floorf(gy);
            float fz = floorf(gz);
            int32_t i0 = (int32_t)fx;
            int32_t j0 = (int32_t)fy;
            int32_t k0 = (int32_t)fz;
            
            /* Interpolation weights */
            float wx = gx - fx;
            float wy = gy - fy;
            float wz = gz - fz;
            
            /* For each feature, do trilinear interpolation over 8 corners */
            for (uint32_t f = 0; f < config->features_per_entry; f++) {
                /* Hash all 8 corners */
                uint32_t h000 = ysu_hash_ijk(i0,   j0,   k0,   config->hashmap_size);
                uint32_t h001 = ysu_hash_ijk(i0,   j0,   k0+1, config->hashmap_size);
                uint32_t h010 = ysu_hash_ijk(i0,   j0+1, k0,   config->hashmap_size);
                uint32_t h011 = ysu_hash_ijk(i0,   j0+1, k0+1, config->hashmap_size);
                uint32_t h100 = ysu_hash_ijk(i0+1, j0,   k0,   config->hashmap_size);
                uint32_t h101 = ysu_hash_ijk(i0+1, j0,   k0+1, config->hashmap_size);
                uint32_t h110 = ysu_hash_ijk(i0+1, j0+1, k0,   config->hashmap_size);
                uint32_t h111 = ysu_hash_ijk(i0+1, j0+1, k0+1, config->hashmap_size);
                
                /* Prefetch all 8 hash-indexed cache lines before reading */
                YSU_PREFETCH(&hashgrid_data[level_offset + h000 * config->features_per_entry + f]);
                YSU_PREFETCH(&hashgrid_data[level_offset + h001 * config->features_per_entry + f]);
                YSU_PREFETCH(&hashgrid_data[level_offset + h010 * config->features_per_entry + f]);
                YSU_PREFETCH(&hashgrid_data[level_offset + h011 * config->features_per_entry + f]);
                YSU_PREFETCH(&hashgrid_data[level_offset + h100 * config->features_per_entry + f]);
                YSU_PREFETCH(&hashgrid_data[level_offset + h101 * config->features_per_entry + f]);
                YSU_PREFETCH(&hashgrid_data[level_offset + h110 * config->features_per_entry + f]);
                YSU_PREFETCH(&hashgrid_data[level_offset + h111 * config->features_per_entry + f]);

                /* Look up feature values at each corner */
                float v000 = hashgrid_data[level_offset + h000 * config->features_per_entry + f];
                float v001 = hashgrid_data[level_offset + h001 * config->features_per_entry + f];
                float v010 = hashgrid_data[level_offset + h010 * config->features_per_entry + f];
                float v011 = hashgrid_data[level_offset + h011 * config->features_per_entry + f];
                float v100 = hashgrid_data[level_offset + h100 * config->features_per_entry + f];
                float v101 = hashgrid_data[level_offset + h101 * config->features_per_entry + f];
                float v110 = hashgrid_data[level_offset + h110 * config->features_per_entry + f];
                float v111 = hashgrid_data[level_offset + h111 * config->features_per_entry + f];
                
                /* Trilinear interpolation */
                /* First lerp in Z */
                float v00 = v000 * (1.0f - wz) + v001 * wz;
                float v01 = v010 * (1.0f - wz) + v011 * wz;
                float v10 = v100 * (1.0f - wz) + v101 * wz;
                float v11 = v110 * (1.0f - wz) + v111 * wz;
                
                /* Then lerp in Y */
                float v0 = v00 * (1.0f - wy) + v01 * wy;
                float v1 = v10 * (1.0f - wy) + v11 * wy;
                
                /* Finally lerp in X */
                float val = v0 * (1.0f - wx) + v1 * wx;

                /* Bug fix: was `level * 2` — hardcoded stride breaks when
                 * features_per_entry != 2. Use config->features_per_entry. */
                features_out[ray][level * config->features_per_entry + f] = val;
            }
        }
    }
}
/* ===== Batched Occupancy Lookup ===== */

void ysu_occupancy_lookup_batch(
    const Vec3 positions[SIMD_BATCH_SIZE],
    const NeRFConfig *config,
    const uint8_t *occ_grid,
    uint8_t occupancy_out[SIMD_BATCH_SIZE]
) {
    for (uint32_t ray = 0; ray < SIMD_BATCH_SIZE; ray++) {
        /* Transform position to occupancy grid coordinates [0, 64) */
        Vec3 p = positions[ray];
        
        /* Normalize to [0, 1] relative to bounds */
        float bounds = config->scale;
        Vec3 normalized;
        normalized.x = (p.x - config->center.x) / bounds + 0.5f;
        normalized.y = (p.y - config->center.y) / bounds + 0.5f;
        normalized.z = (p.z - config->center.z) / bounds + 0.5f;
        
        /* Clamp to [0, 1] */
        normalized.x = fmaxf(0.0f, fminf(1.0f, normalized.x));
        normalized.y = fmaxf(0.0f, fminf(1.0f, normalized.y));
        normalized.z = fmaxf(0.0f, fminf(1.0f, normalized.z));
        
        /* Scale to [0, 63] */
        uint32_t xi = (uint32_t)(normalized.x * 63.0f);
        uint32_t yi = (uint32_t)(normalized.y * 63.0f);
        uint32_t zi = (uint32_t)(normalized.z * 63.0f);
        
        /* Linear index into 64^3 grid */
        uint32_t idx = zi * 64 * 64 + yi * 64 + xi;
        occupancy_out[ray] = occ_grid[idx];
    }
}

/* ===== Batched MLP Inference (cache-friendly loop order) ===== */

void ysu_mlp_inference_batch(
    const float features_in[SIMD_BATCH_SIZE][27],
    const NeRFConfig *config,
    const float *mlp_weights,
    const float *mlp_biases,
    float rgb_out[SIMD_BATCH_SIZE][3],
    float sigma_out[SIMD_BATCH_SIZE]
) {
    uint32_t in_dim = config->mlp_in_dim;
    uint32_t hidden_dim = config->mlp_hidden_dim;
    uint32_t out_dim = config->mlp_out_dim;
    
    /* Guard: batch hidden arrays are sized for hidden_dim up to 128.
     * Wider networks are not supported by this fast path. */
    if (hidden_dim > 128) {
        fprintf(stderr, "[NeRF] ERROR: hidden_dim=%u exceeds batch inference limit (128)\n", hidden_dim);
        return;
    }
    /* Hidden layer output [8 rays][hidden_dim] */
    float hidden[SIMD_BATCH_SIZE][128];
    
    /* Get weight pointers — layout is [in_dim][hidden_dim] (row-major, CPU convention).
     * NOTE: the GPU kernel (mlp_forward.cu) uses [hidden_dim][in_dim] — TRANSPOSED.
     * Weights loaded from .ysub files use this CPU [in][hidden] layout. */
    const float *w0 = mlp_weights;
    const float *b0 = mlp_biases;
    const float *w1 = w0 + hidden_dim * in_dim;
    const float *b1 = b0 + hidden_dim;
    const float *w_out = w1 + hidden_dim * hidden_dim;
    const float *b_out = b1 + hidden_dim;
    
    /* Layer 0: Input -> Hidden (27 -> 64)
     * Restructured: iterate inputs OUTER so w0[i * hidden_dim + h] scans
     * contiguously over h in the inner loop — reads full cache lines. */
    for (uint32_t ray = 0; ray < SIMD_BATCH_SIZE; ray++) {
        for (uint32_t h = 0; h < hidden_dim; h++) {
            hidden[ray][h] = b0[h];
        }
    }
    for (uint32_t i = 0; i < in_dim; i++) {
        const float *w0_row = &w0[i * hidden_dim]; /* contiguous row */
        for (uint32_t ray = 0; ray < SIMD_BATCH_SIZE; ray++) {
            float fi = features_in[ray][i];
            for (uint32_t h = 0; h < hidden_dim; h++) {
                hidden[ray][h] += w0_row[h] * fi;
            }
        }
    }
    for (uint32_t ray = 0; ray < SIMD_BATCH_SIZE; ray++) {
        for (uint32_t h = 0; h < hidden_dim; h++) {
            hidden[ray][h] = fmaxf(0.0f, hidden[ray][h]);
        }
    }
    
    /* Layer 1: Hidden -> Hidden (64 -> 64) with ReLU — same cache-friendly pattern */
    float hidden2[SIMD_BATCH_SIZE][128];
    for (uint32_t ray = 0; ray < SIMD_BATCH_SIZE; ray++) {
        for (uint32_t h = 0; h < hidden_dim; h++) {
            hidden2[ray][h] = b1[h];
        }
    }
    for (uint32_t h_prev = 0; h_prev < hidden_dim; h_prev++) {
        const float *w1_row = &w1[h_prev * hidden_dim]; /* contiguous row */
        for (uint32_t ray = 0; ray < SIMD_BATCH_SIZE; ray++) {
            float val = hidden[ray][h_prev];
            for (uint32_t h = 0; h < hidden_dim; h++) {
                hidden2[ray][h] += w1_row[h] * val;
            }
        }
    }
    for (uint32_t ray = 0; ray < SIMD_BATCH_SIZE; ray++) {
        for (uint32_t h = 0; h < hidden_dim; h++) {
            hidden2[ray][h] = fmaxf(0.0f, hidden2[ray][h]);
        }
    }
    
    /* Output layer: Hidden -> Output — scan w_out rows contiguously.
     * mlp_num_layers > 2 not yet implemented; warn if mismatch. */
    if (config->mlp_num_layers != 2) {
        fprintf(stderr, "[NeRF] WARNING: mlp_num_layers=%u but inference is hardcoded for 2 hidden layers\n",
                config->mlp_num_layers);
    }
    float out_acc[SIMD_BATCH_SIZE][4];
    for (uint32_t ray = 0; ray < SIMD_BATCH_SIZE; ray++) {
        for (uint32_t o = 0; o < out_dim && o < 4; o++) {
            out_acc[ray][o] = b_out[o];
        }
    }
    for (uint32_t h = 0; h < hidden_dim; h++) {
        const float *wout_row = &w_out[h * out_dim]; /* contiguous row */
        for (uint32_t ray = 0; ray < SIMD_BATCH_SIZE; ray++) {
            float hv = hidden2[ray][h];
            for (uint32_t o = 0; o < out_dim && o < 4; o++) {
                out_acc[ray][o] += wout_row[o] * hv;
            }
        }
    }
    /* Apply activations */
    for (uint32_t ray = 0; ray < SIMD_BATCH_SIZE; ray++) {
        for (uint32_t o = 0; o < out_dim; o++) {
            float val = out_acc[ray][o];
            if (o < 3) {
                float sigmoid = 1.0f / (1.0f + expf(-val));
                rgb_out[ray][o] = sigmoid;
            } else {
                float sigma;
                if (val > 20.0f) {
                    sigma = val;
                } else if (val < -20.0f) {
                    sigma = 0.0f;
                } else {
                    sigma = logf(1.0f + expf(val));
                }
                sigma_out[ray] = fminf(50.0f, sigma);
            }
        }
    }
}

/* Single-ray MLP inference optimized for single-lane execution.
 * Uses weight access patterns that favor the stored layout [in_dim, hidden_dim]
 * and [hidden_dim, hidden_dim] to improve cache locality. */
void ysu_mlp_inference_single(
    const float features_in[27],
    const NeRFConfig *config,
    const float *mlp_weights,
    const float *mlp_biases,
    float rgb_out[3],
    float *sigma_out
) {
    uint32_t in_dim = config->mlp_in_dim;
    uint32_t hidden_dim = config->mlp_hidden_dim;
    uint32_t out_dim = config->mlp_out_dim;

    /* Guard: hidden[] and hidden2[] are stack arrays of size 128.
     * Also out_acc[] is size 16. Reject configs that would overflow. */
    if (hidden_dim > 128) {
        fprintf(stderr, "[NeRF] ERROR: hidden_dim=%u exceeds single inference limit (128)\n", hidden_dim);
        return;
    }
    if (out_dim > 16) {
        fprintf(stderr, "[NeRF] ERROR: mlp_out_dim=%u exceeds single inference limit (16)\n", out_dim);
        return;
    }

    /* Warn if the saved model has a depth we can't handle correctly. */
    if (config->mlp_num_layers != 2) {
        fprintf(stderr, "[NeRF] WARNING: mlp_num_layers=%u but single inference is hardcoded for 2 hidden layers\n",
                config->mlp_num_layers);
    }

    /* Pointers into packed weights/biases (same layout as batch version) */
    const float *w0 = mlp_weights; /* [in_dim][hidden_dim] */
    const float *b0 = mlp_biases;  /* [hidden_dim] */
    const float *w1 = w0 + (size_t)hidden_dim * in_dim; /* [hidden_dim][hidden_dim] */
    const float *b1 = b0 + hidden_dim; /* [hidden_dim] */
    const float *w_out = w1 + (size_t)hidden_dim * hidden_dim; /* [hidden_dim][out_dim] */
    const float *b_out = b1 + hidden_dim; /* [out_dim] */

    /* Layer 0: Input -> Hidden
     * Compute hidden = b0 + sum_i features_in[i] * w0[i]
     * We iterate inputs outer so w0 row is contiguous over hidden_dim. */
    float hidden[128]; /* allocate a bit more, but only use hidden_dim */

#ifdef __AVX2__
    /* Vectorized init: load biases into hidden */
    uint32_t h = 0;
    for (; h + 7 < hidden_dim; h += 8) {
        __m256 v = _mm256_loadu_ps(&b0[h]);
        _mm256_storeu_ps(&hidden[h], v);
    }
    for (; h < hidden_dim; ++h) hidden[h] = b0[h];

    /* Vectorized input->hidden: for each input, fused multiply-add across hidden dim */
    for (uint32_t i = 0; i < in_dim; i++) {
        float fi = features_in[i];
        __m256 fvec = _mm256_set1_ps(fi);
        const float *w0_row = &w0[(size_t)i * hidden_dim];
        uint32_t hh = 0;
        for (; hh + 7 < hidden_dim; hh += 8) {
            __m256 wvec = _mm256_loadu_ps(&w0_row[hh]);
            __m256 acc = _mm256_loadu_ps(&hidden[hh]);
            acc = _mm256_add_ps(_mm256_mul_ps(wvec, fvec), acc);
            _mm256_storeu_ps(&hidden[hh], acc);
        }
        for (; hh < hidden_dim; ++hh) hidden[hh] += w0_row[hh] * fi;
    }

    /* ReLU */
    for (uint32_t hh = 0; hh < hidden_dim; ++hh) {
        hidden[hh] = hidden[hh] > 0.0f ? hidden[hh] : 0.0f;
    }
#else
    for (uint32_t h = 0; h < hidden_dim; h++) hidden[h] = b0[h];

    for (uint32_t i = 0; i < in_dim; i++) {
        float fi = features_in[i];
        const float *w0_row = &w0[(size_t)i * hidden_dim];
        for (uint32_t h = 0; h < hidden_dim; h++) {
            hidden[h] += w0_row[h] * fi;
        }
    }
    for (uint32_t h = 0; h < hidden_dim; h++) {
        hidden[h] = hidden[h] > 0.0f ? hidden[h] : 0.0f;
    }
#endif

    /* Layer 1: Hidden -> Hidden
     * Compute hidden2 = b1 + sum_hprev hidden[hprev] * w1[hprev]
     * Vectorize over hidden dimension when AVX2 is available. */
    float hidden2[128];
#ifdef __AVX2__
    uint32_t hh = 0;
    for (; hh + 7 < hidden_dim; hh += 8) {
        __m256 v = _mm256_loadu_ps(&b1[hh]);
        _mm256_storeu_ps(&hidden2[hh], v);
    }
    for (; hh < hidden_dim; ++hh) hidden2[hh] = b1[hh];

    for (uint32_t hprev = 0; hprev < hidden_dim; hprev++) {
        float val = hidden[hprev];
        __m256 val_vec = _mm256_set1_ps(val);
        const float *w1_row = &w1[(size_t)hprev * hidden_dim];
        uint32_t h2 = 0;
        for (; h2 + 7 < hidden_dim; h2 += 8) {
            __m256 wvec = _mm256_loadu_ps(&w1_row[h2]);
            __m256 acc = _mm256_loadu_ps(&hidden2[h2]);
            acc = _mm256_add_ps(_mm256_mul_ps(wvec, val_vec), acc);
            _mm256_storeu_ps(&hidden2[h2], acc);
        }
        for (; h2 < hidden_dim; ++h2) hidden2[h2] += w1_row[h2] * val;
    }

    for (uint32_t h3 = 0; h3 < hidden_dim; ++h3) hidden2[h3] = hidden2[h3] > 0.0f ? hidden2[h3] : 0.0f;
#else
    for (uint32_t h = 0; h < hidden_dim; h++) hidden2[h] = b1[h];

    for (uint32_t hprev = 0; hprev < hidden_dim; hprev++) {
        float val = hidden[hprev];
        const float *w1_row = &w1[(size_t)hprev * hidden_dim];
        for (uint32_t h = 0; h < hidden_dim; h++) {
            hidden2[h] += w1_row[h] * val;
        }
    }
    for (uint32_t h = 0; h < hidden_dim; h++) {
        hidden2[h] = hidden2[h] > 0.0f ? hidden2[h] : 0.0f;
    }
#endif

    /* Output layer: hidden2 -> out
     * We accumulate per-output in a small array for cache-friendly writes. */
    float out_acc[16];
    for (uint32_t o = 0; o < out_dim; o++) out_acc[o] = b_out[o];

    for (uint32_t h = 0; h < hidden_dim; h++) {
        const float *wout_row = &w_out[(size_t)h * out_dim];
        float hv = hidden2[h];
        for (uint32_t o = 0; o < out_dim; o++) {
            out_acc[o] += wout_row[o] * hv;
        }
    }

    /* Apply activations: first 3 are RGB (sigmoid), last is sigma (softplus-like).
     * Fix: initialise sigma_out to 0 before the loop so it is never left
     * unwritten when out_dim < 4 (e.g. RGB-only checkpoint). */
    *sigma_out = 0.0f;
    for (uint32_t o = 0; o < out_dim; o++) {
        float val = out_acc[o];
        if (o < 3) {
            rgb_out[o] = 1.0f / (1.0f + expf(-val));
        } else {
            float sigma;
            if (val > 20.0f) sigma = val;
            else if (val < -20.0f) sigma = 0.0f;
            else sigma = logf(1.0f + expf(val));
            *sigma_out = sigma;
        }
    }
}

/* ===== Adaptive Sampling ===== */

float ysu_adaptive_step_size(
    const Vec3 pos,
    const uint8_t *occ_grid,
    const NeRFConfig *config,
    float base_step
) {
    /* Quantize to occupancy grid */
    float bounds = config->scale;
    Vec3 normalized;
    normalized.x = (pos.x - config->center.x) / bounds + 0.5f;
    normalized.y = (pos.y - config->center.y) / bounds + 0.5f;
    normalized.z = (pos.z - config->center.z) / bounds + 0.5f;
    
    normalized.x = fmaxf(0.0f, fminf(1.0f, normalized.x));
    normalized.y = fmaxf(0.0f, fminf(1.0f, normalized.y));
    normalized.z = fmaxf(0.0f, fminf(1.0f, normalized.z));
    
    uint32_t xi = (uint32_t)(normalized.x * 63.0f);
    uint32_t yi = (uint32_t)(normalized.y * 63.0f);
    uint32_t zi = (uint32_t)(normalized.z * 63.0f);
    
    /* Clamp to valid range */
    if (xi >= 64) xi = 63;
    if (yi >= 64) yi = 63;
    if (zi >= 64) zi = 63;
    
    uint32_t idx = zi * 64 * 64 + yi * 64 + xi;
    uint8_t occupancy = occ_grid[idx];
    
    /* Threshold: if LOW occupancy (empty space), use LARGER step */
    const uint8_t OCCUPANCY_THRESHOLD = 32;  /* Out of 255 */
    
    if (occupancy < OCCUPANCY_THRESHOLD) {
        return base_step * 3.0f;  /* Skip empty space faster */
    }
    return base_step;  /* Fine sampling in occupied regions */
}

bool ysu_ray_should_terminate(float accumulated_alpha) {
    /* Stop when near-full opacity reached */
    if (accumulated_alpha > 0.99f) return true;
    
    return false;
}

/* ===== Volume Integration (Main Rendering Kernel) ===== */

void ysu_volume_integrate_batch(
    const RayBatch *batch,
    const NeRFConfig *config,
    const NeRFData *nerf_data,
    NeRFFramebuffer *output_fb,
    uint32_t num_steps,
    float density_scale,
    float bounds_max
) {
    /* Early return if num_steps is zero — avoids division by zero in base_step
     * and a degenerate ray-march with no samples. */
    if (num_steps == 0) return;

    /* Process each ray independently within batch */
    /* Precompute level scales to avoid repeated pow() calls.
     * Use 20 slots — enough for any foreseeable instant-NGP config (default=16). */
    float level_scales[20];
    uint32_t num_levels_clamped = config->num_levels < 20 ? config->num_levels : 20;
    for (uint32_t l = 0; l < num_levels_clamped; l++) {
        level_scales[l] = config->base_res * powf(config->per_level_scale, (float)l);
    }
    
    /* Parallelize ray loop with OpenMP if available */
    #pragma omp parallel for schedule(dynamic, 1) if(_OPENMP)
    for (int ray_idx_signed = 0; ray_idx_signed < (int)batch->count; ray_idx_signed++) {
        uint32_t ray_idx = (uint32_t)ray_idx_signed;
        if (!batch->active[ray_idx]) continue;
        
        Vec3 origin = batch->origin[ray_idx];
        Vec3 direction = batch->direction[ray_idx];
        float t_min = batch->tmin[ray_idx];
        float t_max = batch->tmax[ray_idx];
        uint32_t pixel_id = batch->pixel_id[ray_idx];
        
        /* Initialize accumulation */
        float accumulated_rgb[3] = {0.0f, 0.0f, 0.0f};
        float accumulated_alpha = 0.0f;
        float base_step = (bounds_max * 2.0f) / (float)num_steps;
        float dir_len = sqrtf(direction.x * direction.x + direction.y * direction.y + direction.z * direction.z);

        /* Ray marching loop — t is accumulated so adaptive step_size actually
         * advances the ray position, not just the alpha weighting. */
        float t = t_min;
        for (uint32_t step = 0; step < num_steps; step++) {
            if (t > t_max) break;
            
            Vec3 pos;
            pos.x = origin.x + direction.x * t;
            pos.y = origin.y + direction.y * t;
            pos.z = origin.z + direction.z * t;
            
            /* Adaptive step size — used both for alpha computation AND to advance t. */
            float step_size = ysu_adaptive_step_size(pos, nerf_data->occupancy_grid, config, base_step);
            
            /* Create feature vector for this ray step */
            float feat[27];
            memset(feat, 0, sizeof(feat));

            /* Normalize position to scene bounds [-1, 1] then to [0, 1] like GPU shader */
            Vec3 norm_pos;
            norm_pos.x = (pos.x - config->center.x) / config->scale;
            norm_pos.y = (pos.y - config->center.y) / config->scale;
            norm_pos.z = (pos.z - config->center.z) / config->scale;
            
            /* Convert to [0, 1] and clamp */
            Vec3 pn;
            pn.x = fmaxf(0.0f, fminf(1.0f, norm_pos.x * 0.5f + 0.5f));
            pn.y = fmaxf(0.0f, fminf(1.0f, norm_pos.y * 0.5f + 0.5f));
            pn.z = fmaxf(0.0f, fminf(1.0f, norm_pos.z * 0.5f + 0.5f));

            /* Hashgrid features per level with trilinear interpolation */
            for (uint32_t level = 0; level < num_levels_clamped; level++) {
                /* Use precomputed level scale */
                float res = level_scales[level];
                uint32_t level_offset = level * config->hashmap_size * config->features_per_entry;
                
                /* Scale position to grid coordinates */
                float gx = pn.x * res;
                float gy = pn.y * res;
                float gz = pn.z * res;
                
                /* Get integer corner and fractional weights */
                float fx = floorf(gx);
                float fy = floorf(gy);
                float fz = floorf(gz);
                int32_t i0 = (int32_t)fx;
                int32_t j0 = (int32_t)fy;
                int32_t k0 = (int32_t)fz;
                
                /* Interpolation weights */
                float wx = gx - fx;
                float wy = gy - fy;
                float wz = gz - fz;
                
                /* For each feature, do trilinear interpolation over 8 corners */
                for (uint32_t f = 0; f < config->features_per_entry; f++) {
                    /* Hash all 8 corners */
                    uint32_t h000 = ysu_hash_ijk(i0,   j0,   k0,   config->hashmap_size);
                    uint32_t h001 = ysu_hash_ijk(i0,   j0,   k0+1, config->hashmap_size);
                    uint32_t h010 = ysu_hash_ijk(i0,   j0+1, k0,   config->hashmap_size);
                    uint32_t h011 = ysu_hash_ijk(i0,   j0+1, k0+1, config->hashmap_size);
                    uint32_t h100 = ysu_hash_ijk(i0+1, j0,   k0,   config->hashmap_size);
                    uint32_t h101 = ysu_hash_ijk(i0+1, j0,   k0+1, config->hashmap_size);
                    uint32_t h110 = ysu_hash_ijk(i0+1, j0+1, k0,   config->hashmap_size);
                    uint32_t h111 = ysu_hash_ijk(i0+1, j0+1, k0+1, config->hashmap_size);
                    
                    /* Prefetch hash-indexed lines before the loads */
                    YSU_PREFETCH(&nerf_data->hashgrid_data[level_offset + h000 * config->features_per_entry]);
                    YSU_PREFETCH(&nerf_data->hashgrid_data[level_offset + h100 * config->features_per_entry]);
                    YSU_PREFETCH(&nerf_data->hashgrid_data[level_offset + h010 * config->features_per_entry]);
                    YSU_PREFETCH(&nerf_data->hashgrid_data[level_offset + h110 * config->features_per_entry]);

                    /* Look up feature values at each corner */
                    float v000 = nerf_data->hashgrid_data[level_offset + h000 * config->features_per_entry + f];
                    float v001 = nerf_data->hashgrid_data[level_offset + h001 * config->features_per_entry + f];
                    float v010 = nerf_data->hashgrid_data[level_offset + h010 * config->features_per_entry + f];
                    float v011 = nerf_data->hashgrid_data[level_offset + h011 * config->features_per_entry + f];
                    float v100 = nerf_data->hashgrid_data[level_offset + h100 * config->features_per_entry + f];
                    float v101 = nerf_data->hashgrid_data[level_offset + h101 * config->features_per_entry + f];
                    float v110 = nerf_data->hashgrid_data[level_offset + h110 * config->features_per_entry + f];
                    float v111 = nerf_data->hashgrid_data[level_offset + h111 * config->features_per_entry + f];
                    
                    /* Trilinear interpolation */
                    float v00 = v000 * (1.0f - wz) + v001 * wz;
                    float v01 = v010 * (1.0f - wz) + v011 * wz;
                    float v10 = v100 * (1.0f - wz) + v101 * wz;
                    float v11 = v110 * (1.0f - wz) + v111 * wz;
                    float v0 = v00 * (1.0f - wy) + v01 * wy;
                    float v1 = v10 * (1.0f - wy) + v11 * wy;
                    float val = v0 * (1.0f - wx) + v1 * wx;
                    
                    /* Bug fix: was hardcoded `level * 2` which is wrong when
                     * features_per_entry != 2. Use features_per_entry as stride. */
                    feat[level * config->features_per_entry + f] = val;
                }
            }

            /* View direction encoded */
            if (dir_len > 1e-6f) {
                feat[24] = direction.x / dir_len * 0.5f + 0.5f;  /* Normalize to [0, 1] */
                feat[25] = direction.y / dir_len * 0.5f + 0.5f;
                feat[26] = direction.z / dir_len * 0.5f + 0.5f;
            }
            
            /* MLP inference for this ray step - use single-ray fast path */
            float rgb_step_scalar[3];
            float sigma_step_scalar = 0.0f;

            ysu_mlp_inference_single(
                feat,
                config,
                nerf_data->mlp_weights,
                nerf_data->mlp_biases,
                rgb_step_scalar,
                &sigma_step_scalar
            );

            /* Volume compositing */
            float sigma = sigma_step_scalar * density_scale;
            float alpha = 1.0f - expf(-sigma * step_size);
            
            /* Accumulate color */
            float weight = alpha * (1.0f - accumulated_alpha);
            accumulated_rgb[0] += rgb_step_scalar[0] * weight;
            accumulated_rgb[1] += rgb_step_scalar[1] * weight;
            accumulated_rgb[2] += rgb_step_scalar[2] * weight;
            accumulated_alpha += weight;
            
            /* Advance ray position by the adaptive step (fixes the bug where t
             * was always t_min + step*base_step, making empty-space skip ineffective). */
            t += step_size;

            /* Early termination */
            if (ysu_ray_should_terminate(accumulated_alpha)) {
                break;
            }
        }
        
        /* Write to framebuffer */
        uint32_t px = pixel_id % output_fb->width;
        uint32_t py = pixel_id / output_fb->width;
        
        if (px < output_fb->width && py < output_fb->height) {
            uint32_t fb_idx = py * output_fb->width + px;
            /* Thread-safe write: each pixel written by one thread */
            output_fb->pixels[fb_idx].rgb.x = accumulated_rgb[0];  /* R */
            output_fb->pixels[fb_idx].rgb.y = accumulated_rgb[1];  /* G */
            output_fb->pixels[fb_idx].rgb.z = accumulated_rgb[2];  /* B */
            output_fb->pixels[fb_idx].alpha = accumulated_alpha;
        }
    }  /* End of parallel region */
}

/* ===== Profiling Utilities ===== */

void ysu_perf_start(uint64_t *start_cycle) {
    *start_cycle = ysu_rdtsc();
}

void ysu_perf_end(uint64_t start_cycle, PerfCounter *counter) {
    uint64_t end_cycle = ysu_rdtsc();
    counter->total_cycles += (end_cycle - start_cycle);
    counter->sample_count++;
}

void ysu_perf_report(const char *name, const PerfCounter *counter) {
    if (counter->sample_count == 0) return;
    
    double avg_cycles = (double)counter->total_cycles / counter->sample_count;
    double avg_us = avg_cycles / 3000.0;  /* ~3 GHz CPU */
    
    printf("[PERF] %s: %.2f cycles/sample, %.2f µs/sample (%lu samples)\n",
           name, avg_cycles, avg_us, counter->sample_count);
}
