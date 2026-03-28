#ifndef LBM_KERNELS_H
#define LBM_KERNELS_H

// Kernel variant registry for all D3Q19 LBM CUDA kernels.
// Each variant maps to a step kernel + init kernel pair in the .cu files.

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Kernel variant enumeration
// ============================================================================

typedef enum {
    // FP32 SoA variants (kernels_soa.cu)
    LBM_FP32_SOA_FUSED,
    LBM_FP32_SOA_MRT_FUSED,
    LBM_FP32_SOA_PULL,
    LBM_FP32_SOA_MRT_PULL,
    LBM_FP32_SOA_TILED,
    LBM_FP32_SOA_MRT_TILED,
    LBM_FP32_SOA_COARSENED,
    LBM_FP32_SOA_MRT_COARSENED,
    LBM_FP32_SOA_COARSENED_F4,
    LBM_FP32_SOA_AA,
    LBM_FP32_SOA_MRT_AA,
    LBM_FP32_SOA_CS,           // kernels_fp32_soa_cs.cu

    // FP16 variants
    LBM_FP16_AOS,              // kernels_fp16.cu
    LBM_FP16_SOA,              // kernels_fp16_soa.cu
    LBM_FP16_SOA_HALF2,        // kernels_fp16_soa_half2.cu

    // BF16 variants
    LBM_BF16_AOS,              // kernels_bf16.cu
    LBM_BF16_SOA,              // kernels_bf16_soa.cu
    LBM_BF16_SOA_BF162,        // kernels_bf16_soa_bf162.cu (HFMA2.BF16_V2 packed FMA)

    // FP8 variants
    LBM_FP8_E4M3_AOS,          // kernels_fp8.cu
    LBM_FP8_E4M3_SOA,          // kernels_fp8_soa.cu
    LBM_FP8_E5M2_AOS,          // kernels_fp8_e5m2.cu
    LBM_FP8_E5M2_SOA,          // kernels_fp8_e5m2_soa.cu

    // INT8 variants
    LBM_INT8_AOS,              // kernels_int8.cu
    LBM_INT8_SOA,              // kernels_int8_soa.cu

    // INT16 variants
    LBM_INT16_AOS,             // kernels_int16.cu
    LBM_INT16_SOA,             // kernels_int16_soa.cu

    // FP64 variants
    LBM_FP64_AOS,              // kernels_fp64.cu
    LBM_FP64_SOA,              // kernels_fp64_soa.cu

    // Double-double FP128
    LBM_DD_SOA,                // kernels_dd.cu

    // Bandwidth ceiling (non-physics)
    LBM_INT4_SOA,              // kernels_int4.cu
    LBM_FP4_SOA,               // kernels_fp4.cu

    LBM_VARIANT_COUNT
} LbmKernelVariant;

// ============================================================================
// Kernel metadata
// ============================================================================

typedef struct {
    const char* name;           // Human-readable name (e.g., "INT8 SoA")
    const char* step_kernel;    // Step kernel function name
    const char* init_kernel;    // Init kernel function name
    int bytes_per_dist;         // Bytes per distribution value (0.5 for nibble)
    int min_sm;                 // Minimum SM architecture (e.g., 89 for FP8)
    int physics_valid;          // 1 if physics-valid, 0 if BW-ceiling only
    int is_soa;                 // 1 if SoA layout, 0 if AoS
    int threads_per_block;      // Default threads per block
    int cells_per_thread;       // Cells processed per thread (1 or 2 or 4)
    int is_aa;                  // 1 if A-A single-buffer scheme
    int is_mrt;                 // 1 if MRT collision (vs BGK)
} LbmKernelInfo;

// Static kernel info table -- indexed by LbmKernelVariant.
// bytes_per_dist: uses 1 for nibble types (actual is 0.5); VRAM calc adjusts.
static const LbmKernelInfo LBM_KERNEL_INFO[LBM_VARIANT_COUNT] = {
    // FP32 SoA variants
    //                          name                    step_kernel                       init_kernel                       B/d  SM  phys soa  tpb  cpt  aa  mrt
    [LBM_FP32_SOA_FUSED]     = {"FP32 SoA Fused",      "lbm_step_soa_fused",             "initialize_uniform_soa_kernel",  4,   50, 1,   1,   128, 1,   0,  0},
    [LBM_FP32_SOA_MRT_FUSED] = {"FP32 SoA MRT",        "lbm_step_soa_mrt_fused",         "initialize_uniform_soa_kernel",  4,   50, 1,   1,   128, 1,   0,  1},
    [LBM_FP32_SOA_PULL]      = {"FP32 SoA Pull",       "lbm_step_soa_pull",              "initialize_uniform_soa_kernel",  4,   50, 1,   1,   128, 1,   0,  0},
    [LBM_FP32_SOA_MRT_PULL]  = {"FP32 SoA MRT Pull",   "lbm_step_soa_mrt_pull",          "initialize_uniform_soa_kernel",  4,   50, 1,   1,   128, 1,   0,  1},
    [LBM_FP32_SOA_TILED]     = {"FP32 SoA Tiled",      "lbm_step_soa_tiled",             "initialize_uniform_soa_kernel",  4,   50, 1,   1,   256, 1,   0,  0},
    [LBM_FP32_SOA_MRT_TILED] = {"FP32 SoA MRT Tiled",  "lbm_step_soa_mrt_tiled",         "initialize_uniform_soa_kernel",  4,   50, 1,   1,   256, 1,   0,  1},
    [LBM_FP32_SOA_COARSENED] = {"FP32 SoA Coarsened",  "lbm_step_soa_coarsened",         "initialize_uniform_soa_kernel",  4,   50, 1,   1,   128, 2,   0,  0},
    [LBM_FP32_SOA_MRT_COARSENED] = {"FP32 SoA MRT Coarsened", "lbm_step_soa_mrt_coarsened", "initialize_uniform_soa_kernel", 4, 50, 1, 1, 128, 2, 0, 1},
    [LBM_FP32_SOA_COARSENED_F4] = {"FP32 SoA Coarsened F4", "lbm_step_soa_coarsened_float4", "initialize_uniform_soa_kernel", 4, 50, 1, 1, 128, 4, 0, 0},
    [LBM_FP32_SOA_AA]        = {"FP32 SoA A-A",        "lbm_step_soa_aa",                "initialize_uniform_soa_kernel",  4,   50, 1,   1,   128, 1,   1,  0},
    [LBM_FP32_SOA_MRT_AA]    = {"FP32 SoA MRT A-A",    "lbm_step_soa_mrt_aa",            "initialize_uniform_soa_kernel",  4,   50, 1,   1,   128, 1,   1,  1},
    [LBM_FP32_SOA_CS]        = {"FP32 SoA CS",         "lbm_step_fp32_soa_cs_kernel",    "initialize_uniform_fp32_soa_cs_kernel", 4, 80, 1, 1, 128, 1, 0, 0},

    // FP16 variants
    [LBM_FP16_AOS]           = {"FP16 AoS",            "lbm_step_fused_fp16_kernel",     "initialize_uniform_fp16_kernel", 2,   70, 1,   0,   128, 1,   0,  0},
    [LBM_FP16_SOA]           = {"FP16 SoA",            "lbm_step_fp16_soa_kernel",       "initialize_uniform_fp16_soa_kernel", 2, 70, 1, 1, 128, 1, 0, 0},
    [LBM_FP16_SOA_HALF2]     = {"FP16 SoA H2",         "lbm_step_fp16_soa_half2_kernel", "initialize_uniform_fp16_soa_half2_kernel", 2, 70, 1, 1, 128, 2, 0, 0},

    // BF16 variants
    [LBM_BF16_AOS]           = {"BF16 AoS",            "lbm_step_fused_bf16_kernel",     "initialize_uniform_bf16_kernel", 2,   80, 1,   0,   128, 1,   0,  0},
    [LBM_BF16_SOA]           = {"BF16 SoA",            "lbm_step_bf16_soa_kernel",       "initialize_uniform_bf16_soa_kernel", 2, 80, 1, 1, 128, 1, 0, 0},
    [LBM_BF16_SOA_BF162]    = {"BF16 SoA BF162",      "lbm_step_bf16_soa_bf162_kernel", "initialize_uniform_bf16_soa_bf162_kernel", 2, 80, 1, 1, 128, 2, 0, 0},

    // FP8 variants
    [LBM_FP8_E4M3_AOS]       = {"FP8 e4m3 AoS",       "lbm_step_fused_fp8_kernel",      "initialize_uniform_fp8_kernel",  1,   89, 1,   0,   128, 1,   0,  0},
    [LBM_FP8_E4M3_SOA]       = {"FP8 e4m3 SoA",       "lbm_step_fp8_soa_kernel",        "initialize_uniform_fp8_soa_kernel", 1, 89, 1, 1, 128, 1, 0, 0},
    [LBM_FP8_E5M2_AOS]       = {"FP8 e5m2 AoS",       "lbm_step_fused_fp8e5m2_kernel",  "initialize_uniform_fp8e5m2_kernel", 1, 89, 1, 0, 128, 1, 0, 0},
    [LBM_FP8_E5M2_SOA]       = {"FP8 e5m2 SoA",       "lbm_step_fp8_e5m2_soa_kernel",   "initialize_uniform_fp8_e5m2_soa_kernel", 1, 89, 1, 1, 128, 1, 0, 0},

    // INT8 variants
    [LBM_INT8_AOS]           = {"INT8 AoS",            "lbm_step_fused_int8_kernel",     "initialize_uniform_int8_kernel", 1,   61, 1,   0,   128, 1,   0,  0},
    [LBM_INT8_SOA]           = {"INT8 SoA",            "lbm_step_int8_soa_kernel",       "initialize_uniform_int8_soa_kernel", 1, 61, 1, 1, 128, 1, 0, 0},

    // INT16 variants
    [LBM_INT16_AOS]          = {"INT16 AoS",           "lbm_step_int16_kernel",          "initialize_uniform_int16_kernel", 2,  50, 1,   0,   128, 1,   0,  0},
    [LBM_INT16_SOA]          = {"INT16 SoA",           "lbm_step_int16_soa_kernel",      "initialize_uniform_int16_soa_kernel", 2, 50, 1, 1, 128, 1, 0, 0},

    // FP64 variants
    [LBM_FP64_AOS]           = {"FP64 AoS",            "lbm_step_fused_fp64_kernel",     "initialize_uniform_fp64_kernel", 8,   60, 1,   0,   128, 1,   0,  0},
    [LBM_FP64_SOA]           = {"FP64 SoA",            "lbm_step_fp64_soa_kernel",       "initialize_uniform_fp64_soa_kernel", 8, 60, 1, 1, 128, 1, 0, 0},

    // Double-double
    [LBM_DD_SOA]             = {"DD FP128 SoA",        "lbm_step_fused_dd_kernel",       "initialize_uniform_dd_kernel",   16,  60, 1,   1,   128, 1,   0,  0},

    // BW ceiling (non-physics)
    [LBM_INT4_SOA]           = {"INT4 Nibble SoA",     "lbm_step_fused_int4_kernel",     "initialize_uniform_int4_kernel", 1,   61, 0,   1,   128, 2,   0,  0},
    [LBM_FP4_SOA]            = {"FP4 E2M1 SoA",       "lbm_step_fp4_kernel",            "initialize_uniform_fp4_kernel",  1,   89, 0,   1,   128, 2,   0,  0},
};

// ============================================================================
// Helper functions
// ============================================================================

// Compute VRAM bytes for distribution buffers (ping + pong, or single for A-A).
static inline size_t lbm_dist_vram_bytes(LbmKernelVariant v, int nx, int ny, int nz) {
    const LbmKernelInfo* info = &LBM_KERNEL_INFO[v];
    size_t n_cells = (size_t)nx * ny * nz;
    size_t dirs = 19;
    size_t bytes_per_dist = (size_t)info->bytes_per_dist;
    // INT4/FP4 use 0.5 bytes per dist but bytes_per_dist is stored as 1;
    // actual nibble packing halves the buffer.
    if (v == LBM_INT4_SOA || v == LBM_FP4_SOA) {
        // Nibble-packed: ceil(19 * n_cells / 2) bytes per buffer
        size_t per_buf = (dirs * n_cells + 1) / 2;
        return info->is_aa ? per_buf : per_buf * 2;
    }
    if (info->is_soa) {
        // SoA: 19 * n_cells * bytes_per_dist per buffer
        size_t per_buf = dirs * n_cells * bytes_per_dist;
        return info->is_aa ? per_buf : per_buf * 2;
    } else {
        // AoS: stride=20 * n_cells * bytes_per_dist per buffer
        size_t per_buf = 20 * n_cells * bytes_per_dist;
        return per_buf * 2; // AoS always uses ping-pong
    }
}

// Compute bytes transferred per cell per step (bandwidth model).
// read 19 + write 19 + aux 8 scalars = 46 scalars * bytes_per_dist.
// Aux (rho, u[3], tau, force[3]) are FP32 regardless of distribution type.
static inline size_t lbm_bytes_per_cell_per_step(LbmKernelVariant v) {
    const LbmKernelInfo* info = &LBM_KERNEL_INFO[v];
    // Distributions: 19 read + 19 write
    size_t dist_bytes = 38 * (size_t)info->bytes_per_dist;
    if (v == LBM_INT4_SOA || v == LBM_FP4_SOA) {
        dist_bytes = 38 / 2; // nibble-packed
    }
    // Auxiliary arrays: 8 FP32 scalars (rho, u[3], tau, force[3])
    size_t aux_bytes = 8 * 4;
    return dist_bytes + aux_bytes;
}

#ifdef __cplusplus
}
#endif

#endif // LBM_KERNELS_H
