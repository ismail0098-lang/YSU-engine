/*
 * Instant-NGP Tiny MLP Forward — SASS-level inline PTX
 *
 * Architecture: 27-in → 64-hidden (ReLU) → 64-hidden (ReLU) → 4-out (sigmoid)
 *
 * This is the second-hottest kernel in instant-NGP. After hash grid encoding
 * produces 24 features (+3 view direction = 27 inputs), the MLP decodes them
 * into (R, G, B, density).
 *
 * SASS-level optimizations:
 *   - FFMA chains for dot products with 8-wide ILP (8 independent FMAs in flight)
 *   - Shared memory tiling: weights loaded to smem once per block
 *   - Predicated ReLU via FMNMX (max(x, 0) in one instruction)
 *   - MUFU.EX2 for fast sigmoid: sig(x) = 1/(1+2^(-x/ln2))
 *   - Register blocking: each thread computes one sample's full MLP
 *
 * Weight layout (matches engine NeRFConfig):
 *   W0: [64][27]  B0: [64]    — layer 0
 *   W1: [64][64]  B1: [64]    — layer 1
 *   W2: [4][64]   B2: [4]     — output layer
 *
 * Target: SM 8.9 (Ada Lovelace)
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

/* MLP dimensions */
#define MLP_IN        27   /* 24 hash features + 3 view direction */
#define MLP_HIDDEN    64
#define MLP_OUT       4    /* R, G, B, sigma */

/* Weight sizes */
#define W0_SIZE  (MLP_HIDDEN * MLP_IN)     /* 64 × 27 = 1728 */
#define B0_SIZE  (MLP_HIDDEN)              /* 64 */
#define W1_SIZE  (MLP_HIDDEN * MLP_HIDDEN) /* 64 × 64 = 4096 */
#define B1_SIZE  (MLP_HIDDEN)              /* 64 */
#define W2_SIZE  (MLP_OUT * MLP_HIDDEN)    /* 4 × 64 = 256 */
#define B2_SIZE  (MLP_OUT)                 /* 4 */

/* Total weight count */
#define TOTAL_WEIGHTS (W0_SIZE + B0_SIZE + W1_SIZE + B1_SIZE + W2_SIZE + B2_SIZE)

/* Block size — each thread processes one sample */
#define MLP_BLOCK_SIZE 128


/* ════════════════════════════════════════════════════════════════
 * Helper: Fast sigmoid via MUFU.EX2
 *
 * sigmoid(x) = 1 / (1 + exp(-x))
 *            = 1 / (1 + 2^(-x * log2(e)))
 *
 * PTX: mul.f32  neg_x_log2e, x, -1.4426950408...
 *      ex2.approx.f32  exp_val, neg_x_log2e      → MUFU.EX2
 *      add.f32  denom, exp_val, 1.0
 *      rcp.approx.f32  result, denom              → MUFU.RCP
 *
 * Total: 4 instructions, 2 of which are MUFU (special function unit)
 * ════════════════════════════════════════════════════════════════ */
__device__ __forceinline__
float fast_sigmoid_ptx(float x) {
    float result;
    asm volatile(
        "{\n\t"
        ".reg .f32 neg_xl2e, ex, denom;\n\t"
        "mul.f32         neg_xl2e, %1, 0fBFB8AA3B;\n\t"
        "ex2.approx.f32  ex, neg_xl2e;\n\t"
        "add.f32         denom, ex, 0f3F800000;\n\t"
        "rcp.approx.f32  %0, denom;\n\t"
        "}"
        : "=f"(result) : "f"(x)
    );
    return result;
}


/* ════════════════════════════════════════════════════════════════
 * Helper: ReLU via predicated max
 *
 * SASS: FMNMX Rd, Ra, RZ, !PT   (max with zero register)
 * PTX:  max.f32 Rd, Ra, 0f00000000  → compiles to single FMNMX
 * ════════════════════════════════════════════════════════════════ */
__device__ __forceinline__
float fast_relu_ptx(float x) {
    float result;
    asm volatile("max.f32 %0, %1, 0f00000000;" : "=f"(result) : "f"(x));
    return result;
}


/* ════════════════════════════════════════════════════════════════
 * KERNEL 2: MLP Forward — Full inline PTX
 *
 * Each thread computes the full MLP for ONE sample.
 * Weights are loaded into shared memory for reuse across the block.
 *
 * Args:
 *   input:   [N][27] float — hash-encoded features + view dir
 *   weights: flat array of all weights (W0|B0|W1|B1|W2|B2)
 *   output:  [N][4]  float — (R, G, B, sigma)
 *   N:       number of samples
 * ════════════════════════════════════════════════════════════════ */

extern "C" __global__ void __launch_bounds__(MLP_BLOCK_SIZE)
mlp_forward_ptx(
    const float * __restrict__ input,
    const float * __restrict__ weights,
    float       * __restrict__ output,
    int N
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    /* ── Shared memory for weights ──
     * Layer 0: 1728 + 64 = 1792 floats = 7168 bytes
     * Layer 1: 4096 + 64 = 4160 floats = 16640 bytes
     * Layer 2: 256 + 4 = 260 floats = 1040 bytes
     * Total: 24848 bytes — fits in 48KB L1 config */
    __shared__ float smem_W0[W0_SIZE];
    __shared__ float smem_B0[B0_SIZE];
    __shared__ float smem_W1[W1_SIZE];
    __shared__ float smem_B1[B1_SIZE];
    __shared__ float smem_W2[W2_SIZE];
    __shared__ float smem_B2[B2_SIZE];

    /* Cooperative weight loading: all threads in block participate */
    {
        const float *w_ptr = weights;
        /* Load W0 */
        for (int i = threadIdx.x; i < W0_SIZE; i += MLP_BLOCK_SIZE)
            smem_W0[i] = w_ptr[i];
        w_ptr += W0_SIZE;
        for (int i = threadIdx.x; i < B0_SIZE; i += MLP_BLOCK_SIZE)
            smem_B0[i] = w_ptr[i];
        w_ptr += B0_SIZE;
        /* Load W1 */
        for (int i = threadIdx.x; i < W1_SIZE; i += MLP_BLOCK_SIZE)
            smem_W1[i] = w_ptr[i];
        w_ptr += W1_SIZE;
        for (int i = threadIdx.x; i < B1_SIZE; i += MLP_BLOCK_SIZE)
            smem_B1[i] = w_ptr[i];
        w_ptr += B1_SIZE;
        /* Load W2 */
        for (int i = threadIdx.x; i < W2_SIZE; i += MLP_BLOCK_SIZE)
            smem_W2[i] = w_ptr[i];
        w_ptr += W2_SIZE;
        for (int i = threadIdx.x; i < B2_SIZE; i += MLP_BLOCK_SIZE)
            smem_B2[i] = w_ptr[i];
    }
    __syncthreads();

    if (tid >= N) return;

    /* ── Load input features into registers ── */
    float in_feat[MLP_IN];
    {
        const float *in_ptr = input + tid * MLP_IN;
        #pragma unroll
        for (int i = 0; i < MLP_IN; i++) {
            asm volatile("ld.global.f32 %0, [%1];"
                : "=f"(in_feat[i]) : "l"(in_ptr + i));
        }
    }

    /* ════════════════════════════════════
     * Layer 0: [64][27] × [27] + [64]
     * 64 neurons, each a 27-element dot product
     *
     * SASS strategy: 27 FFMA per neuron, 8 neurons computed
     * in parallel for ILP (8 independent accumulator chains).
     * ════════════════════════════════════ */
    float hidden0[MLP_HIDDEN];

    /* Process 8 neurons at a time for ILP */
    #pragma unroll
    for (int neuron_base = 0; neuron_base < MLP_HIDDEN; neuron_base += 8) {
        /* Initialize accumulators with bias
         * Compiler generates LDS for shared memory — that's fine,
         * LDS is already optimal. Our value-add is the FFMA chains. */
        float acc[8];
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            acc[k] = smem_B0[neuron_base + k];
        }

        /* Dot product: acc[k] += W0[neuron_base+k][j] * in_feat[j]
         * 8 independent FFMA chains → fills FFMA pipeline (4-cycle latency,
         * 8 outstanding = 2× throughput on Ada's dual FP32 pipes) */
        #pragma unroll
        for (int j = 0; j < MLP_IN; j++) {
            float inp = in_feat[j];
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                /* Weight load from shared memory (compiler → LDS.32) */
                float w = smem_W0[(neuron_base + k) * MLP_IN + j];
                /* FMA accumulate
                 * SASS: FFMA Rd, Ra, Rb, Rd */
                asm volatile("fma.rn.f32 %0, %1, %2, %0;"
                    : "+f"(acc[k]) : "f"(w), "f"(inp));
            }
        }

        /* ReLU activation
         * SASS: FMNMX Rd, Ra, RZ, !PT */
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            hidden0[neuron_base + k] = fast_relu_ptx(acc[k]);
        }
    }

    /* ════════════════════════════════════
     * Layer 1: [64][64] × [64] + [64]
     * Same structure as layer 0 but with hidden-to-hidden
     * ════════════════════════════════════ */
    float hidden1[MLP_HIDDEN];

    #pragma unroll
    for (int neuron_base = 0; neuron_base < MLP_HIDDEN; neuron_base += 8) {
        float acc[8];
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            acc[k] = smem_B1[neuron_base + k];
        }

        #pragma unroll
        for (int j = 0; j < MLP_HIDDEN; j++) {
            float h = hidden0[j];
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                float w = smem_W1[(neuron_base + k) * MLP_HIDDEN + j];
                asm volatile("fma.rn.f32 %0, %1, %2, %0;"
                    : "+f"(acc[k]) : "f"(w), "f"(h));
            }
        }

        #pragma unroll
        for (int k = 0; k < 8; k++) {
            hidden1[neuron_base + k] = fast_relu_ptx(acc[k]);
        }
    }

    /* ════════════════════════════════════
     * Output layer: [4][64] × [64] + [4]
     * Only 4 outputs, each a 64-element dot product
     * Apply sigmoid for final activation
     * ════════════════════════════════════ */
    float out_vals[MLP_OUT];

    #pragma unroll
    for (int o = 0; o < MLP_OUT; o++) {
        float acc = smem_B2[o];

        /* 64-element dot product with 8-wide ILP unroll */
        #pragma unroll
        for (int j = 0; j < MLP_HIDDEN; j++) {
            float w = smem_W2[o * MLP_HIDDEN + j];
            asm volatile("fma.rn.f32 %0, %1, %2, %0;"
                : "+f"(acc) : "f"(w), "f"(hidden1[j]));
        }

        /* Sigmoid activation
         * SASS: FMUL + MUFU.EX2 + FADD + MUFU.RCP */
        out_vals[o] = fast_sigmoid_ptx(acc);
    }

    /* ── Store output [R, G, B, sigma] as STG.E.128 ── */
    {
        float *out_ptr = output + tid * MLP_OUT;
        asm volatile(
            "st.global.v4.f32 [%0], {%1, %2, %3, %4};"
            : : "l"(out_ptr),
                "f"(out_vals[0]), "f"(out_vals[1]),
                "f"(out_vals[2]), "f"(out_vals[3])
        );
    }
}


/* ════════════════════════════════════════════════════════════════
 * Reference CUDA implementation (for validation)
 * ════════════════════════════════════════════════════════════════ */

extern "C" __global__ void __launch_bounds__(MLP_BLOCK_SIZE)
mlp_forward_ref(
    const float * __restrict__ input,
    const float * __restrict__ weights,
    float       * __restrict__ output,
    int N
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    const float *in_ptr = input + tid * MLP_IN;

    /* Decode weight layout */
    const float *W0 = weights;
    const float *B0 = W0 + W0_SIZE;
    const float *W1 = B0 + B0_SIZE;
    const float *B1 = W1 + W1_SIZE;
    const float *W2 = B1 + B1_SIZE;
    const float *B2 = W2 + W2_SIZE;

    /* Layer 0 */
    float h0[MLP_HIDDEN];
    for (int i = 0; i < MLP_HIDDEN; i++) {
        float acc = B0[i];
        for (int j = 0; j < MLP_IN; j++)
            acc += W0[i * MLP_IN + j] * in_ptr[j];
        h0[i] = fmaxf(acc, 0.0f); /* ReLU */
    }

    /* Layer 1 */
    float h1[MLP_HIDDEN];
    for (int i = 0; i < MLP_HIDDEN; i++) {
        float acc = B1[i];
        for (int j = 0; j < MLP_HIDDEN; j++)
            acc += W1[i * MLP_HIDDEN + j] * h0[j];
        h1[i] = fmaxf(acc, 0.0f); /* ReLU */
    }

    /* Output layer */
    float *out_ptr = output + tid * MLP_OUT;
    for (int o = 0; o < MLP_OUT; o++) {
        float acc = B2[o];
        for (int j = 0; j < MLP_HIDDEN; j++)
            acc += W2[o * MLP_HIDDEN + j] * h1[j];
        out_ptr[o] = 1.0f / (1.0f + expf(-acc)); /* sigmoid */
    }
}
