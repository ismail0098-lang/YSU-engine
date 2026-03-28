/*
 * SASS RE Probe: INT8 DP4A Dot Product Accumulate
 * Isolates: DP4A (4-element dot product of packed INT8 vectors)
 *
 * DP4A computes: acc += dot(a[3:0], b[3:0])
 * where a and b are packed as 4x INT8 in a single INT32 register.
 *
 * On Ada Lovelace SM 8.9, DP4A shares the INT32 datapath:
 *   Expected latency: ~4-5 cycles (same as IMAD)
 *   Expected throughput: 64 ops/clock/SM (INT32 pipe)
 *
 * DP4A is the core acceleration in kernels_int8.cu where 5 DP4A calls
 * compute the momentum sums for 19 D3Q19 distributions (packed as
 * 5 groups of 4 INT8 values in AoS stride=20 layout).
 *
 * Key SASS instruction: IDP, IDP4A, or DP4A (varies by assembler)
 *
 * Signedness variants force the four packed INT8 dot-product signedness
 * combinations through inline PTX so cuobjdump can confirm the exact
 * emitted IDP.4A suffixes:
 *   - IDP.4A.U8.U8 (dp4a.u32.u32)
 *   - IDP.4A.S8.S8 (dp4a.s32.s32)
 *   - IDP.4A.S8.U8 (dp4a.s32.u32)
 *   - IDP.4A.U8.S8 (dp4a.u32.s32)
 */

#include <stdint.h>

// DP4A: packed INT8 dot product accumulate
extern "C" __global__ void __launch_bounds__(32)
probe_dp4a(int *out, const int *a, const int *b) {
    int i = threadIdx.x;

    int va = a[i];      // 4 packed INT8 values
    int vb = b[i];      // 4 packed INT8 values
    int acc = 0;

    // Single DP4A: acc += dot4(va, vb)
    acc = __dp4a(va, vb, acc);

    out[i] = acc;
}

// DP4A dependent chain for latency measurement
extern "C" __global__ void __launch_bounds__(32)
probe_dp4a_chain(int *out, const int *a, const int *b) {
    int i = threadIdx.x;
    int va = a[i];
    int vb = b[i];
    int acc = 0;

    // 64-deep dependent DP4A chain (accumulator feeds back)
    #pragma unroll 1
    for (int j = 0; j < 64; j++) {
        acc = __dp4a(va, vb, acc);
    }

    out[i] = acc;
}

// DP4A throughput: 8 independent accumulator streams
extern "C" __global__ void __launch_bounds__(128)
probe_dp4a_throughput(int *out, const int *a, const int *b) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    int va = a[i];
    int vb = b[i];

    // 8 independent accumulators (ILP)
    int acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    int acc4 = 0, acc5 = 0, acc6 = 0, acc7 = 0;

    #pragma unroll 1
    for (int j = 0; j < 64; j++) {
        acc0 = __dp4a(va, vb, acc0);
        acc1 = __dp4a(va, vb, acc1);
        acc2 = __dp4a(va, vb, acc2);
        acc3 = __dp4a(va, vb, acc3);
        acc4 = __dp4a(va, vb, acc4);
        acc5 = __dp4a(va, vb, acc5);
        acc6 = __dp4a(va, vb, acc6);
        acc7 = __dp4a(va, vb, acc7);
    }

    out[i] = acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7;
}

// 5x DP4A momentum pattern (from kernels_int8.cu D3Q19 layout)
extern "C" __global__ void __launch_bounds__(32)
probe_dp4a_momentum(int *out, const int *distributions, const int *weights) {
    int i = threadIdx.x;

    // Simulates the 5-group DP4A momentum accumulation pattern:
    // Each of the 5 groups packs 4 distribution values as INT8x4
    // and dots with 4 lattice velocity components.
    int base = i * 5;
    int rho = 0;
    int mx = 0;
    int my = 0;

    // Group 0: directions 0-3
    rho = __dp4a(distributions[base + 0], weights[0], rho);
    mx  = __dp4a(distributions[base + 0], weights[1], mx);
    my  = __dp4a(distributions[base + 0], weights[2], my);

    // Group 1: directions 4-7
    rho = __dp4a(distributions[base + 1], weights[3], rho);
    mx  = __dp4a(distributions[base + 1], weights[4], mx);
    my  = __dp4a(distributions[base + 1], weights[5], my);

    // Group 2: directions 8-11
    rho = __dp4a(distributions[base + 2], weights[6], rho);
    mx  = __dp4a(distributions[base + 2], weights[7], mx);
    my  = __dp4a(distributions[base + 2], weights[8], my);

    // Group 3: directions 12-15
    rho = __dp4a(distributions[base + 3], weights[9], rho);
    mx  = __dp4a(distributions[base + 3], weights[10], mx);
    my  = __dp4a(distributions[base + 3], weights[11], my);

    // Group 4: directions 16-18 + padding
    rho = __dp4a(distributions[base + 4], weights[12], rho);
    mx  = __dp4a(distributions[base + 4], weights[13], mx);
    my  = __dp4a(distributions[base + 4], weights[14], my);

    out[i * 3 + 0] = rho;
    out[i * 3 + 1] = mx;
    out[i * 3 + 2] = my;
}

// DP4A signedness: U8.U8 via inline PTX (dp4a.u32.u32)
extern "C" __global__ void __launch_bounds__(32)
probe_dp4a_u8_u8(int *out, const uint32_t *a, const uint32_t *b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t va = a[idx];
    uint32_t vb = b[idx];
    int acc = 0;
    asm volatile("dp4a.u32.u32 %0, %1, %2, %3;" : "=r"(acc) : "r"(va), "r"(vb), "r"(acc));
    out[idx] = acc;
}

// DP4A signedness: S8.S8 via inline PTX (dp4a.s32.s32)
extern "C" __global__ void __launch_bounds__(32)
probe_dp4a_s8_s8(int *out, const uint32_t *a, const uint32_t *b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t va = a[idx];
    uint32_t vb = b[idx];
    int acc = 0;
    asm volatile("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(acc) : "r"(va), "r"(vb), "r"(acc));
    out[idx] = acc;
}

// DP4A signedness: S8.U8 via inline PTX (dp4a.s32.u32)
extern "C" __global__ void __launch_bounds__(32)
probe_dp4a_s8_u8(int *out, const uint32_t *a, const uint32_t *b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t va = a[idx];
    uint32_t vb = b[idx];
    int acc = 0;
    asm volatile("dp4a.s32.u32 %0, %1, %2, %3;" : "=r"(acc) : "r"(va), "r"(vb), "r"(acc));
    out[idx] = acc;
}

// DP4A signedness: U8.S8 via inline PTX (dp4a.u32.s32)
extern "C" __global__ void __launch_bounds__(32)
probe_dp4a_u8_s8(int *out, const uint32_t *a, const uint32_t *b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t va = a[idx];
    uint32_t vb = b[idx];
    int acc = 0;
    asm volatile("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(acc) : "r"(va), "r"(vb), "r"(acc));
    out[idx] = acc;
}
