/*
 * SASS RE Probe: stricter uniform 64-bit shift and signed address forms
 *
 * Goal: move closer to the cuDNN-mined contexts for:
 *   - USHF.L.U64.HI
 *   - ULEA.HI.X.SX32
 *
 * The key idea is to keep a 64-bit byte offset warp-uniform for as long as
 * possible, and only introduce the per-thread term at the very end.
 */

#include <cuda_runtime.h>
#include <stdint.h>

__device__ __constant__ uint32_t probe_uniform_shift_table[8] =
    {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u};

extern "C" __global__ void __launch_bounds__(128)
probe_uniform_u64_shifted_ptr(float *out, const float *base, uint64_t uniform_index) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    uint64_t uniform_bytes = (uniform_index & 0xfu) << 5;
    const char *block_ptr = (const char *)base + uniform_bytes;
    const float *ptr = (const float *)(block_ptr + ((size_t)(threadIdx.x & 3) << 2));

    out[i] = ptr[0];
}

extern "C" __global__ void __launch_bounds__(128)
probe_uniform_u64_shifted_ptr_imm(float *out, const float *base, uint64_t uniform_index) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    uint64_t uniform_bytes = (uniform_index & 0x7u) << 6;
    const char *block_ptr = (const char *)base + uniform_bytes;
    const float *ptr = (const float *)(block_ptr + ((size_t)(threadIdx.x & 1) << 2));

    out[i] = ptr[0] * 2.0f;
}

extern "C" __global__ void __launch_bounds__(128)
probe_uniform_sx32_broadcast(float *out, const float *base, int signed_offset_bytes) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    const char *ptr_bytes = (const char *)base + (int64_t)signed_offset_bytes;
    const float *ptr = (const float *)ptr_bytes;
    float v = ptr[0];
    out[i] = v;
}

extern "C" __global__ void __launch_bounds__(128)
probe_uniform_sx32_thread_term(float *out, const float *base, int signed_offset_bytes) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    const char *ptr_bytes = (const char *)base + (int64_t)signed_offset_bytes;
    const float *ptr = (const float *)(ptr_bytes + ((size_t)(threadIdx.x & 3) << 2));

    out[i] = ptr[0];
}

extern "C" __global__ void __launch_bounds__(128)
probe_uniform_ushf_table_shift(float *out, const float *base, int shift_index) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    uint32_t shift = probe_uniform_shift_table[shift_index & 7] & 7u;
    uint64_t uniform_bytes =
        ((uint64_t)probe_uniform_shift_table[(shift_index + 1) & 7]) << shift;
    const char *block_ptr = (const char *)base + uniform_bytes;
    const float *ptr = (const float *)(block_ptr + ((size_t)(threadIdx.x & 1) << 2));

    out[i] = ptr[0];
}

extern "C" __global__ void __launch_bounds__(128)
probe_uniform_ushf_table_shift_imm(float *out, const float *base, int shift_index) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    uint64_t uniform_bytes =
        ((uint64_t)probe_uniform_shift_table[shift_index & 7]) << 6;
    const char *block_ptr = (const char *)base + uniform_bytes;
    const float *ptr = (const float *)(block_ptr + ((size_t)(threadIdx.x & 1) << 2));

    out[i] = ptr[0] * 0.5f;
}
