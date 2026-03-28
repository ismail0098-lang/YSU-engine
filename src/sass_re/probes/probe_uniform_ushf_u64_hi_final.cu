/*
 * SASS RE Probe: final strict uniform U64-high shift follow-up
 *
 * Goal: try several cleaner warp-uniform 64-bit source shapes that may coax
 * the compiler into emitting USHF.L.U64.HI instead of falling back to
 * GPR-space SHF.L.U64.HI or mixed U32 uniform forms.
 */

#include <cuda_runtime.h>
#include <stdint.h>

__device__ __constant__ uint64_t probe_uniform_u64_table[4] = {
    0x0000000100000001ull,
    0x0000000200000003ull,
    0x0000000400000007ull,
    0x000000080000000full,
};

extern "C" __global__ void __launch_bounds__(128)
probe_uniform_ushf_const_imm(uint32_t *out) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    uint64_t x = probe_uniform_u64_table[1];
    uint64_t y = x << 13;
    out[i] = (uint32_t)(y >> 32);
}

extern "C" __global__ void __launch_bounds__(128)
probe_uniform_ushf_const_param(uint32_t *out, uint32_t shift) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    uint64_t x = probe_uniform_u64_table[2];
    uint32_t s = shift & 31u;
    uint64_t y = x << s;
    out[i] = (uint32_t)(y >> 32);
}

extern "C" __global__ void __launch_bounds__(128)
probe_uniform_ushf_param_imm(uint32_t *out, uint64_t x) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    uint64_t y = x << 11;
    out[i] = (uint32_t)(y >> 32);
}

extern "C" __global__ void __launch_bounds__(128)
probe_uniform_ushf_addr_use(float *out, const float *base, uint32_t shift) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    uint64_t bytes = probe_uniform_u64_table[0] << (shift & 7u);
    const char *block_ptr = (const char *)base + bytes;
    const float *ptr = (const float *)(block_ptr + ((size_t)(threadIdx.x & 1) << 2));
    out[i] = ptr[0];
}

extern "C" __global__ void __launch_bounds__(128)
probe_uniform_ushf_dual_const_addr(float *out, const float *base) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    uint64_t x = probe_uniform_u64_table[3];
    uint64_t bytes = x << 6;
    const char *block_ptr = (const char *)base + bytes;
    const float *ptr = (const float *)(block_ptr + ((size_t)(threadIdx.x & 3) << 2));
    out[i] = ptr[0];
}
