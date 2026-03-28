/*
 * SASS RE Probe: direct ldmatrix transposed x4 inline PTX
 *
 * cuDNN library mining first surfaced LDSM.16.MT88.4 near Ada. This probe
 * forces ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 directly so the
 * resulting SASS spelling can be confirmed locally on sm_89.
 */

#include <cuda_runtime.h>
#include <stdint.h>

static __device__ __forceinline__ unsigned smem_addr_ldm(const unsigned short *ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

extern "C" __global__ void __launch_bounds__(32)
probe_ldmatrix_x4(unsigned *out, const unsigned short *src) {
    __shared__ __align__(16) unsigned short smem[8 * 8 * 4];
    int lane = threadIdx.x;
    for (int i = lane; i < 8 * 8 * 4; i += 32) {
        smem[i] = src[i];
    }
    __syncthreads();

    unsigned row = static_cast<unsigned>(lane & 7);
    unsigned matrix = static_cast<unsigned>(lane >> 3);
    unsigned addr = smem_addr_ldm(smem + matrix * 64 + row * 8);
    unsigned r0 = 0, r1 = 0, r2 = 0, r3 = 0;

    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));

    out[lane + 0 * 32] = r0;
    out[lane + 1 * 32] = r1;
    out[lane + 2 * 32] = r2;
    out[lane + 3 * 32] = r3;
}

extern "C" __global__ void __launch_bounds__(32)
probe_ldmatrix_x4_trans(unsigned *out, const unsigned short *src) {
    __shared__ __align__(16) unsigned short smem[8 * 8 * 4];
    int lane = threadIdx.x;
    for (int i = lane; i < 8 * 8 * 4; i += 32) {
        smem[i] = src[i];
    }
    __syncthreads();

    unsigned row = static_cast<unsigned>(lane & 7);
    unsigned matrix = static_cast<unsigned>(lane >> 3);
    unsigned addr = smem_addr_ldm(smem + matrix * 64 + row * 8);
    unsigned r0 = 0, r1 = 0, r2 = 0, r3 = 0;

    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));

    out[lane + 0 * 32] = r0;
    out[lane + 1 * 32] = r1;
    out[lane + 2 * 32] = r2;
    out[lane + 3 * 32] = r3;
}
