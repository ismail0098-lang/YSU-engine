/*
 * SASS RE Probe: Asynchronous Copy (cp.async / LDGSTS)
 * Isolates: Hardware async global->shared memory transfer on Ada SM 8.0+
 *
 * cp.async bypasses the register file entirely: data moves from global memory
 * directly into shared memory via the L2 cache, without consuming registers
 * or ALU cycles. The SM can execute compute instructions while the copy
 * proceeds in the background.
 *
 * Available since Ampere (SM 8.0). On Ada SM 8.9, cp.async is the preferred
 * path for tiled kernels that need halo data in shared memory.
 *
 * Key SASS instructions to look for:
 *   LDGSTS     -- Load Global, Store Shared (hardware DMA path)
 *   LDGDEPBAR  -- cp.async dependency barrier (wait for N outstanding copies)
 *   BAR.SYNC   -- block-level sync (after cp.async group commit)
 *
 * The PTX instructions are:
 *   cp.async.ca.shared.global [dst_smem], [src_gmem], 4;   // 4-byte copy
 *   cp.async.ca.shared.global [dst_smem], [src_gmem], 16;  // 16-byte copy
 *   cp.async.commit_group;
 *   cp.async.wait_group 0;
 */

#include <cuda_pipeline.h>
#include <cooperative_groups.h>

// Basic cp.async: copy 4 bytes from global to shared
extern "C" __global__ void __launch_bounds__(128)
probe_cp_async_basic(float *out, const float *in, int n) {
    __shared__ float smem[128];
    int i = threadIdx.x;

    // Initiate async copy: global -> shared (4 bytes per thread)
    __pipeline_memcpy_async(&smem[i], &in[i], sizeof(float));
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    // Use the data from shared memory
    out[i] = smem[i] * 2.0f;
}

// cp.async with 16-byte (float4) vectorized transfer
extern "C" __global__ void __launch_bounds__(128)
probe_cp_async_vec4(float *out, const float *in, int n) {
    __shared__ float smem[512];  // 128 threads * 4 floats
    int i = threadIdx.x;

    // 16-byte async copy (float4 width)
    __pipeline_memcpy_async(&smem[i * 4], &in[i * 4], 16);
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    float sum = smem[i * 4] + smem[i * 4 + 1] + smem[i * 4 + 2] + smem[i * 4 + 3];
    out[i] = sum;
}

// Multi-stage pipeline: overlap copy of next tile with compute on current tile
extern "C" __global__ void __launch_bounds__(128)
probe_cp_async_pipeline(float *out, const float *in, int n_tiles) {
    __shared__ float smem[2][128];  // Double buffer
    int i = threadIdx.x;

    // Stage 0: initiate first tile copy
    __pipeline_memcpy_async(&smem[0][i], &in[i], sizeof(float));
    __pipeline_commit();

    for (int tile = 1; tile < n_tiles; tile++) {
        int buf_curr = (tile - 1) & 1;
        int buf_next = tile & 1;

        // Initiate next tile copy while waiting for current
        __pipeline_memcpy_async(&smem[buf_next][i], &in[tile * 128 + i], sizeof(float));
        __pipeline_commit();

        // Wait for current tile (group 1 = the one before the latest commit)
        __pipeline_wait_prior(1);
        __syncthreads();

        // Compute on current tile data
        out[(tile - 1) * 128 + i] = smem[buf_curr][i] * 2.0f + 1.0f;
        __syncthreads();
    }

    // Drain last tile
    __pipeline_wait_prior(0);
    __syncthreads();
    out[(n_tiles - 1) * 128 + i] = smem[(n_tiles - 1) & 1][i] * 2.0f + 1.0f;
}
