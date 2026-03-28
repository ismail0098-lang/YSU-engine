/*
 * SASS RE Probe: Cooperative Groups and Grid-Level Synchronization
 * Isolates: cooperative_groups API, grid.sync(), SETCTAID-related ops
 *
 * Cooperative groups enable grid-level synchronization: all blocks
 * across all SMs synchronize before proceeding. This is used for:
 *   - Multi-pass algorithms without kernel re-launch
 *   - Global reductions without atomics
 *   - Iterative solvers with global convergence checks
 *
 * The grid.sync() call compiles to hardware barrier operations that
 * stall all blocks until every block reaches the barrier.
 *
 * Requires: cudaLaunchCooperativeKernel (not <<<>>>)
 * Hardware requirement: SM 6.0+ with cooperative launch support
 *
 * Key SASS:
 *   BAR.SYNC          -- block-level barrier
 *   CCTL              -- cache control (for cooperative memory visibility)
 *   MEMBAR.GPU        -- GPU-scope fence (ensures visibility across SMs)
 *   (Grid sync may compile to ATOM + spin-wait pattern)
 */

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Block-level sync (baseline, generates BAR.SYNC)
extern "C" __global__ void __launch_bounds__(128)
probe_block_sync(float *out, const float *in, int n) {
    __shared__ float smem[128];
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    smem[threadIdx.x] = in[i];
    __syncthreads();  // BAR.SYNC

    // Cross-thread read within block
    float neighbor = smem[(threadIdx.x + 1) % blockDim.x];
    __syncthreads();  // BAR.SYNC

    out[i] = smem[threadIdx.x] + neighbor;
}

// Cooperative groups: thread block group
extern "C" __global__ void __launch_bounds__(128)
probe_cg_thread_block(float *out, const float *in, int n) {
    auto block = cg::this_thread_block();
    int i = block.thread_rank() + block.group_index().x * block.size();
    if (i >= n) return;

    __shared__ float smem[128];
    smem[block.thread_rank()] = in[i];
    block.sync();  // Cooperative group sync (should generate BAR.SYNC)

    float val = smem[block.thread_rank()];
    out[i] = val * 2.0f;
}

// Cooperative groups: warp-level tiled partition
extern "C" __global__ void __launch_bounds__(128)
probe_cg_warp_partition(float *out, const float *in, int n) {
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    float val = in[i];

    // Warp-level reduction via cooperative groups API
    for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
        val += warp.shfl_xor(val, offset);
    }

    if (warp.thread_rank() == 0) {
        out[i / 32] = val;
    }
}

// Cooperative groups: sub-warp tiled partition (8 threads)
extern "C" __global__ void __launch_bounds__(128)
probe_cg_subwarp_partition(float *out, const float *in, int n) {
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<8>(block);

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    float val = in[i];

    // 8-thread tile reduction (3 steps instead of 5)
    for (int offset = tile.size() / 2; offset > 0; offset /= 2) {
        val += tile.shfl_xor(val, offset);
    }

    if (tile.thread_rank() == 0) {
        out[i / 8] = val;
    }
}

// Grid-level sync (requires cooperative kernel launch)
// NOTE: This kernel must be launched via cudaLaunchCooperativeKernel
extern "C" __global__ void __launch_bounds__(128)
probe_cg_grid_sync(float *data, int n, int iterations) {
    auto grid = cg::this_grid();
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    for (int iter = 0; iter < iterations; iter++) {
        // Phase 1: compute
        data[i] = data[i] * 0.999f + 0.001f;

        // Grid sync: ALL blocks across ALL SMs synchronize here
        grid.sync();

        // Phase 2: read neighbor (safe because grid.sync guarantees visibility)
        if (i + 1 < n) {
            data[i] = (data[i] + data[i + 1]) * 0.5f;
        }

        grid.sync();
    }
}

// Coalesced group: only active threads participate
extern "C" __global__ void __launch_bounds__(128)
probe_cg_coalesced(float *out, const float *in, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    float val = in[i];

    // Only threads with val > 0 form a coalesced group
    if (val > 0.0f) {
        auto active = cg::coalesced_threads();
        // Reduce within the active set only
        for (int offset = active.size() / 2; offset > 0; offset /= 2) {
            val += active.shfl_down(val, offset);
        }
        if (active.thread_rank() == 0) {
            atomicAdd(&out[0], val);
        }
    }
}
