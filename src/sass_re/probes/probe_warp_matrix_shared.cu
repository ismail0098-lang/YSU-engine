/*
 * SASS RE Probe: Warp-Cooperative Shared Memory Tiling Patterns
 * Isolates: Multi-warp shared memory coordination, double-buffered tiling
 *
 * Patterns from production LBM/NeRF kernels:
 *   1. Single-warp tile: 32 threads load a 32-element tile
 *   2. Multi-warp tile: 128 threads (4 warps) load a 128-element tile
 *   3. Double-buffered: overlap next tile load with current tile compute
 *   4. Halo exchange: each warp loads its region + ghost cells from neighbors
 *
 * These patterns determine the BSSY/BSYNC and BAR.SYNC scheduling
 * overhead for cooperative data movement.
 */

// Single-warp tile load + compute
extern "C" __global__ void __launch_bounds__(32)
probe_single_warp_tile(float *out, const float *in, int n) {
    __shared__ float tile[32];
    int base = blockIdx.x * 32;
    int tid = threadIdx.x;

    // Load tile
    tile[tid] = in[base + tid];
    // No sync needed (single warp)

    // Compute with neighbor access
    float left  = tile[(tid + 31) % 32];
    float right = tile[(tid + 1) % 32];
    out[base + tid] = 0.25f * left + 0.5f * tile[tid] + 0.25f * right;
}

// Multi-warp tile: 4 warps cooperate on 128-element tile
extern "C" __global__ void __launch_bounds__(128)
probe_multi_warp_tile(float *out, const float *in, int n) {
    __shared__ float tile[128];
    int base = blockIdx.x * 128;
    int tid = threadIdx.x;

    tile[tid] = in[base + tid];
    __syncthreads();  // BAR.SYNC: all 4 warps must finish loading

    float left  = tile[(tid + 127) % 128];
    float right = tile[(tid + 1) % 128];
    out[base + tid] = 0.25f * left + 0.5f * tile[tid] + 0.25f * right;
}

// Double-buffered tiling: overlap load and compute
extern "C" __global__ void __launch_bounds__(128)
probe_double_buffer_tile(float *out, const float *in, int n_tiles) {
    __shared__ float buf[2][128];
    int tid = threadIdx.x;

    // Load first tile
    buf[0][tid] = in[tid];
    __syncthreads();

    for (int t = 1; t < n_tiles; t++) {
        int cur = (t - 1) & 1;
        int nxt = t & 1;

        // Load next tile while computing current
        buf[nxt][tid] = in[t * 128 + tid];

        // Compute on current tile (no sync needed for read)
        float left  = buf[cur][(tid + 127) % 128];
        float right = buf[cur][(tid + 1) % 128];
        out[(t-1) * 128 + tid] = 0.25f * left + 0.5f * buf[cur][tid] + 0.25f * right;

        __syncthreads();  // Ensure next tile load is complete
    }

    // Process last tile
    int last = (n_tiles - 1) & 1;
    float left  = buf[last][(tid + 127) % 128];
    float right = buf[last][(tid + 1) % 128];
    out[(n_tiles - 1) * 128 + tid] = 0.25f * left + 0.5f * buf[last][tid] + 0.25f * right;
}

// D3Q19 halo exchange pattern (8x8x4 tile from kernels_soa.cu tiled variant)
extern "C" __global__ void __launch_bounds__(256)
probe_halo_exchange(float *out, const float *in, int nx, int ny, int nz) {
    // 8x8x4 = 256 cells per block (1 thread per cell)
    // Halo: +1 on each side = 10x10x6 = 600 shared memory elements
    __shared__ float smem[10 * 10 * 6];  // 2400 bytes

    int tx = threadIdx.x % 8;
    int ty = (threadIdx.x / 8) % 8;
    int tz = threadIdx.x / 64;

    int bx = blockIdx.x * 8;
    int by = blockIdx.y * 8;
    int bz = blockIdx.z * 4;

    int gx = bx + tx, gy = by + ty, gz = bz + tz;

    // Load interior
    int si = (tx + 1) + (ty + 1) * 10 + (tz + 1) * 100;
    int gi = gx + gy * nx + gz * nx * ny;
    if (gx < nx && gy < ny && gz < nz)
        smem[si] = in[gi];

    // Load halo faces (simplified: only x-direction for probe)
    if (tx == 0 && gx > 0)
        smem[si - 1] = in[gi - 1];
    if (tx == 7 && gx + 1 < nx)
        smem[si + 1] = in[gi + 1];

    __syncthreads();

    // 3-point stencil using halo data
    if (gx > 0 && gx < nx - 1 && gy < ny && gz < nz) {
        float left  = smem[si - 1];
        float center = smem[si];
        float right = smem[si + 1];
        out[gi] = 0.25f * left + 0.5f * center + 0.25f * right;
    }
}
