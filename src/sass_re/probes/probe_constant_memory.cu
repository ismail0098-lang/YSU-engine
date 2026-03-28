/*
 * SASS RE Probe: Constant Memory and Constant Cache
 * Isolates: LDC (constant memory load), __constant__ broadcast behavior
 *
 * Ada Lovelace constant memory hierarchy:
 *   __constant__: 64 KB per kernel, cached in dedicated constant cache
 *   Constant cache: per-SM, optimized for broadcast (all threads read same addr)
 *   LDC latency: expected ~4-8 cycles for cached, same-address broadcast
 *
 * LDC is distinct from LDG: it uses a separate cache with broadcast optimization.
 * When all threads in a warp read the same constant address, it's a single
 * transaction (no bank conflicts, no coalescing needed).
 *
 * This is relevant for D3Q19 LBM: the lattice weights (D3Q19_WF[19]) and
 * velocity vectors (D3Q19_CX/CY/CZ[19]) are __constant__ arrays that every
 * thread reads identically.
 *
 * Key SASS:
 *   LDC          -- constant memory load (32-bit)
 *   LDC.64       -- constant memory load (64-bit)
 *   ULDC         -- uniform constant load (to uniform register file)
 *   ULDC.64      -- 64-bit uniform constant load
 */

// D3Q19 lattice constants (real values from LBM kernels)
__constant__ float D3Q19_W[19] = {
    1.0f/3.0f,
    1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f,
    1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f
};

__constant__ int D3Q19_CX[19] = {
    0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0
};

__constant__ int D3Q19_CY[19] = {
    0, 0, 0, 1,-1, 0, 0, 1,-1, 0, 0, 0, 0,-1, 1, 1,-1, 1,-1
};

__constant__ float LARGE_CONST[256];  // For non-broadcast access pattern test

// Broadcast constant read: all threads read same constant array element
// Should generate LDC with zero-cost broadcast
extern "C" __global__ void __launch_bounds__(128)
probe_const_broadcast(float *out, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    float sum = 0.0f;
    // All threads read D3Q19_W[d] for the same d -- pure broadcast
    #pragma unroll
    for (int d = 0; d < 19; d++) {
        sum += D3Q19_W[d];
    }
    out[i] = sum;
}

// Per-thread divergent constant read: each thread reads different element
// This breaks the broadcast optimization -- should show higher latency
extern "C" __global__ void __launch_bounds__(128)
probe_const_divergent(float *out, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    // Each thread reads a different constant element
    float val = LARGE_CONST[threadIdx.x % 256];
    out[i] = val;
}

// LBM equilibrium computation using constant memory weights
// This is the actual pattern from D3Q19 kernels
extern "C" __global__ void __launch_bounds__(128)
probe_const_lbm_equilibrium(float *feq, const float *rho,
                            const float *ux, const float *uy,
                            const float *uz, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    float r = rho[i];
    float u = ux[i], v = uy[i], w = uz[i];
    float usq = u*u + v*v + w*w;

    // LDC for weights and velocity vectors (broadcast across warp)
    #pragma unroll
    for (int d = 0; d < 19; d++) {
        float eu = (float)D3Q19_CX[d] * u
                 + (float)D3Q19_CY[d] * v;
        // Horner form: w*rho * ((4.5*eu + 3)*eu + 1 - 1.5*usq)
        float feq_d = D3Q19_W[d] * r * fmaf(fmaf(eu, 4.5f, 3.0f), eu, 1.0f - 1.5f * usq);
        feq[d * n + i] = feq_d;
    }
}

// Constant memory latency chain: dependent reads from constant
extern "C" __global__ void __launch_bounds__(32)
probe_const_chain(float *out) {
    int idx = threadIdx.x % 19;
    float acc = 0.0f;

    // Chain: read constant[idx], use result to index next read
    #pragma unroll 1
    for (int j = 0; j < 512; j++) {
        float w = D3Q19_W[idx];
        acc += w;
        idx = (idx + 1) % 19;
    }
    out[threadIdx.x] = acc;
}
