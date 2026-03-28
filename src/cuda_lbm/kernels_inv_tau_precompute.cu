// Pre-compute inv_tau (1/tau) field to eliminate MUFU.RCP from BGK collision.
//
// From SASS RE measurements:
//   MUFU.RCP latency: 41.53 cycles (SFU pipeline)
//   FFMA latency: 4.53 cycles (FP32 pipeline)
//   BGK collision: ~57 FMA ops/cell
//   MUFU.RCP overhead in BGK: 41.53 / (57*4.53) = ~16% of ALU budget
//
// By pre-computing inv_tau once (when tau changes) and passing it as a
// per-cell array, every BGK step saves one MUFU.RCP per cell.
//
// Memory cost: 4 bytes/cell (FP32 inv_tau field)
//   At 128^3: 2M cells * 4 = 8 MB (vs 304 MB FP32 distributions = 2.6%)
//
// When tau is spatially uniform (constant), inv_tau can be passed as a
// kernel argument instead of an array (zero memory cost).
//
// Usage:
//   1. Call compute_inv_tau_kernel once when tau field changes
//   2. Pass inv_tau array to BGK kernels instead of tau
//   3. Kernel replaces `float inv_tau = 1.0f / tau_local;` with
//      `float inv_tau = __ldg(&inv_tau_arr[idx]);`
//      LDG at ~33 cy (L1 hit) vs MUFU.RCP at 41.53 cy = 20% faster
//      And LDG can be overlapped with other warps (MUFU.RCP cannot).

extern "C" __launch_bounds__(256) __global__
void compute_inv_tau_kernel(
    float* __restrict__ inv_tau_out,
    const float* __restrict__ tau_in,
    int n_cells
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cells) return;

    float tau = tau_in[idx];
    // Guard against division by zero or denormal tau
    if (tau < 0.5f) tau = 0.5f;  // Minimum stable tau
    inv_tau_out[idx] = 1.0f / tau;
}

// Variant: also pre-compute (1 - 0.5/tau) for Guo forcing prefactor
extern "C" __launch_bounds__(256) __global__
void compute_inv_tau_and_prefactor_kernel(
    float* __restrict__ inv_tau_out,
    float* __restrict__ guo_prefactor_out,
    const float* __restrict__ tau_in,
    int n_cells
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cells) return;

    float tau = tau_in[idx];
    if (tau < 0.5f) tau = 0.5f;
    float inv_tau = 1.0f / tau;
    inv_tau_out[idx] = inv_tau;
    guo_prefactor_out[idx] = 1.0f - 0.5f * inv_tau;
}

// Variant: uniform tau (single value, no array needed)
// Sets all cells to the same inv_tau. Use when tau is spatially constant.
extern "C" __launch_bounds__(256) __global__
void set_uniform_inv_tau_kernel(
    float* __restrict__ inv_tau_out,
    float tau_uniform,
    int n_cells
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cells) return;
    inv_tau_out[idx] = 1.0f / tau_uniform;
}
