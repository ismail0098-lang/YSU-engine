/*
 * SASS RE Probe: Uniform Register File and Uniform Datapath
 * Isolates: UIADD3, UMOV, ULOP3, UIMAD, USETP, UPOPC, UCLEA, S2UR
 *
 * Ada Lovelace has a SEPARATE uniform register file (UR0-UR63) that stores
 * warp-uniform values (same value across all 32 lanes). The uniform datapath
 * executes scalar operations once per warp instead of per-thread.
 *
 * The compiler automatically promotes warp-uniform computations to the
 * uniform datapath. This probe creates patterns that should trigger
 * uniform register allocation.
 *
 * Key SASS:
 *   UIADD3  -- uniform integer 3-input add
 *   UMOV    -- uniform register move
 *   ULOP3   -- uniform 3-input logic
 *   UIMAD   -- uniform integer multiply-add
 *   USETP   -- uniform set predicate
 *   UPOPC   -- uniform population count
 *   S2UR    -- special register to uniform register
 *   ULDC    -- uniform constant load (already observed)
 *   UCLEA   -- uniform constant LEA
 *   USHF    -- uniform shift (already observed)
 *   UPLOP3  -- uniform predicate logic
 */

// Warp-uniform address computation (should use UIADD3/UIMAD)
extern "C" __global__ void __launch_bounds__(128)
probe_uniform_address(float *out, const float *in, int stride, int offset) {
    // blockIdx.x and gridDim.x are warp-uniform -> compiler uses UR
    int block_base = blockIdx.x * blockDim.x;  // UIMAD
    int global_offset = block_base + offset;     // UIADD3

    // threadIdx.x is NOT uniform -> stays in regular registers
    int tid = threadIdx.x;
    int addr = global_offset + tid * stride;

    out[addr] = in[addr] * 2.0f;
}

// Uniform control flow (branch condition is warp-uniform)
extern "C" __global__ void __launch_bounds__(128)
probe_uniform_branch(float *out, const float *in, int n, int mode) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    float val = in[i];

    // 'mode' is uniform across the warp (kernel argument)
    // Compiler should use USETP for the uniform predicate
    if (mode == 0) {
        val = val * 2.0f;
    } else if (mode == 1) {
        val = val + 1.0f;
    } else if (mode == 2) {
        val = val * val;
    } else {
        val = -val;
    }

    out[i] = val;
}

// Uniform loop bound (trip count is warp-uniform)
extern "C" __global__ void __launch_bounds__(128)
probe_uniform_loop(float *out, const float *in, int n, int iterations) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    float val = in[i];

    // 'iterations' is uniform -> loop control uses uniform datapath
    for (int j = 0; j < iterations; j++) {
        val = val * 0.999f + 0.001f;
    }

    out[i] = val;
}

// Uniform bit manipulation
extern "C" __global__ void __launch_bounds__(128)
probe_uniform_bitops(unsigned *out, unsigned mask, unsigned shift_amt) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned val = (unsigned)i;

    // mask and shift_amt are uniform -> ULOP3, USHF
    val = val & mask;
    val = val << shift_amt;
    val = val ^ mask;

    out[i] = val;
}

// Force S2UR: read special register into uniform register
extern "C" __global__ void __launch_bounds__(128)
probe_s2ur(unsigned *out) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    // gridDim, blockDim are uniform -> S2UR
    unsigned grid_size = gridDim.x * blockDim.x;  // UIMAD from uniform regs
    unsigned block_id = blockIdx.x;                 // S2UR

    // Use uniform values in computation
    out[i] = (unsigned)i + grid_size + block_id;
}
