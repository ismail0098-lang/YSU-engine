/*
 * SASS RE Probe: Memory Operations
 * Isolates: LDG, STG, LDS, STS, LDGSTS (async), LDL, STL, ATOM, RED, BAR, MEMBAR
 *
 * Memory instructions are the most latency-sensitive.
 * L1 hit: ~34 cycles, L2 hit: ~200 cycles, DRAM: ~400+ cycles.
 * Shared memory: ~23 cycles latency, ~100 bytes/clock bandwidth.
 *
 * LDGSTS: async global->shared copy (Ada Lovelace hardware DMA).
 */

// LDG: global memory load (various widths)
extern "C" __global__ void __launch_bounds__(32)
probe_ldg(float *out, const float *a, const float2 *b, const float4 *c) {
    int i = threadIdx.x;

    // LDG.E.32: 32-bit load
    float v1 = a[i];

    // LDG.E.64: 64-bit load (float2)
    float2 v2 = b[i];

    // LDG.E.128: 128-bit load (float4)
    float4 v4 = c[i];

    // LDG with constant cache hint: __ldg intrinsic
    float v1c = __ldg(&a[i + 32]);

    out[i] = v1 + v2.x + v2.y + v4.x + v4.y + v4.z + v4.w + v1c;
}

// STG: global memory store (various widths)
extern "C" __global__ void __launch_bounds__(32)
probe_stg(float *out1, float2 *out2, float4 *out4, const float *a) {
    int i = threadIdx.x;
    float v = a[i];

    // STG.E.32
    out1[i] = v;

    // STG.E.64
    out2[i] = make_float2(v, v);

    // STG.E.128
    out4[i] = make_float4(v, v, v, v);
}

// LDS / STS: shared memory load / store
extern "C" __global__ void __launch_bounds__(32)
probe_shared(float *out, const float *a) {
    __shared__ float smem[256];
    int i = threadIdx.x;

    // STS: store to shared
    smem[i] = a[i];
    smem[i + 32] = a[i + 32];
    smem[i + 64] = a[i + 64];
    smem[i + 96] = a[i + 96];
    __syncthreads();

    // LDS: load from shared (with different access patterns)
    float v = smem[i];
    v += smem[i ^ 1];    // XOR pattern (bank conflict test)
    v += smem[i ^ 2];
    v += smem[31 - i];   // Reverse pattern

    out[i] = v;
}

// Shared memory bank conflict probe
extern "C" __global__ void __launch_bounds__(32)
probe_bank_conflicts(float *out, const float *a) {
    __shared__ float smem[1024];
    int i = threadIdx.x;

    // No bank conflict: stride-1 access
    smem[i] = a[i];
    __syncthreads();

    float v = smem[i];  // stride-1: no conflict

    // 2-way bank conflict: stride-2
    v += smem[i * 2];

    // Broadcast: all threads read same address (no conflict on Ada)
    v += smem[0];

    // N-way bank conflict: stride-32 (all hit bank 0)
    // (only valid for first thread to avoid OOB)
    if (i == 0) {
        for (int j = 0; j < 32; j++) {
            smem[j * 32] = (float)j;
        }
    }
    __syncthreads();
    v += smem[i * 32 % 1024];

    out[i] = v;
}

// ATOM: atomic operations
extern "C" __global__ void __launch_bounds__(32)
probe_atomics(int *out, int *counter, float *fout, float *fcounter) {
    int i = threadIdx.x;

    // ATOM.E.ADD: atomic add (global)
    int old_add = atomicAdd(counter, 1);

    // ATOM.E.MIN / MAX
    int old_min = atomicMin(counter + 1, i);
    int old_max = atomicMax(counter + 2, i);

    // ATOM.E.CAS: compare-and-swap
    int old_cas = atomicCAS(counter + 3, 0, i);

    // ATOM.E.EXCH: exchange
    int old_exch = atomicExch(counter + 4, i);

    // ATOM.E.AND / OR / XOR
    int old_and = atomicAnd(counter + 5, i);
    int old_or  = atomicOr(counter + 6, i);
    int old_xor = atomicXor(counter + 7, i);

    out[i] = old_add + old_min + old_max + old_cas + old_exch + old_and + old_or + old_xor;
}

// Shared memory atomics
extern "C" __global__ void __launch_bounds__(32)
probe_shared_atomics(int *out) {
    __shared__ int smem[8];
    int i = threadIdx.x;
    if (i < 8) smem[i] = 0;
    __syncthreads();

    // ATOMS: shared memory atomic add
    atomicAdd(&smem[0], 1);
    atomicMin(&smem[1], i);
    atomicMax(&smem[2], i);
    __syncthreads();

    if (i < 8) out[i] = smem[i];
}

// MEMBAR / fence operations
extern "C" __global__ void __launch_bounds__(32)
probe_fences(int *out, int *flag, const int *data) {
    int i = threadIdx.x;

    if (i == 0) {
        // Store data, then fence, then set flag
        // Look for MEMBAR.GL (global fence) in SASS
        out[0] = 42;
        __threadfence();    // MEMBAR.GL
        flag[0] = 1;
    }

    // MEMBAR.CTA: block-level fence
    __shared__ int s;
    if (i == 0) s = data[0];
    __threadfence_block();  // MEMBAR.CTA
    __syncthreads();
    out[i] = s;
}
