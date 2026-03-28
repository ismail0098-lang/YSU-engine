/*
 * SASS RE Probe: L2 Cache Policy and Streaming Store Operations
 * Isolates: STG.CS (streaming store), LDG with __ldg(), L2 eviction hints
 *
 * Ada Lovelace SM 8.9 cache hierarchy:
 *   L1: 128 KB per SM, ~33 cycle latency (data + texture unified)
 *   L2: 48 MB shared, ~200 cycle latency
 *   GDDR6X: 504 GB/s peak, ~400+ cycle latency
 *
 * Cache policy annotations control which cache level receives a line:
 *   __ldg()  : Read-only texture cache path (bypasses L1 writeback pollution)
 *   __stcs() : Streaming store (L2 evict-first, does not pollute L1)
 *   __ldcs() : Streaming load (L2 evict-first)
 *   __stwb() : Write-back store (normal L1+L2 caching)
 *
 * In kernels_fp32_soa_cs.cu, __ldg() for ping reads + __stcs() for pong writes
 * prevents cold pong writes from evicting hot ping data. Measured <3% gain
 * at 128^3 (working set >> 48 MB L2), but significant at 64^3.
 *
 * Key SASS instructions:
 *   STG.E.CS   -- streaming store (evict-first L2 policy)
 *   STG.E.WB   -- write-back store (normal)
 *   STG.E.WT   -- write-through store
 *   LDG.E      -- normal global load
 *   LDG.E.CI   -- constant/immutable load (__ldg path)
 */

// Streaming store (__stcs) vs normal store
extern "C" __global__ void __launch_bounds__(128)
probe_streaming_store(float *out_normal, float *out_streaming,
                      const float *in) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    float val = in[i];

    // Normal store: STG.E (uses L1+L2 write-allocate)
    out_normal[i] = val;

    // Streaming store: STG.E.CS (L2 evict-first, skips L1)
    __stcs(out_streaming + i, val);
}

// Read-only cache path (__ldg) vs normal load
extern "C" __global__ void __launch_bounds__(128)
probe_readonly_load(float *out, const float *normal_in,
                    const float *readonly_in) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    // Normal load: LDG.E
    float v1 = normal_in[i];

    // Read-only texture cache load: LDG.E.CI via __ldg()
    float v2 = __ldg(readonly_in + i);

    out[i] = v1 + v2;
}

// Combined ping/pong cache separation pattern (from kernels_fp32_soa_cs.cu):
// Read ping via __ldg(), write pong via __stcs()
// This prevents pong write-allocate from evicting ping read data in L2.
extern "C" __global__ void __launch_bounds__(128)
probe_pingpong_cache_separation(float *pong, const float *ping, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    // Read from ping buffer via read-only cache path
    float val = __ldg(ping + i);

    // Simple transform (simulates collision)
    val = val * 0.99f + 0.01f;

    // Write to pong buffer via streaming store (L2 evict-first)
    __stcs(pong + i, val);
}

// Vectorized cache policy ops (float4 width)
extern "C" __global__ void __launch_bounds__(128)
probe_cache_policy_vectorized(float4 *out_cs, float4 *out_normal,
                              const float *in) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    // Vectorized read-only load
    float v0 = __ldg(in + i * 4 + 0);
    float v1 = __ldg(in + i * 4 + 1);
    float v2 = __ldg(in + i * 4 + 2);
    float v3 = __ldg(in + i * 4 + 3);

    float4 val = make_float4(v0, v1, v2, v3);

    // Normal vector store
    out_normal[i] = val;

    // Streaming store requires scalar path (no float4 __stcs overload)
    __stcs(reinterpret_cast<float*>(out_cs) + i * 4 + 0, v0);
    __stcs(reinterpret_cast<float*>(out_cs) + i * 4 + 1, v1);
    __stcs(reinterpret_cast<float*>(out_cs) + i * 4 + 2, v2);
    __stcs(reinterpret_cast<float*>(out_cs) + i * 4 + 3, v3);
}
