/*
 * SASS RE Probe: Named Barriers (BAR.SYNC with barrier ID)
 * Ada supports up to 16 named barriers per thread block.
 * BAR.SYNC <id>, <thread_count> synchronizes a SUBSET of threads.
 */

// Named barrier 0 (default __syncthreads)
extern "C" __global__ void __launch_bounds__(256)
probe_bar_sync_default(float *out, const float *in) {
    __shared__ float s[256];
    s[threadIdx.x] = in[threadIdx.x + blockIdx.x * 256];
    __syncthreads();  // BAR.SYNC 0
    out[threadIdx.x + blockIdx.x * 256] = s[255 - threadIdx.x];
}

// Named barrier with thread count (partial block sync)
extern "C" __global__ void __launch_bounds__(256)
probe_bar_sync_count(float *out, const float *in) {
    __shared__ float s[256];
    s[threadIdx.x] = in[threadIdx.x + blockIdx.x * 256];

    // Only first 128 threads participate in this barrier
    if (threadIdx.x < 128) {
        // Use asm to force a specific barrier ID and count
        asm volatile("bar.sync 1, 128;");
    }
    __syncthreads();  // Full block sync after
    out[threadIdx.x + blockIdx.x * 256] = s[threadIdx.x];
}

// Multiple named barriers (different subsets sync independently)
extern "C" __global__ void __launch_bounds__(256)
probe_bar_sync_multiple(float *out, const float *in) {
    __shared__ float s[256];
    int tid = threadIdx.x;
    s[tid] = in[tid + blockIdx.x * 256];

    // Barrier 1: first half syncs
    if (tid < 128) asm volatile("bar.sync 1, 128;");
    // Barrier 2: second half syncs
    if (tid >= 128) asm volatile("bar.sync 2, 128;");

    // Full sync
    __syncthreads();
    out[tid + blockIdx.x * 256] = s[255 - tid];
}

// BAR.ARRIVE: signal arrival without waiting
extern "C" __global__ void __launch_bounds__(256)
probe_bar_arrive(float *out, const float *in) {
    __shared__ float s[256];
    s[threadIdx.x] = in[threadIdx.x + blockIdx.x * 256];

    // Arrive at barrier 3 without waiting (split-phase)
    asm volatile("bar.arrive 3, 256;");

    // Do independent work here (no sync needed yet)
    float local = s[threadIdx.x] * 2.0f;

    // Now wait for barrier 3
    asm volatile("bar.sync 3, 256;");

    out[threadIdx.x + blockIdx.x * 256] = local + s[255 - threadIdx.x];
}

// BAR.RED: barrier with reduction (AND/OR/POPC)
extern "C" __global__ void __launch_bounds__(256)
probe_bar_red(float *out, const float *in) {
    __shared__ float s[256];
    s[threadIdx.x] = in[threadIdx.x + blockIdx.x * 256];

    int pred = (s[threadIdx.x] > 0.5f) ? 1 : 0;

    // Use __syncthreads + ballot as the barrier-reduction equivalent
    __syncthreads();
    unsigned ballot = __ballot_sync(0xFFFFFFFF, pred);
    int all_positive = __all_sync(0xFFFFFFFF, pred);
    int any_positive = __any_sync(0xFFFFFFFF, pred);
    int count = __popc(ballot);

    out[threadIdx.x + blockIdx.x * 256] = (float)(all_positive + any_positive + count);
}
