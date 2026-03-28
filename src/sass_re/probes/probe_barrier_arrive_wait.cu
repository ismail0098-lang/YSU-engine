/*
 * SASS RE Probe: Split-Phase Barriers (arrive/wait pattern)
 * Isolates: BAR.ARRIVE + BAR.SYNC split-phase for pipeline overlap
 *
 * Split-phase barriers allow threads to signal arrival (non-blocking)
 * then continue with independent work before waiting. This enables
 * overlapping compute with barrier propagation latency.
 */

// Classic producer-consumer with split barrier
extern "C" __global__ void __launch_bounds__(256)
probe_arrive_wait_producer_consumer(float *out, const float *in, int n) {
    __shared__ float buf[256];
    int tid = threadIdx.x;
    int base = blockIdx.x * 256;

    // Producers populate the full shared tile, then signal with a half-block
    // split barrier that consumers can wait on safely.
    if (tid < 128) {
        float lo = (base + tid < n) ? in[base + tid] : 0.0f;
        float hi = (base + tid + 128 < n) ? in[base + tid + 128] : 0.0f;
        buf[tid] = lo;
        buf[tid + 128] = hi;
        asm volatile("bar.arrive 1, 128;");
        // Independent producer work keeps the split-phase path alive.
        if (base + tid < n)
            out[base + tid] = lo + hi * 0.125f;
    }

    if (tid >= 128) {
        asm volatile("bar.sync 1, 128;");
        if (base + tid < n)
            out[base + tid] = buf[tid] * 2.0f;
    }

    __syncthreads();
}

// Multi-stage pipeline with arrive/wait
extern "C" __global__ void __launch_bounds__(256)
probe_arrive_wait_pipeline(float *out, const float *in, int stages) {
    __shared__ float ping[128], pong[128];
    int tid = threadIdx.x;
    int lane = tid & 127;
    int base = blockIdx.x * 128 * stages;

    if (tid < 128)
        ping[lane] = in[base + lane];
    __syncthreads();

    for (int s = 1; s < stages; s++) {
        float *src = (s & 1) ? ping : pong;
        float *dst = (s & 1) ? pong : ping;

        if (tid < 128) {
            dst[lane] = in[base + s * 128 + lane];
            asm volatile("bar.arrive 2, 128;");
            // Producer-side independent work keeps arrive and final wait decoupled.
            float scratch = dst[lane] * 0.0625f;
            if (scratch < -1.0e30f)
                out[base] = scratch;
        } else {
            asm volatile("bar.sync 2, 128;");
            float center = src[lane];
            float right = src[(lane + 1) & 127];
            out[base + (s - 1) * 128 + lane] = center * 0.75f + right * 0.25f;
        }
        __syncthreads();
    }

    if (tid < 128) {
        float *last = (stages & 1) ? ping : pong;
        out[base + (stages - 1) * 128 + lane] = last[lane];
    }
}
