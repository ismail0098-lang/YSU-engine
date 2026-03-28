/*
 * SASS RE Probe: Convergence Barriers and Thread Lifecycle
 * Isolates: BMOV, BSSY, BSYNC, BREAK, YIELD, KILL, EXIT variants
 *
 * Ada Lovelace uses a convergence barrier model (not the Volta-style
 * independent thread scheduling). BSSY (barrier set synchronize) and
 * BSYNC (barrier synchronize) manage warp reconvergence after divergence.
 *
 * BMOV manipulates barrier tokens for complex control flow (nested loops
 * with break, switch statements, etc.).
 *
 * Key SASS:
 *   BSSY      -- set barrier for reconvergence point
 *   BSYNC     -- synchronize at barrier (reconverge)
 *   BMOV      -- move/copy barrier token
 *   BREAK     -- exit from loop (break to barrier)
 *   YIELD     -- hint to scheduler to switch warps
 *   EXIT      -- thread termination
 *   BRA       -- branch (conditional/unconditional)
 *   BRA.U     -- uniform branch (all threads take same path)
 */

// Simple divergence: if/else generates BSSY + BSYNC
extern "C" __global__ void __launch_bounds__(32)
probe_simple_diverge(float *out, const float *in) {
    int i = threadIdx.x;
    float val = in[i];

    // Divergent branch: even vs odd threads
    // Should generate: BSSY (set reconvergence), BRA, ..., BSYNC
    if (i % 2 == 0) {
        val = val * 2.0f;
    } else {
        val = val * 3.0f;
    }

    out[i] = val;
}

// Nested divergence: multiple BSSY/BSYNC levels
extern "C" __global__ void __launch_bounds__(32)
probe_nested_diverge(float *out, const float *in) {
    int i = threadIdx.x;
    float val = in[i];

    // Outer divergence (BSSY level 1)
    if (i < 16) {
        // Inner divergence (BSSY level 2)
        if (i < 8) {
            val = val * 4.0f;
        } else {
            val = val * 5.0f;
        }
        // BSYNC level 2
    } else {
        if (i < 24) {
            val = val * 6.0f;
        } else {
            val = val * 7.0f;
        }
    }
    // BSYNC level 1

    out[i] = val;
}

// Loop with break: generates BREAK instruction
extern "C" __global__ void __launch_bounds__(32)
probe_loop_break(float *out, const float *in, float threshold) {
    int i = threadIdx.x;
    float val = in[i];
    float acc = 0.0f;

    // Loop with per-thread early exit (BREAK)
    for (int j = 0; j < 100; j++) {
        acc += val;
        if (acc > threshold) break;  // BREAK instruction
        val *= 0.99f;
    }

    out[i] = acc;
}

// Switch statement: generates BMOV for barrier token management
extern "C" __global__ void __launch_bounds__(32)
probe_switch_bmov(float *out, const float *in, const int *selector) {
    int i = threadIdx.x;
    float val = in[i];
    int sel = selector[i] % 4;

    // Switch generates BMOV to manage multiple reconvergence points
    switch (sel) {
    case 0: val = val * 2.0f; break;
    case 1: val = val + 1.0f; break;
    case 2: val = val * val;  break;
    case 3: val = -val;       break;
    }

    out[i] = val;
}

// Multi-level break: nested loops with independent exits
extern "C" __global__ void __launch_bounds__(32)
probe_multi_break(float *out, const float *in) {
    int i = threadIdx.x;
    float val = in[i];
    float acc = 0.0f;
    int found = 0;

    // Outer loop
    for (int j = 0; j < 10 && !found; j++) {
        // Inner loop with break
        for (int k = 0; k < 10; k++) {
            acc += val * (float)(j * 10 + k);
            if (acc > 100.0f) {
                found = 1;
                break;  // BREAK inner
            }
        }
        val *= 0.9f;
    }
    // BREAK outer (via found flag)

    out[i] = acc;
}

// Early return: thread exits before kernel end
extern "C" __global__ void __launch_bounds__(32)
probe_early_return(float *out, const float *in, int cutoff) {
    int i = threadIdx.x;
    float val = in[i];

    // Early EXIT for some threads
    if (val < 0.0f) {
        out[i] = 0.0f;
        return;  // EXIT (thread terminates early)
    }

    // Remaining threads do more work
    float result = val;
    for (int j = 0; j < 10; j++)
        result = result * 0.9f + 0.1f;

    out[i] = result;
}
