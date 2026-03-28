/*
 * SASS RE Probe: Control Flow
 * Isolates: BRA, BSSY, BSYNC, WARPSYNC, EXIT, YIELD, CALL/RET, BREAK
 *
 * Ada Lovelace uses Independent Thread Scheduling (ITS).
 * Divergence is handled by BSSY (begin sync section) / BSYNC (barrier sync).
 * WARPSYNC replaces old __syncwarp.
 *
 * KEY INSIGHT: Look at the stall counts and yield hints in control words.
 * Each instruction has a control word that encodes:
 *   - Stall count (0-15): how many cycles to wait before issuing next instruction
 *   - Yield hint: whether to switch to another warp after this instruction
 *   - Read/write barriers: which scoreboard barriers to wait on / set
 *   - Predicate guard: which predicate register gates this instruction
 */

// Simple divergence: if/else -> BSSY/BRA/BSYNC
extern "C" __global__ void __launch_bounds__(32)
probe_divergence(float *out, const float *a) {
    int i = threadIdx.x;
    float x = a[i];
    float r;

    // This if/else should produce:
    // BSSY  (mark convergence point)
    // ISETP (compare threadIdx)
    // BRA   (conditional branch)
    // ... then-body ...
    // BRA   (unconditional to convergence)
    // ... else-body ...
    // BSYNC (reconverge)
    if (i < 16) {
        r = x * 2.0f;
    } else {
        r = x * 0.5f;
    }
    out[i] = r;
}

// Nested divergence
extern "C" __global__ void __launch_bounds__(32)
probe_nested_divergence(float *out, const float *a) {
    int i = threadIdx.x;
    float x = a[i];
    float r = 0.0f;

    // Two levels of BSSY/BSYNC nesting
    if (i < 16) {
        if (i < 8) {
            r = x * 4.0f;
        } else {
            r = x * 3.0f;
        }
    } else {
        if (i < 24) {
            r = x * 2.0f;
        } else {
            r = x * 1.0f;
        }
    }
    out[i] = r;
}

// Loop with early exit -- BRA + BREAK
extern "C" __global__ void __launch_bounds__(32)
probe_loop_break(int *out, const int *a) {
    int i = threadIdx.x;
    int x = a[i];
    int sum = 0;

    // The compiler may use BSSY/BSYNC or BRA for loop constructs.
    // The break inside a divergent loop is particularly interesting.
    for (int j = 0; j < 100; j++) {
        sum += x;
        if (sum > 1000) break;  // Early exit: divergent BREAK
        x = x ^ (x << 1);
    }
    out[i] = sum;
}

// WARPSYNC (cooperative warp synchronization)
extern "C" __global__ void __launch_bounds__(32)
probe_warpsync(int *out, const int *a) {
    int i = threadIdx.x;

    int x = a[i];

    // __syncwarp compiles to WARPSYNC instruction
    __syncwarp(0xFFFFFFFF);  // full mask

    // Partial mask sync
    if (i < 16) {
        x = x + 1;
        __syncwarp(0x0000FFFF);  // sync lower 16 threads
    } else {
        x = x - 1;
        __syncwarp(0xFFFF0000);  // sync upper 16 threads
    }

    __syncwarp(0xFFFFFFFF);
    out[i] = x;
}

// Warp shuffle -- SHFL instruction
extern "C" __global__ void __launch_bounds__(32)
probe_shfl(int *out, const int *a) {
    int i = threadIdx.x;
    int x = a[i];

    // SHFL.IDX: read from specific lane
    int from_lane0 = __shfl_sync(0xFFFFFFFF, x, 0);             // SHFL.IDX

    // SHFL.UP: shift up (receive from lane-1)
    int shifted_up = __shfl_up_sync(0xFFFFFFFF, x, 1);          // SHFL.UP

    // SHFL.DOWN: shift down (receive from lane+1)
    int shifted_down = __shfl_down_sync(0xFFFFFFFF, x, 1);      // SHFL.DOWN

    // SHFL.BFLY: butterfly shuffle (XOR lane)
    int butterfly = __shfl_xor_sync(0xFFFFFFFF, x, 1);          // SHFL.BFLY

    // Warp reduction pattern using butterfly shuffles
    int sum = x;
    sum += __shfl_xor_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_xor_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_xor_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_xor_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_xor_sync(0xFFFFFFFF, sum, 1);

    out[i] = from_lane0 + shifted_up + shifted_down + butterfly + sum;
}

// VOTE: warp vote instructions
extern "C" __global__ void __launch_bounds__(32)
probe_vote(int *out, const int *a) {
    int i = threadIdx.x;
    int x = a[i];

    // VOTE.ALL: true if predicate is true for all threads
    int all = __all_sync(0xFFFFFFFF, x > 0);       // VOTE.ALL

    // VOTE.ANY: true if any thread has predicate true
    int any = __any_sync(0xFFFFFFFF, x > 0);       // VOTE.ANY

    // VOTE.BALLOT: returns bitmask of which threads have predicate true
    unsigned ballot = __ballot_sync(0xFFFFFFFF, x > 0);  // VOTE.BALLOT

#if __CUDA_ARCH__ >= 700
    // MATCH: true if all active threads have the same value
    // (Volta+ feature — uses MATCH instruction)
    int match_pred = 0;
    unsigned match = __match_all_sync(0xFFFFFFFF, x, &match_pred);  // MATCH.ALL
#else
    unsigned match = 0;  // MATCH not available on SM < 7.0
#endif

    out[i] = all + any + (int)ballot + (int)match;
}

// Predicated execution (no branch, just predicate guard)
extern "C" __global__ void __launch_bounds__(32)
probe_predication(float *out, const float *a) {
    int i = threadIdx.x;
    float x = a[i];

    // The ternary operator with simple operations often compiles to
    // predicated instructions rather than branches:
    // @P0 FADD Rd, Ra, Rb
    // @!P0 FMUL Rd, Ra, Rb
    float r = (x > 0.0f) ? (x + 1.0f) : (x * 2.0f);
    float s = (x > r) ? (r + x) : (r * x);

    out[i] = r + s;
}
