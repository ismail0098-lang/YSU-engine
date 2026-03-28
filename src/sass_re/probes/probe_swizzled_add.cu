/*
 * SASS RE Probe: FSWZADD -- Swizzled FP32 Add
 * Isolates: Warp-level butterfly add via hardware swizzle (not SHFL)
 *
 * FSWZADD performs an FP32 add where one operand comes from a different
 * lane in the warp, selected by a swizzle pattern. This is a potential
 * alternative to SHFL+FADD for warp-level butterfly reductions.
 *
 * If FSWZADD exists on Ada and is faster than SHFL.BFLY+FADD (~30 cy),
 * it could be a REDUX.FADD equivalent for float reductions.
 *
 * The instruction is triggered by specific warp-level reduction patterns
 * that the compiler recognizes. We use inline PTX to force emission.
 *
 * Key SASS:
 *   FSWZADD   -- swizzled FP32 add (single instruction butterfly)
 *
 * Note: FSWZADD may not exist on Ada (it was observed on Volta/Turing).
 * If the compiler emits SHFL+FADD instead, that itself is a finding.
 */

// Attempt to trigger FSWZADD via warp shuffle reduction pattern
extern "C" __global__ void __launch_bounds__(32)
probe_fswzadd_butterfly(float *out, const float *in) {
    int i = threadIdx.x;
    float val = in[i];

    // Classic butterfly reduction that may compile to FSWZADD on some archs
    // On Ada, this likely compiles to SHFL.BFLY + FADD pairs
    val += __shfl_xor_sync(0xFFFFFFFF, val, 16);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 8);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 4);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 2);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 1);

    out[i] = val;
}

// Explicit PTX for swizzled add (may or may not be supported)
// If not supported, ptxas will decompose to SHFL+FADD
extern "C" __global__ void __launch_bounds__(32)
probe_fswzadd_explicit(float *out, const float *in) {
    int i = threadIdx.x;
    float val = in[i];

    // Try to emit FSWZADD via shfl.sync + add in one PTX statement
    // The actual instruction selection depends on ptxas
    asm volatile(
        "{"
        ".reg .f32 tmp;"
        "shfl.sync.bfly.b32 tmp, %1, 16, 0x1f, 0xffffffff;"
        "add.f32 %0, %1, tmp;"
        "}"
        : "=f"(val) : "f"(val)
    );
    asm volatile(
        "{"
        ".reg .f32 tmp;"
        "shfl.sync.bfly.b32 tmp, %1, 8, 0x1f, 0xffffffff;"
        "add.f32 %0, %1, tmp;"
        "}"
        : "=f"(val) : "f"(val)
    );
    asm volatile(
        "{"
        ".reg .f32 tmp;"
        "shfl.sync.bfly.b32 tmp, %1, 4, 0x1f, 0xffffffff;"
        "add.f32 %0, %1, tmp;"
        "}"
        : "=f"(val) : "f"(val)
    );
    asm volatile(
        "{"
        ".reg .f32 tmp;"
        "shfl.sync.bfly.b32 tmp, %1, 2, 0x1f, 0xffffffff;"
        "add.f32 %0, %1, tmp;"
        "}"
        : "=f"(val) : "f"(val)
    );
    asm volatile(
        "{"
        ".reg .f32 tmp;"
        "shfl.sync.bfly.b32 tmp, %1, 1, 0x1f, 0xffffffff;"
        "add.f32 %0, %1, tmp;"
        "}"
        : "=f"(val) : "f"(val)
    );

    out[i] = val;
}

// Comparison: SHFL-based float reduction vs integer REDUX.SUM
// Measures the gap that FSWZADD (if it existed) would close
extern "C" __global__ void __launch_bounds__(32)
probe_float_vs_int_reduce(float *fout, int *iout, const float *fin) {
    int i = threadIdx.x;
    float fval = fin[i];
    int ival = (int)(fval * 1000.0f);

    // Float reduction: must use SHFL+FADD (no REDUX.FADD)
    fval += __shfl_xor_sync(0xFFFFFFFF, fval, 16);
    fval += __shfl_xor_sync(0xFFFFFFFF, fval, 8);
    fval += __shfl_xor_sync(0xFFFFFFFF, fval, 4);
    fval += __shfl_xor_sync(0xFFFFFFFF, fval, 2);
    fval += __shfl_xor_sync(0xFFFFFFFF, fval, 1);

    // Integer reduction: uses REDUX.SUM (single instruction)
    ival = __reduce_add_sync(0xFFFFFFFF, ival);

    fout[i] = fval;
    iout[i] = ival;
}
