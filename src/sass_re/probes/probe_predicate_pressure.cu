/*
 * SASS RE Probe: Predicate Register Pressure
 * Isolates: What happens when code needs >6 simultaneous predicates
 *
 * Ada Lovelace SM 8.9 has 8 predicate registers per thread: P0-P7.
 * P7 is typically reserved as PT (always-true). Practical limit: 7 usable.
 *
 * When a kernel needs more simultaneous predicate values than registers
 * available, the compiler must spill predicates to GPRs (via ISETP -> GPR
 * + SEL or P2R/R2P instructions).
 *
 * This probe creates kernels with 2, 4, 6, and 8+ simultaneous predicate
 * conditions to observe:
 *   - ISETP/FSETP for predicate generation
 *   - @P0/@!P0 predicated execution
 *   - P2R (predicate to register) / R2P (register to predicate) for spills
 *   - SEL (select based on predicate) as spill workaround
 *
 * Key SASS to look for:
 *   ISETP   -- integer set predicate
 *   FSETP   -- float set predicate
 *   PLOP3   -- predicate logic (3-input predicate boolean)
 *   P2R     -- pack predicates into GPR (spill path)
 *   R2P     -- unpack GPR into predicates (reload path)
 *   SEL     -- conditional select (alternative to predicated exec)
 */

// 2 predicates: well within budget
extern "C" __global__ void __launch_bounds__(32)
probe_pred_2(float *out, const float *a, const float *b) {
    int i = threadIdx.x;
    float va = a[i], vb = b[i];

    float result = 0.0f;
    if (va > 0.0f)          // P0
        result += va;
    if (vb > 0.0f)          // P1
        result += vb;

    out[i] = result;
}

// 4 predicates: still within budget
extern "C" __global__ void __launch_bounds__(32)
probe_pred_4(float *out, const float *a, const float *b,
             const float *c, const float *d) {
    int i = threadIdx.x;
    float va = a[i], vb = b[i], vc = c[i], vd = d[i];

    float result = 0.0f;
    if (va > 0.0f) result += va;     // P0
    if (vb > 0.0f) result += vb;     // P1
    if (vc > 0.0f) result += vc;     // P2
    if (vd > 0.0f) result += vd;     // P3

    out[i] = result;
}

// 6 predicates: approaching limit
extern "C" __global__ void __launch_bounds__(32)
probe_pred_6(float *out, const float *a, const float *b,
             const float *c, const float *d,
             const float *e, const float *f) {
    int i = threadIdx.x;
    float va = a[i], vb = b[i], vc = c[i];
    float vd = d[i], ve = e[i], vf = f[i];

    // Force all 6 predicates to be live simultaneously
    // by using them in a single expression chain
    float result = 0.0f;
    int p0 = (va > 0.0f);    // P0
    int p1 = (vb > 0.0f);    // P1
    int p2 = (vc > 0.5f);    // P2
    int p3 = (vd < 1.0f);    // P3
    int p4 = (ve > -1.0f);   // P4
    int p5 = (vf != 0.0f);   // P5

    // All predicates used together -- compiler must keep all live
    if (p0 && p1) result += va + vb;
    if (p2 && p3) result += vc + vd;
    if (p4 && p5) result += ve + vf;
    if (p0 && p2 && p4) result *= 2.0f;
    if (p1 && p3 && p5) result *= 0.5f;

    out[i] = result;
}

// 8+ predicates: should exceed budget, forcing P2R/R2P or SEL spills
extern "C" __global__ void __launch_bounds__(32)
probe_pred_8(float *out, const float *in, int n) {
    int i = threadIdx.x;
    float v[8];
    for (int k = 0; k < 8; k++)
        v[k] = in[k * 32 + i];

    // 8 independent predicates, all needed simultaneously
    int p0 = (v[0] > 0.0f);
    int p1 = (v[1] > 0.0f);
    int p2 = (v[2] > 0.5f);
    int p3 = (v[3] < 1.0f);
    int p4 = (v[4] > -1.0f);
    int p5 = (v[5] != 0.0f);
    int p6 = (v[6] >= 0.0f);
    int p7 = (v[7] <= 2.0f);

    // Force all 8 to be live by using complex predicate logic
    float result = 0.0f;
    if (p0) result += v[0];
    if (p1) result += v[1];
    if (p2) result += v[2];
    if (p3) result += v[3];
    if (p4) result += v[4];
    if (p5) result += v[5];
    if (p6) result += v[6];
    if (p7) result += v[7];

    // Cross-predicate combinations
    if (p0 && p4) result += v[0] * v[4];
    if (p1 && p5) result += v[1] * v[5];
    if (p2 && p6) result += v[2] * v[6];
    if (p3 && p7) result += v[3] * v[7];

    // Triple combinations
    if (p0 && p1 && p2) result *= 1.1f;
    if (p3 && p4 && p5) result *= 0.9f;
    if (p6 && p7 && p0) result += 1.0f;

    out[i] = result;
}

// 12+ predicates: definitely spilling
extern "C" __global__ void __launch_bounds__(32)
probe_pred_12(float *out, const float *in) {
    int i = threadIdx.x;
    float v[12];
    for (int k = 0; k < 12; k++)
        v[k] = in[k * 32 + i];

    // 12 independent predicates
    int p[12];
    for (int k = 0; k < 12; k++)
        p[k] = (v[k] > (float)k * 0.1f);

    // Use all in a reduction pattern that prevents reordering
    float result = 0.0f;
    for (int k = 0; k < 12; k++)
        if (p[k]) result += v[k];

    // Cross-combinations requiring multiple predicates live
    for (int k = 0; k < 6; k++)
        if (p[k] && p[k + 6]) result += v[k] * v[k + 6];

    out[i] = result;
}
