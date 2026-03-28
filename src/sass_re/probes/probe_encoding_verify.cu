/*
 * SASS RE Probe: Instruction Encoding Bit Field Verification
 * Isolates: Systematic register/immediate/modifier encoding patterns
 *
 * Strategy: emit identical instructions with only one operand changed,
 * then diff the SASS binary encoding to pin down exact bit fields.
 *
 * Test matrix:
 *   1. FADD with R0-R15 as destination -> identifies dest register bits
 *   2. FADD with R0-R15 as source A -> identifies src A register bits
 *   3. FADD with R0-R15 as source B -> identifies src B register bits
 *   4. IADD3 with various immediates -> identifies immediate field
 *   5. LOP3 with different LUT values -> identifies LUT constant field
 *   6. Predicated instructions -> identifies predicate guard field
 *   7. FFMA with different rounding modes -> identifies modifier bits
 *
 * The __asm volatile blocks prevent instruction reordering/elimination.
 * Each kernel function tests ONE dimension of variation.
 *
 * After disassembly, use encoding_analysis.py to XOR-diff pairs and
 * extract the varying bit positions.
 */

/* ── Destination register sweep (R0-R15) ──────────── */
/* FADD Rdst, R16, R17 -- only Rdst varies */
extern "C" __global__ void __launch_bounds__(32)
probe_enc_dest_regs(float *out) {
    float a, b;
    asm volatile("mov.f32 %0, 1.0;" : "=f"(a));
    asm volatile("mov.f32 %0, 2.0;" : "=f"(b));

    float r0, r1, r2, r3, r4, r5, r6, r7;
    float r8, r9, r10, r11, r12, r13, r14, r15;

    asm volatile("add.f32 %0, %1, %2;" : "=f"(r0)  : "f"(a), "f"(b));
    asm volatile("add.f32 %0, %1, %2;" : "=f"(r1)  : "f"(a), "f"(b));
    asm volatile("add.f32 %0, %1, %2;" : "=f"(r2)  : "f"(a), "f"(b));
    asm volatile("add.f32 %0, %1, %2;" : "=f"(r3)  : "f"(a), "f"(b));
    asm volatile("add.f32 %0, %1, %2;" : "=f"(r4)  : "f"(a), "f"(b));
    asm volatile("add.f32 %0, %1, %2;" : "=f"(r5)  : "f"(a), "f"(b));
    asm volatile("add.f32 %0, %1, %2;" : "=f"(r6)  : "f"(a), "f"(b));
    asm volatile("add.f32 %0, %1, %2;" : "=f"(r7)  : "f"(a), "f"(b));
    asm volatile("add.f32 %0, %1, %2;" : "=f"(r8)  : "f"(a), "f"(b));
    asm volatile("add.f32 %0, %1, %2;" : "=f"(r9)  : "f"(a), "f"(b));
    asm volatile("add.f32 %0, %1, %2;" : "=f"(r10) : "f"(a), "f"(b));
    asm volatile("add.f32 %0, %1, %2;" : "=f"(r11) : "f"(a), "f"(b));
    asm volatile("add.f32 %0, %1, %2;" : "=f"(r12) : "f"(a), "f"(b));
    asm volatile("add.f32 %0, %1, %2;" : "=f"(r13) : "f"(a), "f"(b));
    asm volatile("add.f32 %0, %1, %2;" : "=f"(r14) : "f"(a), "f"(b));
    asm volatile("add.f32 %0, %1, %2;" : "=f"(r15) : "f"(a), "f"(b));

    // Store all to prevent DCE (compiler must keep all register assignments)
    out[0]  = r0;  out[1]  = r1;  out[2]  = r2;  out[3]  = r3;
    out[4]  = r4;  out[5]  = r5;  out[6]  = r6;  out[7]  = r7;
    out[8]  = r8;  out[9]  = r9;  out[10] = r10; out[11] = r11;
    out[12] = r12; out[13] = r13; out[14] = r14; out[15] = r15;
}

/* ── LOP3 LUT value sweep (0x00-0xFF) ────────────── */
/* LOP3.LUT R0, R1, R2, R3, <LUT> -- only LUT varies */
extern "C" __global__ void __launch_bounds__(32)
probe_enc_lop3_lut(unsigned *out, const unsigned *in) {
    unsigned a = in[0], b = in[1], c = in[2];
    unsigned r_and, r_or, r_xor, r_nand, r_xnor, r_oab, r_mux, r_maj;

    // LUT 0x80 = AND (a & b & c)
    asm volatile("lop3.b32 %0, %1, %2, %3, 0x80;" : "=r"(r_and) : "r"(a), "r"(b), "r"(c));
    // LUT 0xFE = OR (a | b | c)
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xFE;" : "=r"(r_or) : "r"(a), "r"(b), "r"(c));
    // LUT 0x96 = XOR (a ^ b ^ c)
    asm volatile("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(r_xor) : "r"(a), "r"(b), "r"(c));
    // LUT 0x7F = NAND ~(a & b & c)
    asm volatile("lop3.b32 %0, %1, %2, %3, 0x7F;" : "=r"(r_nand) : "r"(a), "r"(b), "r"(c));
    // LUT 0x69 = XNOR ~(a ^ b ^ c)
    asm volatile("lop3.b32 %0, %1, %2, %3, 0x69;" : "=r"(r_xnor) : "r"(a), "r"(b), "r"(c));
    // LUT 0xCA = MUX (c ? a : b) -- used in hash grid spatial hash
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xCA;" : "=r"(r_mux) : "r"(a), "r"(b), "r"(c));
    // LUT 0xE8 = MAJ (majority vote)
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xE8;" : "=r"(r_maj) : "r"(a), "r"(b), "r"(c));
    // LUT 0x00 = zero
    asm volatile("lop3.b32 %0, %1, %2, %3, 0x00;" : "=r"(r_oab) : "r"(a), "r"(b), "r"(c));

    out[0] = r_and; out[1] = r_or; out[2] = r_xor; out[3] = r_nand;
    out[4] = r_xnor; out[5] = r_oab; out[6] = r_mux; out[7] = r_maj;
}

/* ── Predicate guard sweep (via C conditionals -> FSETP + @Pn FADD) ── */
extern "C" __global__ void __launch_bounds__(32)
probe_enc_predicates(float *out, const float *in) {
    float a = in[threadIdx.x];
    float r = 0.0f;

    // Each condition generates FSETP + predicated FADD in SASS
    // The compiler assigns P0-P6 sequentially; look for @P0, @P1, etc.
    if (a > 0.0f) r += 1.0f;    // @P0
    if (a > 1.0f) r += 2.0f;    // @P1
    if (a > 2.0f) r += 4.0f;    // @P2
    if (a > 3.0f) r += 8.0f;    // @P3
    if (a > 4.0f) r += 16.0f;   // @P4
    if (a > 5.0f) r += 32.0f;   // @P5
    if (a <= 0.0f) r += 64.0f;  // @!P0 (negated first predicate)

    out[threadIdx.x] = r;
}

/* ── FFMA rounding mode sweep ─────────────────────── */
extern "C" __global__ void __launch_bounds__(32)
probe_enc_rounding(float *out, const float *in) {
    float a = in[0], b = in[1], c = in[2];
    float rn, rz, rp, rm;

    // Default: round to nearest even (.rn)
    asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rn) : "f"(a), "f"(b), "f"(c));
    // Round toward zero (.rz)
    asm volatile("fma.rz.f32 %0, %1, %2, %3;" : "=f"(rz) : "f"(a), "f"(b), "f"(c));
    // Round toward +inf (.rp)
    asm volatile("fma.rp.f32 %0, %1, %2, %3;" : "=f"(rp) : "f"(a), "f"(b), "f"(c));
    // Round toward -inf (.rm)
    asm volatile("fma.rm.f32 %0, %1, %2, %3;" : "=f"(rm) : "f"(a), "f"(b), "f"(c));

    out[0] = rn; out[1] = rz; out[2] = rp; out[3] = rm;
}

/* ── Immediate width sweep ────────────────────────── */
extern "C" __global__ void __launch_bounds__(32)
probe_enc_immediates(int *out, const int *in) {
    int base = in[threadIdx.x];
    int r;

    // Various immediate sizes to identify field width
    asm volatile("add.s32 %0, %1, 0;"    : "=r"(r) : "r"(base));   out[0]  = r;
    asm volatile("add.s32 %0, %1, 1;"    : "=r"(r) : "r"(base));   out[1]  = r;
    asm volatile("add.s32 %0, %1, 127;"  : "=r"(r) : "r"(base));   out[2]  = r;
    asm volatile("add.s32 %0, %1, 128;"  : "=r"(r) : "r"(base));   out[3]  = r;
    asm volatile("add.s32 %0, %1, 255;"  : "=r"(r) : "r"(base));   out[4]  = r;
    asm volatile("add.s32 %0, %1, 256;"  : "=r"(r) : "r"(base));   out[5]  = r;
    asm volatile("add.s32 %0, %1, 65535;": "=r"(r) : "r"(base));   out[6]  = r;
    asm volatile("add.s32 %0, %1, 65536;": "=r"(r) : "r"(base));   out[7]  = r;
}
