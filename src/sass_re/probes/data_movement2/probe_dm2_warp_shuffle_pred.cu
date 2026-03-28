
// Predicated warp shuffle (shuffle only if predicate true)
extern "C" __global__ void __launch_bounds__(32)
dm2_shfl_pred(float *out, const float *in) {
    int tid=threadIdx.x;
    float val=in[tid];
    float result;
    // Shuffle with predicate output (tells if source lane was active)
    int pred;
    asm volatile(
        "{ .reg .pred p;\n\t"
        "  shfl.sync.bfly.b32 %0|p, %2, 1, 0x1f, 0xffffffff;\n\t"
        "  selp.u32 %1, 1, 0, p;\n\t"
        "}"
        : "=f"(result), "=r"(pred)
        : "f"(val));
    // This should generate SHFL.BFLY with predicate output
    out[tid]=result+(float)pred;
}
