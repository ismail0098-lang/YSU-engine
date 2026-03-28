
// Explicit branch via inline PTX (force specific branch types)
extern "C" __global__ void __launch_bounds__(128)
cf_ptx_branch(float *out, const float *in, int n) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i>=n)return;
    float v=in[i];
    const float thresh = 0.5f;
    const float mul2 = 2.0f;
    const float add1 = 1.0f;
    // Uniform branch via PTX (all threads same direction)
    asm volatile(
        "{\n\t"
        "  .reg .pred p;\n\t"
        "  setp.gt.f32 p, %1, %2;\n\t"
        "  @!p bra L_false_%=;\n\t"
        "  add.f32 %0, %1, %4;\n\t"
        "  bra L_end_%=;\n"
        "L_false_%=:\n\t"
        "  mul.f32 %0, %1, %3;\n"
        "L_end_%=:\n\t"
        "}\n"
        : "=f"(v) : "f"(v), "f"(thresh), "f"(mul2), "f"(add1));
    out[i]=v;
}
