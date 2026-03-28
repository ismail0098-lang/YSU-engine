
// MOV instruction variants: immediate, register, predicated
extern "C" __global__ void __launch_bounds__(32)
dm_mov_imm(unsigned *out) {
    int tid=threadIdx.x;
    unsigned r;
    // Small immediate (fits in instruction encoding)
    asm volatile("mov.u32 %0, 42;":"=r"(r)); out[tid*4]=r;
    // Large immediate (32-bit, requires MOV32I?)
    asm volatile("mov.u32 %0, 0xDEADBEEF;":"=r"(r)); out[tid*4+1]=r;
    // Predicated MOV
    asm volatile("{.reg .pred p; setp.gt.u32 p, %1, 16; @p mov.u32 %0, 999; @!p mov.u32 %0, 0;}"
        :"=r"(r):"r"((unsigned)tid)); out[tid*4+2]=r;
    // Double MOV (64-bit via register pair)
    unsigned long long wide;
    unsigned lo, hi;
    asm volatile("mov.u64 %0, 0xCAFEBABEDEADBEEF;" : "=l"(wide));
    lo = (unsigned)(wide & 0xffffffffull);
    hi = (unsigned)(wide >> 32);
    out[tid*4+3]=lo^hi;
}
