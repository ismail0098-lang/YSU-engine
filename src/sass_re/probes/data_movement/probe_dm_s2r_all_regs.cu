
// S2R: read ALL available special registers
extern "C" __global__ void __launch_bounds__(128)
dm_s2r_all(unsigned *out) {
    int tid=threadIdx.x;
    unsigned v;
    unsigned long long wide;
    asm volatile("mov.u32 %0, %%tid.x;":"=r"(v)); out[tid*16]=v;
    asm volatile("mov.u32 %0, %%tid.y;":"=r"(v)); out[tid*16+1]=v;
    asm volatile("mov.u32 %0, %%tid.z;":"=r"(v)); out[tid*16+2]=v;
    asm volatile("mov.u32 %0, %%ntid.x;":"=r"(v)); out[tid*16+3]=v;
    asm volatile("mov.u32 %0, %%ntid.y;":"=r"(v)); out[tid*16+4]=v;
    asm volatile("mov.u32 %0, %%ntid.z;":"=r"(v)); out[tid*16+5]=v;
    asm volatile("mov.u32 %0, %%ctaid.x;":"=r"(v)); out[tid*16+6]=v;
    asm volatile("mov.u32 %0, %%ctaid.y;":"=r"(v)); out[tid*16+7]=v;
    asm volatile("mov.u32 %0, %%ctaid.z;":"=r"(v)); out[tid*16+8]=v;
    asm volatile("mov.u32 %0, %%nctaid.x;":"=r"(v)); out[tid*16+9]=v;
    asm volatile("mov.u32 %0, %%laneid;":"=r"(v)); out[tid*16+10]=v;
    asm volatile("mov.u32 %0, %%warpid;":"=r"(v)); out[tid*16+11]=v;
    asm volatile("mov.u32 %0, %%smid;":"=r"(v)); out[tid*16+12]=v;
    asm volatile("mov.u32 %0, %%nsmid;":"=r"(v)); out[tid*16+13]=v;
    asm volatile("mov.u32 %0, %%clock;":"=r"(v)); out[tid*16+14]=v;
    asm volatile("mov.u64 %0, %%globaltimer;":"=l"(wide)); out[tid*16+15]=(unsigned)wide;
}
