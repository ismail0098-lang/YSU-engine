
// Cross-lane data exchange patterns (beyond basic shuffle)
extern "C" __global__ void __launch_bounds__(32)
dm_cross_lane(float *out, const float *in) {
    int tid=threadIdx.x;
    float val=in[tid];
    // Rotate left by 1
    float rot1=__shfl_sync(0xFFFFFFFF,val,(tid+1)%32);
    // Rotate right by 1
    float rot_r1=__shfl_sync(0xFFFFFFFF,val,(tid+31)%32);
    // Reverse order
    float rev=__shfl_sync(0xFFFFFFFF,val,31-tid);
    // Interleave even/odd
    float partner=__shfl_xor_sync(0xFFFFFFFF,val,1);
    // Broadcast from lane 0 to all
    float bc=__shfl_sync(0xFFFFFFFF,val,0);
    // Shift by warp-dynamic amount
    int shift_amt=tid%8;
    float dyn=__shfl_sync(0xFFFFFFFF,val,(tid+shift_amt)%32);
    out[tid]=rot1+rot_r1+rev+partner+bc+dyn;
}
