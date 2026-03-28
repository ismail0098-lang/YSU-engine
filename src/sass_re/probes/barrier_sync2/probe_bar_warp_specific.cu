
// Warp-specific barrier patterns: different masks, partial warps
extern "C" __global__ void __launch_bounds__(128)
bar_warp_quad_sync(float *out, const float *in) {
    int tid=threadIdx.x, lane=tid&31;
    float val=in[tid+blockIdx.x*128];
    // Quad sync: 4-lane groups
    unsigned quad_mask=0xFu<<((lane/4)*4);
    __syncwarp(quad_mask);
    float q=__shfl_sync(quad_mask,val,(lane&~3)); // Broadcast from quad leader
    // Octet sync
    unsigned oct_mask=0xFFu<<((lane/8)*8);
    __syncwarp(oct_mask);
    float o=__shfl_sync(oct_mask,val,(lane&~7));
    out[tid+blockIdx.x*128]=q+o;
}
