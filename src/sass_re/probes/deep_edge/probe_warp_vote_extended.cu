
// Extended warp vote patterns: MATCH.ANY + VOTE.BALLOT + __activemask
extern "C" __global__ void __launch_bounds__(32)
vote_activemask_diverge(unsigned *out, const float *in) {
    int tid=threadIdx.x;
    float v=in[tid];
    // Full mask
    unsigned full=__activemask();
    out[tid]=full;
    // After divergence
    if(v>0.5f){
        unsigned partial=__activemask(); // Should show fewer lanes
        out[32+tid]=partial;
    }
    // match_any: which lanes have the same value?
    int bucket=(int)(v*4.0f); // 0-3 buckets
    unsigned match=__match_any_sync(0xFFFFFFFF,bucket);
    out[64+tid]=match;
}
