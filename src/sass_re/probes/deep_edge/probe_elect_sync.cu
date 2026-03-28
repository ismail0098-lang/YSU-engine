
#include <cooperative_groups.h>

// ELECT.SYNC: elect a single lane from active mask
extern "C" __global__ void __launch_bounds__(32)
elect_sync(int *out, const int *in) {
    int tid=threadIdx.x;
    int val=in[tid];
    // Only lanes with val>0 participate
    if(val>0){
        auto active=cooperative_groups::coalesced_threads();
        // The "elected" lane is thread_rank()==0
        if(active.thread_rank()==0){
            out[blockIdx.x]=val; // Only elected thread writes
        }
    }
}
