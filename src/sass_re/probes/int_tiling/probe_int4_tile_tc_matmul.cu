
#include <mma.h>
using namespace nvcuda;
// INT4 tensor core tiled matmul (8x8x32)
extern "C" __global__ void __launch_bounds__(32)
int4_tile_tc(int *D, const void *A, const void *B, int K_tiles) {
    using namespace nvcuda::wmma::experimental;
    wmma::fragment<wmma::matrix_a,8,8,32,precision::s4,wmma::row_major> fA;
    wmma::fragment<wmma::matrix_b,8,8,32,precision::s4,wmma::col_major> fB;
    wmma::fragment<wmma::accumulator,8,8,32,int> fC;
    wmma::fill_fragment(fC,0);
    for(int k=0;k<K_tiles;k++){
        wmma::load_matrix_sync(fA,(const char*)A+k*128,32);
        wmma::load_matrix_sync(fB,(const char*)B+k*128,32);
        wmma::mma_sync(fC,fA,fB,fC);
    }
    wmma::store_matrix_sync(D,fC,8,wmma::mem_row_major);
}
