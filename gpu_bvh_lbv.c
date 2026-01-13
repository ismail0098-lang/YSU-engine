// gpu_bvh_lbv.c
#include "gpu_bvh_lbv.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    uint32_t key;   // 30-bit morton
    uint32_t id;    // triangle id
    float c[3];     // centroid
    float bmin[3];
    float bmax[3];
} Prim;

static inline float fmin3(float a,float b,float c){ return fminf(a,fminf(b,c)); }
static inline float fmax3(float a,float b,float c){ return fmaxf(a,fmaxf(b,c)); }

static inline void bbox_init(float mn[3], float mx[3]){
    mn[0]=mn[1]=mn[2]= +INFINITY;
    mx[0]=mx[1]=mx[2]= -INFINITY;
}
static inline void bbox_expand(float mn[3], float mx[3], const float p[3]){
    mn[0]=fminf(mn[0],p[0]); mn[1]=fminf(mn[1],p[1]); mn[2]=fminf(mn[2],p[2]);
    mx[0]=fmaxf(mx[0],p[0]); mx[1]=fmaxf(mx[1],p[1]); mx[2]=fmaxf(mx[2],p[2]);
}

// bit interleave (10-bit -> 30-bit spread)
static inline uint32_t expand_bits_10(uint32_t v){
    v &= 1023u;
    v = (v | (v << 16)) & 0x030000FFu;
    v = (v | (v <<  8)) & 0x0300F00Fu;
    v = (v | (v <<  4)) & 0x030C30C3u;
    v = (v | (v <<  2)) & 0x09249249u;
    return v;
}

static inline uint32_t morton3(float x, float y, float z){
    // clamp [0,1)
    x = fminf(fmaxf(x, 0.0f), 0.999999f);
    y = fminf(fmaxf(y, 0.0f), 0.999999f);
    z = fminf(fmaxf(z, 0.0f), 0.999999f);
    uint32_t xx = (uint32_t)(x * 1024.0f);
    uint32_t yy = (uint32_t)(y * 1024.0f);
    uint32_t zz = (uint32_t)(z * 1024.0f);
    uint32_t xb = expand_bits_10(xx);
    uint32_t yb = expand_bits_10(yy);
    uint32_t zb = expand_bits_10(zz);
    return (xb << 0) | (yb << 1) | (zb << 2);
}

static inline int clz32_u(uint32_t x){
#if defined(__GNUC__) || defined(__clang__)
    return x ? __builtin_clz(x) : 32;
#else
    // portable fallback
    int n=0;
    while(n<32 && (x & 0x80000000u)==0){ x<<=1; n++; }
    return n;
#endif
}

// Karras common prefix
static inline int common_prefix(const Prim* p, int i, int j, int n){
    if(j < 0 || j >= n) return -1;
    uint32_t ki = p[i].key;
    uint32_t kj = p[j].key;
    if(ki == kj){
        // tie-breaker by id
        uint32_t di = p[i].id;
        uint32_t dj = p[j].id;
        uint32_t x = di ^ dj;
        return 32 + (x ? clz32_u(x) : 32);
    }
    return clz32_u(ki ^ kj);
}

static void determine_range(const Prim* p, int i, int n, int* out_first, int* out_last){
    int cpL = common_prefix(p, i, i-1, n);
    int cpR = common_prefix(p, i, i+1, n);
    int d = (cpR > cpL) ? 1 : -1;
    int cpMin = common_prefix(p, i, i - d, n);

    int lmax = 2;
    while(common_prefix(p, i, i + lmax*d, n) > cpMin) lmax <<= 1;

    int l = 0;
    for(int t = lmax >> 1; t > 0; t >>= 1){
        if(common_prefix(p, i, i + (l + t)*d, n) > cpMin) l += t;
    }

    int j = i + l*d;
    int first = i;
    int last  = j;
    if(first > last){ int tmp=first; first=last; last=tmp; }
    *out_first = first;
    *out_last  = last;
}

static int find_split(const Prim* p, int first, int last, int n){
    if(first == last) return first;
    int cp = common_prefix(p, first, last, n);
    int split = first;
    int step = last - first;
    do {
        step = (step + 1) >> 1;
        int mid = split + step;
        if(mid < last){
            int cpm = common_prefix(p, first, mid, n);
            if(cpm > cp) split = mid;
        }
    } while(step > 1);
    return split;
}

// bottom-up bbox compute (post-order) - iterative, stack allocated
static void compute_internal_bboxes(GPUBVHNode* nodes, uint32_t root){
    typedef struct { uint32_t n; uint8_t state; } Stack;
    Stack st[256];
    int sp=0;
    st[sp++] = (Stack){root, 0};

    while(sp){
        Stack* top = &st[sp-1];
        uint32_t ni = top->n;
        GPUBVHNode* nd = &nodes[ni];

        if(nd->left < 0 && nd->right < 0){
            sp--;
            continue;
        }

        if(top->state == 0){
            top->state = 1;
            st[sp++] = (Stack){(uint32_t)nd->left, 0};
            continue;
        }
        if(top->state == 1){
            top->state = 2;
            st[sp++] = (Stack){(uint32_t)nd->right, 0};
            continue;
        }

        GPUBVHNode* L = &nodes[(uint32_t)nd->left];
        GPUBVHNode* R = &nodes[(uint32_t)nd->right];

        nd->bmin[0]=fminf(L->bmin[0], R->bmin[0]);
        nd->bmin[1]=fminf(L->bmin[1], R->bmin[1]);
        nd->bmin[2]=fminf(L->bmin[2], R->bmin[2]);
        nd->bmin[3]=0.0f;

        nd->bmax[0]=fmaxf(L->bmax[0], R->bmax[0]);
        nd->bmax[1]=fmaxf(L->bmax[1], R->bmax[1]);
        nd->bmax[2]=fmaxf(L->bmax[2], R->bmax[2]);
        nd->bmax[3]=0.0f;

        sp--;
    }
}

bool gpu_build_bvh_from_tri_vec4_lbv(
    const float* tri_vec4,
    uint32_t tri_count,
    GPUBVHNode** out_nodes,
    uint32_t* out_node_count,
    int32_t** out_indices,
    uint32_t* out_index_count
){
    if(!tri_vec4 || tri_count == 0 || !out_nodes || !out_node_count || !out_indices || !out_index_count)
        return false;

    Prim* prim = (Prim*)calloc(tri_count, sizeof(Prim));
    Prim* tmp  = (Prim*)calloc(tri_count, sizeof(Prim));
    if(!prim || !tmp){ free(prim); free(tmp); return false; }

    float cmin[3], cmax[3]; bbox_init(cmin,cmax);

    for(uint32_t i=0;i<tri_count;i++){
        const float* t = tri_vec4 + (size_t)i * 12; // 3*vec4
        float a[3] = {t[0], t[1], t[2]};
        float b[3] = {t[4], t[5], t[6]};
        float c[3] = {t[8], t[9], t[10]};

        float mn[3] = { fmin3(a[0],b[0],c[0]), fmin3(a[1],b[1],c[1]), fmin3(a[2],b[2],c[2]) };
        float mx[3] = { fmax3(a[0],b[0],c[0]), fmax3(a[1],b[1],c[1]), fmax3(a[2],b[2],c[2]) };

        float cen[3] = { (a[0]+b[0]+c[0])/3.0f, (a[1]+b[1]+c[1])/3.0f, (a[2]+b[2]+c[2])/3.0f };

        bbox_expand(cmin,cmax,cen);

        prim[i].id = i;
        prim[i].c[0]=cen[0]; prim[i].c[1]=cen[1]; prim[i].c[2]=cen[2];
        prim[i].bmin[0]=mn[0]; prim[i].bmin[1]=mn[1]; prim[i].bmin[2]=mn[2];
        prim[i].bmax[0]=mx[0]; prim[i].bmax[1]=mx[1]; prim[i].bmax[2]=mx[2];
    }

    float extent[3] = { cmax[0]-cmin[0], cmax[1]-cmin[1], cmax[2]-cmin[2] };
    if(extent[0] < 1e-20f) extent[0]=1.0f;
    if(extent[1] < 1e-20f) extent[1]=1.0f;
    if(extent[2] < 1e-20f) extent[2]=1.0f;

    for(uint32_t i=0;i<tri_count;i++){
        float nx = (prim[i].c[0]-cmin[0]) / extent[0];
        float ny = (prim[i].c[1]-cmin[1]) / extent[1];
        float nz = (prim[i].c[2]-cmin[2]) / extent[2];
        prim[i].key = morton3(nx,ny,nz);
    }

    // radix sort 30-bit, 6 passes of 5 bits
    Prim* a=prim;
    Prim* b=tmp;
    const uint32_t RAD=32u;
    for(uint32_t pass=0; pass<6; pass++){
        uint32_t shift = pass*5u;
        uint32_t count[RAD]; memset(count,0,sizeof(count));
        for(uint32_t i=0;i<tri_count;i++){
            uint32_t bin = (a[i].key >> shift) & (RAD-1u);
            count[bin]++;
        }
        uint32_t sum=0;
        for(uint32_t bin=0;bin<RAD;bin++){
            uint32_t c = count[bin];
            count[bin]=sum;
            sum += c;
        }
        for(uint32_t i=0;i<tri_count;i++){
            uint32_t bin = (a[i].key >> shift) & (RAD-1u);
            b[count[bin]++] = a[i];
        }
        Prim* sw=a; a=b; b=sw;
    }
    if(a != prim) memcpy(prim, a, sizeof(Prim)*tri_count);

    uint32_t n = tri_count;
    uint32_t leafBase = (n > 1) ? (n - 1) : 0;
    uint32_t nodeCount = (n > 1) ? (2*n - 1) : 1;

    GPUBVHNode* nodes = (GPUBVHNode*)calloc(nodeCount, sizeof(GPUBVHNode));
    int32_t* indices  = (int32_t*)malloc(sizeof(int32_t)*n);
    if(!nodes || !indices){
        free(nodes); free(indices); free(prim); free(tmp);
        return false;
    }

    // Leaves: one triangle per leaf, in morton order
    for(uint32_t i=0;i<n;i++){
        indices[i] = (int32_t)prim[i].id;

        GPUBVHNode* L = &nodes[leafBase + i];
        L->bmin[0]=prim[i].bmin[0]; L->bmin[1]=prim[i].bmin[1]; L->bmin[2]=prim[i].bmin[2]; L->bmin[3]=0.0f;
        L->bmax[0]=prim[i].bmax[0]; L->bmax[1]=prim[i].bmax[1]; L->bmax[2]=prim[i].bmax[2]; L->bmax[3]=0.0f;
        L->left = -1; L->right = -1;
        L->triOffset = (int32_t)i;
        L->triCount  = 1;
    }

    if(n == 1){
        *out_nodes = nodes;
        *out_node_count = 1;
        *out_indices = indices;
        *out_index_count = 1;
        free(prim); free(tmp);
        return true;
    }

    // Internal nodes 0..n-2
    for(uint32_t i=0;i<n-1;i++){
        int first=0,last=0;
        determine_range(prim, (int)i, (int)n, &first, &last);
        int split = find_split(prim, first, last, (int)n);

        int leftIndex  = (split == first) ? (int)(leafBase + (uint32_t)split) : split;
        int rightIndex = (split + 1 == last) ? (int)(leafBase + (uint32_t)(split + 1)) : (split + 1);

        GPUBVHNode* N = &nodes[i];
        N->left = leftIndex;
        N->right = rightIndex;
        N->triOffset = 0;
        N->triCount = 0;
        N->bmin[0]=N->bmin[1]=N->bmin[2]=+INFINITY; N->bmin[3]=0.0f;
        N->bmax[0]=N->bmax[1]=N->bmax[2]=-INFINITY; N->bmax[3]=0.0f;
    }

    compute_internal_bboxes(nodes, 0);

    *out_nodes = nodes;
    *out_node_count = nodeCount;
    *out_indices = indices;
    *out_index_count = n;

    free(prim);
    free(tmp);
    return true;
}
