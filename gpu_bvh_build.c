// gpu_bvh_build.c
#include "gpu_bvh_build.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct { float x,y,z; } v3;

static v3 v3_min(v3 a, v3 b){ v3 r={ fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z)}; return r; }
static v3 v3_max(v3 a, v3 b){ v3 r={ fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z)}; return r; }
static v3 v3_add(v3 a, v3 b){ v3 r={ a.x+b.x, a.y+b.y, a.z+b.z}; return r; }
static v3 v3_mul(v3 a, float s){ v3 r={ a.x*s, a.y*s, a.z*s}; return r; }

typedef struct {
    v3 bmin;
    v3 bmax;
    v3 centroid;
} TriInfo;

typedef struct {
    GPUBVHNode* nodes;
    uint32_t count;
    uint32_t cap;
} NodeVec;

static void nv_reserve(NodeVec* nv, uint32_t want){
    if(nv->cap >= want) return;
    uint32_t nc = nv->cap ? nv->cap : 1024;
    while(nc < want) nc *= 2;
    nv->nodes = (GPUBVHNode*)realloc(nv->nodes, (size_t)nc * sizeof(GPUBVHNode));
    nv->cap = nc;
}

static uint32_t nv_push(NodeVec* nv, const GPUBVHNode* n){
    nv_reserve(nv, nv->count + 1);
    nv->nodes[nv->count] = *n;
    return nv->count++;
}

static int g_sort_axis = 0;
static const TriInfo* g_tri = NULL;
static const int32_t* g_idx = NULL;

static int cmp_centroid(const void* a, const void* b){
    int32_t ia = *(const int32_t*)a;
    int32_t ib = *(const int32_t*)b;
    float ca, cb;
    if(g_sort_axis==0){ ca = g_tri[ia].centroid.x; cb = g_tri[ib].centroid.x; }
    else if(g_sort_axis==1){ ca = g_tri[ia].centroid.y; cb = g_tri[ib].centroid.y; }
    else { ca = g_tri[ia].centroid.z; cb = g_tri[ib].centroid.z; }
    return (ca < cb) ? -1 : (ca > cb) ? 1 : 0;
}

static void compute_bounds_range(
    const TriInfo* tri,
    const int32_t* idx,
    uint32_t start,
    uint32_t end,
    v3* out_min,
    v3* out_max
){
    v3 mn = (v3){ 1e30f, 1e30f, 1e30f };
    v3 mx = (v3){-1e30f,-1e30f,-1e30f };
    for(uint32_t i=start;i<end;i++){
        int32_t t = idx[i];
        mn = v3_min(mn, tri[t].bmin);
        mx = v3_max(mx, tri[t].bmax);
    }
    *out_min = mn;
    *out_max = mx;
}

static uint32_t build_node(
    NodeVec* nv,
    const TriInfo* tri,
    int32_t* idx,
    uint32_t start,
    uint32_t end
){
    const uint32_t LEAF_MAX = 8;
    uint32_t ntris = end - start;

    v3 mn, mx;
    compute_bounds_range(tri, idx, start, end, &mn, &mx);

    GPUBVHNode node;
    memset(&node, 0, sizeof(node));
    node.bmin[0]=mn.x; node.bmin[1]=mn.y; node.bmin[2]=mn.z; node.bmin[3]=0.0f;
    node.bmax[0]=mx.x; node.bmax[1]=mx.y; node.bmax[2]=mx.z; node.bmax[3]=0.0f;
    node.left = -1;
    node.right = -1;
    node.triOffset = (int32_t)start;
    node.triCount  = (int32_t)ntris;

    uint32_t my_index = nv_push(nv, &node);

    if(ntris <= LEAF_MAX){
        // leaf
        return my_index;
    }

    // split axis: longest extent
    v3 ext = (v3){ mx.x - mn.x, mx.y - mn.y, mx.z - mn.z };
    int axis = 0;
    if(ext.y > ext.x) axis = 1;
    if(ext.z > (axis==0 ? ext.x : ext.y)) axis = 2;

    // sort indices in [start,end) by centroid along axis
    g_sort_axis = axis;
    g_tri = tri;
    qsort(idx + start, (size_t)ntris, sizeof(int32_t), cmp_centroid);

    uint32_t mid = start + ntris/2;

    uint32_t L = build_node(nv, tri, idx, start, mid);
    uint32_t R = build_node(nv, tri, idx, mid, end);

    nv->nodes[my_index].left  = (int32_t)L;
    nv->nodes[my_index].right = (int32_t)R;
    // internal nodes: triOffset/triCount artık kullanılmayacak, ama debug için kalabilir
    nv->nodes[my_index].triOffset = -1;
    nv->nodes[my_index].triCount  = 0;

    return my_index;
}

int gpu_build_bvh_from_tri_vec4(
    const float* tri_data,
    uint32_t tri_count,
    GPUBVHNode** out_nodes,
    uint32_t* out_node_count,
    int32_t** out_indices,
    uint32_t* out_index_count
){
    if(!tri_data || tri_count == 0 || !out_nodes || !out_node_count || !out_indices || !out_index_count)
        return 0;

    TriInfo* tri = (TriInfo*)malloc((size_t)tri_count * sizeof(TriInfo));
    if(!tri) return 0;

    // tri_info fill
    for(uint32_t i=0;i<tri_count;i++){
        const float* t = tri_data + (size_t)i * 12u;

        v3 p0 = { t[0],  t[1],  t[2]  };
        v3 p1 = { t[4],  t[5],  t[6]  };
        v3 p2 = { t[8],  t[9],  t[10] };

        v3 mn = v3_min(p0, v3_min(p1, p2));
        v3 mx = v3_max(p0, v3_max(p1, p2));
        v3 c  = v3_mul(v3_add(v3_add(p0,p1),p2), 1.0f/3.0f);

        tri[i].bmin = mn;
        tri[i].bmax = mx;
        tri[i].centroid = c;
    }

    int32_t* idx = (int32_t*)malloc((size_t)tri_count * sizeof(int32_t));
    if(!idx){ free(tri); return 0; }
    for(uint32_t i=0;i<tri_count;i++) idx[i] = (int32_t)i;

    NodeVec nv;
    memset(&nv, 0, sizeof(nv));

    // root build (creates all nodes)
    (void)build_node(&nv, tri, idx, 0, tri_count);

    free(tri);

    *out_nodes = nv.nodes;
    *out_node_count = nv.count;
    *out_indices = idx;
    *out_index_count = tri_count;
    return 1;
}
