#define VK_USE_PLATFORM_WIN32_KHR
#include "gpu_bvh_lbv.h"
#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <math.h>

// Include denoiser
#include "bilateral_denoise.h"
#include "neural_denoise.h"

// Hybrid scheduler scaffolding
#include "nerf_batch.h"
#include "nerf_scheduler.h"

// ---------------- NeRF model headers ----------------
#define NERF_HASHGRID_MAGIC 0x3147484Eu // 'NHG1'
#define NERF_OCC_MAGIC      0x31474F4Eu // 'NOG1'

typedef struct NerfHashGridHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t levels;
    uint32_t features;
    uint32_t hashmap_size;
    uint32_t base_resolution;
    float    per_level_scale;
    uint32_t mlp_in;
    uint32_t mlp_hidden;
    uint32_t mlp_layers;
    uint32_t mlp_out;
    uint32_t flags;
    uint32_t reserved[3];
} NerfHashGridHeader;

typedef struct NerfOccHeader {
    uint32_t magic;
    uint32_t dim;
    float    scale;
    float    threshold;
} NerfOccHeader;

typedef struct NerfHashGridBlob {
    NerfHashGridHeader hdr;
    void* data;
    size_t bytes;
} NerfHashGridBlob;

typedef struct NerfOccBlob {
    NerfOccHeader hdr;
    void* data;
    size_t bytes;
} NerfOccBlob;

static float ysu_u32_to_f(uint32_t v){
    float f;
    memcpy(&f, &v, sizeof(float));
    return f;
}

static void* read_file_blob(const char* path, size_t* out_bytes){
    if(out_bytes) *out_bytes = 0;
    FILE* f = fopen(path, "rb");
    if(!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if(sz <= 0){ fclose(f); return NULL; }
    void* buf = malloc((size_t)sz);
    if(!buf){ fclose(f); return NULL; }
    if(fread(buf, 1, (size_t)sz, f) != (size_t)sz){
        fclose(f);
        free(buf);
        return NULL;
    }
    fclose(f);
    if(out_bytes) *out_bytes = (size_t)sz;
    return buf;
}

// ---------------- ENV helpers ----------------
static int ysu_env_bool(const char* key, int defv){
    const char* v = getenv(key);
    if(!v || !v[0]) return defv;
    // trim leading spaces
    while(*v==' '||*v=='\t'||*v=='\r'||*v=='\n') v++;
    if(!*v) return defv;
    if(!strcmp(v,"1") || !strcmp(v,"true") || !strcmp(v,"TRUE") || !strcmp(v,"on") || !strcmp(v,"ON")) return 1;
    if(!strcmp(v,"0") || !strcmp(v,"false") || !strcmp(v,"FALSE") || !strcmp(v,"off") || !strcmp(v,"OFF")) return 0;
    return defv;
}
static int ysu_env_int(const char* key, int defv){
    const char* v = getenv(key);
    if(!v || !v[0]) return defv;
    return atoi(v);
}


static float ysu_env_float(const char* key, float defv){
    const char* v = getenv(key);
    if(!v || !v[0]) return defv;
    // accept both "1.25" and "1,25"
    char buf[128];
    size_t n = strlen(v);
    if(n >= sizeof(buf)) n = sizeof(buf)-1;
    memcpy(buf, v, n);
    buf[n] = 0;
    for(size_t i=0;i<n;i++) if(buf[i]==',') buf[i]='.';
    return (float)atof(buf);
}


// ---------------- Minimal Vec3 helpers (for camera math) ----------------
typedef struct { float x,y,z; } CamVec3;

static CamVec3 cam_v3_add(CamVec3 a, CamVec3 b){ CamVec3 r = { a.x + b.x, a.y + b.y, a.z + b.z }; return r; }
static CamVec3 cam_v3_sub(CamVec3 a, CamVec3 b){ CamVec3 r = { a.x - b.x, a.y - b.y, a.z - b.z }; return r; }
static CamVec3 cam_v3_scale(CamVec3 a, float s){ CamVec3 r = { a.x * s, a.y * s, a.z * s }; return r; }
static float cam_v3_dot(CamVec3 a, CamVec3 b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
static CamVec3 cam_v3_cross(CamVec3 a, CamVec3 b){ CamVec3 r = { a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x }; return r; }
static CamVec3 cam_v3_norm(CamVec3 a){ float len2 = cam_v3_dot(a,a); if(len2 <= 1e-12f) return (CamVec3){0.f,1.f,0.f}; float inv = 1.0f / sqrtf(len2); return cam_v3_scale(a, inv); }

static CamVec3 cam_from_yaw_pitch(float yaw, float pitch){
    float cp = cosf(pitch);
    CamVec3 f = { sinf(yaw) * cp, sinf(pitch), cosf(yaw) * cp };
    return cam_v3_norm(f);
}

static void cam_build_basis(float yaw, float pitch, CamVec3* forward_out, CamVec3* right_out, CamVec3* up_out){
    CamVec3 forward = cam_from_yaw_pitch(yaw, pitch);
    CamVec3 world_up = {0.f, 1.f, 0.f};
    CamVec3 right = cam_v3_norm(cam_v3_cross(world_up, forward)); // Corrected: Y x Z = X
    CamVec3 up = cam_v3_norm(cam_v3_cross(forward, right));     // Corrected: Z x X = Y
    if(forward_out) *forward_out = forward;
    if(right_out) *right_out = right;
    if(up_out) *up_out = up;
}

// ---------------- CPU Depth Prepass for Depth-Conditioned NeRF ----------------
// This function estimates depth per pixel using the occupancy grid
// and fills the depth_hints buffer for GPU narrow-band sampling

typedef struct DepthHintVec4 {
    float depth;      // Estimated depth
    float delta;      // Sampling half-width
    float confidence; // 0.0 = miss, 1.0 = hit
    float flags;      // Reserved
} DepthHintVec4;

// Sample occupancy grid at world position p
// The occupancy file stores uint8 values (0 or 1) per voxel, not bits
// Header is 16 bytes (4 words), followed by dim^3 bytes of occupancy data
static int occ_sample_cpu(const uint8_t* occ_data, uint32_t occ_dim, 
                          float cx, float cy, float cz, float scale,
                          float px, float py, float pz) {
    if(!occ_data || occ_dim == 0) return 0;
    
    // Transform to local coordinates [-1,1]
    float invScale = 1.0f / (scale > 1e-6f ? scale : 1.0f);
    float lx = (px - cx) * invScale;
    float ly = (py - cy) * invScale;
    float lz = (pz - cz) * invScale;
    
    // Check bounds
    if(lx < -1.0f || lx > 1.0f || ly < -1.0f || ly > 1.0f || lz < -1.0f || lz > 1.0f)
        return 0;
    
    // Map to grid indices [0, dim-1]
    float nx = (lx * 0.5f + 0.5f) * (float)(occ_dim - 1);
    float ny = (ly * 0.5f + 0.5f) * (float)(occ_dim - 1);
    float nz = (lz * 0.5f + 0.5f) * (float)(occ_dim - 1);
    
    int ix = (int)(nx + 0.5f);
    int iy = (int)(ny + 0.5f);
    int iz = (int)(nz + 0.5f);
    
    if(ix < 0 || ix >= (int)occ_dim || 
       iy < 0 || iy >= (int)occ_dim || 
       iz < 0 || iz >= (int)occ_dim)
        return 0;
    
    // Occupancy is stored as bytes (uint8): 16-byte header + dim^3 bytes
    // Index: z * dim^2 + y * dim + x
    uint32_t byte_idx = 16 + (uint32_t)(iz * occ_dim * occ_dim + iy * occ_dim + ix);
    
    return occ_data[byte_idx] != 0;
}

// Compute depth hints for all pixels using occupancy grid raymarching
// Compute depth hints at 1/4 resolution for speed, then replicate to full buffer
// This is 16x faster than full resolution
static void compute_depth_hints_occ(
    DepthHintVec4* hints, int W, int H,
    CamVec3 cam_pos, CamVec3 cam_forward, CamVec3 cam_right, CamVec3 cam_up,
    float fov_y, float aspect,
    const uint8_t* occ_data, uint32_t occ_dim,
    float cx, float cy, float cz, float scale,
    float t_near, float t_far, int coarse_steps,
    int merge)
{
    if(!hints || !occ_data || occ_dim == 0) return;
    
    float half_h = tanf(fov_y * 0.5f);
    float half_w = half_h * aspect;
    float step_size = (t_far - t_near) / (float)coarse_steps;
    // Adaptive delta: keep a sane minimum but grow with step_size to tolerate CPU/GPU mismatch.
    float delta = fmaxf(1.5f, step_size * 4.0f); // adaptive half-width
    
    // Compute at 1/4 resolution (4x4 pixel blocks). Use BLOCK=2 to reduce block artifacts.
    const int BLOCK = 2;
    int lw = (W + BLOCK - 1) / BLOCK;
    int lh = (H + BLOCK - 1) / BLOCK;
    
    for(int ly = 0; ly < lh; ly++) {
        int py = ly * BLOCK + BLOCK/2; // Sample at block center
        if(py >= H) py = H - 1;
        float v = 1.0f - 2.0f * ((float)py + 0.5f) / (float)H;
        
        for(int lx = 0; lx < lw; lx++) {
            int px = lx * BLOCK + BLOCK/2;
            if(px >= W) px = W - 1;
            float u = 2.0f * ((float)px + 0.5f) / (float)W - 1.0f;
            
            // Compute ray direction
            CamVec3 rd;
            rd.x = cam_forward.x + u * half_w * cam_right.x + v * half_h * cam_up.x;
            rd.y = cam_forward.y + u * half_w * cam_right.y + v * half_h * cam_up.y;
            rd.z = cam_forward.z + u * half_w * cam_right.z + v * half_h * cam_up.z;
            rd = cam_v3_norm(rd);
            
            // March through occupancy grid
            float found_depth = 0.0f;
            int found = 0;
            
            for(int i = 0; i < coarse_steps && !found; i++) {
                float t = t_near + ((float)i + 0.5f) * step_size;
                float wx = cam_pos.x + rd.x * t;
                float wy = cam_pos.y + rd.y * t;
                float wz = cam_pos.z + rd.z * t;
                
                if(occ_sample_cpu(occ_data, occ_dim, cx, cy, cz, scale, wx, wy, wz)) {
                    found_depth = t;
                    found = 1;
                }
            }
            
            // Fill entire block with same hint
            DepthHintVec4 hint;
            if(found) {
                hint.depth = found_depth;
                hint.delta = delta;
                hint.confidence = 1.0f;
                hint.flags = 0.0f;
            } else {
                hint.depth = 4.0f;
                hint.delta = 2.0f;
                hint.confidence = 0.0f;
                hint.flags = 0.0f;
            }
            
            // Replicate hint to block, but only mark the sampled center as confident.
            // Other pixels in the block get confidence=0 so they fallback to full sampling,
            // avoiding blocky artifacts while still preserving speedup for center pixels.
            for(int by = 0; by < BLOCK; by++) {
                int fy = ly * BLOCK + by;
                if(fy >= H) break;
                for(int bx = 0; bx < BLOCK; bx++) {
                    int fx = lx * BLOCK + bx;
                    if(fx >= W) break;
                    DepthHintVec4 out_hint = hint;
                    // center sample offset used above was BLOCK/2
                    if(!(by == (BLOCK/2) && bx == (BLOCK/2))) {
                        // non-center pixels: low confidence -> shader will ignore
                        out_hint.confidence = 0.0f;
                    }
                    // Merge behavior: if merge==1, only overwrite when new confidence is higher
                    if(merge) {
                        DepthHintVec4 prev = hints[fy * W + fx];
                        if(out_hint.confidence > prev.confidence) {
                            hints[fy * W + fx] = out_hint;
                        }
                    } else {
                        hints[fy * W + fx] = out_hint;
                    }
                }
            }
        }
    }
}


#include "gpu_bvh.h"
#include "gpu_bvh_build.h"

// ---------------- OBJ loader (minimal) ----------------
// Supports lines:
//   v x y z
//   f i j k [l ...]   where each index can be "i", "i/j", "i//n", "i/j/n" and can be negative.
// Faces with >3 verts are triangulated as a fan.
// Output format matches tri.comp: vec4 per vertex, repeating (p0,p1,p2) per triangle.
typedef struct { float x,y,z; } ObjV3;

static void die(const char* what, VkResult r){
    fprintf(stderr, "[VK] %s failed: %d\n", what, (int)r);
    exit(1);
}

static uint8_t* read_file(const char* path, size_t* out_sz){
    FILE* f = fopen(path, "rb");
    if(!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if(sz <= 0){ fclose(f); return NULL; }
    uint8_t* data = (uint8_t*)malloc((size_t)sz);
    if(!data){ fclose(f); return NULL; }
    if(fread(data, 1, (size_t)sz, f) != (size_t)sz){
        fclose(f); free(data); return NULL;
    }
    fclose(f);
    if(out_sz) *out_sz = (size_t)sz;
    return data;
}

static int parse_int(const char** s, int* out){
    while(**s==' ' || **s=='\t') (*s)++;
    int sign = 1;
    if(**s=='-'){ sign=-1; (*s)++; }
    if(**s<'0' || **s>'9') return 0;
    int v=0;
    while(**s>='0' && **s<='9'){ v = v*10 + (**s - '0'); (*s)++; }
    *out = v*sign;
    return 1;
}

static int parse_face_vi(const char** s, int* vi_out){
    int vi;
    if(!parse_int(s, &vi)) return 0;

    // skip /vt/vn forms
    if(**s=='/'){
        (*s)++;
        if(**s=='/'){
            (*s)++;
            int tmp;
            parse_int(s,&tmp);
        } else {
            int tmp;
            parse_int(s,&tmp);
            if(**s=='/'){
                (*s)++;
                parse_int(s,&tmp);
            }
        }
    }
    while(**s && **s!=' ' && **s!='\t' && **s!='\r' && **s!='\n') (*s)++;
    *vi_out = vi;
    return 1;
}

static int obj_index_to_zero(int idx, int vcount){
    if(idx > 0) return idx - 1;
    if(idx < 0) return vcount + idx;
    return -1;
}

// Push constants shared with shaders/tri.comp
typedef struct {
    int W;
    int H;
    int frame;
    int seed;
    int triCount;
    int nodeCount;
    int useBVH;
    int cullBackface;
    int rootCount;
    int enableCounters;
    float alpha;
    int resetAccum;
    int enableNerfProxy;
    float nerfStrength;
    float nerfDensity;
    float nerfBlend;
    float nerfBounds;
    float nerfCenterX;
    float nerfCenterY;
    float nerfCenterZ;
    float nerfScale;
    int nerfSkipOcc;
    int nerfSteps;
    int renderMode; // 0=mesh, 1=probe (placeholder), 2=hybrid (mesh+nerf proxy)
    int depthHintsReady;
} PushConstants;

// Camera uniform buffer (separate binding for stability)
typedef struct {
    float pos[4];      // xyz = camera position, w = unused
    float forward[4];  // xyz = forward, w = unused
    float right[4];    // xyz = right, w = unused
    float up[4];       // xyz = up, w = unused
} CameraUBO;

_Static_assert(sizeof(PushConstants) <= 128, "PushConstants exceeds Vulkan push constant limit");


// =========================
// BVH OFFLINE CACHE (optional)
// Set env: YSU_GPU_BVH_CACHE="path/to/file.bvhbin"
// =========================
#define BVH_CACHE_MAGIC 0x48565359u  // 'YSVH'
#define BVH_CACHE_VER   1u
#if defined(_WIN32)
#include <windows.h>
#endif
#include <sys/stat.h>

// ---------------- Timing helpers ----------------
static uint64_t ysu_now_us(void){
#if defined(_WIN32)
    static LARGE_INTEGER freq;
    static int inited = 0;
    if(!inited){ QueryPerformanceFrequency(&freq); inited = 1; }
    LARGE_INTEGER c; QueryPerformanceCounter(&c);
    return (uint64_t)((c.QuadPart * 1000000ULL) / (uint64_t)freq.QuadPart);
#else
    // fallback: microseconds not guaranteed
    return (uint64_t)(clock() * (1000000.0 / (double)CLOCKS_PER_SEC));
#endif
}

// ---------------- Triangle cache (binary) ----------------
// Speeds up huge OBJ loads by caching the vec4 triangle stream to disk.
// Env: YSU_GPU_TRI_CACHE="path/to/file.tri"
// If not set, and YSU_GPU_BVH_CACHE is set, we auto-use "<bvh_cache>.tri".
#define TRI_CACHE_MAGIC "YSUTRI1"
typedef struct TriCacheHeader {
    char magic[8];        // "YSUTRI1"
    uint32_t tri_count;
    uint64_t obj_size;
    uint64_t obj_mtime;
} TriCacheHeader;

static int ysu_stat_file(const char* path, uint64_t* out_size, uint64_t* out_mtime){
#if defined(_WIN32)
    struct _stat64 st;
    if(_stat64(path, &st) != 0) return 0;
    if(out_size) *out_size = (uint64_t)st.st_size;
    if(out_mtime) *out_mtime = (uint64_t)st.st_mtime;
    return 1;
#else
    struct stat st;
    if(stat(path, &st) != 0) return 0;
    if(out_size) *out_size = (uint64_t)st.st_size;
    if(out_mtime) *out_mtime = (uint64_t)st.st_mtime;
    return 1;
#endif
}

static int tri_cache_load(const char* tri_cache_path, const char* obj_path,
                          float** out_tri_data, int* out_tri_count){
    if(!tri_cache_path || !tri_cache_path[0] || !obj_path || !obj_path[0]) return 0;

    uint64_t obj_size=0, obj_mtime=0;
    if(!ysu_stat_file(obj_path, &obj_size, &obj_mtime)) return 0;

    FILE* f = fopen(tri_cache_path, "rb");
    if(!f) return 0;

    TriCacheHeader h;
    if(fread(&h, 1, sizeof(h), f) != sizeof(h)){ fclose(f); return 0; }
    if(memcmp(h.magic, TRI_CACHE_MAGIC, 7) != 0){ fclose(f); return 0; }
    if(h.obj_size != obj_size || h.obj_mtime != obj_mtime){ fclose(f); return 0; }
    if(h.tri_count == 0){ fclose(f); return 0; }

    size_t floats = (size_t)h.tri_count * 12u;
    float* data = (float*)malloc(floats * sizeof(float));
    if(!data){ fclose(f); return 0; }

    if(fread(data, sizeof(float), floats, f) != floats){
        fclose(f); free(data); return 0;
    }
    fclose(f);

    *out_tri_data = data;
    *out_tri_count = (int)h.tri_count;
    return 1;
}

static int tri_cache_save(const char* tri_cache_path, const char* obj_path,
                          const float* tri_data, int tri_count){
    if(!tri_cache_path || !tri_cache_path[0] || !obj_path || !obj_path[0]) return 0;
    if(!tri_data || tri_count <= 0) return 0;

    uint64_t obj_size=0, obj_mtime=0;
    if(!ysu_stat_file(obj_path, &obj_size, &obj_mtime)) return 0;

    FILE* f = fopen(tri_cache_path, "wb");
    if(!f) return 0;

    TriCacheHeader h;
    memset(&h, 0, sizeof(h));
    memcpy(h.magic, TRI_CACHE_MAGIC, 7);
    h.tri_count = (uint32_t)tri_count;
    h.obj_size  = obj_size;
    h.obj_mtime = obj_mtime;

    if(fwrite(&h, 1, sizeof(h), f) != sizeof(h)){ fclose(f); return 0; }

    size_t floats = (size_t)tri_count * 12u;
    if(fwrite(tri_data, sizeof(float), floats, f) != floats){ fclose(f); return 0; }

    fclose(f);
    return 1;
}


typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t triCount;
    uint32_t rootCount;
    uint32_t nodeCount;
    uint32_t indexCount;
} BVHCacheHeader;

static int bvh_cache_load(
    const char* path,
    uint32_t triCount,
    int32_t** out_roots, uint32_t* out_rootCount,
    GPUBVHNode** out_nodes, uint32_t* out_nodeCount,
    int32_t** out_indices, uint32_t* out_indexCount
){
    FILE* f = fopen(path, "rb");
    if(!f) return 0;

    BVHCacheHeader h;
    if(fread(&h, sizeof(h), 1, f) != 1){ fclose(f); return 0; }
    if(h.magic != BVH_CACHE_MAGIC || h.version != BVH_CACHE_VER){ fclose(f); return 0; }
    if(h.triCount != triCount){ fclose(f); return 0; }

    int32_t* roots = (int32_t*)malloc((size_t)h.rootCount * sizeof(int32_t));
    GPUBVHNode* nodes = (GPUBVHNode*)malloc((size_t)h.nodeCount * sizeof(GPUBVHNode));
    int32_t* idx = (int32_t*)malloc((size_t)h.indexCount * sizeof(int32_t));
    if(!roots || !nodes || !idx){
        fclose(f);
        free(roots); free(nodes); free(idx);
        return 0;
    }

    if(fread(roots, sizeof(int32_t), (size_t)h.rootCount, f) != h.rootCount){ fclose(f); free(roots); free(nodes); free(idx); return 0; }
    if(fread(nodes, sizeof(GPUBVHNode), (size_t)h.nodeCount, f) != h.nodeCount){ fclose(f); free(roots); free(nodes); free(idx); return 0; }
    if(fread(idx, sizeof(int32_t), (size_t)h.indexCount, f) != h.indexCount){ fclose(f); free(roots); free(nodes); free(idx); return 0; }

    fclose(f);
    *out_roots = roots; *out_rootCount = h.rootCount;
    *out_nodes = nodes; *out_nodeCount = h.nodeCount;
    *out_indices = idx; *out_indexCount = h.indexCount;
    return 1;
}

static int bvh_cache_save(
    const char* path,
    uint32_t triCount,
    const int32_t* roots, uint32_t rootCount,
    const GPUBVHNode* nodes, uint32_t nodeCount,
    const int32_t* indices, uint32_t indexCount
){
    FILE* f = fopen(path, "wb");
    if(!f){
        fprintf(stderr, "[GPU] BVH CACHE SAVE fopen failed: '%s' errno=%d\n", path, errno);
        return 0;
    }

    BVHCacheHeader h;
    h.magic = BVH_CACHE_MAGIC;
    h.version = BVH_CACHE_VER;
    h.triCount = triCount;
    h.rootCount = rootCount;
    h.nodeCount = nodeCount;
    h.indexCount = indexCount;

    if(fwrite(&h, sizeof(h), 1, f) != 1){ fclose(f); return 0; }
    if(fwrite(roots, sizeof(int32_t), (size_t)rootCount, f) != rootCount){ fclose(f); return 0; }
    if(fwrite(nodes, sizeof(GPUBVHNode), (size_t)nodeCount, f) != nodeCount){ fclose(f); return 0; }
    if(fwrite(indices, sizeof(int32_t), (size_t)indexCount, f) != indexCount){ fclose(f); return 0; }

    fclose(f);
    return 1;
}

static int load_obj_as_tri_vec4(const char* path, float** out_tri_data, int* out_tri_count){
    *out_tri_data = NULL;
    *out_tri_count = 0;

    FILE* f = fopen(path, "rb");
    if(!f) return 0;

    ObjV3* verts = NULL;
    int vcap = 0, vcount = 0;

    int* faces = NULL;
    int* face_off = NULL;
    int* face_len = NULL;
    int fcap = 0, fcount = 0;

    char line[4096];
    while(fgets(line, sizeof(line), f)){
        if(line[0] == 'v' && (line[1]==' ' || line[1]=='\t')){
            float x=0,y=0,z=0;
            if(sscanf(line+1, "%f %f %f", &x, &y, &z) == 3){
                if(vcount >= vcap){
                    vcap = vcap ? vcap*2 : 1024;
                    verts = (ObjV3*)realloc(verts, (size_t)vcap * sizeof(ObjV3));
                    if(!verts){ fclose(f); return 0; }
                }
                verts[vcount++] = (ObjV3){x,y,z};
            }
        } else if(line[0] == 'f' && (line[1]==' ' || line[1]=='\t')){
            const char* s = line+1;

            int tmpIdx[256];
            int n = 0;
            while(*s){
                while(*s==' ' || *s=='\t') s++;
                if(*s=='\0' || *s=='\r' || *s=='\n') break;
                if(n >= 256) break;
                int vi=0;
                if(!parse_face_vi(&s, &vi)) break;
                tmpIdx[n++] = vi;
            }
            if(n < 3) continue;

            if(fcount >= fcap){
                fcap = fcap ? fcap*2 : 1024;
                face_off = (int*)realloc(face_off, (size_t)fcap * sizeof(int));
                face_len = (int*)realloc(face_len, (size_t)fcap * sizeof(int));
                if(!face_off || !face_len){ fclose(f); return 0; }
            }

            int start = 0;
            if(faces){
                start = face_off[fcount-1] + face_len[fcount-1];
            }
            faces = (int*)realloc(faces, (size_t)(start + n) * sizeof(int));
            if(!faces){ fclose(f); return 0; }

            for(int i=0;i<n;i++) faces[start+i] = tmpIdx[i];
            face_off[fcount] = start;
            face_len[fcount] = n;
            fcount++;
        }
    }
    fclose(f);

    // Count triangles after fan triangulation
    int tri_count = 0;
    for(int fi=0; fi<fcount; fi++){
        int n = face_len[fi];
        tri_count += (n - 2);
    }
    if(tri_count <= 0 || vcount <= 0){
        free(verts); free(faces); free(face_off); free(face_len);
        return 0;
    }

    // Allocate tri_data as vec4 stream: p0,p1,p2 repeating => 3 vec4 per tri => 12 floats
    float* tri_data = (float*)malloc((size_t)tri_count * 12u * sizeof(float));
    if(!tri_data){
        free(verts); free(faces); free(face_off); free(face_len);
        return 0;
    }

    int t = 0;
    for(int fi=0; fi<fcount; fi++){
        int start = face_off[fi];
        int n = face_len[fi];

        int i0 = obj_index_to_zero(faces[start+0], vcount);
        if(i0 < 0 || i0 >= vcount) continue;
        ObjV3 a = verts[i0];

        for(int k=1; k+1<n; k++){
            int i1 = obj_index_to_zero(faces[start+k], vcount);
            int i2 = obj_index_to_zero(faces[start+k+1], vcount);
            if(i1<0||i1>=vcount||i2<0||i2>=vcount) continue;

            ObjV3 b = verts[i1];
            ObjV3 c = verts[i2];

            size_t base = (size_t)t * 12u;
            // p0
            tri_data[base+0]=a.x; tri_data[base+1]=a.y; tri_data[base+2]=a.z; tri_data[base+3]=0.0f;
            // p1
            tri_data[base+4]=b.x; tri_data[base+5]=b.y; tri_data[base+6]=b.z; tri_data[base+7]=0.0f;
            // p2
            tri_data[base+8]=c.x; tri_data[base+9]=c.y; tri_data[base+10]=c.z; tri_data[base+11]=0.0f;
            t++;
        }
    }

    free(verts); free(faces); free(face_off); free(face_len);

    if(t <= 0){
        free(tri_data);
        return 0;
    }

    *out_tri_data = tri_data;
    *out_tri_count = t;
    return 1;
}

// ---------------- Vulkan helper funcs ----------------

static VkBuffer create_buffer(VkDevice dev, VkDeviceSize size, VkBufferUsageFlags usage){
    VkBufferCreateInfo bci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size = size;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer b = 0;
    VkResult r = vkCreateBuffer(dev, &bci, NULL, &b);
    if(r!=VK_SUCCESS) die("vkCreateBuffer", r);
    return b;
}

static uint32_t find_mem_type(VkPhysicalDevice phy, uint32_t type_bits, VkMemoryPropertyFlags want){
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phy, &mp);
    for(uint32_t i=0;i<mp.memoryTypeCount;i++){
        if((type_bits & (1u<<i)) && ((mp.memoryTypes[i].propertyFlags & want) == want))
            return i;
    }
    fprintf(stderr, "No suitable memory type\n");
    exit(1);
}

static VkDeviceMemory alloc_bind_buffer_mem(VkPhysicalDevice phy, VkDevice dev, VkBuffer buf, VkMemoryPropertyFlags props){
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(dev, buf, &mr);

    VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = find_mem_type(phy, mr.memoryTypeBits, props);

    VkDeviceMemory mem = 0;
    VkResult r = vkAllocateMemory(dev, &mai, NULL, &mem);
    if(r!=VK_SUCCESS) die("vkAllocateMemory", r);

    r = vkBindBufferMemory(dev, buf, mem, 0);
    if(r!=VK_SUCCESS) die("vkBindBufferMemory", r);
    return mem;
}

static VkImage create_image(VkDevice dev, int W, int H, VkFormat fmt, VkImageUsageFlags usage){
    VkImageCreateInfo ici = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = fmt;
    ici.extent.width = (uint32_t)W;
    ici.extent.height = (uint32_t)H;
    ici.extent.depth = 1;
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = usage;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImage img = 0;
    VkResult r = vkCreateImage(dev, &ici, NULL, &img);
    if(r!=VK_SUCCESS) die("vkCreateImage", r);
    return img;
}

static VkDeviceMemory alloc_bind_image_mem(VkPhysicalDevice phy, VkDevice dev, VkImage img, VkMemoryPropertyFlags props){
    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(dev, img, &mr);

    VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = find_mem_type(phy, mr.memoryTypeBits, props);

    VkDeviceMemory mem = 0;
    VkResult r = vkAllocateMemory(dev, &mai, NULL, &mem);
    if(r!=VK_SUCCESS) die("vkAllocateMemory(img)", r);

    r = vkBindImageMemory(dev, img, mem, 0);
    if(r!=VK_SUCCESS) die("vkBindImageMemory", r);
    return mem;
}

static VkCommandPool create_cmd_pool(VkDevice dev, uint32_t qfi){
    VkCommandPoolCreateInfo ci = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    ci.queueFamilyIndex = qfi;
    VkCommandPool pool = 0;
    VkResult r = vkCreateCommandPool(dev, &ci, NULL, &pool);
    if(r!=VK_SUCCESS) die("vkCreateCommandPool", r);
    return pool;
}

static VkCommandBuffer alloc_cmd(VkDevice dev, VkCommandPool pool){
    VkCommandBufferAllocateInfo ai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    ai.commandPool = pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cb = 0;
    VkResult r = vkAllocateCommandBuffers(dev, &ai, &cb);
    if(r!=VK_SUCCESS) die("vkAllocateCommandBuffers", r);
    return cb;
}

static void submit_and_wait(VkDevice dev, VkQueue q, VkCommandBuffer cb){
    VkResult r = vkEndCommandBuffer(cb);
    if(r!=VK_SUCCESS) die("vkEndCommandBuffer", r);

    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;

        r = vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE);
    if(r!=VK_SUCCESS) die("vkQueueSubmit", r);

    r = vkQueueWaitIdle(q);
    if(r!=VK_SUCCESS) die("vkQueueWaitIdle", r);
}

int main(void){
    // --- Config ---
    int W = 4096;
    int H = 2048;
    int spp = 128;
    int seed = 1337;
    int frames = 1;

    int window_enabled = ysu_env_bool("YSU_GPU_WINDOW", 0);
    int no_io = ysu_env_bool("YSU_GPU_NO_IO", 0);      // skip readback/file writes for timing
    int minimal = ysu_env_bool("YSU_GPU_MINIMAL", 0);  // force 1 frame, 1 spp, no tonemap/IO
    int fast_mode = ysu_env_bool("YSU_GPU_FAST", 0);   // aggressive optimization for realtime (lower quality)
    const char* env_w = getenv("YSU_GPU_W");
    const char* env_h = getenv("YSU_GPU_H");
    const char* env_spp = getenv("YSU_GPU_SPP");
    const char* env_seed = getenv("YSU_GPU_SEED");
    const char* env_frames = getenv("YSU_GPU_FRAMES");
    const char* obj_path = getenv("YSU_GPU_OBJ");
    const char* env_use_bvh = getenv("YSU_GPU_USE_BVH"); // default 1
    const char* env_cull = getenv("YSU_GPU_CULL"); // default 1 (backface culling)
    const char* env_render_scale = getenv("YSU_GPU_RENDER_SCALE"); // 0.5 = half-res, 0.25 = quarter-res
    int ts_enabled = ysu_env_bool("YSU_GPU_TS", 0); // enable GPU timestamps for denoise timing
    double ts_period_ns = 0.0;

    // Render mode: 0=mesh, 1=probe (placeholder), 2=hybrid (mesh+nerf proxy), 3=nerf-only, 4-25=debug modes, 26-27=depth-conditioned
    int render_mode = ysu_env_int("YSU_RENDER_MODE", 0);
    if(render_mode < 0) render_mode = 0;
    if(render_mode > 27) render_mode = 27;  // Increased for depth-conditioned modes 26-27

    // Interactive camera controls
    float cam_speed = ysu_env_float("YSU_CAM_SPEED", 3.0f);           // units per second for WASD
    float mouse_sens = ysu_env_float("YSU_CAM_MOUSE_SENS", 0.0025f);  // radians per pixel for mouse look
    int cam_mouse_lock = ysu_env_bool("YSU_CAM_MOUSE_LOCK", 1);       // lock/hide cursor for FPS look
    // Deadzone to ignore sub-pixel mouse jitter (helps prevent accumulation resets)
    float mouse_deadzone = ysu_env_float("YSU_CAM_MOUSE_DEADZONE", 0.5f); // pixels
    // Epsilon thresholds for detecting camera movement (position/orientation)
    float cam_reset_eps_pos = ysu_env_float("YSU_CAM_RESET_EPS_POS", 1e-4f);
    float cam_reset_eps_ang = ysu_env_float("YSU_CAM_RESET_EPS_ANG", 1e-4f);

    // Hybrid Mesh + NeRF proxy (procedural placeholder for research)
    int nerf_proxy_enabled = ysu_env_bool("YSU_NERF_PROXY", 0);
    float nerf_strength = ysu_env_float("YSU_NERF_STRENGTH", 1.0f);
    float nerf_density = ysu_env_float("YSU_NERF_DENSITY", 1.0f);
    float nerf_blend = ysu_env_float("YSU_NERF_BLEND", 0.35f); // 0=mesh only, 1=nerf only
    float nerf_bounds = ysu_env_float("YSU_NERF_BOUNDS", 8.0f); // max distance for proxy volume
    const char* nerf_center_x_env = getenv("YSU_NERF_CENTER_X");
    const char* nerf_center_y_env = getenv("YSU_NERF_CENTER_Y");
    const char* nerf_center_z_env = getenv("YSU_NERF_CENTER_Z");
    const char* nerf_scale_env = getenv("YSU_NERF_SCALE");
    float nerf_center_x = (nerf_center_x_env != NULL) ? ysu_env_float("YSU_NERF_CENTER_X", 0.0f) : 0.0f;
    float nerf_center_y = (nerf_center_y_env != NULL) ? ysu_env_float("YSU_NERF_CENTER_Y", 0.0f) : 0.0f;
    float nerf_center_z = (nerf_center_z_env != NULL) ? ysu_env_float("YSU_NERF_CENTER_Z", 0.0f) : 0.0f;
    float nerf_scale = (nerf_scale_env != NULL) ? ysu_env_float("YSU_NERF_SCALE", 1.0f) : 1.0f;
    if(nerf_scale <= 0.0f) nerf_scale = 1.0f;
    int nerf_skip_occ = ysu_env_bool("YSU_NERF_SKIP_OCC", 0);
    int nerf_steps = ysu_env_int("YSU_NERF_STEPS", 6);
    if(nerf_steps < 1) nerf_steps = 1;
    if(nerf_steps > 1024) nerf_steps = 1024;

    const char* nerf_hash_path = getenv("YSU_NERF_HASHGRID");
    const char* nerf_occ_path = getenv("YSU_NERF_OCC");
    NerfHashGridBlob nerf_hash = {0};
    NerfOccBlob nerf_occ = {0};

    if(nerf_hash_path && nerf_hash_path[0]){
        nerf_hash.data = read_file_blob(nerf_hash_path, &nerf_hash.bytes);
        if(nerf_hash.data && nerf_hash.bytes >= sizeof(NerfHashGridHeader)){
            memcpy(&nerf_hash.hdr, nerf_hash.data, sizeof(NerfHashGridHeader));
            if(nerf_hash.hdr.magic != NERF_HASHGRID_MAGIC){
                fprintf(stderr, "[NERF] hashgrid magic mismatch (0x%08X)\n", nerf_hash.hdr.magic);
            } else {
                fprintf(stderr, "[NERF] hashgrid loaded: L=%u F=%u H=%u base=%u layers=%u hidden=%u\n",
                        nerf_hash.hdr.levels, nerf_hash.hdr.features, nerf_hash.hdr.hashmap_size,
                        nerf_hash.hdr.base_resolution, nerf_hash.hdr.mlp_layers, nerf_hash.hdr.mlp_hidden);
                if(nerf_hash.hdr.version >= 2){
                    float cx = ysu_u32_to_f(nerf_hash.hdr.reserved[0]);
                    float cy = ysu_u32_to_f(nerf_hash.hdr.reserved[1]);
                    float cz = ysu_u32_to_f(nerf_hash.hdr.reserved[2]);
                    float sc = ysu_u32_to_f(nerf_hash.hdr.flags);
                    fprintf(stderr, "[NERF] hashgrid xform: center=(%.3f, %.3f, %.3f) scale=%.3f\n", cx, cy, cz, sc);
                    if(!nerf_center_x_env) nerf_center_x = cx;
                    if(!nerf_center_y_env) nerf_center_y = cy;
                    if(!nerf_center_z_env) nerf_center_z = cz;
                    if(!nerf_scale_env) nerf_scale = sc;
                }
            }
        } else {
            fprintf(stderr, "[NERF] failed to read hashgrid: %s\n", nerf_hash_path);
        }
    }

    if(nerf_occ_path && nerf_occ_path[0]){
        nerf_occ.data = read_file_blob(nerf_occ_path, &nerf_occ.bytes);
        if(nerf_occ.data && nerf_occ.bytes >= sizeof(NerfOccHeader)){
            memcpy(&nerf_occ.hdr, nerf_occ.data, sizeof(NerfOccHeader));
            if(nerf_occ.hdr.magic != NERF_OCC_MAGIC){
                fprintf(stderr, "[NERF] occ magic mismatch (0x%08X)\n", nerf_occ.hdr.magic);
            } else {
                /* Validate that the file is large enough for 16-byte header + dim^3 voxels.
                 * Without this check, a truncated file would cause occ_sample_cpu to
                 * read out of bounds. */
                uint32_t od = nerf_occ.hdr.dim;
                size_t expected = (size_t)16 + (size_t)od * od * od;
                if(nerf_occ.bytes < expected){
                    fprintf(stderr, "[NERF] occupancy file truncated: %zu < %zu (dim=%u)\n",
                            nerf_occ.bytes, expected, od);
                    free(nerf_occ.data);
                    nerf_occ.data = NULL;
                    nerf_occ.bytes = 0;
                } else {
                    fprintf(stderr, "[NERF] occupancy loaded: dim=%u threshold=%.3f\n",
                            nerf_occ.hdr.dim, nerf_occ.hdr.threshold);
                }
            }
        } else {
            fprintf(stderr, "[NERF] failed to read occupancy: %s\n", nerf_occ_path);
        }
    }

    // Scheduler config (AVX2 batch size default 4096)
    NerfScheduleConfig sched_cfg;
    sched_cfg.batch_size = (uint32_t)ysu_env_int("YSU_NERF_BATCH", 4096);
    if(sched_cfg.batch_size < 8) sched_cfg.batch_size = 8;
    sched_cfg.cpu_share = ysu_env_float("YSU_NERF_CPU_SHARE", 0.25f);
    if(sched_cfg.cpu_share < 0.0f) sched_cfg.cpu_share = 0.0f;
    if(sched_cfg.cpu_share > 1.0f) sched_cfg.cpu_share = 1.0f;
    sched_cfg.fovea_radius = ysu_env_float("YSU_NERF_FOVEA", 0.35f);

    NerfScheduleQueues sched_q = {0};
    int sched_ok = nerf_scheduler_init(&sched_q, sched_cfg.batch_size);
    if(!sched_ok){
        fprintf(stderr, "[NERF] scheduler init failed (batch=%u)\n", sched_cfg.batch_size);
    }

    if(env_w) W = atoi(env_w);
    if(env_h) H = atoi(env_h);
    if(env_spp) spp = atoi(env_spp);
    if(env_seed) seed = atoi(env_seed);
    if(env_frames) frames = atoi(env_frames);
    if(frames < 1) frames = 1;

    // In window mode, always use 1 frame per iteration for responsive input
    // (YSU_GPU_FRAMES is ignored in window mode; use ESC to quit)
    if(window_enabled) frames = 1;

    // Render scale: 0.5 = render at half resolution, 0.25 = quarter res
    // Applied directly to W/H (output resolution matches render resolution)
    float render_scale = 0.5f;  // Default 0.5 = 2x speedup
    if(env_render_scale) render_scale = ysu_env_float("YSU_GPU_RENDER_SCALE", 0.5f);
    if(render_scale < 0.1f) render_scale = 0.1f;  // clamp to reasonable values
    if(render_scale > 1.0f) render_scale = 1.0f;

    // Default camera position: match Lego dataset frame 0
    // Pos: (-0.05, 3.85, 1.21) looking at (0,0,0)
    CamVec3 cam_pos = {-0.05f, 3.85f, 1.21f};
    float cam_yaw = 3.10f;     // ~178 degrees
    float cam_pitch = -1.26f;  // -72 degrees (looking down)

    // Apply render scale BEFORE shader setup
    if(render_scale < 1.0f){
        W = (int)(W * render_scale);
        H = (int)(H * render_scale);
        fprintf(stderr, "[GPU] render scale %.2f -> %dx%d\n", render_scale, W, H);
    }

    // Minimal mode: isolates core compute for timing (no tonemap/readback/window)
    if(minimal){
        frames = 1;
        spp = 1;
        window_enabled = 0;
    }

    // Fast mode: aggressive optimization for 60 FPS (lower quality but real-time)
    if(fast_mode){
        // Reduce render resolution to 960×540, denoise at quarter res, then upscale
        // This trades quality for speed: ~4x faster rendering + denoise
        if(!env_spp) spp = 1;  // force 1 sample if not explicitly set
        fprintf(stderr, "[GPU] FAST mode: reduced quality for realtime performance\n");
    }

    int use_bvh = 1;
    if(env_use_bvh) use_bvh = atoi(env_use_bvh);

    int cull_backface = 1;
    if(env_cull) cull_backface = atoi(env_cull);

    fprintf(stderr, "[GPU] W=%d H=%d SPP=%d seed=%d renderScale=%.2f\n", W, H, spp, seed, render_scale);

    // Benchmark toggles (important for performance tests)
    int write_enabled    = ysu_env_bool("YSU_GPU_WRITE", 1);
    int readback_enabled = ysu_env_bool("YSU_GPU_READBACK", write_enabled);
    if(minimal){ write_enabled = 0; readback_enabled = 0; }
    int counters_enabled = ysu_env_bool("YSU_GPU_COUNTERS", 1);
        if(no_io){ write_enabled = 0; readback_enabled = 0; }
        if(!write_enabled) readback_enabled = 0;
        fprintf(stderr, "[GPU] toggles: WRITE=%d READBACK=%d COUNTERS_READ=%d MINIMAL=%d NO_IO=%d\n",
            write_enabled, readback_enabled, counters_enabled, minimal, no_io);

    // Output / tonemap
    const char* outfmt = getenv("YSU_GPU_OUTFMT");
    if(!outfmt || !outfmt[0]) outfmt = "ppm";

    int tonemap_enabled = ysu_env_bool("YSU_GPU_TONEMAP", 0);
    if(window_enabled) tonemap_enabled = 1; // swapchain wants UNORM output
    if(minimal){ tonemap_enabled = 0; }
    float tm_exposure = ysu_env_float("YSU_GPU_EXPOSURE", 1.0f);
    float tm_gamma    = ysu_env_float("YSU_GPU_GAMMA",    2.2f);
    if(tonemap_enabled){
        fprintf(stderr, "[GPU] tonemap: ENABLED exposure=%.3f gamma=%.3f outfmt=%s\n", tm_exposure, tm_gamma, outfmt);
    }


    // --- Load triangles (vec4 stream) ---
    float* tri_data = NULL;
    int tri_count = 0;

    // Triangle cache: avoids slow OBJ parsing on repeat runs.
    const char* tri_cache_path = getenv("YSU_GPU_TRI_CACHE");
    char tri_cache_auto[1024];
    if((!tri_cache_path || !tri_cache_path[0])) {
        const char* bvh_cache_env = getenv("YSU_GPU_BVH_CACHE");
        if(bvh_cache_env && bvh_cache_env[0]) {
            // auto: "<bvh_cache>.tri"
            snprintf(tri_cache_auto, sizeof(tri_cache_auto), "%s.tri", bvh_cache_env);
            tri_cache_path = tri_cache_auto;
        }
    }

    uint64_t t_obj0 = ysu_now_us();
    int tri_cache_hit = 0;

    if(obj_path && obj_path[0]){
        if(tri_cache_path && tri_cache_path[0]) {
            if(tri_cache_load(tri_cache_path, obj_path, &tri_data, &tri_count)) {
                tri_cache_hit = 1;
                fprintf(stderr, "[GPU] TRI CACHE HIT: %s tris=%d\n", tri_cache_path, tri_count);
            } else {
                fprintf(stderr, "[GPU] TRI CACHE MISS: %s\n", tri_cache_path);
            }
        }

        if(!tri_data || tri_count <= 0){
            if(!load_obj_as_tri_vec4(obj_path, &tri_data, &tri_count)){
                fprintf(stderr, "[GPU] OBJ load failed: %s (falling back to cube)\n", obj_path);
            } else if(tri_cache_path && tri_cache_path[0]) {
                if(tri_cache_save(tri_cache_path, obj_path, tri_data, tri_count)) {
                    fprintf(stderr, "[GPU] TRI CACHE SAVED: %s\n", tri_cache_path);
                }
            }
        }
    }

    uint64_t t_obj1 = ysu_now_us();
    fprintf(stderr, "[GPU] OBJ/TRI load time: %.3f ms%s\n",
            (double)(t_obj1 - t_obj0)/1000.0,
            tri_cache_hit ? " (cached)" : "");


    if(!tri_data || tri_count <= 0){
        // Fallback cube (12 tris), centered at (0,0,-3)
        // FIXED: Triangle winding order corrected for front-facing normals (outward pointing)
        tri_count = 12;
        tri_data = (float*)calloc((size_t)tri_count * 12u, sizeof(float));

        ObjV3 v[8] = {
            {-1,-1,-4}, {+1,-1,-4}, {+1,+1,-4}, {-1,+1,-4},
            {-1,-1,-2}, {+1,-1,-2}, {+1,+1,-2}, {-1,+1,-2}
        };
        int idx[12][3] = {
            {2,1,0},{3,2,0},  // Front face: normal now points toward camera
            {5,6,4},{6,7,4},  // Back face
            {5,4,0},{1,5,0},  // Bottom face
            {6,2,3},{7,6,3},  // Top face
            {7,3,0},{4,7,0},  // Left face
            {6,5,1},{2,6,1}   // Right face
        };
        for(int i=0;i<12;i++){
            ObjV3 a=v[idx[i][0]], b=v[idx[i][1]], c=v[idx[i][2]];
            size_t base=(size_t)i*12u;
            tri_data[base+0]=a.x; tri_data[base+1]=a.y; tri_data[base+2]=a.z; tri_data[base+3]=0;
            tri_data[base+4]=b.x; tri_data[base+5]=b.y; tri_data[base+6]=b.z; tri_data[base+7]=0;
            tri_data[base+8]=c.x; tri_data[base+9]=c.y; tri_data[base+10]=c.z; tri_data[base+11]=0;
        }
    }

    // --- Build BVH (CPU) ---
// NOTE: BVH build is expensive. For very large meshes we build multiple BVHs (chunks) and traverse
// multiple roots on the GPU. This avoids the "15M+ stalls forever" problem.
GPUBVHNode* bvh_nodes = NULL;
int32_t* bvh_indices = NULL;
uint32_t bvh_node_count = 0;
uint32_t bvh_index_count = 0;

int32_t* bvh_roots = NULL;
uint32_t bvh_root_count = 0;

// Optional BVH cache (skips expensive build on repeat runs)
    uint64_t t_bvh0 = ysu_now_us();
int cache_hit = 0;
const char* cache_path = getenv("YSU_GPU_BVH_CACHE");
if(use_bvh != 0 && cache_path && cache_path[0]){
    if(bvh_cache_load(cache_path, (uint32_t)tri_count,
                      &bvh_roots, &bvh_root_count,
                      &bvh_nodes, &bvh_node_count,
                      &bvh_indices, &bvh_index_count)){
        cache_hit = 1;
        fprintf(stderr, "[GPU] BVH CACHE HIT: %s roots=%u nodes=%u indices=%u\n",
                cache_path, bvh_root_count, bvh_node_count, bvh_index_count);
    } else {
        fprintf(stderr, "[GPU] BVH CACHE MISS: %s\n", cache_path);
    }
}


if(use_bvh != 0 && !cache_hit){
    const char* env_chunk = getenv("YSU_GPU_BVH_CHUNK_TRIS");
    int chunk_tris = env_chunk ? atoi(env_chunk) : 3000000;
    if(chunk_tris < 100000) chunk_tris = 100000;

    if(tri_count <= chunk_tris){
        // Single BVH
        if(!gpu_build_bvh_from_tri_vec4_lbv(
                tri_data,
                (uint32_t)tri_count,
                &bvh_nodes,
                &bvh_node_count,
                &bvh_indices,
                &bvh_index_count))
        {
            fprintf(stderr, "[GPU] BVH build failed\n");
            exit(1);
        }
        bvh_root_count = 1;
        bvh_roots = (int32_t*)malloc(sizeof(int32_t));
        bvh_roots[0] = 0;
    } else {
        // Chunked BVH build
        int chunks = (tri_count + chunk_tris - 1) / chunk_tris;
        bvh_root_count = (uint32_t)chunks;
        bvh_roots = (int32_t*)malloc((size_t)chunks * sizeof(int32_t));
        if(!bvh_roots){ fprintf(stderr,"[GPU] OOM roots\n"); exit(1); }

        // First pass: build each chunk, keep temporary arrays
        GPUBVHNode** chunk_nodes = (GPUBVHNode**)calloc((size_t)chunks, sizeof(GPUBVHNode*));
        int32_t**   chunk_idx   = (int32_t**)calloc((size_t)chunks, sizeof(int32_t*));
        uint32_t*   chunk_ncnt  = (uint32_t*)calloc((size_t)chunks, sizeof(uint32_t));
        uint32_t*   chunk_icnt  = (uint32_t*)calloc((size_t)chunks, sizeof(uint32_t));
        if(!chunk_nodes||!chunk_idx||!chunk_ncnt||!chunk_icnt){
            fprintf(stderr,"[GPU] OOM chunk arrays\n"); exit(1);
        }

        uint32_t total_nodes = 0;
        uint32_t total_idx   = 0;

        for(int ci=0; ci<chunks; ci++){
            int start = ci * chunk_tris;
            int count = chunk_tris;
            if(start + count > tri_count) count = tri_count - start;

            const float* tri_ptr = tri_data + (size_t)start * 12u;

            fprintf(stderr, "[GPU] BVH chunk %d/%d: tris=%d (start=%d)\n", ci+1, chunks, count, start);

            if(!gpu_build_bvh_from_tri_vec4_lbv(
                    tri_ptr,
                    (uint32_t)count,
                    &chunk_nodes[ci],
                    &chunk_ncnt[ci],
                    &chunk_idx[ci],
                    &chunk_icnt[ci]))
            {
                fprintf(stderr, "[GPU] BVH build failed on chunk %d\n", ci);
                exit(1);
            }

            total_nodes += chunk_ncnt[ci];
            total_idx   += chunk_icnt[ci];
        }

        // Allocate combined arrays
        bvh_nodes = (GPUBVHNode*)malloc((size_t)total_nodes * sizeof(GPUBVHNode));
        bvh_indices = (int32_t*)malloc((size_t)total_idx * sizeof(int32_t));
        if(!bvh_nodes || !bvh_indices){
            fprintf(stderr,"[GPU] OOM combined BVH arrays\n"); exit(1);
        }

        // Second pass: copy + fixup indices
        uint32_t node_off = 0;
        uint32_t idx_off  = 0;

        for(int ci=0; ci<chunks; ci++){
            int start_tri = ci * chunk_tris;

            // record root for this chunk
            bvh_roots[ci] = (int32_t)node_off;

            // copy nodes and fix child pointers + triOffset
            for(uint32_t n=0; n<chunk_ncnt[ci]; n++){
                GPUBVHNode nd = chunk_nodes[ci][n];

                if(nd.left  >= 0) nd.left  += (int32_t)node_off;
                if(nd.right >= 0) nd.right += (int32_t)node_off;

                nd.triOffset += (int32_t)idx_off;

                bvh_nodes[node_off + n] = nd;
            }

            // copy indices and convert to GLOBAL triangle IDs
            for(uint32_t j=0; j<chunk_icnt[ci]; j++){
                bvh_indices[idx_off + j] = chunk_idx[ci][j] + start_tri;
            }

            node_off += chunk_ncnt[ci];
            idx_off  += chunk_icnt[ci];

            free(chunk_nodes[ci]);
            free(chunk_idx[ci]);
        }

        free(chunk_nodes);
        free(chunk_idx);
        free(chunk_ncnt);
        free(chunk_icnt);

        bvh_node_count = total_nodes;
        bvh_index_count = total_idx;
    }
} else if(use_bvh == 0) {
    // No BVH: keep buffers minimal to avoid huge CPU work.
    bvh_node_count = 1;
    bvh_index_count = 1;
    bvh_nodes = (GPUBVHNode*)calloc(1, sizeof(GPUBVHNode));
    bvh_indices = (int32_t*)calloc(1, sizeof(int32_t));
    bvh_root_count = 1;
    bvh_roots = (int32_t*)malloc(sizeof(int32_t));
    bvh_roots[0] = 0;
}

int node_count = (int)bvh_node_count;
fprintf(stderr, "[GPU] BVH roots=%u nodes=%d indices=%u tris=%d useBVH=%d\n",
        bvh_root_count, node_count, bvh_index_count, tri_count, use_bvh);


    // Save BVH cache if requested (only when we actually built it)
    if(use_bvh != 0 && !cache_hit && cache_path && cache_path[0]){
        if(bvh_cache_save(cache_path, (uint32_t)tri_count,
                          bvh_roots, bvh_root_count,
                          bvh_nodes, bvh_node_count,
                          bvh_indices, bvh_index_count)){
            fprintf(stderr, "[GPU] BVH CACHE SAVED: %s\n", cache_path);
        } else {
            fprintf(stderr, "[GPU] BVH CACHE SAVE FAILED: %s\n", cache_path);
        }
    }

    uint64_t t_bvh1 = ysu_now_us();
    fprintf(stderr, "[GPU] BVH total (cache/load/build) time: %.3f ms%s\n",
            (double)(t_bvh1 - t_bvh0)/1000.0,
            cache_hit ? " (cache hit)" : "");

    
// --- Window/GLFW (optional) ---
GLFWwindow* window = NULL;
const char** inst_exts = NULL;
uint32_t inst_ext_count = 0;

if(window_enabled){
    if(!glfwInit()){
        fprintf(stderr,"[GLFW] glfwInit failed -> headless\n");
        window_enabled = 0;
    }else{
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        window = glfwCreateWindow(W, H, "YSU Vulkan (compute)", NULL, NULL);
        if(!window){
            fprintf(stderr,"[GLFW] glfwCreateWindow failed -> headless\n");
            glfwTerminate();
            window_enabled = 0;
        }else{
            inst_exts = glfwGetRequiredInstanceExtensions(&inst_ext_count);
            if(!inst_exts || !inst_ext_count){
                fprintf(stderr,"[GLFW] glfwGetRequiredInstanceExtensions failed -> headless\n");
                window_enabled = 0;
            }
        }
    }
}

// --- Vulkan instance ---
    VkApplicationInfo app = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.pApplicationName = "YSU GPU Demo";
    app.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ici.pApplicationInfo = &app;

    VkInstance inst = 0;
if(window_enabled){
    ici.enabledExtensionCount = inst_ext_count;
    ici.ppEnabledExtensionNames = inst_exts;
}

    // (validation layers and debug prints removed for production build)

    VkResult r = vkCreateInstance(&ici, NULL, &inst);
    if(r!=VK_SUCCESS) die("vkCreateInstance", r);


    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if(window_enabled){
        r = glfwCreateWindowSurface(inst, window, NULL, &surface);
        if(r!=VK_SUCCESS){
            fprintf(stderr,"[GLFW] glfwCreateWindowSurface failed -> headless\n");
            window_enabled = 0;
            surface = VK_NULL_HANDLE;
        }
    }
    // --- Pick physical device ---
    uint32_t pcount=0;
    r = vkEnumeratePhysicalDevices(inst, &pcount, NULL);
    if(r!=VK_SUCCESS || pcount==0) die("vkEnumeratePhysicalDevices(count)", r);

    VkPhysicalDevice* phys = (VkPhysicalDevice*)malloc(sizeof(VkPhysicalDevice)*pcount);
    r = vkEnumeratePhysicalDevices(inst, &pcount, phys);
    if(r!=VK_SUCCESS) die("vkEnumeratePhysicalDevices(list)", r);
    VkPhysicalDevice phy = phys[0];
    free(phys);

    if(ts_enabled){
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(phy, &props);
        ts_period_ns = (double)props.limits.timestampPeriod;
    }

    // --- Find compute queue family ---
    uint32_t qfcount=0;
    vkGetPhysicalDeviceQueueFamilyProperties(phy, &qfcount, NULL);
    VkQueueFamilyProperties* qfp = (VkQueueFamilyProperties*)malloc(sizeof(VkQueueFamilyProperties)*qfcount);
    vkGetPhysicalDeviceQueueFamilyProperties(phy, &qfcount, qfp);

    // Prefer a queue that can do compute and present when window is enabled.
    uint32_t qfi = 0xFFFFFFFFu;

if(window_enabled && surface!=VK_NULL_HANDLE){
    // Prefer a present-capable graphics queue (ensures transfer/graphics support)
    // 1) graphics+compute+present
    for(uint32_t i=0;i<qfcount;i++){
        VkBool32 present = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(phy, i, surface, &present);
        if(present && (qfp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT)){
            qfi=i; break;
        }
    }
    // 2) graphics+present (fallback to graphics if compute not available)
    if(qfi==0xFFFFFFFFu){
        for(uint32_t i=0;i<qfcount;i++){
            VkBool32 present = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(phy, i, surface, &present);
            if(present && (qfp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)){
                qfi=i; break;
            }
        }
    }
    // 3) compute+present (last resort)
    if(qfi==0xFFFFFFFFu){
        for(uint32_t i=0;i<qfcount;i++){
            VkBool32 present = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(phy, i, surface, &present);
            if(present && (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT)){
                qfi=i; break;
            }
        }
    }
}else{
    // headless: any compute queue is fine (graphics queues also support compute on most GPUs)
    for(uint32_t i=0;i<qfcount;i++){
        if(qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT){ qfi=i; break; }
    }
}

if(qfi==0xFFFFFFFFu){
    // last resort: queue 0
    qfi = 0;
}

// chosen queue family index determined (diagnostic prints removed)

free(qfp);
if(qfi==0xFFFFFFFFu){ fprintf(stderr,"No compute queue\n"); exit(1); }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    qci.queueFamilyIndex = qfi;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;

    VkDeviceCreateInfo dci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;


    const char* dev_exts[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
    if(window_enabled){
        dci.enabledExtensionCount = 1;
        dci.ppEnabledExtensionNames = dev_exts;
    }
    VkDevice dev = 0;
    r = vkCreateDevice(phy, &dci, NULL, &dev);
    if(r!=VK_SUCCESS) die("vkCreateDevice", r);

    VkQueryPool qp_ts = VK_NULL_HANDLE;
    if(ts_enabled){
        VkQueryPoolCreateInfo qpci = { VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
        qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpci.queryCount = 3; // h-begin, h-end, v-end
        r = vkCreateQueryPool(dev, &qpci, NULL, &qp_ts);
        if(r!=VK_SUCCESS){
            fprintf(stderr, "[GPU] timestamp query pool creation failed, disabling timestamps\n");
            qp_ts = VK_NULL_HANDLE;
            ts_enabled = 0;
        } else {
            fprintf(stderr, "[GPU] timestamp query pool created (period=%.2f ns)\n", ts_period_ns);
        }
    } else {
        fprintf(stderr, "[GPU] timestamps disabled\n");
    }

    // --- Swapchain (optional) ---
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
VkFormat sw_format = VK_FORMAT_B8G8R8A8_UNORM;
VkExtent2D sw_extent = { (uint32_t)W, (uint32_t)H };
uint32_t sw_count = 0;
VkImage* sw_images = NULL;
VkImageView* sw_views = NULL;

if(window_enabled && surface!=VK_NULL_HANDLE){
    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(phy, surface, &caps);

    uint32_t fmt_count = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(phy, surface, &fmt_count, NULL);
    VkSurfaceFormatKHR* fmts = (VkSurfaceFormatKHR*)calloc(fmt_count ? fmt_count : 1, sizeof(VkSurfaceFormatKHR));
    if(fmt_count) vkGetPhysicalDeviceSurfaceFormatsKHR(phy, surface, &fmt_count, fmts);

    VkSurfaceFormatKHR chosen = { VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
    if(fmt_count){
        chosen = fmts[0];
        for(uint32_t i=0;i<fmt_count;i++){
            if(fmts[i].format==VK_FORMAT_B8G8R8A8_UNORM || fmts[i].format==VK_FORMAT_R8G8B8A8_UNORM){
                chosen = fmts[i]; break;
            }
        }
    }
    free(fmts);
    sw_format = chosen.format;

    uint32_t pm_count = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(phy, surface, &pm_count, NULL);
    VkPresentModeKHR* pms = (VkPresentModeKHR*)calloc(pm_count ? pm_count : 1, sizeof(VkPresentModeKHR));
    if(pm_count) vkGetPhysicalDeviceSurfacePresentModesKHR(phy, surface, &pm_count, pms);
    VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR; // always available
    for(uint32_t i=0;i<pm_count;i++){
        if(pms[i]==VK_PRESENT_MODE_MAILBOX_KHR){ present_mode = pms[i]; break; }
    }
    free(pms);

    // extent
    if(caps.currentExtent.width != 0xFFFFFFFFu){
        sw_extent = caps.currentExtent;
    }else{
        sw_extent.width  = (uint32_t)W;
        sw_extent.height = (uint32_t)H;
        if(sw_extent.width  < caps.minImageExtent.width)  sw_extent.width  = caps.minImageExtent.width;
        if(sw_extent.height < caps.minImageExtent.height) sw_extent.height = caps.minImageExtent.height;
        if(sw_extent.width  > caps.maxImageExtent.width)  sw_extent.width  = caps.maxImageExtent.width;
        if(sw_extent.height > caps.maxImageExtent.height) sw_extent.height = caps.maxImageExtent.height;
    }

    uint32_t desired = 2;
    if(desired < caps.minImageCount) desired = caps.minImageCount;
    if(caps.maxImageCount && desired > caps.maxImageCount) desired = caps.maxImageCount;

    VkSwapchainCreateInfoKHR sci = { VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
    sci.surface = surface;
    sci.minImageCount = desired;
    sci.imageFormat = sw_format;
    sci.imageColorSpace = chosen.colorSpace;
    sci.imageExtent = sw_extent;
    sci.imageArrayLayers = 1;
    // allow transfer->present and sampling/attachment use where needed
    sci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    sci.preTransform = caps.currentTransform;
    sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sci.presentMode = present_mode;
    sci.clipped = VK_TRUE;

    r = vkCreateSwapchainKHR(dev, &sci, NULL, &swapchain);
    if(r!=VK_SUCCESS) die("vkCreateSwapchainKHR", r);

    vkGetSwapchainImagesKHR(dev, swapchain, &sw_count, NULL);
    sw_images = (VkImage*)calloc(sw_count, sizeof(VkImage));
    vkGetSwapchainImagesKHR(dev, swapchain, &sw_count, sw_images);

    sw_views = (VkImageView*)calloc(sw_count, sizeof(VkImageView));
    for(uint32_t i=0;i<sw_count;i++){
        VkImageViewCreateInfo vci = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        vci.image = sw_images[i];
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = sw_format;
        vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vci.subresourceRange.levelCount = 1;
        vci.subresourceRange.layerCount = 1;
        r = vkCreateImageView(dev, &vci, NULL, &sw_views[i]);
        if(r!=VK_SUCCESS) die("vkCreateImageView", r);
    }
    // If running with a window, prefer to run compute at the swapchain extent
    if(window_enabled && swapchain!=VK_NULL_HANDLE){
        W = (int)sw_extent.width;
        H = (int)sw_extent.height;
        fprintf(stderr, "[GPU] Window mode: using swapchain extent %dx%d for compute\n", W, H);
    }
    }

    VkQueue queue = 0;
    vkGetDeviceQueue(dev, qfi, 0, &queue);

    VkCommandPool pool = create_cmd_pool(dev, qfi);
    VkCommandBuffer cb = alloc_cmd(dev, pool);

    // ---------- Create storage images (out + accum) ----------
    VkFormat fmt = VK_FORMAT_R32G32B32A32_SFLOAT;

    VkImage out_img = create_image(dev, W, H, fmt,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
    );
    VkDeviceMemory out_mem = alloc_bind_image_mem(phy, dev, out_img, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkImage accum_img = create_image(dev, W, H, fmt,
        VK_IMAGE_USAGE_STORAGE_BIT
    );
    VkDeviceMemory accum_mem = alloc_bind_image_mem(phy, dev, accum_img, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // ---------- Optional LDR image for GPU tonemap ----------
    VkImage ldr_img = VK_NULL_HANDLE;
    VkDeviceMemory ldr_mem = VK_NULL_HANDLE;
    VkImageView ldr_view = VK_NULL_HANDLE;
    VkFormat ldr_fmt = VK_FORMAT_R8G8B8A8_UNORM;  // Always use RGBA8 (shader expects this, not BGRA8)
    if(tonemap_enabled){
        ldr_img = create_image(dev, W, H, ldr_fmt,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
        );
        ldr_mem = alloc_bind_image_mem(phy, dev, ldr_img, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    }

    VkImageViewCreateInfo ivci = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    ivci.format = fmt;
    ivci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    ivci.subresourceRange.levelCount = 1;
    ivci.subresourceRange.layerCount = 1;

    ivci.image = out_img;
    VkImageView out_view = 0;
    r = vkCreateImageView(dev, &ivci, NULL, &out_view);
    if(r!=VK_SUCCESS) die("vkCreateImageView(out)", r);

    ivci.image = accum_img;
    VkImageView accum_view = 0;
    r = vkCreateImageView(dev, &ivci, NULL, &accum_view);
    if(r!=VK_SUCCESS) die("vkCreateImageView(accum)", r);

    if(tonemap_enabled){
        ivci.image = ldr_img;
        ivci.format = ldr_fmt; // ldr image uses ldr_fmt (may be swapchain format)
        r = vkCreateImageView(dev, &ivci, NULL, &ldr_view);
        if(r!=VK_SUCCESS) die("vkCreateImageView(ldr)", r);
    }

    // ---------- Triangle buffer (binding=2) ----------
    VkDeviceSize tri_bytes = (VkDeviceSize)((size_t)tri_count * 3u * 4u * sizeof(float));
    if(tri_bytes == 0){
        tri_count = 1;
        tri_bytes = 16;
        tri_data = (float*)calloc(1, (size_t)tri_bytes);
    }

    VkBuffer tri_buf = create_buffer(dev, tri_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    VkDeviceMemory tri_mem = alloc_bind_buffer_mem(
        phy, dev, tri_buf,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    void* tri_map = NULL;
    r = vkMapMemory(dev, tri_mem, 0, tri_bytes, 0, &tri_map);
    if(r!=VK_SUCCESS) die("vkMapMemory(tri)", r);
    if(tri_count>0 && tri_data) memcpy(tri_map, tri_data, (size_t)tri_bytes);
    vkUnmapMemory(dev, tri_mem);
    if(tri_data){ free(tri_data); tri_data=NULL; }

    // ---------- BVH buffer (binding=3) ----------
    VkDeviceSize bvh_bytes = (VkDeviceSize)sizeof(GPUBVHNode) * (VkDeviceSize)node_count;
    VkBuffer bvh_buf = create_buffer(dev, bvh_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    VkDeviceMemory bvh_mem = alloc_bind_buffer_mem(
        phy, dev, bvh_buf,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    void* bvh_map = NULL;
    r = vkMapMemory(dev, bvh_mem, 0, bvh_bytes, 0, &bvh_map);
    if(r!=VK_SUCCESS) die("vkMapMemory(bvh)", r);
    memcpy(bvh_map, bvh_nodes, (size_t)bvh_bytes);
    vkUnmapMemory(dev, bvh_mem);

    // ---------- Triangle index buffer (binding=4) ----------
    // bvh_indices contains tri indices in the BVH leaf order. Size = tri_count (int32).
    VkDeviceSize idx_bytes = (VkDeviceSize)sizeof(int32_t) * (VkDeviceSize)bvh_index_count;
    if(idx_bytes == 0) idx_bytes = 4;

    VkBuffer idx_buf = create_buffer(dev, idx_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    VkDeviceMemory idx_mem = alloc_bind_buffer_mem(
        phy, dev, idx_buf,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    void* idx_map = NULL;
    r = vkMapMemory(dev, idx_mem, 0, idx_bytes, 0, &idx_map);
    if(r!=VK_SUCCESS) die("vkMapMemory(idx)", r);
    if(bvh_indices && bvh_index_count) memcpy(idx_map, bvh_indices, (size_t)idx_bytes);
    vkUnmapMemory(dev, idx_mem);

    // ---------- Counters buffer (binding=5) ----------
    // counters[0] = nodeVisits, counters[1] = triTests
    VkDeviceSize ctr_bytes = 8;
    VkBuffer ctr_buf = create_buffer(dev, ctr_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    VkDeviceMemory ctr_mem = alloc_bind_buffer_mem(
        phy, dev, ctr_buf,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    // init counters to 0
    void* ctr_map = NULL;
    r = vkMapMemory(dev, ctr_mem, 0, ctr_bytes, 0, &ctr_map);
    if(r!=VK_SUCCESS) die("vkMapMemory(ctr init)", r);
    memset(ctr_map, 0, (size_t)ctr_bytes);
    vkUnmapMemory(dev, ctr_mem);

    
    // ---------- BVH roots buffer (binding=6) ----------
    // Contains one int per BVH root (for chunked BVH builds).
    if(bvh_root_count == 0){ bvh_root_count = 1; }
    VkDeviceSize roots_bytes = (VkDeviceSize)sizeof(int32_t) * (VkDeviceSize)bvh_root_count;

    VkBuffer roots_buf = create_buffer(dev, roots_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    VkDeviceMemory roots_mem = alloc_bind_buffer_mem(
        phy, dev, roots_buf,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    void* roots_map = NULL;
    r = vkMapMemory(dev, roots_mem, 0, roots_bytes, 0, &roots_map);
    if(r!=VK_SUCCESS) die("vkMapMemory(roots)", r);
    if(bvh_roots) memcpy(roots_map, bvh_roots, (size_t)roots_bytes);
    else memset(roots_map, 0, (size_t)roots_bytes);
    vkUnmapMemory(dev, roots_mem);
    
    // ---------- NeRF buffers (hashgrid + occupancy) ----------
    VkDeviceSize nerf_hash_bytes = (nerf_hash.data && nerf_hash.bytes > 0) ? (VkDeviceSize)nerf_hash.bytes : 4;
    VkDeviceSize nerf_occ_bytes = (nerf_occ.data && nerf_occ.bytes > 0) ? (VkDeviceSize)nerf_occ.bytes : 4;
    
    VkBuffer nerf_hash_buf = create_buffer(dev, nerf_hash_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    VkDeviceMemory nerf_hash_mem = alloc_bind_buffer_mem(
        phy, dev, nerf_hash_buf,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    
    VkBuffer nerf_occ_buf = create_buffer(dev, nerf_occ_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    VkDeviceMemory nerf_occ_mem = alloc_bind_buffer_mem(
        phy, dev, nerf_occ_buf,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    
    // Upload blobs (or zeros if missing)
    void* nerf_hash_map = NULL;
    r = vkMapMemory(dev, nerf_hash_mem, 0, nerf_hash_bytes, 0, &nerf_hash_map);
    if(r!=VK_SUCCESS) die("vkMapMemory(nerf_hash)", r);
    if(nerf_hash.data && nerf_hash.bytes > 0){
        memcpy(nerf_hash_map, nerf_hash.data, (size_t)nerf_hash_bytes);
    } else {
        memset(nerf_hash_map, 0, (size_t)nerf_hash_bytes);
    }
    vkUnmapMemory(dev, nerf_hash_mem);
    
    void* nerf_occ_map = NULL;
    r = vkMapMemory(dev, nerf_occ_mem, 0, nerf_occ_bytes, 0, &nerf_occ_map);
    if(r!=VK_SUCCESS) die("vkMapMemory(nerf_occ)", r);
    if(nerf_occ.data && nerf_occ.bytes > 0){
        memcpy(nerf_occ_map, nerf_occ.data, (size_t)nerf_occ_bytes);
    } else {
        memset(nerf_occ_map, 0, (size_t)nerf_occ_bytes);
    }
    vkUnmapMemory(dev, nerf_occ_mem);

    // ---------- Depth hints buffer (binding=10) for CPU-guided sparse sampling ----------
    // Create a full-size buffer for per-pixel depth hints
    // Each hint is vec4: (depth, delta, confidence, flags)
    VkDeviceSize depth_hints_bytes = (VkDeviceSize)(W * H * 16); // 16 bytes per pixel (vec4)
    VkBuffer depth_hints_buf = create_buffer(dev, depth_hints_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    VkDeviceMemory depth_hints_mem = alloc_bind_buffer_mem(
        phy, dev, depth_hints_buf,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    // Initialize with zeros (fallback to full-range sampling)
    void* depth_hints_map = NULL;
    r = vkMapMemory(dev, depth_hints_mem, 0, depth_hints_bytes, 0, &depth_hints_map);
    if(r!=VK_SUCCESS) die("vkMapMemory(depth_hints)", r);
    memset(depth_hints_map, 0, (size_t)depth_hints_bytes);
    // Keep mapped for per-frame updates
    // vkUnmapMemory(dev, depth_hints_mem); // Keep mapped!

    // We can free CPU BVH arrays after upload
    if(bvh_nodes){ free(bvh_nodes); bvh_nodes = NULL; }
    if(bvh_indices){ free(bvh_indices); bvh_indices = NULL; }
    if(bvh_roots){ free(bvh_roots); bvh_roots = NULL; }

    // ---------- Camera uniform buffer (binding=7) ----------
    VkDeviceSize cam_bytes = (VkDeviceSize)sizeof(CameraUBO);
    VkBuffer cam_ubo = create_buffer(dev, cam_bytes, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    VkDeviceMemory cam_ubo_mem = alloc_bind_buffer_mem(
        phy, dev, cam_ubo,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    
    void* cam_ubo_mapped = NULL;
    r = vkMapMemory(dev, cam_ubo_mem, 0, cam_bytes, 0, &cam_ubo_mapped);
    if(r!=VK_SUCCESS) die("vkMapMemory(cam_ubo)", r);
    // Camera will be updated per-frame, so leave mapped


    // ---------- Descriptor set layout: binding0 outImg, binding1 accumImg, binding2 triBuf, binding3 bvhBuf, binding7 camUBO, binding10 depthHints ----------
    VkDescriptorSetLayoutBinding binds[11];
    memset(binds, 0, sizeof(binds));

    binds[0].binding = 0;
    binds[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    binds[0].descriptorCount = 1;
    binds[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    binds[1].binding = 1;
    binds[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    binds[1].descriptorCount = 1;
    binds[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    binds[2].binding = 2;
    binds[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binds[2].descriptorCount = 1;
    binds[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    binds[3].binding = 3;
    binds[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binds[3].descriptorCount = 1;
    binds[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    binds[4].binding = 4;
    binds[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binds[4].descriptorCount = 1;
    binds[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    binds[5].binding = 5;
    binds[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binds[5].descriptorCount = 1;
    binds[5].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    binds[6].binding = 6;
    binds[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binds[6].descriptorCount = 1;
    binds[6].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    
    binds[7].binding = 7;
    binds[7].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    binds[7].descriptorCount = 1;
    binds[7].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    binds[8].binding = 8;
    binds[8].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binds[8].descriptorCount = 1;
    binds[8].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    binds[9].binding = 9;
    binds[9].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binds[9].descriptorCount = 1;
    binds[9].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    // binding 10: depth hints buffer for CPU-guided sparse sampling
    binds[10].binding = 10;
    binds[10].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binds[10].descriptorCount = 1;
    binds[10].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo dlci = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    dlci.bindingCount = 11;
    dlci.pBindings = binds;

    VkDescriptorSetLayout dsl = 0;
    r = vkCreateDescriptorSetLayout(dev, &dlci, NULL, &dsl);
    if(r!=VK_SUCCESS) die("vkCreateDescriptorSetLayout", r);

    VkDescriptorPoolSize dps[3];
    dps[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    dps[0].descriptorCount = 2;
    dps[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    dps[1].descriptorCount = 8; // tri + bvh + idx + counters + roots + nerf hash + nerf occ + depth hints
    dps[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    dps[2].descriptorCount = 1; // camera UBO

    VkDescriptorPoolCreateInfo dpci = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    dpci.maxSets = 1;
    dpci.poolSizeCount = 3;
    dpci.pPoolSizes = dps;

    VkDescriptorPool dp = 0;
    r = vkCreateDescriptorPool(dev, &dpci, NULL, &dp);
    if(r!=VK_SUCCESS) die("vkCreateDescriptorPool", r);

    VkDescriptorSetAllocateInfo dsai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dsai.descriptorPool = dp;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &dsl;

    VkDescriptorSet ds = 0;
    r = vkAllocateDescriptorSets(dev, &dsai, &ds);
    if(r!=VK_SUCCESS) die("vkAllocateDescriptorSets", r);

    VkDescriptorImageInfo dii0 = {0};
    dii0.imageView = out_view;
    dii0.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorImageInfo dii1 = {0};
    dii1.imageView = accum_view;
    dii1.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorBufferInfo dbi_tri = {0};
    dbi_tri.buffer = tri_buf;
    dbi_tri.offset = 0;
    dbi_tri.range  = tri_bytes;

    VkDescriptorBufferInfo dbi_bvh = {0};
    dbi_bvh.buffer = bvh_buf;
    dbi_bvh.offset = 0;
    dbi_bvh.range  = bvh_bytes;

    VkDescriptorBufferInfo dbi_idx = {0};
    dbi_idx.buffer = idx_buf;
    dbi_idx.offset = 0;
    dbi_idx.range  = idx_bytes;

    VkDescriptorBufferInfo dbi_ctr = {0};
    dbi_ctr.buffer = ctr_buf;
    dbi_ctr.offset = 0;
    dbi_ctr.range  = ctr_bytes;

    VkDescriptorBufferInfo dbi_roots = {0};
    dbi_roots.buffer = roots_buf;
    dbi_roots.offset = 0;
    dbi_roots.range  = roots_bytes;
    
    VkDescriptorBufferInfo dbi_cam = {0};
    dbi_cam.buffer = cam_ubo;
    dbi_cam.offset = 0;
    dbi_cam.range  = cam_bytes;

    VkDescriptorBufferInfo dbi_nerf_hash = {0};
    dbi_nerf_hash.buffer = nerf_hash_buf;
    dbi_nerf_hash.offset = 0;
    dbi_nerf_hash.range  = nerf_hash_bytes;

    VkDescriptorBufferInfo dbi_nerf_occ = {0};
    dbi_nerf_occ.buffer = nerf_occ_buf;
    dbi_nerf_occ.offset = 0;
    dbi_nerf_occ.range  = nerf_occ_bytes;

    VkDescriptorBufferInfo dbi_depth_hints = {0};
    dbi_depth_hints.buffer = depth_hints_buf;
    dbi_depth_hints.offset = 0;
    dbi_depth_hints.range  = depth_hints_bytes;

    VkWriteDescriptorSet ws[11];
    memset(ws, 0, sizeof(ws));

    ws[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    ws[0].dstSet = ds;
    ws[0].dstBinding = 0;
    ws[0].descriptorCount = 1;
    ws[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    ws[0].pImageInfo = &dii0;

    ws[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    ws[1].dstSet = ds;
    ws[1].dstBinding = 1;
    ws[1].descriptorCount = 1;
    ws[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    ws[1].pImageInfo = &dii1;

    ws[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    ws[2].dstSet = ds;
    ws[2].dstBinding = 2;
    ws[2].descriptorCount = 1;
    ws[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ws[2].pBufferInfo = &dbi_tri;

    ws[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    ws[3].dstSet = ds;
    ws[3].dstBinding = 3;
    ws[3].descriptorCount = 1;
    ws[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ws[3].pBufferInfo = &dbi_bvh;


    ws[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    ws[4].dstSet = ds;
    ws[4].dstBinding = 4;
    ws[4].descriptorCount = 1;
    ws[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ws[4].pBufferInfo = &dbi_idx;

    ws[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    ws[5].dstSet = ds;
    ws[5].dstBinding = 5;
    ws[5].descriptorCount = 1;
    ws[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ws[5].pBufferInfo = &dbi_ctr;

    ws[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    ws[6].dstSet = ds;
    ws[6].dstBinding = 6;
    ws[6].descriptorCount = 1;
    ws[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ws[6].pBufferInfo = &dbi_roots;
    
    ws[7].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    ws[7].dstSet = ds;
    ws[7].dstBinding = 7;
    ws[7].descriptorCount = 1;
    ws[7].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    ws[7].pBufferInfo = &dbi_cam;

    ws[8].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    ws[8].dstSet = ds;
    ws[8].dstBinding = 8;
    ws[8].descriptorCount = 1;
    ws[8].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ws[8].pBufferInfo = &dbi_nerf_hash;

    ws[9].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    ws[9].dstSet = ds;
    ws[9].dstBinding = 9;
    ws[9].descriptorCount = 1;
    ws[9].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ws[9].pBufferInfo = &dbi_nerf_occ;

    ws[10].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    ws[10].dstSet = ds;
    ws[10].dstBinding = 10;
    ws[10].descriptorCount = 1;
    ws[10].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ws[10].pBufferInfo = &dbi_depth_hints;

    vkUpdateDescriptorSets(dev, 11, ws, 0, NULL);

    // ---------- Pipeline layout + push constants ----------
    VkPushConstantRange pcr = {0};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = (uint32_t)sizeof(PushConstants); // camera now in uniform buffer

    VkPipelineLayoutCreateInfo plci = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;

    VkPipelineLayout pl = 0;
    r = vkCreatePipelineLayout(dev, &plci, NULL, &pl);
    if(r!=VK_SUCCESS) die("vkCreatePipelineLayout", r);

    // ---------- Shader module ----------
    // Try multiple paths to find the shader
    const char* spv_paths[] = {
        "shaders/tri.comp.spv",
        "../shaders/tri.comp.spv",
        "./shaders/tri.comp.spv"
    };
    uint8_t* spv = NULL;
    size_t spv_sz = 0;
    for(int i=0; i<3; i++){
        spv = read_file(spv_paths[i], &spv_sz);
        if(spv) break;
    }
    if(!spv){
        fprintf(stderr, "Cannot read shaders/tri.comp.spv from any search path\n");
        exit(1);
    }

    VkShaderModuleCreateInfo smci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smci.codeSize = spv_sz;
    smci.pCode = (const uint32_t*)spv;

    VkShaderModule sm = 0;
    r = vkCreateShaderModule(dev, &smci, NULL, &sm);
    if(r!=VK_SUCCESS) die("vkCreateShaderModule", r);
    free(spv);

    VkPipelineShaderStageCreateInfo ssci = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    ssci.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ssci.module = sm;
    ssci.pName = "main";

    VkComputePipelineCreateInfo cpci = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpci.stage = ssci;
    cpci.layout = pl;

    VkPipeline pipe = 0;
    r = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe);
    if(r!=VK_SUCCESS) die("vkCreateComputePipelines", r);


    // ---------- Optional tonemap compute pipeline (HDR->LDR RGBA8) ----------
    VkShaderModule sm_tm = VK_NULL_HANDLE;
    VkPipelineLayout pl_tm = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl_tm = VK_NULL_HANDLE;
    VkDescriptorPool dp_tm = VK_NULL_HANDLE;
    VkDescriptorSet ds_tm = VK_NULL_HANDLE;
    VkPipeline pipe_tm = VK_NULL_HANDLE;

    if(tonemap_enabled){
        // Try multiple paths for tonemap shader
        const char* tm_spv_paths[] = {
            "shaders/tonemap.comp.spv",
            "../shaders/tonemap.comp.spv",
            "./shaders/tonemap.comp.spv"
        };
        uint8_t* tm_spv = NULL;
        size_t tm_spv_sz = 0;
        for(int i=0; i<3; i++){
            tm_spv = read_file(tm_spv_paths[i], &tm_spv_sz);
            if(tm_spv) break;
        }
        if(!tm_spv){
            fprintf(stderr, "Cannot read shaders/tonemap.comp.spv from any search path\n");
            exit(1);
        }
        VkShaderModuleCreateInfo smci_tm = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        smci_tm.codeSize = tm_spv_sz;
        smci_tm.pCode = (const uint32_t*)tm_spv;
        r = vkCreateShaderModule(dev, &smci_tm, NULL, &sm_tm);
        if(r!=VK_SUCCESS) die("vkCreateShaderModule(tonemap)", r);
        free(tm_spv);

        VkDescriptorSetLayoutBinding b_tm[2];
        memset(b_tm, 0, sizeof(b_tm));
        b_tm[0].binding = 0;
        b_tm[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        b_tm[0].descriptorCount = 1;
        b_tm[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        b_tm[1].binding = 1;
        b_tm[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        b_tm[1].descriptorCount = 1;
        b_tm[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo dslci_tm = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        dslci_tm.bindingCount = 2;
        dslci_tm.pBindings = b_tm;
        r = vkCreateDescriptorSetLayout(dev, &dslci_tm, NULL, &dsl_tm);
        if(r!=VK_SUCCESS) die("vkCreateDescriptorSetLayout(tonemap)", r);

        VkPushConstantRange pcr_tm = {0};
        pcr_tm.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr_tm.offset = 0;
        pcr_tm.size = 16; // int W,H + float exposure,gamma

        VkPipelineLayoutCreateInfo plci_tm = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        plci_tm.setLayoutCount = 1;
        plci_tm.pSetLayouts = &dsl_tm;
        plci_tm.pushConstantRangeCount = 1;
        // ensure push constant range is large enough for tonemap shader (avoid VUID errors)
        pcr_tm.size = 32;
        plci_tm.pPushConstantRanges = &pcr_tm;
        r = vkCreatePipelineLayout(dev, &plci_tm, NULL, &pl_tm);
        if(r!=VK_SUCCESS) die("vkCreatePipelineLayout(tonemap)", r);

        VkDescriptorPoolSize ps_tm = { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 };
        VkDescriptorPoolCreateInfo dpci_tm = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        dpci_tm.maxSets = 1;
        dpci_tm.poolSizeCount = 1;
        dpci_tm.pPoolSizes = &ps_tm;
        r = vkCreateDescriptorPool(dev, &dpci_tm, NULL, &dp_tm);
        if(r!=VK_SUCCESS) die("vkCreateDescriptorPool(tonemap)", r);

        VkDescriptorSetAllocateInfo dsai_tm = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        dsai_tm.descriptorPool = dp_tm;
        dsai_tm.descriptorSetCount = 1;
        dsai_tm.pSetLayouts = &dsl_tm;
        r = vkAllocateDescriptorSets(dev, &dsai_tm, &ds_tm);
        if(r!=VK_SUCCESS) die("vkAllocateDescriptorSets(tonemap)", r);

        VkDescriptorImageInfo di_hdr = {0};
        di_hdr.imageView = out_view;
        di_hdr.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo di_ldr = {0};
        di_ldr.imageView = ldr_view;
        di_ldr.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet wtm[2];
        memset(wtm, 0, sizeof(wtm));
        wtm[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wtm[0].dstSet = ds_tm;
        wtm[0].dstBinding = 0;
        wtm[0].descriptorCount = 1;
        wtm[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        wtm[0].pImageInfo = &di_hdr;

        wtm[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wtm[1].dstSet = ds_tm;
        wtm[1].dstBinding = 1;
        wtm[1].descriptorCount = 1;
        wtm[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        wtm[1].pImageInfo = &di_ldr;

        vkUpdateDescriptorSets(dev, 2, wtm, 0, NULL);

        VkPipelineShaderStageCreateInfo ss_tm = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
        ss_tm.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ss_tm.module = sm_tm;
        ss_tm.pName = "main";

        VkComputePipelineCreateInfo cpci_tm = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        cpci_tm.stage = ss_tm;
        cpci_tm.layout = pl_tm;
        r = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci_tm, NULL, &pipe_tm);
        if(r!=VK_SUCCESS) die("vkCreateComputePipelines(tonemap)", r);
    }

    // ---------- Optional GPU denoiser (separable bilateral filter) ----------
    VkShaderModule sm_denoise = VK_NULL_HANDLE;
    VkShaderModule sm_blend = VK_NULL_HANDLE;  // temporal blend shader
    VkPipelineLayout pl_denoise = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl_denoise = VK_NULL_HANDLE;
    VkDescriptorPool dp_denoise = VK_NULL_HANDLE;
    VkDescriptorSet ds_denoise_h = VK_NULL_HANDLE;  // horizontal pass
    VkDescriptorSet ds_denoise_v = VK_NULL_HANDLE;  // vertical pass
    VkDescriptorPool dp_blend = VK_NULL_HANDLE;     // temporal blend descriptor pool
    VkDescriptorSet ds_blend = VK_NULL_HANDLE;      // temporal blend descriptor set
    VkImage denoise_temp = VK_NULL_HANDLE;          // temporary image for 2-pass filter
    VkDeviceMemory denoise_temp_mem = VK_NULL_HANDLE;
    VkImageView denoise_temp_view = VK_NULL_HANDLE;
    VkImage denoise_history = VK_NULL_HANDLE;       // previous frame denoised output (for temporal blend)
    VkDeviceMemory denoise_history_mem = VK_NULL_HANDLE;
    VkImageView denoise_history_view = VK_NULL_HANDLE;
    VkPipeline pipe_denoise = VK_NULL_HANDLE;
    VkPipeline pipe_blend = VK_NULL_HANDLE;  // temporal blend pipeline
    
    int gpu_denoise_enabled = ysu_env_bool("YSU_GPU_DENOISE", 0);
    int denoise_radius = ysu_env_int("YSU_GPU_DENOISE_RADIUS", fast_mode ? 1 : 3);
    float denoise_sigma_s = ysu_env_float("YSU_GPU_DENOISE_SIGMA_S", fast_mode ? 0.8f : 1.5f);
    float denoise_sigma_r = ysu_env_float("YSU_GPU_DENOISE_SIGMA_R", fast_mode ? 0.05f : 0.1f);
    int denoise_skip = ysu_env_int("YSU_GPU_DENOISE_SKIP", 1);  // denoise every Nth frame (1=every frame, 2=every 2nd, etc.)
    int denoise_history_reset = ysu_env_bool("YSU_GPU_DENOISE_HISTORY_RESET", 0);  // reset history buffer periodically
    int denoise_history_reset_frame = ysu_env_int("YSU_GPU_DENOISE_HISTORY_RESET_FRAME", 60);  // reset history every N frames
    int adaptive_denoise_enabled = ysu_env_bool("YSU_GPU_DENOISE_ADAPTIVE", 0);  // adjust denoise_skip based on frame variance
    int adaptive_denoise_min = ysu_env_int("YSU_GPU_DENOISE_ADAPTIVE_MIN", 1);  // minimum denoise_skip in low-noise regions
    int adaptive_denoise_max = ysu_env_int("YSU_GPU_DENOISE_ADAPTIVE_MAX", 8);  // maximum denoise_skip in high-noise regions
    int temporal_denoise_enabled = ysu_env_bool("YSU_GPU_TEMPORAL_DENOISE", 1);  // blend with previous frame denoised output
    float temporal_denoise_weight = ysu_env_float("YSU_GPU_TEMPORAL_DENOISE_WEIGHT", 0.7f);  // 0.7 = 70% history, 30% current
    int cpu_denoise_enabled = ysu_env_bool("YSU_NEURAL_DENOISE", 0);
    
    // Temporal accumulation: skip readback, blend frames on GPU
    int temporal_enabled = ysu_env_bool("YSU_GPU_TEMPORAL", 1);  // default ON for ~60% speedup
    int readback_skip = ysu_env_int("YSU_GPU_READBACK_SKIP", 4); // readback every Nth frame
    if(temporal_enabled){
        readback_skip = ysu_env_int("YSU_GPU_READBACK_SKIP", readback_skip);
        fprintf(stderr, "[GPU] Temporal mode: ENABLED (readback every %d frames)\n", readback_skip);
    }
    
    if(gpu_denoise_enabled && !cpu_denoise_enabled){
        // Only use GPU denoiser if CPU denoiser is disabled (avoid double denoising)
        fprintf(stderr, "[GPU] GPU denoiser: ENABLED (radius=%d sigma_s=%.2f sigma_r=%.4f skip=%d)\n", 
                denoise_radius, denoise_sigma_s, denoise_sigma_r, denoise_skip);
        if(denoise_history_reset) {
            fprintf(stderr, "[GPU]   History reset: ENABLED (every %d frames)\n", denoise_history_reset_frame);
        }
        if(adaptive_denoise_enabled) {
            fprintf(stderr, "[GPU]   Adaptive denoise: ENABLED (skip range %d-%d based on variance)\n", 
                    adaptive_denoise_min, adaptive_denoise_max);
        }
        if(temporal_denoise_enabled) {
            fprintf(stderr, "[GPU]   Temporal denoising: ENABLED (weight=%.2f, blend history with current)\n", temporal_denoise_weight);
        }
        
        // Create temporary intermediate image for separable filter
        VkImageCreateInfo denoise_ici = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        denoise_ici.imageType = VK_IMAGE_TYPE_2D;
        denoise_ici.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        denoise_ici.extent.width = (uint32_t)W;
        denoise_ici.extent.height = (uint32_t)H;
        denoise_ici.extent.depth = 1;
        denoise_ici.mipLevels = 1;
        denoise_ici.arrayLayers = 1;
        denoise_ici.samples = VK_SAMPLE_COUNT_1_BIT;
        denoise_ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        denoise_ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        denoise_ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        denoise_ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        
        r = vkCreateImage(dev, &denoise_ici, NULL, &denoise_temp);
        if(r!=VK_SUCCESS) die("vkCreateImage(denoise_temp)", r);
        
        VkMemoryRequirements denoise_mr;
        vkGetImageMemoryRequirements(dev, denoise_temp, &denoise_mr);
        
        // Find suitable memory type for denoise temp image
        VkPhysicalDeviceMemoryProperties mp_denoise;
        vkGetPhysicalDeviceMemoryProperties(phy, &mp_denoise);
        
        int mem_type_idx = 0;
        for(uint32_t i=0; i<mp_denoise.memoryTypeCount; i++){
            if((denoise_mr.memoryTypeBits & (1u<<i)) && (mp_denoise.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)){
                mem_type_idx = (int)i;
                break;
            }
        }
        
        VkMemoryAllocateInfo denoise_mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        denoise_mai.allocationSize = denoise_mr.size;
        denoise_mai.memoryTypeIndex = (uint32_t)mem_type_idx;
        r = vkAllocateMemory(dev, &denoise_mai, NULL, &denoise_temp_mem);
        if(r!=VK_SUCCESS) die("vkAllocateMemory(denoise_temp)", r);
        
        r = vkBindImageMemory(dev, denoise_temp, denoise_temp_mem, 0);
        if(r!=VK_SUCCESS) die("vkBindImageMemory(denoise_temp)", r);
        
        VkImageViewCreateInfo denoise_ivci = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        denoise_ivci.image = denoise_temp;
        denoise_ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        denoise_ivci.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        denoise_ivci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        denoise_ivci.subresourceRange.baseMipLevel = 0;
        denoise_ivci.subresourceRange.levelCount = 1;
        denoise_ivci.subresourceRange.baseArrayLayer = 0;
        denoise_ivci.subresourceRange.layerCount = 1;
        r = vkCreateImageView(dev, &denoise_ivci, NULL, &denoise_temp_view);
        if(r!=VK_SUCCESS) die("vkCreateImageView(denoise_temp)", r);
        
        // Create denoise history image (previous frame denoised output for temporal blending)
        if(temporal_denoise_enabled) {
            VkImageCreateInfo denoise_hist_ici = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
            denoise_hist_ici.imageType = VK_IMAGE_TYPE_2D;
            denoise_hist_ici.format = VK_FORMAT_R32G32B32A32_SFLOAT;
            denoise_hist_ici.extent.width = (uint32_t)W;
            denoise_hist_ici.extent.height = (uint32_t)H;
            denoise_hist_ici.extent.depth = 1;
            denoise_hist_ici.mipLevels = 1;
            denoise_hist_ici.arrayLayers = 1;
            denoise_hist_ici.samples = VK_SAMPLE_COUNT_1_BIT;
            denoise_hist_ici.tiling = VK_IMAGE_TILING_OPTIMAL;
            denoise_hist_ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            denoise_hist_ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            denoise_hist_ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            
            r = vkCreateImage(dev, &denoise_hist_ici, NULL, &denoise_history);
            if(r!=VK_SUCCESS) die("vkCreateImage(denoise_history)", r);
            
            VkMemoryRequirements denoise_hist_mr;
            vkGetImageMemoryRequirements(dev, denoise_history, &denoise_hist_mr);
            
            VkMemoryAllocateInfo denoise_hist_mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
            denoise_hist_mai.allocationSize = denoise_hist_mr.size;
            denoise_hist_mai.memoryTypeIndex = (uint32_t)mem_type_idx;
            r = vkAllocateMemory(dev, &denoise_hist_mai, NULL, &denoise_history_mem);
            if(r!=VK_SUCCESS) die("vkAllocateMemory(denoise_history)", r);
            
            r = vkBindImageMemory(dev, denoise_history, denoise_history_mem, 0);
            if(r!=VK_SUCCESS) die("vkBindImageMemory(denoise_history)", r);
            
            VkImageViewCreateInfo denoise_hist_ivci = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
            denoise_hist_ivci.image = denoise_history;
            denoise_hist_ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            denoise_hist_ivci.format = VK_FORMAT_R32G32B32A32_SFLOAT;
            denoise_hist_ivci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            denoise_hist_ivci.subresourceRange.baseMipLevel = 0;
            denoise_hist_ivci.subresourceRange.levelCount = 1;
            denoise_hist_ivci.subresourceRange.baseArrayLayer = 0;
            denoise_hist_ivci.subresourceRange.layerCount = 1;
            r = vkCreateImageView(dev, &denoise_hist_ivci, NULL, &denoise_history_view);
            if(r!=VK_SUCCESS) die("vkCreateImageView(denoise_history)", r);
            
            fprintf(stderr, "[GPU] Denoise history buffer created for temporal denoising\n");
        }
        
        // Load denoise shader
        const char* denoise_spv_paths[] = {
            "shaders/denoise.comp.spv",
            "../shaders/denoise.comp.spv",
            "./shaders/denoise.comp.spv"
        };
        uint8_t* denoise_spv = NULL;
        size_t denoise_spv_sz = 0;
        for(int i=0; i<3; i++){
            denoise_spv = read_file(denoise_spv_paths[i], &denoise_spv_sz);
            if(denoise_spv) break;
        }
        if(!denoise_spv){
            fprintf(stderr, "[GPU] Cannot find denoise.comp.spv, GPU denoiser disabled\n");
            gpu_denoise_enabled = 0;
        } else {
            VkShaderModuleCreateInfo denoise_smci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
            denoise_smci.codeSize = denoise_spv_sz;
            denoise_smci.pCode = (const uint32_t*)denoise_spv;
            r = vkCreateShaderModule(dev, &denoise_smci, NULL, &sm_denoise);
            if(r!=VK_SUCCESS) die("vkCreateShaderModule(denoise)", r);
            free(denoise_spv);
            
            // Descriptor set layout for denoiser (2 storage images: in, out)
            VkDescriptorSetLayoutBinding denoise_binds[2];
            memset(denoise_binds, 0, sizeof(denoise_binds));
            denoise_binds[0].binding = 0;
            denoise_binds[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            denoise_binds[0].descriptorCount = 1;
            denoise_binds[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            
            denoise_binds[1].binding = 1;
            denoise_binds[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            denoise_binds[1].descriptorCount = 1;
            denoise_binds[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            
            VkDescriptorSetLayoutCreateInfo denoise_dslci = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            denoise_dslci.bindingCount = 2;
            denoise_dslci.pBindings = denoise_binds;
            r = vkCreateDescriptorSetLayout(dev, &denoise_dslci, NULL, &dsl_denoise);
            if(r!=VK_SUCCESS) die("vkCreateDescriptorSetLayout(denoise)", r);
            
            // Push constants for denoiser (W, H, pass, radius, sigma_s, sigma_r, pad)
            VkPushConstantRange denoise_pcr = {0};
            denoise_pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            denoise_pcr.offset = 0;
            denoise_pcr.size = 32;  // int W, H, pass, radius + float sigma_s, sigma_r, pad, pad
            
            VkPipelineLayoutCreateInfo denoise_plci = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            denoise_plci.setLayoutCount = 1;
            denoise_plci.pSetLayouts = &dsl_denoise;
            denoise_plci.pushConstantRangeCount = 1;
            denoise_plci.pPushConstantRanges = &denoise_pcr;
            r = vkCreatePipelineLayout(dev, &denoise_plci, NULL, &pl_denoise);
            if(r!=VK_SUCCESS) die("vkCreatePipelineLayout(denoise)", r);
            
            // Descriptor pool for both passes
            VkDescriptorPoolSize denoise_ps = { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4 };  // 2 sets * 2 bindings
            VkDescriptorPoolCreateInfo denoise_dpci = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
            denoise_dpci.maxSets = 2;  // horizontal + vertical pass
            denoise_dpci.poolSizeCount = 1;
            denoise_dpci.pPoolSizes = &denoise_ps;
            r = vkCreateDescriptorPool(dev, &denoise_dpci, NULL, &dp_denoise);
            if(r!=VK_SUCCESS) die("vkCreateDescriptorPool(denoise)", r);
            
            // Allocate descriptor sets for both passes
            VkDescriptorSetAllocateInfo denoise_dsai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            denoise_dsai.descriptorPool = dp_denoise;
            denoise_dsai.descriptorSetCount = 1;
            denoise_dsai.pSetLayouts = &dsl_denoise;
            
            r = vkAllocateDescriptorSets(dev, &denoise_dsai, &ds_denoise_h);
            if(r!=VK_SUCCESS) die("vkAllocateDescriptorSets(denoise_h)", r);
            
            r = vkAllocateDescriptorSets(dev, &denoise_dsai, &ds_denoise_v);
            if(r!=VK_SUCCESS) die("vkAllocateDescriptorSets(denoise_v)", r);
            
            // Horizontal pass: in=out_img, out=temp
            VkDescriptorImageInfo denoise_di_h_in = {0};
            denoise_di_h_in.imageView = out_view;
            denoise_di_h_in.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            
            VkDescriptorImageInfo denoise_di_h_out = {0};
            denoise_di_h_out.imageView = denoise_temp_view;
            denoise_di_h_out.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            
            VkWriteDescriptorSet denoise_w_h[2];
            memset(denoise_w_h, 0, sizeof(denoise_w_h));
            denoise_w_h[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            denoise_w_h[0].dstSet = ds_denoise_h;
            denoise_w_h[0].dstBinding = 0;
            denoise_w_h[0].descriptorCount = 1;
            denoise_w_h[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            denoise_w_h[0].pImageInfo = &denoise_di_h_in;
            
            denoise_w_h[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            denoise_w_h[1].dstSet = ds_denoise_h;
            denoise_w_h[1].dstBinding = 1;
            denoise_w_h[1].descriptorCount = 1;
            denoise_w_h[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            denoise_w_h[1].pImageInfo = &denoise_di_h_out;
            
            vkUpdateDescriptorSets(dev, 2, denoise_w_h, 0, NULL);
            
            // Vertical pass: in=temp, out=out_img
            VkDescriptorImageInfo denoise_di_v_in = {0};
            denoise_di_v_in.imageView = denoise_temp_view;
            denoise_di_v_in.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            
            VkDescriptorImageInfo denoise_di_v_out = {0};
            denoise_di_v_out.imageView = out_view;
            denoise_di_v_out.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            
            VkWriteDescriptorSet denoise_w_v[2];
            memset(denoise_w_v, 0, sizeof(denoise_w_v));
            denoise_w_v[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            denoise_w_v[0].dstSet = ds_denoise_v;
            denoise_w_v[0].dstBinding = 0;
            denoise_w_v[0].descriptorCount = 1;
            denoise_w_v[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            denoise_w_v[0].pImageInfo = &denoise_di_v_in;
            
            denoise_w_v[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            denoise_w_v[1].dstSet = ds_denoise_v;
            denoise_w_v[1].dstBinding = 1;
            denoise_w_v[1].descriptorCount = 1;
            denoise_w_v[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            denoise_w_v[1].pImageInfo = &denoise_di_v_out;
            
            vkUpdateDescriptorSets(dev, 2, denoise_w_v, 0, NULL);
            
            // Compute pipeline
            VkPipelineShaderStageCreateInfo denoise_ssci = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
            denoise_ssci.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            denoise_ssci.module = sm_denoise;
            denoise_ssci.pName = "main";
            
            VkComputePipelineCreateInfo denoise_cpci = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
            denoise_cpci.stage = denoise_ssci;
            denoise_cpci.layout = pl_denoise;
            r = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &denoise_cpci, NULL, &pipe_denoise);
            if(r!=VK_SUCCESS) die("vkCreateComputePipelines(denoise)", r);
            
            fprintf(stderr, "[GPU] GPU denoiser pipeline created\n");
        }
    }

    // ---------- Command buffer record ----------
    // Reset counters before dispatch
    void* ctr_reset = NULL;
    r = vkMapMemory(dev, ctr_mem, 0, ctr_bytes, 0, &ctr_reset);
    if(r!=VK_SUCCESS) die("vkMapMemory(ctr reset)", r);
    memset(ctr_reset, 0, (size_t)ctr_bytes);
    vkUnmapMemory(dev, ctr_mem);

    // --- Present loop (optional window mode) ---
    if(window_enabled && swapchain!=VK_NULL_HANDLE){
    // NOTE: swapchain extent may differ from requested W/H; we keep compute at W/H
    VkSemaphore image_avail = VK_NULL_HANDLE;
    VkSemaphore render_done = VK_NULL_HANDLE;
    VkFence in_flight = VK_NULL_HANDLE;

    VkSemaphoreCreateInfo sci = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    r = vkCreateSemaphore(dev, &sci, NULL, &image_avail); if(r!=VK_SUCCESS) die("vkCreateSemaphore", r);
    r = vkCreateSemaphore(dev, &sci, NULL, &render_done); if(r!=VK_SUCCESS) die("vkCreateSemaphore", r);
    r = vkCreateFence(dev, &fci, NULL, &in_flight);       if(r!=VK_SUCCESS) die("vkCreateFence", r);

    if(cam_mouse_lock && window){
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    }

    double last_time = glfwGetTime();
    double last_mouse_x = 0.0, last_mouse_y = 0.0;
    int mouse_initialized = 0;

    // Track camera state for detecting movement (to reset accumulation)
    CamVec3 prev_cam_pos = cam_pos;
    float prev_cam_yaw = cam_yaw;
    float prev_cam_pitch = cam_pitch;
    int pending_reset = 0; // Reset accumulation on next frame

    int frame_id = 0;
    int ts_first = 1;
    int render_count = 0;
    // FPS tracking (window title)
    uint64_t fps_last_us = ysu_now_us();
    int fps_frames = 0;
    float fps_value = 0.0f;

    // Debug: build a small ray batch and schedule once (placeholder)
    int scheduler_logged = 0;
    fprintf(stderr, "[GPU] Entering render loop (press Esc to quit)\n");
    fflush(stderr);
    while(window && !glfwWindowShouldClose(window)){
        render_count++;
        fps_frames++;
        glfwPollEvents();

        // Update FPS in window title every ~0.5s
        if(window){
            uint64_t fps_now_us = ysu_now_us();
            double fps_dt = (double)(fps_now_us - fps_last_us) / 1000000.0;
            if(fps_dt >= 0.5){
                fps_value = (float)((double)fps_frames / fps_dt);
                char title[128];
                snprintf(title, sizeof(title), "YSU GPU Demo - %.1f FPS", fps_value);
                glfwSetWindowTitle(window, title);
                fps_frames = 0;
                fps_last_us = fps_now_us;
            }
        }

        // --- Input: mouse look + WASD movement ---
        double now = glfwGetTime();
        float dt = (float)(now - last_time);
        if(dt > 0.1f) dt = 0.1f; // clamp to avoid huge jumps when debugging
        last_time = now;

        double mx, my;
        glfwGetCursorPos(window, &mx, &my);
        if(!mouse_initialized){
            last_mouse_x = mx;
            last_mouse_y = my;
            mouse_initialized = 1;
        }
        float dx = (float)(mx - last_mouse_x);
        float dy = (float)(my - last_mouse_y);
        // Apply deadzone: treat tiny movements as zero to avoid micro jitter
        if(fabsf(dx) < mouse_deadzone) dx = 0.0f;
        if(fabsf(dy) < mouse_deadzone) dy = 0.0f;
        last_mouse_x = mx;
        last_mouse_y = my;

        if(cam_mouse_lock){
            cam_yaw   += dx * mouse_sens;
            cam_pitch -= dy * mouse_sens;
            const float pitch_limit = 1.55f; // ~89 degrees
            if(cam_pitch >  pitch_limit) cam_pitch =  pitch_limit;
            if(cam_pitch < -pitch_limit) cam_pitch = -pitch_limit;
        }

        if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS){
            glfwSetWindowShouldClose(window, 1);
        }

        CamVec3 forward, right, up;
        cam_build_basis(cam_yaw, cam_pitch, &forward, &right, &up);
        CamVec3 flat_forward = cam_v3_norm((CamVec3){ forward.x, 0.0f, forward.z });
        if(fabsf(flat_forward.x) + fabsf(flat_forward.y) + fabsf(flat_forward.z) < 1e-6f){
            flat_forward = forward;
        }

        // Scheduler wiring (debug): build tiny ray batch and split once
        if(!scheduler_logged && sched_ok){
            NerfRayBatch frame_rays = {0};
            uint32_t test_count = 256;
            if(nerf_ray_batch_init(&frame_rays, test_count)){
                frame_rays.count = test_count;
                for(uint32_t i = 0; i < test_count; i++){
                    frame_rays.pix[i] = i;
                    frame_rays.ox[i] = cam_pos.x;
                    frame_rays.oy[i] = cam_pos.y;
                    frame_rays.oz[i] = cam_pos.z;
                    frame_rays.dx[i] = forward.x;
                    frame_rays.dy[i] = forward.y;
                    frame_rays.dz[i] = forward.z;
                    frame_rays.tmin[i] = 0.01f;
                    frame_rays.tmax[i] = 100.0f;
                }
                nerf_schedule_split(&sched_cfg, &frame_rays, &sched_q);
                fprintf(stderr, "[NERF] sched split: gpu=%u cpu=%u (batch=%u)\n",
                        sched_q.gpu.count, sched_q.cpu.count, sched_cfg.batch_size);
                nerf_ray_batch_free(&frame_rays);
                scheduler_logged = 1;
            }
        }

        float step = cam_speed * dt;
        if(glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) cam_pos = cam_v3_add(cam_pos, cam_v3_scale(flat_forward,  step));
        if(glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) cam_pos = cam_v3_add(cam_pos, cam_v3_scale(flat_forward, -step));
        if(glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) cam_pos = cam_v3_add(cam_pos, cam_v3_scale(right,       -step));
        if(glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) cam_pos = cam_v3_add(cam_pos, cam_v3_scale(right,        step));
        if(glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)        cam_pos.y -= step;  // Up
        if(glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) cam_pos.y += step; // Down
        
        // Update camera UBO
        CameraUBO* cam_data = (CameraUBO*)cam_ubo_mapped;
        cam_data->pos[0] = cam_pos.x; cam_data->pos[1] = cam_pos.y; cam_data->pos[2] = cam_pos.z; cam_data->pos[3] = 1.0f;
        cam_data->forward[0] = forward.x; cam_data->forward[1] = forward.y; cam_data->forward[2] = forward.z; cam_data->forward[3] = 0.0f;
        cam_data->right[0] = right.x; cam_data->right[1] = right.y; cam_data->right[2] = right.z; cam_data->right[3] = 0.0f;
        cam_data->up[0] = up.x; cam_data->up[1] = up.y; cam_data->up[2] = up.z; cam_data->up[3] = 0.0f;

        // Detect camera movement: reset accumulation if position or orientation changed
        int camera_moved = 0;
          // Use configurable epsilons for stability (avoid resets on floating noise)
          if(fabsf(cam_pos.x - prev_cam_pos.x) > cam_reset_eps_pos ||
              fabsf(cam_pos.y - prev_cam_pos.y) > cam_reset_eps_pos ||
              fabsf(cam_pos.z - prev_cam_pos.z) > cam_reset_eps_pos ||
              fabsf(cam_yaw - prev_cam_yaw) > cam_reset_eps_ang ||
              fabsf(cam_pitch - prev_cam_pitch) > cam_reset_eps_ang) {
            camera_moved = 1;
            prev_cam_pos = cam_pos;
            prev_cam_yaw = cam_yaw;
            prev_cam_pitch = cam_pitch;
            pending_reset = 1; // Flag for next frame to reset accumulation
        }

        if(render_count == 1) { fprintf(stderr, "[GPU] After input, waiting for fence\n"); fflush(stderr); }
        r = vkWaitForFences(dev, 1, &in_flight, VK_TRUE, UINT64_MAX);
        if(r!=VK_SUCCESS) die("vkWaitForFences", r);
        if(ts_enabled && !ts_first && qp_ts!=VK_NULL_HANDLE){
            uint64_t ts[3] = {0,0,0};
            VkResult qr = vkGetQueryPoolResults(dev, qp_ts, 0, 3, sizeof(ts), ts, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            if(qr==VK_SUCCESS){
                double h_ms = (double)(ts[1]-ts[0]) * ts_period_ns * 1e-6;
                double v_ms = (double)(ts[2]-ts[1]) * ts_period_ns * 1e-6;
                fprintf(stderr, "[GPU][denoise] h=%.3f ms v=%.3f ms\n", h_ms, v_ms);
            }
        }
        ts_first = 0;

        vkResetFences(dev, 1, &in_flight);

        uint32_t imageIndex = 0;
        r = vkAcquireNextImageKHR(dev, swapchain, UINT64_MAX, image_avail, VK_NULL_HANDLE, &imageIndex);
        if(r==VK_ERROR_OUT_OF_DATE_KHR){ 
            // Swapchain out of date (window resized, etc.) - skip this frame and continue
            fprintf(stderr, "[GPU] Swapchain out of date, skipping frame\n");
            continue;
        }
        if(r!=VK_SUCCESS && r!=VK_SUBOPTIMAL_KHR) die("vkAcquireNextImageKHR", r);

        vkResetCommandBuffer(cb, 0);

        if(render_count == 1) { fprintf(stderr, "[GPU] Recording command buffer\n"); fflush(stderr); }
        VkCommandBufferBeginInfo cbi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        r = vkBeginCommandBuffer(cb, &cbi);
        if(r!=VK_SUCCESS) die("vkBeginCommandBuffer", r);

        // If pending reset from camera movement, clear the accumulation buffers first
        if(pending_reset){
            VkImageMemoryBarrier clear_bar = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
            clear_bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            clear_bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            clear_bar.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            clear_bar.subresourceRange.levelCount = 1;
            clear_bar.subresourceRange.layerCount = 1;
            // First: clear out_img
            clear_bar.image = out_img;
            clear_bar.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            clear_bar.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            clear_bar.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            clear_bar.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0, 0, NULL, 0, NULL, 1, &clear_bar);
            
            VkClearColorValue clear_val = {{0.0f, 0.0f, 0.0f, 0.0f}};
            VkImageSubresourceRange clear_range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(cb, out_img, VK_IMAGE_LAYOUT_GENERAL, &clear_val, 1, &clear_range);
            
            // Transition back to GENERAL for compute
            clear_bar.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            clear_bar.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            clear_bar.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            clear_bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 0, NULL, 0, NULL, 1, &clear_bar);

            // Second: clear accum_img (history buffer) to ensure clean accumulation start
            clear_bar.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            clear_bar.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            clear_bar.image = accum_img;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0, 0, NULL, 0, NULL, 1, &clear_bar);
            vkCmdClearColorImage(cb, accum_img, VK_IMAGE_LAYOUT_GENERAL, &clear_val, 1, &clear_range);
            clear_bar.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            clear_bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 0, NULL, 0, NULL, 1, &clear_bar);
        }

        uint32_t gx = (uint32_t)((W + 15) / 16);
        uint32_t gy = (uint32_t)((H + 15) / 16);

        // --- CPU DEPTH PREPASS for mode 26 (Depth-Conditioned NeRF) ---
        // Compute depth hints using occupancy grid raymarching
        // Note: Only compute when camera is STATIONARY (not during movement) for perf
        static int depth_prepass_initialized = 0;
        static int frames_since_move = 0;
        if(render_mode == 26 && depth_hints_map && nerf_occ.data) {
            
            if(camera_moved) {
                frames_since_move = 0;
                // Clear hints during movement (full range sampling)
                if(depth_prepass_initialized) {
                    memset(depth_hints_map, 0, (size_t)(W * H * 16));
                    depth_prepass_initialized = 0;
                }
            } else {
                frames_since_move++;
            }
            
            // Only compute depth hints after camera has been stationary for 5 frames
            // This avoids expensive CPU work during interactive movement
            if(!camera_moved && frames_since_move >= 5 && !depth_prepass_initialized) {
                float fov_y = 0.8f; // ~45 degrees (matching training)
                float aspect = (float)W / (float)H;
                compute_depth_hints_occ(
                    (DepthHintVec4*)depth_hints_map, W, H,
                    cam_pos, forward, right, up,
                    fov_y, aspect,
                    (const uint8_t*)nerf_occ.data, nerf_occ.hdr.dim,
                    nerf_center_x, nerf_center_y, nerf_center_z, nerf_scale,
                    2.0f, 6.0f, 64, // t_near, t_far, coarse_steps (increased density)
                    0 // merge = false (initial computation)
                );
                // Count how many hints have confidence > 0
                int hit_count = 0;
                DepthHintVec4* h = (DepthHintVec4*)depth_hints_map;
                for(int i = 0; i < W*H; i++) {
                    if(h[i].confidence > 0.5f) hit_count++;
                }
                fprintf(stderr, "[GPU] depth prepass: %d/%d hits (%.1f%%) - sparse sampling active\n", 
                    hit_count, W*H, 100.0f*hit_count/(float)(W*H));
                depth_prepass_initialized = 1;
            } else if(!camera_moved && frames_since_move >= 5 && depth_prepass_initialized && frames_since_move <= 60) {
                // Accumulate additional hints over subsequent stationary frames to raise hit-rate.
                // Use merge=1 so we only strengthen existing hints without overwriting better ones.
                float fov_y = 0.8f;
                float aspect = (float)W / (float)H;
                compute_depth_hints_occ(
                    (DepthHintVec4*)depth_hints_map, W, H,
                    cam_pos, forward, right, up,
                    fov_y, aspect,
                    (const uint8_t*)nerf_occ.data, nerf_occ.hdr.dim,
                    nerf_center_x, nerf_center_y, nerf_center_z, nerf_scale,
                    2.0f, 6.0f, // t_near, t_far (same range)
                    16, // fewer coarse steps for incremental accumulation
                    1   // merge = true (accumulate)
                );
                int hit_count = 0;
                DepthHintVec4* h = (DepthHintVec4*)depth_hints_map;
                for(int i = 0; i < W*H; i++) {
                    if(h[i].confidence > 0.5f) hit_count++;
                }
                fprintf(stderr, "[GPU] depth prepass (merge): %d/%d hits (%.1f%%) - accumulated\n", 
                    hit_count, W*H, 100.0f*hit_count/(float)(W*H));
            }
        }

        // We render 1 sample per frame in interactive window mode for responsive input
        // (YSU_GPU_FRAMES is ignored in window mode; use ESC to quit)
        {
            PushConstants pc_push = {0};
            pc_push.W = W;
            pc_push.H = H;
            pc_push.frame = frame_id;
            pc_push.seed = seed;
            pc_push.triCount = (int)tri_count;
            pc_push.nodeCount = node_count;
            pc_push.useBVH = use_bvh;
            pc_push.cullBackface = cull_backface;
            pc_push.rootCount = (int)bvh_root_count;
            pc_push.enableCounters = counters_enabled;
            pc_push.alpha = 0.0f;
            pc_push.resetAccum = pending_reset ? 1 : 0; // Reset accumulation if pending from camera movement
            pc_push.enableNerfProxy = (render_mode == 2) ? 1 : nerf_proxy_enabled;
            pc_push.nerfStrength = nerf_strength;
            pc_push.nerfDensity = nerf_density;
            pc_push.nerfBlend = nerf_blend;
            pc_push.nerfBounds = nerf_bounds;
            pc_push.nerfCenterX = nerf_center_x;
            pc_push.nerfCenterY = nerf_center_y;
            pc_push.nerfCenterZ = nerf_center_z;
            pc_push.nerfScale = nerf_scale;
            pc_push.nerfSkipOcc = nerf_skip_occ;
            // Enforce occupancy skipping off for depth-conditioned mode (mode 26)
            // because the occ grid can mismatch NeRF density and cause black pixels.
            if(render_mode == 26) pc_push.nerfSkipOcc = 1;
            // Inform shader whether depth prepass hints are initialized
            pc_push.depthHintsReady = depth_prepass_initialized ? 1 : 0;
            pc_push.nerfSteps = nerf_steps;
            pc_push.renderMode = render_mode;
            
            // Apply pending reset after this frame
            if(pending_reset) {
                frame_id = 0; // Reset frame counter for fresh accumulation
                pending_reset = 0; // Clear the flag
            }

            // Bind pipeline and descriptor sets every frame (command buffer is reset each frame)
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
            
            vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc_push);
            vkCmdDispatch(cb, gx, gy, 1);
        }
        if(render_count == 1) { fprintf(stderr, "[GPU] Dispatch complete\n"); fflush(stderr); }

        // --- Barrier between raytrace and tonemap ---
        if(render_count == 1) { fprintf(stderr, "[GPU] Adding barrier after raytrace\n"); fflush(stderr); }
        if(tonemap_enabled){
            VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
            mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cb,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &mb, 0, NULL, 0, NULL
            );

            // --- GPU Tonemap (HDR out_img -> LDR ldr_img) ---
            if(render_count == 1) { fprintf(stderr, "[GPU] Binding tonemap pipeline\n"); fflush(stderr); }
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_tm);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl_tm, 0, 1, &ds_tm, 0, NULL);
            struct { int W; int H; float exposure; float gamma; } tm_pc;
            tm_pc.W = W; tm_pc.H = H; tm_pc.exposure = tm_exposure; tm_pc.gamma = tm_gamma;
            vkCmdPushConstants(cb, pl_tm, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(tm_pc), &tm_pc);
            uint32_t tgx = (uint32_t)((W + 15) / 16);
            uint32_t tgy = (uint32_t)((H + 15) / 16);
            vkCmdDispatch(cb, tgx, tgy, 1);
        }

        // --- GPU Denoiser (separable bilateral filter) ---
        // Skip denoising on certain frames for performance boost (denoise_skip=1: every frame, 2: every 2nd, etc.)
        // But always denoise on frame 0 (immediate denoise)
        int frame_skip_value = denoise_skip;
        if(adaptive_denoise_enabled) {
            // Adaptive: use minimum denoise_skip (more frequent) at start, reduce frequency over time
            // This ensures initial quality while allowing speedup after convergence
            int warmup_frames = 30;
            if(frame_id < warmup_frames) {
                frame_skip_value = adaptive_denoise_min;  // Denoise frequently at start
            } else {
                frame_skip_value = adaptive_denoise_max;  // Denoise less frequently after warmup
            }
        }
        int should_denoise = (frame_id == 0) ||  // Always denoise first frame (immediate denoise)
                             (frame_skip_value <= 1) || 
                             ((frame_id % frame_skip_value) == 0);
        
        // Reset history buffer every N frames if enabled
        int should_reset_history = denoise_history_reset && 
                                   (frame_id > 0) && 
                                   ((frame_id % denoise_history_reset_frame) == 0);
        
        if(gpu_denoise_enabled && pipe_denoise != VK_NULL_HANDLE && should_denoise){
            if(ts_enabled && qp_ts!=VK_NULL_HANDLE){
                vkCmdResetQueryPool(cb, qp_ts, 0, 3);
            }

            VkImageMemoryBarrier denoise_bar_pre = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
            denoise_bar_pre.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            denoise_bar_pre.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            denoise_bar_pre.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            denoise_bar_pre.subresourceRange.levelCount = 1;
            denoise_bar_pre.subresourceRange.layerCount = 1;
            denoise_bar_pre.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            denoise_bar_pre.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            denoise_bar_pre.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            denoise_bar_pre.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            denoise_bar_pre.image = out_img;

            vkCmdPipelineBarrier(cb,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 0, NULL, 0, NULL, 1, &denoise_bar_pre
            );

            uint32_t denoise_gx = (uint32_t)((W + 15) / 16);
            uint32_t denoise_gy = (uint32_t)((H + 15) / 16);

            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_denoise);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl_denoise, 0, 1, &ds_denoise_h, 0, NULL);

            struct { int W; int H; int pass; int radius; float sigma_s; float sigma_r; } denoise_pc_h;
            denoise_pc_h.W = W;
            denoise_pc_h.H = H;
            denoise_pc_h.pass = 0;
            denoise_pc_h.radius = denoise_radius;
            denoise_pc_h.sigma_s = denoise_sigma_s;
            denoise_pc_h.sigma_r = denoise_sigma_r;
            vkCmdPushConstants(cb, pl_denoise, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(denoise_pc_h), &denoise_pc_h);
            if(ts_enabled && qp_ts!=VK_NULL_HANDLE) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, qp_ts, 0);
            vkCmdDispatch(cb, denoise_gx, denoise_gy, 1);
            if(ts_enabled && qp_ts!=VK_NULL_HANDLE) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, qp_ts, 1);

            VkImageMemoryBarrier denoise_bar_pass = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
            denoise_bar_pass.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            denoise_bar_pass.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            denoise_bar_pass.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            denoise_bar_pass.subresourceRange.levelCount = 1;
            denoise_bar_pass.subresourceRange.layerCount = 1;
            denoise_bar_pass.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            denoise_bar_pass.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            denoise_bar_pass.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            denoise_bar_pass.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            denoise_bar_pass.image = denoise_temp;

            vkCmdPipelineBarrier(cb,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 0, NULL, 0, NULL, 1, &denoise_bar_pass
            );

            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl_denoise, 0, 1, &ds_denoise_v, 0, NULL);

            struct { int W; int H; int pass; int radius; float sigma_s; float sigma_r; } denoise_pc_v;
            denoise_pc_v.W = W;
            denoise_pc_v.H = H;
            denoise_pc_v.pass = 1;
            denoise_pc_v.radius = denoise_radius;
            denoise_pc_v.sigma_s = denoise_sigma_s;
            denoise_pc_v.sigma_r = denoise_sigma_r;
            vkCmdPushConstants(cb, pl_denoise, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(denoise_pc_v), &denoise_pc_v);
            vkCmdDispatch(cb, denoise_gx, denoise_gy, 1);
            if(ts_enabled && qp_ts!=VK_NULL_HANDLE) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, qp_ts, 2);
        }

        // --- Tonemap to RGBA8 (required for swapchain) ---
        if(tonemap_enabled){
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_tm);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl_tm, 0, 1, &ds_tm, 0, NULL);

            // push for tonemap.comp expects W,H, exposure, gamma (see shader)
            struct { int W,H; float exposure; float gamma; } tpc;
            tpc.W=W; tpc.H=H; tpc.exposure=tm_exposure; tpc.gamma=tm_gamma;
            vkCmdPushConstants(cb, pl_tm, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(tpc), &tpc);
            vkCmdDispatch(cb, gx, gy, 1);
        }

        // --- Copy to swapchain image ---
        if(render_count == 1) { fprintf(stderr, "[GPU] Copying to swapchain\n"); fflush(stderr); }
        VkImage src = tonemap_enabled ? ldr_img : out_img;
        VkImage dst = sw_images[imageIndex];

        VkImageMemoryBarrier b0[2] = {0};
        b0[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b0[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        b0[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        b0[0].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        b0[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        b0[0].image = src;
        b0[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        b0[0].subresourceRange.levelCount = 1;
        b0[0].subresourceRange.layerCount = 1;

        b0[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b0[1].srcAccessMask = 0;
        b0[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b0[1].oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        b0[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b0[1].image = dst;
        b0[1].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        b0[1].subresourceRange.levelCount = 1;
        b0[1].subresourceRange.layerCount = 1;

        vkCmdPipelineBarrier(
            cb,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0,NULL, 0,NULL, 2, b0
        );

        VkImageBlit blit = {0};
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.layerCount = 1;
        blit.srcOffsets[1].x = (int32_t)sw_extent.width;
        blit.srcOffsets[1].y = (int32_t)sw_extent.height;
        blit.srcOffsets[1].z = 1;
        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.layerCount = 1;
        blit.dstOffsets[1].x = (int32_t)sw_extent.width;
        blit.dstOffsets[1].y = (int32_t)sw_extent.height;
        blit.dstOffsets[1].z = 1;

        vkCmdBlitImage(cb, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);

            VkImageMemoryBarrier b1[2] = {0};
        b1[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b1[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        b1[0].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        b1[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        b1[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b1[0].image = src;
        b1[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        b1[0].subresourceRange.levelCount = 1;
        b1[0].subresourceRange.layerCount = 1;

        b1[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b1[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b1[1].dstAccessMask = 0;
        b1[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b1[1].newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        b1[1].image = dst;
        b1[1].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        b1[1].subresourceRange.levelCount = 1;
        b1[1].subresourceRange.layerCount = 1;

        vkCmdPipelineBarrier(
            cb,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0,NULL, 0,NULL, 2, b1
        );

        r = vkEndCommandBuffer(cb);
        if(r!=VK_SUCCESS) die("vkEndCommandBuffer", r);

        if(render_count == 1) { fprintf(stderr, "[GPU] Submitting command buffer\n"); fflush(stderr); }
        VkPipelineStageFlags waitStages = VK_PIPELINE_STAGE_TRANSFER_BIT;
        VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
        si.waitSemaphoreCount = 1;
        si.pWaitSemaphores = &image_avail;
        si.pWaitDstStageMask = &waitStages;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cb;
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = &render_done;

        r = vkQueueSubmit(queue, 1, &si, in_flight);
        if(r!=VK_SUCCESS) die("vkQueueSubmit", r);

        VkPresentInfoKHR pi = { VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
        pi.waitSemaphoreCount = 1;
        pi.pWaitSemaphores = &render_done;
        pi.swapchainCount = 1;
        pi.pSwapchains = &swapchain;
        pi.pImageIndices = &imageIndex;

        r = vkQueuePresentKHR(queue, &pi);
        if(r==VK_SUBOPTIMAL_KHR){
            fprintf(stderr, "[GPU] vkQueuePresentKHR: SUBOPTIMAL (continuing)\n");
        } else if(r==VK_ERROR_OUT_OF_DATE_KHR){
            fprintf(stderr, "[GPU] vkQueuePresentKHR: OUT_OF_DATE (skipping frame)\n");
            continue; // Skip this frame and continue loop instead of breaking
        } else if(r!=VK_SUCCESS){
            die("vkQueuePresentKHR", r);
        }
        if(render_count == 1) { fprintf(stderr, "[GPU] First frame presented successfully\n"); fflush(stderr); }

        // optionally dump a single-frame PPM and exit (use YSU_GPU_DUMP_ONESHOT=1)
        // NOTE: Disabled in window mode to keep interactive loop running
        if(!window_enabled && ysu_env_bool("YSU_GPU_DUMP_ONESHOT", 0)){
            // Record a small command buffer to copy the HDR/LDR image to a host-visible buffer
            vkDeviceWaitIdle(dev);
            vkResetCommandBuffer(cb, 0);
            VkCommandBufferBeginInfo cbi_dump = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
            r = vkBeginCommandBuffer(cb, &cbi_dump); if(r!=VK_SUCCESS) die("vkBeginCommandBuffer(dump)", r);

            VkImage src_copy_img = tonemap_enabled ? ldr_img : out_img;
            uint32_t dump_w = sw_extent.width;
            uint32_t dump_h = sw_extent.height;
            VkImageMemoryBarrier out_bar = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
            out_bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            out_bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            out_bar.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            out_bar.subresourceRange.levelCount = 1;
            out_bar.subresourceRange.layerCount = 1;
            out_bar.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            out_bar.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            out_bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            out_bar.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            out_bar.image = src_copy_img;

            vkCmdPipelineBarrier(cb,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 0,NULL, 0,NULL, 1, &out_bar
            );

            VkDeviceSize dump_bytes = (VkDeviceSize)dump_w * (VkDeviceSize)dump_h * (tonemap_enabled ? 4 : 16);
            VkBuffer dump_buf = create_buffer(dev, dump_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
            VkDeviceMemory dump_mem = alloc_bind_buffer_mem(
                phy, dev, dump_buf,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
            );

            VkBufferImageCopy bic = {0};
            bic.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            bic.imageSubresource.layerCount = 1;
            bic.imageExtent.width = dump_w;
            bic.imageExtent.height = dump_h;
            bic.imageExtent.depth = 1;
            vkCmdCopyImageToBuffer(cb, src_copy_img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dump_buf, 1, &bic);

            // restore image layout to GENERAL for safety
            VkImageMemoryBarrier ib_restore = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
            ib_restore.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            ib_restore.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            ib_restore.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            ib_restore.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            ib_restore.image = src_copy_img;
            ib_restore.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            ib_restore.subresourceRange.levelCount = 1;
            ib_restore.subresourceRange.layerCount = 1;

            vkCmdPipelineBarrier(cb,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 0,NULL, 0,NULL, 1, &ib_restore
            );

            r = vkEndCommandBuffer(cb); if(r!=VK_SUCCESS) die("vkEndCommandBuffer(dump)", r);
            submit_and_wait(dev, queue, cb);

            void* dump_map = NULL;
            r = vkMapMemory(dev, dump_mem, 0, dump_bytes, 0, &dump_map);
            if(r!=VK_SUCCESS) die("vkMapMemory(dump)", r);

            FILE* outf = fopen("window_dump.ppm", "wb");
            if(!outf){ fprintf(stderr, "Cannot write window_dump.ppm\n"); }
            else {
                // Convert to Vec3 for denoising
                Vec3 *pixels_cpu = (Vec3*)malloc((size_t)dump_w * (size_t)dump_h * sizeof(Vec3));
                if(!pixels_cpu){
                    fprintf(stderr, "[GPU] malloc failed for window dump denoise buffer\n");
                    fclose(outf);
                    vkUnmapMemory(dev, dump_mem);
                } else {
                    if(tonemap_enabled){
                        const uint8_t* px8 = (const uint8_t*)dump_map;
                        for(int i=0;i<dump_w*dump_h;i++){
                            // ldr_img uses RGBA8 format
                            pixels_cpu[i].x = (float)(px8[i*4 + 0]) / 255.0f;  // R
                            pixels_cpu[i].y = (float)(px8[i*4 + 1]) / 255.0f;  // G
                            pixels_cpu[i].z = (float)(px8[i*4 + 2]) / 255.0f;  // B
                        }
                    } else {
                        const float* px = (const float*)dump_map;
                        for(int i=0;i<dump_w*dump_h;i++){
                            pixels_cpu[i].x = px[i*4 + 0];
                            pixels_cpu[i].y = px[i*4 + 1];
                            pixels_cpu[i].z = px[i*4 + 2];
                            if(pixels_cpu[i].x < 0) pixels_cpu[i].x = 0; if(pixels_cpu[i].x > 1) pixels_cpu[i].x = 1;
                            if(pixels_cpu[i].y < 0) pixels_cpu[i].y = 0; if(pixels_cpu[i].y > 1) pixels_cpu[i].y = 1;
                            if(pixels_cpu[i].z < 0) pixels_cpu[i].z = 0; if(pixels_cpu[i].z > 1) pixels_cpu[i].z = 1;
                        }
                    }

                    // Apply denoiser
                    ysu_neural_denoise_maybe(pixels_cpu, dump_w, dump_h);

                    // Write denoised pixels
                    fprintf(outf, "P6\n%d %d\n255\n", dump_w, dump_h);
                    for(int i=0;i<dump_w*dump_h;i++){
                        unsigned char rgb[3];
                        float r = pixels_cpu[i].x;
                        float g = pixels_cpu[i].y;
                        float b = pixels_cpu[i].z;
                        if(r < 0) r = 0; if(g < 0) g = 0; if(b < 0) b = 0;
                        if(r > 1) r = 1; if(g > 1) g = 1; if(b > 1) b = 1;
                        rgb[0] = (unsigned char)(255.0f * r);
                        rgb[1] = (unsigned char)(255.0f * g);
                        rgb[2] = (unsigned char)(255.0f * b);
                        fwrite(rgb, 1, 3, outf);
                    }

                    free(pixels_cpu);
                    fclose(outf);
                    fprintf(stderr, "[GPU] wrote window_dump.ppm (%dx%d)\n", dump_w, dump_h);
                }
            }

            vkUnmapMemory(dev, dump_mem);
            vkDestroyBuffer(dev, dump_buf, NULL);
            vkFreeMemory(dev, dump_mem, NULL);

            // break out of present loop after dump
            break;
        }

        // History reset: clear denoise_history buffer periodically (for camera cuts, scene changes, etc.)
        if(should_reset_history && denoise_history != VK_NULL_HANDLE) {
            VkImageMemoryBarrier hist_reset_bar = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
            hist_reset_bar.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            hist_reset_bar.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            hist_reset_bar.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            hist_reset_bar.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            hist_reset_bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            hist_reset_bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            hist_reset_bar.image = denoise_history;
            hist_reset_bar.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            hist_reset_bar.subresourceRange.levelCount = 1;
            hist_reset_bar.subresourceRange.layerCount = 1;
            
            // Clear history buffer to black (0.0, 0.0, 0.0, 0.0)
            VkClearColorValue clear_color = {{0.0f, 0.0f, 0.0f, 0.0f}};
            VkImageSubresourceRange subres = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            
            vkCmdPipelineBarrier(cb,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &hist_reset_bar);
            vkCmdClearColorImage(cb, denoise_history, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear_color, 1, &subres);
            
            // Transition back to GENERAL for next frame
            hist_reset_bar.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            hist_reset_bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            hist_reset_bar.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            hist_reset_bar.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            vkCmdPipelineBarrier(cb,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, &hist_reset_bar);
            
            fprintf(stderr, "[GPU] History reset at frame %d\n", frame_id);
        }

        // continue to next frame
        frame_id += frames;
    }

    fprintf(stderr, "[GPU] Exited render loop after %d frames rendered (last FPS: %.1f)\n", render_count, fps_value);
    vkDeviceWaitIdle(dev);
    vkDestroyFence(dev, in_flight, NULL);
    vkDestroySemaphore(dev, render_done, NULL);
    vkDestroySemaphore(dev, image_avail, NULL);

    } else {
    // Headless mode: record and submit command buffer once
    VkCommandBufferBeginInfo cbi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    r = vkBeginCommandBuffer(cb, &cbi);
    if(r!=VK_SUCCESS) die("vkBeginCommandBuffer", r);

    // Transition images to GENERAL
    VkImageMemoryBarrier imb[2];
    memset(imb, 0, sizeof(imb));
    for(int i=0;i<2;i++){
        imb[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        imb[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        imb[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        imb[i].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imb[i].subresourceRange.levelCount = 1;
        imb[i].subresourceRange.layerCount = 1;
        imb[i].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imb[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        imb[i].srcAccessMask = 0;
        imb[i].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    }
    imb[0].image = out_img;
    imb[1].image = accum_img;

    vkCmdPipelineBarrier(cb,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, NULL, 0, NULL, 2, imb
    );
    if(tonemap_enabled){
        VkImageMemoryBarrier ib_tm = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        ib_tm.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ib_tm.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        ib_tm.srcAccessMask = 0;
        ib_tm.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        ib_tm.image = ldr_img;
        ib_tm.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        ib_tm.subresourceRange.levelCount = 1;
        ib_tm.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(cb,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, NULL, 0, NULL, 1, &ib_tm
        );
    }


    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);

    uint32_t gx = (uint32_t)((W + 15) / 16);
    uint32_t gy = (uint32_t)((H + 15) / 16);

    // Progressive frames
    for(int f=0; f<frames; f++){
        // Scripted walk path for headless runs (fixed to look at origin)
        float walkTime = (float)f * 0.01f;
        CamVec3 ro = { sinf(walkTime * 0.3f) * 4.0f, 1.2f, cosf(walkTime * 0.2f) * 4.0f };
        CamVec3 target = { 0.0f, 0.0f, 0.0f };
        CamVec3 forward = cam_v3_norm(cam_v3_sub(target, ro));
        CamVec3 upv = {0.f, 1.f, 0.f};
        CamVec3 right = cam_v3_norm(cam_v3_cross(forward, upv));
        upv = cam_v3_norm(cam_v3_cross(right, forward));
        
        // Update camera UBO
        CameraUBO* cam_data = (CameraUBO*)cam_ubo_mapped;
        cam_data->pos[0] = ro.x; cam_data->pos[1] = ro.y; cam_data->pos[2] = ro.z; cam_data->pos[3] = 1.0f;
        cam_data->forward[0] = forward.x; cam_data->forward[1] = forward.y; cam_data->forward[2] = forward.z; cam_data->forward[3] = 0.0f;
        cam_data->right[0] = right.x; cam_data->right[1] = right.y; cam_data->right[2] = right.z; cam_data->right[3] = 0.0f;
        cam_data->up[0] = upv.x; cam_data->up[1] = upv.y; cam_data->up[2] = upv.z; cam_data->up[3] = 0.0f;

        PushConstants pc_push = {0};
        pc_push.W = W;
        pc_push.H = H;
        pc_push.frame = f;
        pc_push.seed = seed;
        pc_push.triCount = (int)tri_count;
        pc_push.nodeCount = node_count;
        pc_push.useBVH = use_bvh;
        pc_push.cullBackface = cull_backface;
        pc_push.rootCount = (int)bvh_root_count;
        pc_push.enableCounters = counters_enabled;
        pc_push.alpha = 0.0f;
        pc_push.resetAccum = 0;
        pc_push.enableNerfProxy = (render_mode == 2) ? 1 : nerf_proxy_enabled;
        pc_push.nerfStrength = nerf_strength;
        pc_push.nerfDensity = nerf_density;
        pc_push.nerfBlend = nerf_blend;
        pc_push.nerfBounds = nerf_bounds;
        pc_push.nerfCenterX = nerf_center_x;
        pc_push.nerfCenterY = nerf_center_y;
        pc_push.nerfCenterZ = nerf_center_z;
        pc_push.nerfScale = nerf_scale;
        pc_push.nerfSkipOcc = nerf_skip_occ;
        pc_push.nerfSteps = nerf_steps;
        pc_push.renderMode = render_mode;

        vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc_push);
        vkCmdDispatch(cb, gx, gy, 1);
    }

    // ---------- GPU tonemap (HDR out_img -> LDR ldr_img) ----------
    if(tonemap_enabled){
        VkImageMemoryBarrier ib = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        ib.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        ib.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        ib.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        ib.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        ib.image = out_img;
        ib.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        ib.subresourceRange.levelCount = 1;
        ib.subresourceRange.layerCount = 1;

        vkCmdPipelineBarrier(cb,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, NULL, 0, NULL, 1, &ib
        );

        // --- GPU Denoiser (separable bilateral filter) ---
        if(gpu_denoise_enabled && pipe_denoise != VK_NULL_HANDLE){
            if(ts_enabled && qp_ts!=VK_NULL_HANDLE){
                vkCmdResetQueryPool(cb, qp_ts, 0, 3);
            }

            VkImageMemoryBarrier denoise_bar_pre = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
            denoise_bar_pre.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            denoise_bar_pre.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            denoise_bar_pre.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            denoise_bar_pre.subresourceRange.levelCount = 1;
            denoise_bar_pre.subresourceRange.layerCount = 1;
            denoise_bar_pre.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            denoise_bar_pre.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            denoise_bar_pre.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            denoise_bar_pre.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            denoise_bar_pre.image = out_img;

            vkCmdPipelineBarrier(cb,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 0, NULL, 0, NULL, 1, &denoise_bar_pre
            );

            uint32_t denoise_gx = (uint32_t)((W + 15) / 16);
            uint32_t denoise_gy = (uint32_t)((H + 15) / 16);

            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_denoise);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl_denoise, 0, 1, &ds_denoise_h, 0, NULL);

            struct { int W; int H; int pass; int radius; float sigma_s; float sigma_r; } denoise_pc_h;
            denoise_pc_h.W = W;
            denoise_pc_h.H = H;
            denoise_pc_h.pass = 0;
            denoise_pc_h.radius = denoise_radius;
            denoise_pc_h.sigma_s = denoise_sigma_s;
            denoise_pc_h.sigma_r = denoise_sigma_r;
            vkCmdPushConstants(cb, pl_denoise, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(denoise_pc_h), &denoise_pc_h);
            if(ts_enabled && qp_ts!=VK_NULL_HANDLE) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, qp_ts, 0);
            vkCmdDispatch(cb, denoise_gx, denoise_gy, 1);
            if(ts_enabled && qp_ts!=VK_NULL_HANDLE) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, qp_ts, 1);

            VkImageMemoryBarrier denoise_bar_pass = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
            denoise_bar_pass.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            denoise_bar_pass.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            denoise_bar_pass.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            denoise_bar_pass.subresourceRange.levelCount = 1;
            denoise_bar_pass.subresourceRange.layerCount = 1;
            denoise_bar_pass.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            denoise_bar_pass.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            denoise_bar_pass.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            denoise_bar_pass.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            denoise_bar_pass.image = denoise_temp;

            vkCmdPipelineBarrier(cb,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 0, NULL, 0, NULL, 1, &denoise_bar_pass
            );

            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl_denoise, 0, 1, &ds_denoise_v, 0, NULL);

            struct { int W; int H; int pass; int radius; float sigma_s; float sigma_r; } denoise_pc_v;
            denoise_pc_v.W = W;
            denoise_pc_v.H = H;
            denoise_pc_v.pass = 1;
            denoise_pc_v.radius = denoise_radius;
            denoise_pc_v.sigma_s = denoise_sigma_s;
            denoise_pc_v.sigma_r = denoise_sigma_r;
            vkCmdPushConstants(cb, pl_denoise, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(denoise_pc_v), &denoise_pc_v);
            vkCmdDispatch(cb, denoise_gx, denoise_gy, 1);
            if(ts_enabled && qp_ts!=VK_NULL_HANDLE) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, qp_ts, 2);
        }

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_tm);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl_tm, 0, 1, &ds_tm, 0, NULL);

        struct { int W; int H; float exposure; float gamma; } tm_pc;
        tm_pc.W = W; tm_pc.H = H; tm_pc.exposure = tm_exposure; tm_pc.gamma = tm_gamma;
        vkCmdPushConstants(cb, pl_tm, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(tm_pc), &tm_pc);

        uint32_t tgx = (uint32_t)((W + 15) / 16);
        uint32_t tgy = (uint32_t)((H + 15) / 16);
        vkCmdDispatch(cb, tgx, tgy, 1);
    }

    // --- GPU Denoiser (separable bilateral filter) ---
    if(gpu_denoise_enabled && pipe_denoise != VK_NULL_HANDLE){
        // Transition out_img for denoising
        VkImageMemoryBarrier denoise_bar_pre = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        denoise_bar_pre.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        denoise_bar_pre.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        denoise_bar_pre.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        denoise_bar_pre.subresourceRange.levelCount = 1;
        denoise_bar_pre.subresourceRange.layerCount = 1;
        denoise_bar_pre.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        denoise_bar_pre.newLayout = VK_IMAGE_LAYOUT_GENERAL;  // Already GENERAL, but safe barrier
        denoise_bar_pre.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        denoise_bar_pre.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        denoise_bar_pre.image = out_img;

        vkCmdPipelineBarrier(cb,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, NULL, 0, NULL, 1, &denoise_bar_pre
        );

        uint32_t denoise_gx = (uint32_t)((W + 15) / 16);
        uint32_t denoise_gy = (uint32_t)((H + 15) / 16);

        // --- Horizontal pass ---
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_denoise);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl_denoise, 0, 1, &ds_denoise_h, 0, NULL);

        struct { int W; int H; int pass; int radius; float sigma_s; float sigma_r; } denoise_pc_h;
        denoise_pc_h.W = W;
        denoise_pc_h.H = H;
        denoise_pc_h.pass = 0;  // horizontal
        denoise_pc_h.radius = denoise_radius;
        denoise_pc_h.sigma_s = denoise_sigma_s;
        denoise_pc_h.sigma_r = denoise_sigma_r;
        vkCmdPushConstants(cb, pl_denoise, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(denoise_pc_h), &denoise_pc_h);
        vkCmdDispatch(cb, denoise_gx, denoise_gy, 1);

        // Barrier between passes
        VkImageMemoryBarrier denoise_bar_pass = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        denoise_bar_pass.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        denoise_bar_pass.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        denoise_bar_pass.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        denoise_bar_pass.subresourceRange.levelCount = 1;
        denoise_bar_pass.subresourceRange.layerCount = 1;
        denoise_bar_pass.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        denoise_bar_pass.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        denoise_bar_pass.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        denoise_bar_pass.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        denoise_bar_pass.image = denoise_temp;

        vkCmdPipelineBarrier(cb,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, NULL, 0, NULL, 1, &denoise_bar_pass
        );

        // --- Vertical pass ---
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_denoise);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl_denoise, 0, 1, &ds_denoise_v, 0, NULL);

        struct { int W; int H; int pass; int radius; float sigma_s; float sigma_r; } denoise_pc_v;
        denoise_pc_v.W = W;
        denoise_pc_v.H = H;
        denoise_pc_v.pass = 1;  // vertical
        denoise_pc_v.radius = denoise_radius;
        denoise_pc_v.sigma_s = denoise_sigma_s;
        denoise_pc_v.sigma_r = denoise_sigma_r;
        vkCmdPushConstants(cb, pl_denoise, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(denoise_pc_v), &denoise_pc_v);
        vkCmdDispatch(cb, denoise_gx, denoise_gy, 1);

        fprintf(stderr, "[GPU] denoiser dispatch done (bilateral h+v passes)\n");
    }



    // Readback resources (created only if READBACK enabled)
    VkDeviceSize out_bytes = 0;
    VkBuffer read_buf = VK_NULL_HANDLE;
    VkDeviceMemory read_mem = VK_NULL_HANDLE;

    // Temporal mode: skip readback for most frames, only readback every N frames
    int do_readback = readback_enabled && (!temporal_enabled || (frames % readback_skip == 0));

    if(do_readback){
        // Transition out_img for copy: GENERAL -> TRANSFER_SRC_OPTIMAL
        VkImageMemoryBarrier out_bar = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        VkImage src_copy_img = tonemap_enabled ? ldr_img : out_img;
        out_bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        out_bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        out_bar.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        out_bar.subresourceRange.levelCount = 1;
        out_bar.subresourceRange.layerCount = 1;
        out_bar.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        out_bar.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        out_bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        out_bar.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        out_bar.image = src_copy_img;

        vkCmdPipelineBarrier(cb,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, NULL, 0, NULL, 1, &out_bar
        );

        // Readback buffer
        out_bytes = (VkDeviceSize)W * (VkDeviceSize)H * (tonemap_enabled ? 4 : 16); // RGBA8 or RGBA32F
        read_buf = create_buffer(dev, out_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        read_mem = alloc_bind_buffer_mem(
            phy, dev, read_buf,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        );

        VkBufferImageCopy bic = {0};
        bic.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        bic.imageSubresource.layerCount = 1;
        bic.imageExtent.width = (uint32_t)W;
        bic.imageExtent.height = (uint32_t)H;
        bic.imageExtent.depth = 1;

        vkCmdCopyImageToBuffer(cb, src_copy_img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, read_buf, 1, &bic);
    } else {
        if(temporal_enabled && readback_enabled){
            fprintf(stderr, "[GPU] temporal mode: skipped readback (will readback every %d frames)\n", readback_skip);
        } else {
            fprintf(stderr, "[GPU] readback disabled (set YSU_GPU_READBACK=1 / YSU_GPU_WRITE=1 to enable)\n");
        }
    }

    submit_and_wait(dev, queue, cb);

    // ---------- Read counters ----------
    if(counters_enabled){
        uint32_t counters[2] = {0,0};
        void* ctr_read = NULL;
        r = vkMapMemory(dev, ctr_mem, 0, ctr_bytes, 0, &ctr_read);
        if(r!=VK_SUCCESS) die("vkMapMemory(ctr read)", r);
        memcpy(counters, ctr_read, sizeof(counters));
        vkUnmapMemory(dev, ctr_mem);

        fprintf(stderr, "[GPU] Counters: nodeVisits=%u triTests=%u\n",
                counters[0], counters[1]);
    } else {
        fprintf(stderr, "[GPU] Counters: disabled\n");
    }

    // ---------- Read timestamps ----------
    if(ts_enabled && qp_ts!=VK_NULL_HANDLE){
        uint64_t ts[3] = {0,0,0};
        VkResult qr = vkGetQueryPoolResults(dev, qp_ts, 0, 3, sizeof(ts), ts, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        if(qr==VK_SUCCESS){
            double h_ms = (double)(ts[1]-ts[0]) * ts_period_ns * 1e-6;
            double v_ms = (double)(ts[2]-ts[1]) * ts_period_ns * 1e-6;
            fprintf(stderr, "[GPU][denoise] h=%.3f ms v=%.3f ms\n", h_ms, v_ms);
        } else {
            fprintf(stderr, "[GPU][denoise] timestamp readback failed (qr=%d)\n", qr);
        }
    }

    // ---------- Dump output_gpu.ppm ----------
    if(write_enabled && readback_enabled && read_buf!=VK_NULL_HANDLE){
        void* out_map = NULL;
        r = vkMapMemory(dev, read_mem, 0, out_bytes, 0, &out_map);
        if(r!=VK_SUCCESS) die("vkMapMemory(read)", r);

        FILE* outf = fopen("output_gpu.ppm", "wb");
        if(!outf){
            fprintf(stderr,"Cannot write output_gpu.ppm\n");
            return 1;
        }

        if(strcmp(outfmt,"ppm")!=0){
            fprintf(stderr, "[GPU] unsupported OUTFMT=%s (only ppm in this build)\n", outfmt);
        }

        fprintf(outf, "P6\n%d %d\n255\n", W, H);

        // Convert GPU output to Vec3 array for denoising
        Vec3 *pixels_cpu = (Vec3*)malloc((size_t)W * (size_t)H * sizeof(Vec3));
        if(!pixels_cpu){
            fprintf(stderr, "[GPU] malloc failed for denoising buffer\n");
            fclose(outf);
            vkUnmapMemory(dev, read_mem);
            return 1;
        }

        if(tonemap_enabled){
            const uint8_t* px8 = (const uint8_t*)out_map; // BGRA8
            for(int i=0;i<W*H;i++){
                // Convert from BGRA8 to linear RGB
                pixels_cpu[i].x = (float)(px8[i*4 + 2]) / 255.0f;  // R
                pixels_cpu[i].y = (float)(px8[i*4 + 1]) / 255.0f;  // G
                pixels_cpu[i].z = (float)(px8[i*4 + 0]) / 255.0f;  // B
            }
        } else {
            const float* px = (const float*)out_map; // RGBA32F
            for(int i=0;i<W*H;i++){
                pixels_cpu[i].x = px[i*4 + 0];
                pixels_cpu[i].y = px[i*4 + 1];
                pixels_cpu[i].z = px[i*4 + 2];
                // Clamp to [0,1]
                if(pixels_cpu[i].x < 0) pixels_cpu[i].x = 0; if(pixels_cpu[i].x > 1) pixels_cpu[i].x = 1;
                if(pixels_cpu[i].y < 0) pixels_cpu[i].y = 0; if(pixels_cpu[i].y > 1) pixels_cpu[i].y = 1;
                if(pixels_cpu[i].z < 0) pixels_cpu[i].z = 0; if(pixels_cpu[i].z > 1) pixels_cpu[i].z = 1;
            }
        }

        // Apply denoiser if enabled
        ysu_neural_denoise_maybe(pixels_cpu, W, H);

        // Write denoised pixels to PPM
        for(int i=0;i<W*H;i++){
            unsigned char rgb[3];
            float r = pixels_cpu[i].x;
            float g = pixels_cpu[i].y;
            float b = pixels_cpu[i].z;
            if(r < 0) r = 0; if(g < 0) g = 0; if(b < 0) b = 0;
            if(r > 1) r = 1; if(g > 1) g = 1; if(b > 1) b = 1;
            rgb[0] = (unsigned char)(255.0f * r);
            rgb[1] = (unsigned char)(255.0f * g);
            rgb[2] = (unsigned char)(255.0f * b);
            fwrite(rgb, 1, 3, outf);
        }

        free(pixels_cpu);

        fclose(outf);

        vkUnmapMemory(dev, read_mem);

        fprintf(stderr, "[GPU] wrote output_gpu.ppm (%dx%d %s) SPP=%d useBVH=%d nodes=%d tris=%d\n",
            W, H, tonemap_enabled ? "RGBA8" : "RGBA32F", spp, use_bvh, node_count, tri_count
        );
    } else {
        fprintf(stderr, "[GPU] output write skipped (WRITE=%d READBACK=%d read_buf=%s)\n", write_enabled, readback_enabled, (read_buf!=VK_NULL_HANDLE) ? "ok" : "null");
    }

    // ---------- Cleanup ----------
    if(read_buf!=VK_NULL_HANDLE) vkDestroyBuffer(dev, read_buf, NULL);
    if(read_mem!=VK_NULL_HANDLE) vkFreeMemory(dev, read_mem, NULL);

    vkDestroyPipeline(dev, pipe, NULL);
    vkDestroyShaderModule(dev, sm, NULL);

    vkDestroyDescriptorPool(dev, dp, NULL);
    vkDestroyPipelineLayout(dev, pl, NULL);
    vkDestroyDescriptorSetLayout(dev, dsl, NULL);

    if(pipe_tm!=VK_NULL_HANDLE) vkDestroyPipeline(dev, pipe_tm, NULL);
    if(sm_tm!=VK_NULL_HANDLE) vkDestroyShaderModule(dev, sm_tm, NULL);
    if(dp_tm!=VK_NULL_HANDLE) vkDestroyDescriptorPool(dev, dp_tm, NULL);
    if(pl_tm!=VK_NULL_HANDLE) vkDestroyPipelineLayout(dev, pl_tm, NULL);
    if(dsl_tm!=VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(dev, dsl_tm, NULL);

    // GPU denoiser cleanup
    if(pipe_denoise!=VK_NULL_HANDLE) vkDestroyPipeline(dev, pipe_denoise, NULL);
    if(sm_denoise!=VK_NULL_HANDLE) vkDestroyShaderModule(dev, sm_denoise, NULL);
    if(dp_denoise!=VK_NULL_HANDLE) vkDestroyDescriptorPool(dev, dp_denoise, NULL);
    if(pl_denoise!=VK_NULL_HANDLE) vkDestroyPipelineLayout(dev, pl_denoise, NULL);
    if(dsl_denoise!=VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(dev, dsl_denoise, NULL);
    if(denoise_temp_view!=VK_NULL_HANDLE) vkDestroyImageView(dev, denoise_temp_view, NULL);
    if(denoise_temp!=VK_NULL_HANDLE) vkDestroyImage(dev, denoise_temp, NULL);
    if(denoise_temp_mem!=VK_NULL_HANDLE) vkFreeMemory(dev, denoise_temp_mem, NULL);

    if(qp_ts!=VK_NULL_HANDLE) vkDestroyQueryPool(dev, qp_ts, NULL);
    }

cleanup:
    vkDestroyBuffer(dev, ctr_buf, NULL);
    vkFreeMemory(dev, ctr_mem, NULL);

    vkDestroyBuffer(dev, roots_buf, NULL);
    vkFreeMemory(dev, roots_mem, NULL);

    vkDestroyBuffer(dev, nerf_hash_buf, NULL);
    vkFreeMemory(dev, nerf_hash_mem, NULL);
    vkDestroyBuffer(dev, nerf_occ_buf, NULL);
    vkFreeMemory(dev, nerf_occ_mem, NULL);
    vkDestroyBuffer(dev, depth_hints_buf, NULL);
    vkFreeMemory(dev, depth_hints_mem, NULL);

    vkDestroyBuffer(dev, idx_buf, NULL);
    vkFreeMemory(dev, idx_mem, NULL);

    vkDestroyBuffer(dev, bvh_buf, NULL);
    vkFreeMemory(dev, bvh_mem, NULL);

    vkDestroyBuffer(dev, tri_buf, NULL);
    vkFreeMemory(dev, tri_mem, NULL);

    vkDestroyImageView(dev, accum_view, NULL);
    vkDestroyImageView(dev, out_view, NULL);
    if(ldr_view!=VK_NULL_HANDLE) vkDestroyImageView(dev, ldr_view, NULL);
    vkDestroyImage(dev, accum_img, NULL);
    vkDestroyImage(dev, out_img, NULL);
    if(ldr_img!=VK_NULL_HANDLE) vkDestroyImage(dev, ldr_img, NULL);
    vkFreeMemory(dev, accum_mem, NULL);
    vkFreeMemory(dev, out_mem, NULL);
    if(ldr_mem!=VK_NULL_HANDLE) vkFreeMemory(dev, ldr_mem, NULL);

    vkFreeCommandBuffers(dev, pool, 1, &cb);
    vkDestroyCommandPool(dev, pool, NULL);

    // --- Window resources ---
    if(sw_views){
        for(uint32_t i=0;i<sw_count;i++){
            if(sw_views[i]) vkDestroyImageView(dev, sw_views[i], NULL);
        }
        free(sw_views);
        sw_views = NULL;
    }
    if(swapchain!=VK_NULL_HANDLE) vkDestroySwapchainKHR(dev, swapchain, NULL);
    if(sw_images){ free(sw_images); sw_images=NULL; }
    if(surface!=VK_NULL_HANDLE) vkDestroySurfaceKHR(inst, surface, NULL);
    if(window){ glfwDestroyWindow(window); window=NULL; }
    if(window_enabled){ glfwTerminate(); }

    if(sched_ok){
        nerf_scheduler_free(&sched_q);
    }

    if(nerf_hash.data){ free(nerf_hash.data); nerf_hash.data = NULL; }
    if(nerf_occ.data){ free(nerf_occ.data); nerf_occ.data = NULL; }

    vkDestroyDevice(dev, NULL);
    vkDestroyInstance(inst, NULL);

    return 0;
}
