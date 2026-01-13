#define VK_USE_PLATFORM_WIN32_KHR
#include "gpu_bvh_lbv.h"
#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>

// Include denoiser
#include "bilateral_denoise.h"
#include "neural_denoise.h"

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
    const char* env_w = getenv("YSU_GPU_W");
    const char* env_h = getenv("YSU_GPU_H");
    const char* env_spp = getenv("YSU_GPU_SPP");
    const char* env_seed = getenv("YSU_GPU_SEED");
    const char* env_frames = getenv("YSU_GPU_FRAMES");
    const char* obj_path = getenv("YSU_GPU_OBJ");
    const char* env_use_bvh = getenv("YSU_GPU_USE_BVH"); // default 1
    const char* env_cull = getenv("YSU_GPU_CULL"); // default 1 (backface culling)

    if(env_w) W = atoi(env_w);
    if(env_h) H = atoi(env_h);
    if(env_spp) spp = atoi(env_spp);
    if(env_seed) seed = atoi(env_seed);
    if(env_frames) frames = atoi(env_frames);
    if(frames < 1) frames = 1;

    int use_bvh = 1;
    if(env_use_bvh) use_bvh = atoi(env_use_bvh);

    int cull_backface = 1;
    if(env_cull) cull_backface = atoi(env_cull);

    fprintf(stderr, "[GPU] W=%d H=%d SPP=%d seed=%d\n", W, H, spp, seed);

    // Benchmark toggles (important for performance tests)
    int write_enabled    = ysu_env_bool("YSU_GPU_WRITE", 1);
    int readback_enabled = ysu_env_bool("YSU_GPU_READBACK", write_enabled);
    int counters_enabled = ysu_env_bool("YSU_GPU_COUNTERS", 1);
    if(!write_enabled) readback_enabled = 0;
    fprintf(stderr, "[GPU] toggles: WRITE=%d READBACK=%d COUNTERS_READ=%d\n", write_enabled, readback_enabled, counters_enabled);

    // Output / tonemap
    const char* outfmt = getenv("YSU_GPU_OUTFMT");
    if(!outfmt || !outfmt[0]) outfmt = "ppm";

    int tonemap_enabled = ysu_env_bool("YSU_GPU_TONEMAP", 0);
    if(window_enabled) tonemap_enabled = 1; // swapchain wants UNORM output
    float tm_exposure = ysu_env_float("YSU_GPU_EXPOSURE", 3.0f);
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
        tri_count = 12;
        tri_data = (float*)calloc((size_t)tri_count * 12u, sizeof(float));

        ObjV3 v[8] = {
            {-1,-1,-4}, {+1,-1,-4}, {+1,+1,-4}, {-1,+1,-4},
            {-1,-1,-2}, {+1,-1,-2}, {+1,+1,-2}, {-1,+1,-2}
        };
        int idx[12][3] = {
            {0,1,2},{0,2,3},
            {4,6,5},{4,7,6},
            {0,4,5},{0,5,1},
            {3,2,6},{3,6,7},
            {0,3,7},{0,7,4},
            {1,5,6},{1,6,2}
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


    // We can free CPU BVH arrays after upload
    if(bvh_nodes){ free(bvh_nodes); bvh_nodes = NULL; }
    if(bvh_indices){ free(bvh_indices); bvh_indices = NULL; }
    if(bvh_roots){ free(bvh_roots); bvh_roots = NULL; }


    // ---------- Descriptor set layout: binding0 outImg, binding1 accumImg, binding2 triBuf, binding3 bvhBuf ----------
    VkDescriptorSetLayoutBinding binds[7];
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

    VkDescriptorSetLayoutCreateInfo dlci = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    dlci.bindingCount = 7;
    dlci.pBindings = binds;

    VkDescriptorSetLayout dsl = 0;
    r = vkCreateDescriptorSetLayout(dev, &dlci, NULL, &dsl);
    if(r!=VK_SUCCESS) die("vkCreateDescriptorSetLayout", r);

    VkDescriptorPoolSize dps[2];
    dps[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    dps[0].descriptorCount = 2;
    dps[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    dps[1].descriptorCount = 5; // tri + bvh + idx + counters + roots

    VkDescriptorPoolCreateInfo dpci = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    dpci.maxSets = 1;
    dpci.poolSizeCount = 2;
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

    VkWriteDescriptorSet ws[7];
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

    vkUpdateDescriptorSets(dev, 7, ws, 0, NULL);

    // ---------- Pipeline layout + push constants ----------
    VkPushConstantRange pcr = {0};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = 64; // allow future expansion (we use 9 ints now incl. rootCount)

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

    int frame_id = 0;
    while(window && !glfwWindowShouldClose(window)){
        glfwPollEvents();

        vkWaitForFences(dev, 1, &in_flight, VK_TRUE, UINT64_MAX);
        vkResetFences(dev, 1, &in_flight);

        uint32_t imageIndex = 0;
        r = vkAcquireNextImageKHR(dev, swapchain, UINT64_MAX, image_avail, VK_NULL_HANDLE, &imageIndex);
        if(r==VK_ERROR_OUT_OF_DATE_KHR) break;
        if(r!=VK_SUCCESS && r!=VK_SUBOPTIMAL_KHR) die("vkAcquireNextImageKHR", r);

        vkResetCommandBuffer(cb, 0);

        VkCommandBufferBeginInfo cbi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        r = vkBeginCommandBuffer(cb, &cbi);
        if(r!=VK_SUCCESS) die("vkBeginCommandBuffer", r);

        // --- Dispatch raytrace compute ---
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);

        // Optional bypass: clear out_img to a solid color (useful to test copy/present path)
        int bypass_shader = ysu_env_bool("YSU_GPU_BYPASS_SHADER", 0);
        if(bypass_shader){
            VkClearColorValue ccol;
            ccol.float32[0] = 1.0f; ccol.float32[1] = 0.0f; ccol.float32[2] = 1.0f; ccol.float32[3] = 1.0f; // magenta
            VkImageSubresourceRange rrange = {0};
            rrange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            rrange.baseMipLevel = 0; rrange.levelCount = 1;
            rrange.baseArrayLayer = 0; rrange.layerCount = 1;
            vkCmdClearColorImage(cb, out_img, VK_IMAGE_LAYOUT_GENERAL, &ccol, 1, &rrange);
        }

        uint32_t gx = (uint32_t)((W + 15) / 16);
        uint32_t gy = (uint32_t)((H + 15) / 16);

        // We accumulate 'frames' samples per present tick (YSU_GPU_FRAMES)
        for(int f=0; f<frames; f++){
            // push constants layout matches shaders/tri.comp (ints + floats packed)
            int enable_counters = counters_enabled;

            // IMPORTANT: keep this push layout in sync with tri.comp
            // (This file's original push constant code is kept as-is.)
            int push_i[16] = {0};
            push_i[0]=W;
            push_i[1]=H;
            push_i[2]=frame_id + f;
            push_i[3]=seed;
            push_i[4]=tri_count;
            push_i[5]=node_count;
            push_i[6]=use_bvh;
            push_i[7]=cull_backface;
            push_i[8]=(int)bvh_root_count;
            push_i[9]=enable_counters;
            // push_i[10..] reserved

            if(!bypass_shader){
                vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_i), push_i);
                vkCmdDispatch(cb, gx, gy, 1);
            }
        }

        // --- Barrier between raytrace and tonemap ---
        if(tonemap_enabled){
            VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
            mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cb,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &mb, 0, NULL, 0, NULL
            );
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

        VkImageCopy ic = {0};
        ic.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        ic.srcSubresource.layerCount = 1;
        ic.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        ic.dstSubresource.layerCount = 1;
        ic.extent.width = sw_extent.width;
        ic.extent.height = sw_extent.height;
        ic.extent.depth = 1;
        vkCmdCopyImage(cb, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &ic);

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
        if(r==VK_ERROR_OUT_OF_DATE_KHR || r==VK_SUBOPTIMAL_KHR) break;
        if(r!=VK_SUCCESS) die("vkQueuePresentKHR", r);

        // optionally dump a single-frame PPM and exit (use YSU_GPU_DUMP_ONESHOT=1)
        if(ysu_env_bool("YSU_GPU_DUMP_ONESHOT", 0)){
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

        // continue to next frame

        frame_id += frames;
    }

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
        int enable_counters = counters_enabled;
        int push_i[16] = {0};
        push_i[0]=W;
        push_i[1]=H;
        push_i[2]=f;
        push_i[3]=seed;
        push_i[4]=(int)tri_count;
        push_i[5]=node_count;
        push_i[6]=use_bvh;
        push_i[7]=cull_backface;
        push_i[8]=(int)bvh_root_count;
        push_i[9]=enable_counters;
        vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_i), push_i);
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

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_tm);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl_tm, 0, 1, &ds_tm, 0, NULL);

        struct { int W; int H; float exposure; float gamma; } tm_pc;
        tm_pc.W = W; tm_pc.H = H; tm_pc.exposure = tm_exposure; tm_pc.gamma = tm_gamma;
        vkCmdPushConstants(cb, pl_tm, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(tm_pc), &tm_pc);

        uint32_t tgx = (uint32_t)((W + 15) / 16);
        uint32_t tgy = (uint32_t)((H + 15) / 16);
        vkCmdDispatch(cb, tgx, tgy, 1);
    }



    // Readback resources (created only if READBACK enabled)
    VkDeviceSize out_bytes = 0;
    VkBuffer read_buf = VK_NULL_HANDLE;
    VkDeviceMemory read_mem = VK_NULL_HANDLE;

    if(readback_enabled){
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
        fprintf(stderr, "[GPU] readback disabled (set YSU_GPU_READBACK=1 / YSU_GPU_WRITE=1 to enable)\n");
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
    }

cleanup:
    vkDestroyBuffer(dev, ctr_buf, NULL);
    vkFreeMemory(dev, ctr_mem, NULL);

    vkDestroyBuffer(dev, roots_buf, NULL);
    vkFreeMemory(dev, roots_mem, NULL);

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

    vkDestroyDevice(dev, NULL);
    vkDestroyInstance(inst, NULL);

    return 0;
}
