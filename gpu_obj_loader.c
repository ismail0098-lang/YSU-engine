#include "gpu_obj_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

typedef struct { float x,y,z; } V3;

static void* xrealloc(void* p, size_t n) {
    void* q = realloc(p, n);
    if(!q) { fprintf(stderr, "[OBJ] OOM realloc %zu\n", n); exit(1); }
    return q;
}

static int is_space_or_end(char c){ return c=='\0' || isspace((unsigned char)c); }

static int parse_int(const char** s, int* out) {
    while(isspace((unsigned char)**s)) (*s)++;
    int sign = 1;
    if(**s=='-'){ sign=-1; (*s)++; }
    if(!isdigit((unsigned char)**s)) return 0;
    int v=0;
    while(isdigit((unsigned char)**s)){ v = v*10 + (**s - '0'); (*s)++; }
    *out = v*sign;
    return 1;
}

// Parse a single face vertex like:
// "12", "12/3", "12//7", "12/3/7", "-1/.."
// We only care about position index (vi).
static int parse_face_vi(const char** s, int* vi_out) {
    int vi;
    if(!parse_int(s, &vi)) return 0;
    // skip optional /... parts
    if(**s=='/') {
        (*s)++;
        if(**s=='/') { // v//n
            (*s)++;
            // skip normal index
            int tmp;
            parse_int(s, &tmp);
        } else {
            // skip vt
            int tmp;
            parse_int(s, &tmp);
            if(**s=='/') {
                (*s)++;
                // skip vn
                parse_int(s, &tmp);
            }
        }
    }
    *vi_out = vi;
    // advance until space/end
    while(!is_space_or_end(**s)) (*s)++;
    return 1;
}

static int obj_index_to_zero_based(int idx, size_t vcount) {
    // OBJ: 1-based positive, negative counts from end
    if(idx > 0) return idx - 1;
    if(idx < 0) return (int)vcount + idx; // idx is negative
    return -1;
}

static void push_tri(GPUTriangle** tris, size_t* tcount, size_t* tcap,
                     V3 a, V3 b, V3 c)
{
    if(*tcount + 1 > *tcap) {
        *tcap = (*tcap == 0) ? 1024 : (*tcap * 2);
        *tris = (GPUTriangle*)xrealloc(*tris, (*tcap) * sizeof(GPUTriangle));
    }
    GPUTriangle* t = &(*tris)[(*tcount)++];
    t->v0[0]=a.x; t->v0[1]=a.y; t->v0[2]=a.z; t->v0[3]=0.0f;
    t->v1[0]=b.x; t->v1[1]=b.y; t->v1[2]=b.z; t->v1[3]=0.0f;
    t->v2[0]=c.x; t->v2[1]=c.y; t->v2[2]=c.z; t->v2[3]=0.0f;
}

void gpu_make_fallback_cube(GPUTriangle** out_tris, size_t* out_count) {
    // Cube centered at (0,0,-3), size 2
    V3 v[8] = {
        {-1,-1,-4}, {+1,-1,-4}, {+1,+1,-4}, {-1,+1,-4},
        {-1,-1,-2}, {+1,-1,-2}, {+1,+1,-2}, {-1,+1,-2}
    };
    // 12 triangles (two per face)
    static const int idx[12][3] = {
        {0,1,2},{0,2,3}, // back
        {4,6,5},{4,7,6}, // front
        {0,4,5},{0,5,1}, // bottom
        {3,2,6},{3,6,7}, // top
        {0,3,7},{0,7,4}, // left
        {1,5,6},{1,6,2}  // right
    };

    GPUTriangle* tris = (GPUTriangle*)malloc(12 * sizeof(GPUTriangle));
    if(!tris){ fprintf(stderr,"[OBJ] OOM cube\n"); exit(1); }
    for(int i=0;i<12;i++){
        V3 a = v[idx[i][0]];
        V3 b = v[idx[i][1]];
        V3 c = v[idx[i][2]];
        tris[i].v0[0]=a.x; tris[i].v0[1]=a.y; tris[i].v0[2]=a.z; tris[i].v0[3]=0;
        tris[i].v1[0]=b.x; tris[i].v1[1]=b.y; tris[i].v1[2]=b.z; tris[i].v1[3]=0;
        tris[i].v2[0]=c.x; tris[i].v2[1]=c.y; tris[i].v2[2]=c.z; tris[i].v2[3]=0;
    }
    *out_tris = tris;
    *out_count = 12;
}

int gpu_load_obj_triangles(const char* path, GPUTriangle** out_tris, size_t* out_count) {
    *out_tris = NULL;
    *out_count = 0;

    FILE* f = fopen(path, "rb");
    if(!f){
        fprintf(stderr, "[OBJ] Cannot open: %s\n", path);
        return 0;
    }

    V3* verts = NULL;
    size_t vcount = 0, vcap = 0;

    GPUTriangle* tris = NULL;
    size_t tcount = 0, tcap = 0;

    char line[4096];
    while(fgets(line, (int)sizeof(line), f)) {
        const char* s = line;
        while(isspace((unsigned char)*s)) s++;
        if(*s=='#' || *s=='\0') continue;

        // vertex
        if(s[0]=='v' && isspace((unsigned char)s[1])) {
            s++;
            float x,y,z;
            if(sscanf(s, "%f %f %f", &x, &y, &z) == 3) {
                if(vcount + 1 > vcap) {
                    vcap = (vcap==0) ? 4096 : (vcap*2);
                    verts = (V3*)xrealloc(verts, vcap * sizeof(V3));
                }
                verts[vcount++] = (V3){x,y,z};
            }
            continue;
        }

        // face
        if(s[0]=='f' && isspace((unsigned char)s[1])) {
            s++;
            // collect all position indices in this face
            int face_vi[128];
            int fn = 0;
            while(*s) {
                while(isspace((unsigned char)*s)) s++;
                if(*s=='\0' || *s=='\n' || *s=='\r') break;
                if(fn >= (int)(sizeof(face_vi)/sizeof(face_vi[0]))) break;
                int vi;
                if(!parse_face_vi(&s, &vi)) break;
                face_vi[fn++] = vi;
            }
            if(fn < 3) continue;

            // fan triangulation: (0, i, i+1)
            int i0 = obj_index_to_zero_based(face_vi[0], vcount);
            if(i0 < 0 || i0 >= (int)vcount) continue;
            V3 a = verts[i0];

            for(int i=1; i+1<fn; i++){
                int i1 = obj_index_to_zero_based(face_vi[i], vcount);
                int i2 = obj_index_to_zero_based(face_vi[i+1], vcount);
                if(i1<0||i1>=(int)vcount||i2<0||i2>=(int)vcount) continue;
                V3 b = verts[i1];
                V3 c = verts[i2];
                push_tri(&tris, &tcount, &tcap, a, c, b);  // winding flip
            }
            continue;
        }
    }

    fclose(f);

    if(vcount == 0 || tcount == 0) {
        fprintf(stderr, "[OBJ] No triangles loaded from: %s\n", path);
        free(verts);
        free(tris);
        return 0;
    }

    free(verts);

    *out_tris = tris;
    *out_count = tcount;
    fprintf(stderr, "[OBJ] Loaded %zu triangles from %s\n", tcount, path);
    return 1;
}
