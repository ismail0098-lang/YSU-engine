// ysub_to_ppm.c - convert YSUB float32 buffer to PPM (for quick inspection)
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
    char     magic[4];   // "YSUB"
    uint32_t version;    // 1
    uint32_t width;
    uint32_t height;
    uint32_t channels;   // 3
    uint32_t dtype;      // 1 = float32
} YSU_BinHeader;

static float clamp01(float x){ return x < 0.f ? 0.f : (x > 1.f ? 1.f : x); }

// basit gamma
static float gamma22(float x){ return powf(clamp01(x), 1.f/2.2f); }

int main(int argc, char** argv){
    const char* in  = (argc > 1) ? argv[1] : "output_color.ysub";
    const char* out = (argc > 2) ? argv[2] : "ysub_preview.ppm";

    FILE* f = fopen(in, "rb");
    if(!f){ printf("cannot open %s\n", in); return 1; }

    YSU_BinHeader h;
    if(fread(&h, sizeof(h), 1, f) != 1){ printf("bad header\n"); return 1; }
    if(h.magic[0]!='Y'||h.magic[1]!='S'||h.magic[2]!='U'||h.magic[3]!='B'){ printf("not YSUB\n"); return 1; }
    if(h.dtype != 1 || h.channels != 3){ printf("expected float32 RGB\n"); return 1; }

    size_t npx = (size_t)h.width * (size_t)h.height;
    float* rgb = (float*)malloc(npx * 3 * sizeof(float));
    if(!rgb){ printf("alloc fail\n"); return 1; }

    if(fread(rgb, sizeof(float), npx*3, f) != npx*3){ printf("read fail\n"); return 1; }
    fclose(f);

    FILE* o = fopen(out, "wb");
    if(!o){ printf("cannot open %s\n", out); return 1; }

    fprintf(o, "P6\n%u %u\n255\n", h.width, h.height);

    for(size_t i=0;i<npx;i++){
        float r = gamma22(rgb[i*3+0]);
        float g = gamma22(rgb[i*3+1]);
        float b = gamma22(rgb[i*3+2]);
        unsigned char px[3] = {
            (unsigned char)(r*255.f + 0.5f),
            (unsigned char)(g*255.f + 0.5f),
            (unsigned char)(b*255.f + 0.5f)
        };
        fwrite(px, 1, 3, o);
    }

    fclose(o);
    free(rgb);

    printf("wrote %s\n", out);
    return 0;
}
