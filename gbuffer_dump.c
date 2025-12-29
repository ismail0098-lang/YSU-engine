// gbuffer_dump.c - minimal binary dumps for neural pipelines

#include "gbuffer_dump.h"

#include <stdint.h>
#include <stdio.h>

typedef struct {
    char     magic[4];   // "YSUB"
    uint32_t version;    // 1
    uint32_t width;
    uint32_t height;
    uint32_t channels;   // 3 RGB, 1 single
    uint32_t dtype;      // 1 = float32
} YSU_BinHeader;

static int ysu_write_header(FILE *f, uint32_t w, uint32_t h, uint32_t c) {
    if (!f) return 0;
    YSU_BinHeader hdr;
    hdr.magic[0] = 'Y';
    hdr.magic[1] = 'S';
    hdr.magic[2] = 'U';
    hdr.magic[3] = 'B';
    hdr.version  = 1u;
    hdr.width    = w;
    hdr.height   = h;
    hdr.channels = c;
    hdr.dtype    = 1u; // float32
    size_t n = fwrite(&hdr, sizeof(hdr), 1, f);
    return (n == 1) ? 1 : 0;
}

int ysu_dump_rgb32(const char *path, const Vec3 *rgb, int width, int height)
{
    if (!path || !rgb || width <= 0 || height <= 0) return 0;

    FILE *f = fopen(path, "wb");
    if (!f) return 0;

    if (!ysu_write_header(f, (uint32_t)width, (uint32_t)height, 3u)) {
        fclose(f);
        return 0;
    }

    // Interleaved RGB float32
    for (int i = 0; i < width * height; ++i) {
        float v[3] = { rgb[i].x, rgb[i].y, rgb[i].z };
        if (fwrite(v, sizeof(float), 3, f) != 3) {
            fclose(f);
            return 0;
        }
    }

    fclose(f);
    return 1;
}

int ysu_dump_f32(const char *path, const float *buf, int width, int height)
{
    if (!path || !buf || width <= 0 || height <= 0) return 0;

    FILE *f = fopen(path, "wb");
    if (!f) return 0;

    if (!ysu_write_header(f, (uint32_t)width, (uint32_t)height, 1u)) {
        fclose(f);
        return 0;
    }

    size_t count = (size_t)width * (size_t)height;
    if (fwrite(buf, sizeof(float), count, f) != count) {
        fclose(f);
        return 0;
    }

    fclose(f);
    return 1;
}
