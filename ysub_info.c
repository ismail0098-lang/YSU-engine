#include <stdio.h>
#include <stdint.h>

typedef struct {
    char     magic[4];
    uint32_t version;
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    uint32_t dtype;
} YSU_BinHeader;

int main(int argc, char** argv){
    const char* in = (argc > 1) ? argv[1] : "output_color.ysub";
    FILE* f = fopen(in, "rb");
    if(!f){ printf("cannot open %s\n", in); return 1; }

    YSU_BinHeader h;
    if(fread(&h, sizeof(h), 1, f) != 1){ printf("bad header\n"); return 1; }
    fclose(f);

    printf("magic=%c%c%c%c version=%u w=%u h=%u ch=%u dtype=%u\n",
           h.magic[0], h.magic[1], h.magic[2], h.magic[3],
           h.version, h.width, h.height, h.channels, h.dtype);
    return 0;
}
