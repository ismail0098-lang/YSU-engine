#include <stdio.h>
#include <stdint.h>

int main() {
    FILE *f = fopen("window_dump.ppm", "rb");
    if (!f) { printf("Cannot open window_dump.ppm\n"); return 1; }
    
    char magic[3] = {0};
    int w, h;
    fscanf(f, "%2s %d %d %d", magic, &w, &h, (int[]){0});
    
    printf("PPM Magic: %s\n", magic);
    printf("Dimensions: %d x %d\n", w, h);
    printf("Expected size: %d bytes data + ~50 header\n", w * h * 3);
    
    // Read pixel data and compute stats
    uint8_t pixel[3];
    uint32_t pixCount = 0;
    uint32_t minR = 255, minG = 255, minB = 255;
    uint32_t maxR = 0, maxG = 0, maxB = 0;
    uint64_t sumR = 0, sumG = 0, sumB = 0;
    uint32_t zeroCount = 0;  // pixels that are all black
    
    while (fread(pixel, 1, 3, f) == 3) {
        minR = pixel[0] < minR ? pixel[0] : minR;
        minG = pixel[1] < minG ? pixel[1] : minG;
        minB = pixel[2] < minB ? pixel[2] : minB;
        maxR = pixel[0] > maxR ? pixel[0] : maxR;
        maxG = pixel[1] > maxG ? pixel[1] : maxG;
        maxB = pixel[2] > maxB ? pixel[2] : maxB;
        sumR += pixel[0];
        sumG += pixel[1];
        sumB += pixel[2];
        if (pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 0) zeroCount++;
        pixCount++;
    }
    
    printf("\nPixel Statistics:\n");
    printf("Total pixels: %u\n", pixCount);
    printf("Black pixels: %u (%.1f%%)\n", zeroCount, 100.0f * zeroCount / pixCount);
    printf("R: min=%u max=%u avg=%.1f\n", minR, maxR, (float)sumR / pixCount);
    printf("G: min=%u max=%u avg=%.1f\n", minG, maxG, (float)sumG / pixCount);
    printf("B: min=%u max=%u avg=%.1f\n", minB, maxB, (float)sumB / pixCount);
    
    fclose(f);
    return 0;
}
