#ifndef IMAGE_H
#define IMAGE_H

#include "vec3.h"

// Write PPM image (P6 binary). If env YSU_POSTFX=1 or YSU_BLOOM=1 is set,
// Bloom+Tonemap (ACES) is applied using postprocess.c.
void image_write_ppm(const char *filename, int width, int height, Vec3 *pixels);

// Convert HDR Vec3 buffer to 8-bit RGB (malloc eder, free sende)
unsigned char* image_rgb_from_hdr(const Vec3 *pixels, int width, int height);

// Write PNG (8-bit RGB). Requires stb_image_write.h in project folder.
void image_write_png(const char *filename, int width, int height, const unsigned char *rgb_u8);

#endif
