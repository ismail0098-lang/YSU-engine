#ifndef IMAGE_H
#define IMAGE_H

#include "vec3.h"

// Write PPM image (ASCII P3 format)
void image_write_ppm(const char *filename, int width, int height, Vec3 *pixels);

#endif
