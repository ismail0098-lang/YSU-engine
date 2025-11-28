#ifndef COLOR_H
#define COLOR_H

#include "vec3.h"

// Convert a Vec3 color (0–1 range) into 0–255 RGB components
void color_to_rgb(Vec3 color, int *r, int *g, int *b);

// Apply simple gamma correction (gamma = 2.2)
Vec3 color_gamma(Vec3 c);

// Clamp color components (0–1)
Vec3 color_clamp(Vec3 c);

#endif
