#include <math.h>  // ← for powf(), fmin(), fmax()
#include "color.h"

// Clamp helper
static float clamp(float x, float min, float max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

// Apply gamma correction
Vec3 color_gamma(Vec3 c) {
    // gamma = 2.2 → corrected = c^(1/2.2)
    float gamma = 1.0f / 2.2f;

    return vec3(
        powf(clamp(c.x, 0.0f, 1.0f), gamma),
        powf(clamp(c.y, 0.0f, 1.0f), gamma),
        powf(clamp(c.z, 0.0f, 1.0f), gamma)
    );
}

// Clamp all components to 0–1
Vec3 color_clamp(Vec3 c) {
    return vec3(
        clamp(c.x, 0.0f, 1.0f),
        clamp(c.y, 0.0f, 1.0f),
        clamp(c.z, 0.0f, 1.0f)
    );
}

// Convert Vec3 → 0–255 RGB integers
void color_to_rgb(Vec3 color, int *r, int *g, int *b) {
    Vec3 c = color_clamp(color);
    c = color_gamma(c);

    *r = (int)(255.999f * c.x);
    *g = (int)(255.999f * c.y);
    *b = (int)(255.999f * c.z);
}
