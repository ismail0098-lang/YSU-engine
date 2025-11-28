#include <math.h>
#include "vec3.h"

// Constructor
Vec3 vec3(float x, float y, float z) {
    Vec3 v = { x, y, z };
    return v;
}

// Add
Vec3 vec3_add(Vec3 a, Vec3 b) {
    return vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

// Subtract
Vec3 vec3_sub(Vec3 a, Vec3 b) {
    return vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

// Component-wise multiply
Vec3 vec3_mul(Vec3 a, Vec3 b) {
    return vec3(a.x * b.x, a.y * b.y, a.z * b.z);
}

// Scale
Vec3 vec3_scale(Vec3 v, float s) {
    return vec3(v.x * s, v.y * s, v.z * s);
}

// Dot product
float vec3_dot(Vec3 a, Vec3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

// Cross product
Vec3 vec3_cross(Vec3 a, Vec3 b) {
    return vec3(
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x
    );
}

// Length
float vec3_length(Vec3 v) {
    return sqrtf(vec3_dot(v, v));
}

// Normalize
Vec3 vec3_normalize(Vec3 v) {
    float len = vec3_length(v);
    if (len == 0.0f) return vec3(0,0,0);
    return vec3_scale(v, 1.0f / len);
}

// Reflection: reflect v across normal n
Vec3 vec3_reflect(Vec3 v, Vec3 n) {
    // r = v - 2*(v•n)*n
    float d = vec3_dot(v, n);
    return vec3_sub(v, vec3_scale(n, 2.0f * d));
}
