#ifndef VEC3_H
#define VEC3_H

typedef struct {
    float x;
    float y;
    float z;
} Vec3;

// Constructor
Vec3 vec3(float x, float y, float z);

// Basic operations
Vec3 vec3_add(Vec3 a, Vec3 b);
Vec3 vec3_sub(Vec3 a, Vec3 b);
Vec3 vec3_mul(Vec3 a, Vec3 b);       // component-wise multiply
Vec3 vec3_scale(Vec3 v, float s);

// Vector math
float vec3_dot(Vec3 a, Vec3 b);
Vec3 vec3_cross(Vec3 a, Vec3 b);
float vec3_length(Vec3 v);
Vec3 vec3_normalize(Vec3 v);

// Utility
Vec3 vec3_reflect(Vec3 v, Vec3 n);

#endif
