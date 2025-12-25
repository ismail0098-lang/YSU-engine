#include <math.h>
#include <stdlib.h>
#include "vec3.h"

Vec3 vec3(float x, float y, float z)
{
    Vec3 v;
    v.x = x;
    v.y = y;
    v.z = z;
    return v;
}

Vec3 vec3_add(Vec3 a, Vec3 b)
{
    return vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

Vec3 vec3_sub(Vec3 a, Vec3 b)
{
    return vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

Vec3 vec3_mul(Vec3 a, Vec3 b)
{
    return vec3(a.x * b.x, a.y * b.y, a.z * b.z);
}

Vec3 vec3_scale(Vec3 a, float s)
{
    return vec3(a.x * s, a.y * s, a.z * s);
}

float vec3_dot(Vec3 a, Vec3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 vec3_cross(Vec3 a, Vec3 b)
{
    return vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

float vec3_length_squared(Vec3 a)
{
    return a.x * a.x + a.y * a.y + a.z * a.z;
}

float vec3_length(Vec3 a)
{
    return sqrtf(vec3_length_squared(a));
}

Vec3 vec3_normalize(Vec3 a)
{
    float len = vec3_length(a);
    if (len <= 0.0f) {
        return vec3(0.0f, 0.0f, 0.0f);
    }
    float inv = 1.0f / len;
    return vec3(a.x * inv, a.y * inv, a.z * inv);
}

Vec3 vec3_reflect(Vec3 v, Vec3 n)
{
    // v - 2 * dot(v, n) * n
    float d = vec3_dot(v, n);
    return vec3_sub(v, vec3_scale(n, 2.0f * d));
}

// Eski kodun kullandığı isim: unit(v) = normalize(v)
Vec3 vec3_unit(Vec3 a)
{
    return vec3_normalize(a);
}

// [min,max] aralığında rastgele vektör
static float rand_float01(void)
{
    return (float)rand() / (float)RAND_MAX;
}

Vec3 vec3_random(float min, float max)
{
    float range = max - min;
    float rx = min + range * rand_float01();
    float ry = min + range * rand_float01();
    float rz = min + range * rand_float01();
    return vec3(rx, ry, rz);
}
