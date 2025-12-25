#ifndef VEC3_H
#define VEC3_H

typedef struct {
    float x;
    float y;
    float z;
} Vec3;

// Temel vektör fonksiyonları
Vec3 vec3(float x, float y, float z);
Vec3 vec3_add(Vec3 a, Vec3 b);
Vec3 vec3_sub(Vec3 a, Vec3 b);
Vec3 vec3_mul(Vec3 a, Vec3 b);       // bileşen çarpımı
Vec3 vec3_scale(Vec3 a, float s);    // skaler çarpım
float vec3_dot(Vec3 a, Vec3 b);
Vec3 vec3_cross(Vec3 a, Vec3 b);
float vec3_length(Vec3 a);
Vec3 vec3_normalize(Vec3 a);
Vec3 vec3_reflect(Vec3 v, Vec3 n);

// --- Eski kodun istediği ekstra helper'lar ---
float vec3_length_squared(Vec3 a);
Vec3 vec3_unit(Vec3 a);
Vec3 vec3_random(float min, float max);

#endif
