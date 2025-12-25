// material.h
#ifndef MATERIAL_H
#define MATERIAL_H

#include <stdbool.h>
#include "vec3.h"
#include "ray.h"

// --------------------------------------
// Material Types
// --------------------------------------
typedef enum {
    MAT_LAMBERTIAN = 0,
    MAT_METAL,
    MAT_DIELECTRIC,  // Cam
    MAT_EMISSIVE     // Işık saçan
} MaterialType;

// --------------------------------------
// Material Struct
// --------------------------------------
typedef struct Material {
    MaterialType type;
    Vec3 albedo;    // temel renk
    float fuzz;     // metal için pürüzlülük (0 = çok düzgün)
    float ref_idx;  // dielektrik için kırılma indisi
    Vec3 emission;  // emissive ise yayılan renk
} Material;

// --------------------------------------
// Scatter
// --------------------------------------
// in          : gelen ışın
// hit_point   : çarpma noktası
// normal      : yüzey normali
// scattered   : çıkan yeni ray
// attenuation : rengin çarpanı
bool material_scatter(const Material *mat,
                      Ray in,
                      Vec3 hit_point,
                      Vec3 normal,
                      Ray *scattered,
                      Vec3 *attenuation);

#endif // MATERIAL_H
