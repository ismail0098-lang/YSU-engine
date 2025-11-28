#ifndef MATERIAL_H
#define MATERIAL_H

#include "vec3.h"
#include "ray.h"

typedef enum {
    MAT_LAMBERTIAN = 0,
    MAT_METAL      = 1
} MaterialType;

typedef struct {
    MaterialType type;
    Vec3 albedo;   // temel renk
    float fuzz;    // metal için yansıma bulanıklığı (0 = ayna, 1 = çok dağınık)
} Material;

// Işık saçılımını hesapla.
// Girdi:  mat       -> vurulan materyal
//         in_ray    -> gelen ışın
//         hit_point -> çarpışma noktası
//         normal    -> yüzey normali
// Çıktı: scattered  -> saçılan yeni ışın
//        attenuation-> renk çarpanı (albedo)
// Dönüş: 1 = saçıldı, 0 = yutuldu (absorbe)
int material_scatter(const Material *mat,
                     Ray in_ray,
                     Vec3 hit_point,
                     Vec3 normal,
                     Ray *scattered,
                     Vec3 *attenuation);

#endif
