#include <math.h>
#include <stdlib.h>
#include "material.h"

// 0–1 arası rastgele float
static float rand_float() {
    return (float)rand() / (float)RAND_MAX;
}

// Rastgele vektör [-1,1] küpünde
static Vec3 random_in_unit_sphere() {
    while (1) {
        Vec3 p = vec3(rand_float()*2.0f - 1.0f,
                      rand_float()*2.0f - 1.0f,
                      rand_float()*2.0f - 1.0f);
        if (vec3_dot(p, p) >= 1.0f) continue;
        return p;
    }
}

// Ünitel normal etrafında rastgele yön
static Vec3 random_unit_vector() {
    Vec3 p = random_in_unit_sphere();
    return vec3_normalize(p);
}

int material_scatter(const Material *mat,
                     Ray in_ray,
                     Vec3 hit_point,
                     Vec3 normal,
                     Ray *scattered,
                     Vec3 *attenuation)
{
    if (!mat || !scattered || !attenuation) return 0;

    switch (mat->type) {
        case MAT_LAMBERTIAN: {
            // Diffuse: normal etrafında rastgele bir yön
            Vec3 scatter_dir = vec3_add(normal, random_unit_vector());

            // "degenerate" durumda fallback yap
            if (vec3_length(scatter_dir) < 1e-8f) {
                scatter_dir = normal;
            }

            *scattered = ray_create(hit_point, vec3_normalize(scatter_dir));
            *attenuation = mat->albedo;
            return 1;
        }

        case MAT_METAL: {
            // Metal: yansıma + fuzz
            Vec3 reflected = vec3_reflect(vec3_normalize(in_ray.direction), normal);
            Vec3 fuzz_vec = vec3_scale(random_in_unit_sphere(), mat->fuzz);
            Vec3 dir = vec3_add(reflected, fuzz_vec);

            *scattered = ray_create(hit_point, vec3_normalize(dir));
            *attenuation = mat->albedo;

            // Eğer içe doğru saçıldıysa, ışığı yut
            if (vec3_dot(scattered->direction, normal) <= 0.0f) {
                return 0;
            }
            return 1;
        }

        default:
            return 0;
    }
}
