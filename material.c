// material.c
#include <math.h>
#include <stdlib.h>
#include <time.h>

#include "material.h"
#include "vec3.h"
#include "ray.h"

static float rand01(void) {
    return (float)rand() / (float)RAND_MAX;
}

static Vec3 random_in_unit_sphere(void) {
    while (1) {
        Vec3 p = vec3_random(-1.0f, 1.0f);
        if (vec3_length_squared(p) >= 1.0f) continue;
        return p;
    }
}

static Vec3 random_unit_vector(void) {
    return vec3_unit(random_in_unit_sphere());
}

static Vec3 reflect(Vec3 v, Vec3 n) {
    return vec3_reflect(v, n);
}

static Vec3 refract(Vec3 uv, Vec3 n, float etai_over_etat) {
    float cos_theta = fminf(vec3_dot(vec3_scale(uv, -1.0f), n), 1.0f);
    Vec3 r_out_perp  = vec3_scale(vec3_add(uv, vec3_scale(n, cos_theta)), etai_over_etat);
    float k = 1.0f - vec3_length_squared(r_out_perp);
    if (k < 0.0f) {
        // total internal reflection
        return reflect(uv, n);
    }
    Vec3 r_out_parallel = vec3_scale(n, -sqrtf(fabsf(k)));
    return vec3_add(r_out_perp, r_out_parallel);
}

static float schlick(float cosine, float ref_idx) {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * powf(1.0f - cosine, 5.0f);
}

bool material_scatter(const Material *mat,
                      Ray in,
                      Vec3 hit_point,
                      Vec3 normal,
                      Ray *scattered,
                      Vec3 *attenuation)
{
    // Emissive: sadece ışık, scatter yok
    if (mat->type == MAT_EMISSIVE) {
        *attenuation = vec3(0.0f, 0.0f, 0.0f);
        return false;
    }

    Vec3 unit_dir = vec3_unit(in.direction);

    switch (mat->type) {
        case MAT_LAMBERTIAN: {
            Vec3 scatter_dir = vec3_add(normal, random_unit_vector());

            // çok küçük vektör (nümerik sorun) olursa normal'e geri düş
            if (vec3_length_squared(scatter_dir) < 1e-8f) {
                scatter_dir = normal;
            }

            *scattered = ray(hit_point, scatter_dir);
            *attenuation = mat->albedo;
        } break;

        case MAT_METAL: {
            Vec3 reflected = reflect(unit_dir, normal);
            Vec3 perturbed = vec3_add(reflected,
                                      vec3_scale(random_in_unit_sphere(), mat->fuzz));
            *scattered = ray(hit_point, perturbed);
            *attenuation = mat->albedo;

            // metal için: içeri doğru yansıma varsa absorbe et
            if (vec3_dot(scattered->direction, normal) <= 0.0f) {
                return false;
            }
        } break;

        case MAT_DIELECTRIC: {
            *attenuation = vec3(1.0f, 1.0f, 1.0f); // cam renksiz kabul

            float refraction_ratio = vec3_dot(unit_dir, normal) > 0.0f
                                   ? mat->ref_idx
                                   : 1.0f / mat->ref_idx;

            float cos_theta = fminf(vec3_dot(vec3_scale(unit_dir, -1.0f), normal), 1.0f);
            float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));

            bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
            Vec3 direction;

            if (cannot_refract || schlick(cos_theta, refraction_ratio) > rand01()) {
                direction = reflect(unit_dir, normal);
            } else {
                direction = refract(unit_dir, normal, 1.0f / refraction_ratio);
            }

            *scattered = ray(hit_point, direction);
        } break;

        default:
            return false;
    }

    return true;
}
