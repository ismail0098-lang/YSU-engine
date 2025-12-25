// primitives.c
#include "primitives.h"
#include <math.h>

// Bu dosya yalnızca "diğer primitive" fonksiyonlarını sağlar.
// triangle_make() ve hit_triangle() zaten triangle.c içinde tanımlı.
// Bu yüzden burada TEKRAR tanımlamıyoruz (aksi halde multiple definition olur).

static HitRecord no_hit_record(void) {
    HitRecord rec;
    rec.hit = 0;
    rec.t = 0.0f;
    rec.point  = vec3(0.0f, 0.0f, 0.0f);
    rec.normal = vec3(0.0f, 0.0f, 0.0f);
    rec.material_index = -1;

    // primitives.h alanları:
    rec.u = 0.0f;
    rec.v = 0.0f;
    rec.b0 = 0.0f;
    rec.b1 = 0.0f;
    rec.b2 = 0.0f;

    return rec;
}

// primitives.h prototipleri void* olduğu için şimdilik "stub".
// Sonra Plane/Cylinder/Box structlarını tanımlayıp gerçek hit fonksiyonlarını yazabiliriz.

HitRecord hit_plane(void* pl, Ray r, float t_min, float t_max) {
    (void)pl; (void)r; (void)t_min; (void)t_max;
    return no_hit_record();
}

HitRecord hit_cylinder(void* cy, Ray r, float t_min, float t_max) {
    (void)cy; (void)r; (void)t_min; (void)t_max;
    return no_hit_record();
}

HitRecord hit_box(void* b, Ray r, float t_min, float t_max) {
    (void)b; (void)r; (void)t_min; (void)t_max;
    return no_hit_record();
}
