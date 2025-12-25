#ifndef BVH_H
#define BVH_H

#include <stdbool.h>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"

// -----------------------------
//   Axis-Aligned Bounding Box
// -----------------------------
typedef struct {
    Vec3 minimum;
    Vec3 maximum;
} aabb;

// -----------------------------
//        BVH Node Yapısı
// -----------------------------
typedef struct bvh_node {
    aabb box;              // Bu node'un kapsayan AABB'si
    int start;             // spheres[start ... start+count-1]
    int count;             // Leaf ise >0, iç node ise 0
    struct bvh_node* left; // İç node
    struct bvh_node* right;// İç node
} bvh_node;

// -----------------------------
//      AABB Yardımcı Fonksiyonları
// -----------------------------
aabb aabb_surrounding(aabb b0, aabb b1);
aabb sphere_bounds(const Sphere* s);
bool aabb_hit(const aabb* box, const Ray* r, double t_min, double t_max);

// -----------------------------
//      BVH Build & Hit Test
// -----------------------------
bvh_node* bvh_build(Sphere* spheres, int start, int end);

bool bvh_hit(
    const bvh_node* node,
    const Sphere* spheres,
    const Ray* r,
    double t_min,
    double t_max,
    HitRecord* rec
);

// -----------------------------
//         Bellek Temizleme
// -----------------------------
void bvh_free(bvh_node* node);

#endif // BVH_H