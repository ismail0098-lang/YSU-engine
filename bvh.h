#ifndef BVH_H
#define BVH_H

#include <stdbool.h>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"

// Basit Axis-Aligned Bounding Box
typedef struct {
    vec3 minimum;
    vec3 maximum;
} aabb;

// BVH node'u: küre dizisi üstünde aralık tutuyor
typedef struct bvh_node {
    aabb box;
    int start;              // spheres[start ... start+count-1]
    int count;              // leaf ise >0, iç node ise 0
    struct bvh_node* left;  // iç node
    struct bvh_node* right; // iç node
} bvh_node;

// AABB yardımcı fonksiyonları
aabb aabb_surrounding(aabb b0, aabb b1);
aabb sphere_bounds(const sphere* s);
bool aabb_hit(const aabb* box, const ray* r, double t_min, double t_max);

// BVH kurma & vurma
bvh_node* bvh_build(sphere* spheres, int start, int end);
bool bvh_hit(
    const bvh_node* node,
    const sphere* spheres,
    const ray* r,
    double t_min,
    double t_max,
    hit_record* rec
);

// Bellek temizleme
void bvh_free(bvh_node* node);

#endif // BVH_H
