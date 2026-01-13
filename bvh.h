#ifndef BVH_H
#define BVH_H

#include <stdbool.h>
#include <stdint.h>

#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "primitives.h"   // HitRecord

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
    aabb box;
    int start;              // spheres[start ... start+count-1]
    int count;              // Leaf ise >0, iç node ise 0
    struct bvh_node* left;
    struct bvh_node* right;

    // ===== HEDEF-0 ÖLÇÜM =====
    uint32_t visit_count;
    uint32_t useful_count;
    uint32_t depth;

    // ===== PASS-2 (policy) =====
    uint32_t id;            // preorder node id (CSV policy için)
    uint8_t  prune;         // 1 => bu subtree prune edilecek
} bvh_node;

// -----------------------------
//      Global ölçüm sayaçları
// -----------------------------
extern uint64_t g_bvh_node_visits;
extern uint64_t g_bvh_aabb_tests;

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
//      CSV dump (HEDEF-0)
//  (PASS-2 için öneri: dump'a node_id ekleyeceğiz)
// -----------------------------
void bvh_dump_stats(const char* path, const bvh_node* root);

// -----------------------------
//      PASS-2 Helpers
// -----------------------------
// Root'a preorder id atar (policy dosyasındaki node_id bununla eşleşir)
void bvh_assign_ids(bvh_node* root);

// CSV formatı:
// node_id,prune
// 12,1
// 19,0
// ...
// Return: işaretlenen prune node sayısı
int bvh_load_policy_csv(const char* path, bvh_node* root);

// -----------------------------
//         Bellek Temizleme
// -----------------------------
void bvh_free(bvh_node* node);

#endif // BVH_H
