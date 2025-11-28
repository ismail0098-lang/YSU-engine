#include "bvh.h"
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <stdbool.h>

// ------ YARDIMCI FONKSIYONLAR ------

static double min3(double a, double b, double c) {
    if (a < b && a < c) return a;
    if (b < c) return b;
    return c;
}

static double max3(double a, double b, double c) {
    if (a > b && a > c) return a;
    if (b > c) return b;
    return c;
}

aabb sphere_bounds(const sphere* s) {
    vec3 r = vec3_new(s->radius, s->radius, s->radius);
    aabb box;
    box.minimum = vec3_sub(s->center, r);
    box.maximum = vec3_add(s->center, r);
    return box;
}

aabb aabb_surrounding(aabb b0, aabb b1) {
    aabb box;
    box.minimum = vec3(
        fmin(b0.minimum.x, b1.minimum.x),
        fmin(b0.minimum.y, b1.minimum.y),
        fmin(b0.minimum.z, b1.minimum.z)
    );
    box.maximum = vec3(
        fmax(b0.maximum.x, b1.maximum.x),
        fmax(b0.maximum.y, b1.maximum.y),
        fmax(b0.maximum.z, b1.maximum.z)
    );
    return box;
}

// AABB - ray intersection (slab method)
bool aabb_hit(const aabb* box, const ray* r, double t_min, double t_max) {
    const double EPS = 1e-12;
    double origin[3]    = { r->origin.x,    r->origin.y,    r->origin.z    };
    double direction[3] = { r->direction.x, r->direction.y, r->direction.z };
    double bmin[3]      = { box->minimum.x, box->minimum.y, box->minimum.z };
    double bmax[3]      = { box->maximum.x, box->maximum.y, box->maximum.z };

    for (int a = 0; a < 3; a++) {
        if (fabs(direction[a]) < EPS) {
            // Ray is parallel to slab. If origin not within slab -> miss.
            if (origin[a] < bmin[a] || origin[a] > bmax[a]) return false;
            // otherwise this axis imposes no interval constraint
            continue;
        }

        double invD = 1.0 / direction[a];
        double t0 = (bmin[a] - origin[a]) * invD;
        double t1 = (bmax[a] - origin[a]) * invD;
        if (t0 > t1) {
            double tmp = t0; t0 = t1; t1 = tmp;
        }
        if (t0 > t_min) t_min = t0;
        if (t1 < t_max) t_max = t1;
        if (t_max <= t_min) return false;
    }
    return true;
}

// ------ BVH KURMA ------

// Küre merkezine göre eksen seçmek için comparator
typedef struct {
    sphere* spheres;
    int axis;
} sphere_sort_context;

static int sphere_compare_func(const void* a, const void* b, void* ctx_ptr) {
    const sphere_sort_context* ctx = (const sphere_sort_context*)ctx_ptr;
    const sphere* sA = &ctx->spheres[*(const int*)a];
    const sphere* sB = &ctx->spheres[*(const int*)b];

    double cA, cB;
    if (ctx->axis == 0) {
        cA = sA->center.x;
        cB = sB->center.x;
    } else if (ctx->axis == 1) {
        cA = sA->center.y;
        cB = sB->center.y;
    } else {
        cA = sA->center.z;
        cB = sB->center.z;
    }

    if (cA < cB) return -1;
    if (cA > cB) return  1;
    return 0;
}

// Basitlik için: spheres[start..end) dizisini yerinde sıralıyoruz
bvh_node* bvh_build(sphere* spheres, int start, int end) {
    int object_span = end - start;

    bvh_node* node = (bvh_node*)malloc(sizeof(bvh_node));
    node->left = NULL;
    node->right = NULL;
    node->start = start;
    node->count = object_span;

    if (object_span <= 2) {
        // Leaf node: bir veya iki obje
        aabb box = sphere_bounds(&spheres[start]);
        if (object_span == 2) {
            aabb box2 = sphere_bounds(&spheres[start + 1]);
            box = aabb_surrounding(box, box2);
        }
        node->box = box;
        return node;
    }

    // Split axis: en geniş boyut
    aabb total_box = sphere_bounds(&spheres[start]);
    for (int i = start + 1; i < end; i++) {
        total_box = aabb_surrounding(total_box, sphere_bounds(&spheres[i]));
    }

    double dx = total_box.maximum.x - total_box.minimum.x;
    double dy = total_box.maximum.y - total_box.minimum.y;
    double dz = total_box.maximum.z - total_box.minimum.z;

    int axis;
    if (dx > dy && dx > dz)      axis = 0;
    else if (dy > dz)           axis = 1;
    else                        axis = 2;

    // Qsort ile eksene göre sırala
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
    // C11: qsort_s / qsort_r platforma göre değişiyor.
    // En garantisi: basit qsort kullanıp context'i global yapmamak için
    // spheres[start..end) dilimini direkt sıralıyoruz.
#endif

    // Sade yaklaşım: insertion sort tarzı basit sort
    for (int i = start; i < end - 1; i++) {
        for (int j = i + 1; j < end; j++) {
            double ci, cj;
            if (axis == 0) {
                ci = spheres[i].center.x;
                cj = spheres[j].center.x;
            } else if (axis == 1) {
                ci = spheres[i].center.y;
                cj = spheres[j].center.y;
            } else {
                ci = spheres[i].center.z;
                cj = spheres[j].center.z;
            }
            if (ci > cj) {
                sphere tmp = spheres[i];
                spheres[i] = spheres[j];
                spheres[j] = tmp;
            }
        }
    }

    int mid = start + object_span / 2;

    node->left  = bvh_build(spheres, start, mid);
    node->right = bvh_build(spheres, mid, end);

    node->count = 0; // iç node
    node->box   = aabb_surrounding(node->left->box, node->right->box);

    return node;
}

// ------ BVH HIT ------

// Burada projendeki sphere-intersection fonksiyonunu kullanıyoruz.
// Eğer sende ismi "hit_sphere" değilse, aşağıdaki satırı ona göre değiştir.
extern bool hit_sphere(
    const sphere* s,
    const ray* r,
    double t_min,
    double t_max,
    hit_record* rec
);

bool bvh_hit(
    const bvh_node* node,
    const sphere* spheres,
    const ray* r,
    double t_min,
    double t_max,
    hit_record* rec
) {
    if (!node) return false;

    if (!aabb_hit(&node->box, r, t_min, t_max))
        return false;

    bool hit_anything = false;
    double closest = t_max;
    hit_record temp_rec;

    if (node->count > 0) {
        // Leaf: doğrudan küreler üzerinde dolaş
        for (int i = 0; i < node->count; i++) {
            const sphere* s = &spheres[node->start + i];
            if (hit_sphere(s, r, t_min, closest, &temp_rec)) {
                hit_anything = true;
                closest = temp_rec.t;
                *rec = temp_rec;
            }
        }
        return hit_anything;
    }

    // İç node: önce sol, sonra sağ
    if (node->left && bvh_hit(node->left, spheres, r, t_min, closest, &temp_rec)) {
        hit_anything = true;
        closest = temp_rec.t;
        *rec = temp_rec;
    }

    if (node->right && bvh_hit(node->right, spheres, r, t_min, closest, &temp_rec)) {
        hit_anything = true;
        *rec = temp_rec;
    }

    return hit_anything;
}

// ------ BELLEK TEMIZLEME ------

void bvh_free(bvh_node* node) {
    if (!node) return;
    bvh_free(node->left);
    bvh_free(node->right);
    free(node);
}
