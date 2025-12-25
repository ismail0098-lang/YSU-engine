#include "bvh.h"
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <stdbool.h>

// ----- Helpers -----

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

aabb sphere_bounds(const Sphere* s) {
    Vec3 r = vec3(s->radius, s->radius, s->radius);
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

// AABB - Ray intersection (slab method)
bool aabb_hit(const aabb* box, const Ray* r, double t_min, double t_max) {
    const double EPS = 1e-12;
    double origin[3]    = { r->origin.x,    r->origin.y,    r->origin.z    };
    double direction[3] = { r->direction.x, r->direction.y, r->direction.z };
    double bmin[3]      = { box->minimum.x, box->minimum.y, box->minimum.z };
    double bmax[3]      = { box->maximum.x, box->maximum.y, box->maximum.z };

    for (int a = 0; a < 3; ++a) {
        if (fabs(direction[a]) < EPS) {
            // parallel: if origin not inside slab -> miss
            if (origin[a] < bmin[a] || origin[a] > bmax[a]) return false;
            continue;
        }
        double invD = 1.0 / direction[a];
        double t0 = (bmin[a] - origin[a]) * invD;
        double t1 = (bmax[a] - origin[a]) * invD;
        if (t0 > t1) { double tmp = t0; t0 = t1; t1 = tmp; }
        if (t0 > t_min) t_min = t0;
        if (t1 < t_max) t_max = t1;
        if (t_max <= t_min) return false;
    }
    return true;
}

// ----- BVH Build -----

bvh_node* bvh_build(Sphere* spheres, int start, int end) {
    if (start >= end) return NULL;

    int object_span = end - start;
    bvh_node* node = (bvh_node*)malloc(sizeof(bvh_node));
    if (!node) return NULL;

    node->left = node->right = NULL;
    node->start = start;
    node->count = object_span;

    if (object_span <= 2) {
        // leaf
        aabb box = sphere_bounds(&spheres[start]);
        if (object_span == 2) {
            aabb box2 = sphere_bounds(&spheres[start+1]);
            box = aabb_surrounding(box, box2);
        }
        node->box = box;
        return node;
    }

    // compute bounding box and choose axis
    aabb total_box = sphere_bounds(&spheres[start]);
    for (int i = start + 1; i < end; ++i)
        total_box = aabb_surrounding(total_box, sphere_bounds(&spheres[i]));

    double dx = total_box.maximum.x - total_box.minimum.x;
    double dy = total_box.maximum.y - total_box.minimum.y;
    double dz = total_box.maximum.z - total_box.minimum.z;

    int axis = 2;
    if (dx > dy && dx > dz) axis = 0;
    else if (dy > dz) axis = 1;

    // simple in-place sort by center coordinate along axis
    for (int i = start; i < end - 1; ++i) {
        for (int j = i + 1; j < end; ++j) {
            double ci = (axis==0) ? spheres[i].center.x : (axis==1) ? spheres[i].center.y : spheres[i].center.z;
            double cj = (axis==0) ? spheres[j].center.x : (axis==1) ? spheres[j].center.y : spheres[j].center.z;
            if (ci > cj) { Sphere tmp = spheres[i]; spheres[i] = spheres[j]; spheres[j] = tmp; }
        }
    }

    int mid = start + object_span / 2;
    node->left  = bvh_build(spheres, start, mid);
    node->right = bvh_build(spheres, mid, end);
    node->count = 0;
    node->box = aabb_surrounding(node->left->box, node->right->box);
    return node;
}

// ----- BVH Hit -----
// Use your project's sphere intersection. render.c uses sphere_intersect(...) that returns HitRecord.
// Declare it here so BVH can reuse it.

extern HitRecord sphere_intersect(Sphere s, Ray r, float t_min, float t_max);

bool bvh_hit(
    const bvh_node* node,
    const Sphere* spheres,
    const Ray* r,
    double t_min,
    double t_max,
    HitRecord* rec
) {
    if (!node) return false;
    if (!aabb_hit(&node->box, r, t_min, t_max)) return false;

    bool hit_any = false;
    double closest = t_max;
    HitRecord tmp;

    if (node->count > 0) {
        for (int i = 0; i < node->count; ++i) {
            const Sphere* s = &spheres[node->start + i];
            HitRecord hr = sphere_intersect(*s, *r, (float)t_min, (float)closest);
            if (hr.hit && hr.t < closest) {
                hit_any = true;
                closest = hr.t;
                *rec = hr;
            }
        }
        return hit_any;
    }

    if (node->left && bvh_hit(node->left, spheres, r, t_min, closest, &tmp)) {
        hit_any = true;
        closest = tmp.t;
        *rec = tmp;
    }
    if (node->right && bvh_hit(node->right, spheres, r, t_min, closest, &tmp)) {
        hit_any = true;
        *rec = tmp;
    }
    return hit_any;
}

void bvh_free(bvh_node* node) {
    if (!node) return;
    bvh_free(node->left);
    bvh_free(node->right);
    free(node);
}
