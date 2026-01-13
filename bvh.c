// bvh.c (PASS-2 policy-aware + güvenli sayım + opsiyonel fiziksel pruning)
#include "bvh.h"
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <string.h>

// ===== HEDEF-0 GLOBAL SAYAÇLAR =====
uint64_t g_bvh_node_visits = 0;
uint64_t g_bvh_aabb_tests  = 0;

// ============================================================
// AABB helpers
// ============================================================

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

// slab test (ölçüm dahil)
bool aabb_hit(const aabb* box, const Ray* r, double t_min, double t_max) {
    g_bvh_aabb_tests++;

    const double EPS = 1e-12;
    const double o[3]  = { r->origin.x,    r->origin.y,    r->origin.z };
    const double d[3]  = { r->direction.x, r->direction.y, r->direction.z };
    const double mn[3] = { box->minimum.x, box->minimum.y, box->minimum.z };
    const double mx[3] = { box->maximum.x, box->maximum.y, box->maximum.z };

    for (int i = 0; i < 3; ++i) {
        double di = d[i];
        if (fabs(di) < EPS) {
            if (o[i] < mn[i] || o[i] > mx[i]) return false;
            continue;
        }

        double invD = 1.0 / di;
        double t0 = (mn[i] - o[i]) * invD;
        double t1 = (mx[i] - o[i]) * invD;
        if (t0 > t1) { double tmp = t0; t0 = t1; t1 = tmp; }

        if (t0 > t_min) t_min = t0;
        if (t1 < t_max) t_max = t1;
        if (t_max <= t_min) return false;
    }
    return true;
}

// child ordering için (SAYAÇ ARTIRMAZ)
static inline double aabb_entry_tmin_no_count(const aabb* box, const Ray* r) {
    const double EPS = 1e-12;
    double t_min = -1e30;
    double t_max =  1e30;

    const double o[3]  = { r->origin.x,    r->origin.y,    r->origin.z };
    const double d[3]  = { r->direction.x, r->direction.y, r->direction.z };
    const double mn[3] = { box->minimum.x, box->minimum.y, box->minimum.z };
    const double mx[3] = { box->maximum.x, box->maximum.y, box->maximum.z };

    for (int i = 0; i < 3; ++i) {
        double di = d[i];
        if (fabs(di) < EPS) {
            if (o[i] < mn[i] || o[i] > mx[i]) return 1e30;
            continue;
        }
        double invD = 1.0 / di;
        double t0 = (mn[i] - o[i]) * invD;
        double t1 = (mx[i] - o[i]) * invD;
        if (t0 > t1) { double tmp = t0; t0 = t1; t1 = tmp; }

        if (t0 > t_min) t_min = t0;
        if (t1 < t_max) t_max = t1;
        if (t_max < t_min) return 1e30;
    }
    return t_min;
}

// ============================================================
// BVH BUILD (median split)  - zip'indeki mevcut yaklaşım
// ============================================================

static void sort_spheres_axis(Sphere* spheres, int start, int end, int axis) {
    // Basit sort (1100 sphere için OK).
    for (int i = start; i < end - 1; ++i) {
        for (int j = i + 1; j < end; ++j) {
            double ci = (axis == 0) ? spheres[i].center.x :
                        (axis == 1) ? spheres[i].center.y : spheres[i].center.z;
            double cj = (axis == 0) ? spheres[j].center.x :
                        (axis == 1) ? spheres[j].center.y : spheres[j].center.z;
            if (ci > cj) {
                Sphere tmp = spheres[i];
                spheres[i] = spheres[j];
                spheres[j] = tmp;
            }
        }
    }
}

static bvh_node* bvh_build_rec(Sphere* spheres, int start, int end, uint32_t depth) {
    bvh_node* node = (bvh_node*)malloc(sizeof(bvh_node));
    if (!node) return NULL;

    node->left = node->right = NULL;
    node->start = start;
    node->count = end - start;
    node->depth = depth;

    node->visit_count  = 0;
    node->useful_count = 0;

    node->id = 0;
    node->prune = 0;

    // Leaf: 1-2 sphere
    if (node->count <= 2) {
        aabb box = sphere_bounds(&spheres[start]);
        if (node->count == 2) {
            box = aabb_surrounding(box, sphere_bounds(&spheres[start + 1]));
        }
        node->box = box;
        return node;
    }

    // Range bounds
    aabb box = sphere_bounds(&spheres[start]);
    for (int i = start + 1; i < end; ++i) {
        box = aabb_surrounding(box, sphere_bounds(&spheres[i]));
    }
    node->box = box;

    // Axis: largest extent
    Vec3 ext = vec3_sub(box.maximum, box.minimum);
    int axis = 0;
    if (ext.y > ext.x) axis = 1;
    if (ext.z > ((axis == 0) ? ext.x : ext.y)) axis = 2;

    sort_spheres_axis(spheres, start, end, axis);

    int mid = start + (end - start) / 2;

    node->left  = bvh_build_rec(spheres, start, mid, depth + 1);
    node->right = bvh_build_rec(spheres, mid, end, depth + 1);

    // internal node marker
    node->count = 0;
    node->start = 0;

    if (node->left && node->right) {
        node->box = aabb_surrounding(node->left->box, node->right->box);
    } else if (node->left) {
        node->box = node->left->box;
    } else if (node->right) {
        node->box = node->right->box;
    }

    return node;
}

bvh_node* bvh_build(Sphere* spheres, int start, int end) {
    return bvh_build_rec(spheres, start, end, 0);
}

// ============================================================
// PASS-2: Assign stable preorder node ids (root=0, left, right)
// ============================================================

static void bvh_assign_ids_rec(bvh_node* n, uint32_t* cur) {
    if (!n) return;
    n->id = (*cur)++;
    bvh_assign_ids_rec(n->left, cur);
    bvh_assign_ids_rec(n->right, cur);
}

void bvh_assign_ids(bvh_node* root) {
    uint32_t cur = 0;
    bvh_assign_ids_rec(root, &cur);
}

// ============================================================
// PASS-2: Load policy CSV: "node_id,prune" (header optional)
// ============================================================

typedef struct {
    uint32_t id;
    uint8_t  prune;
} PolicyPair;

static int policy_pair_cmp(const void* a, const void* b) {
    const PolicyPair* pa = (const PolicyPair*)a;
    const PolicyPair* pb = (const PolicyPair*)b;
    if (pa->id < pb->id) return -1;
    if (pa->id > pb->id) return  1;
    return 0;
}

static uint8_t policy_lookup_sorted(const PolicyPair* arr, int n, uint32_t id) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        uint32_t v = arr[mid].id;
        if (v == id) return arr[mid].prune;
        if (v < id) lo = mid + 1;
        else hi = mid - 1;
    }
    return 0; // default: prune kapalı
}

static void bvh_apply_policy_rec(bvh_node* n, const PolicyPair* arr, int n_pairs) {
    if (!n) return;
    n->prune = policy_lookup_sorted(arr, n_pairs, n->id) ? 1 : 0;
    bvh_apply_policy_rec(n->left,  arr, n_pairs);
    bvh_apply_policy_rec(n->right, arr, n_pairs);
}

// ✅ Stack overflow yok: pruned sayımı recursion
static int bvh_count_pruned_rec(const bvh_node* n) {
    if (!n) return 0;
    int c = n->prune ? 1 : 0;
    return c + bvh_count_pruned_rec(n->left) + bvh_count_pruned_rec(n->right);
}

// ============================================================
// Opsiyonel: Fiziksel pruning (default KAPALI)
// ============================================================

#ifndef YSU_POLICY_PHYSICAL_PRUNE
#define YSU_POLICY_PHYSICAL_PRUNE 0
#endif

static void bvh_prune_subtrees_inplace(bvh_node* n) {
    if (!n) return;

    if (n->prune) {
        if (n->left)  { bvh_free(n->left);  n->left  = NULL; }
        if (n->right) { bvh_free(n->right); n->right = NULL; }

        // leaf gibi boşalt (güvenli)
        n->start = 0;
        n->count = 0;
        return;
    }

    bvh_prune_subtrees_inplace(n->left);
    bvh_prune_subtrees_inplace(n->right);
}

int bvh_load_policy_csv(const char* path, bvh_node* root) {
    if (!path || !path[0] || !root) return 0;

    FILE* f = fopen(path, "r");
    if (!f) {
        printf("[BVH] policy: cannot open %s\n", path);
        return 0;
    }

    int cap = 1024;
    int n = 0;
    PolicyPair* pairs = (PolicyPair*)malloc(sizeof(PolicyPair) * (size_t)cap);
    if (!pairs) { fclose(f); return 0; }

    char line[256];

    // header skip: ilk satır digit değilse header say
    long pos0 = ftell(f);
    if (fgets(line, sizeof(line), f)) {
        if (line[0] >= '0' && line[0] <= '9') {
            fseek(f, pos0, SEEK_SET);
        }
    } else {
        fclose(f);
        free(pairs);
        return 0;
    }

    while (fgets(line, sizeof(line), f)) {
        char* p = line;
        while (*p==' ' || *p=='\t') p++;
        if (!(*p>='0' && *p<='9')) continue;

        uint32_t id = 0;
        int pr = 0;
        if (sscanf(p, "%u,%d", &id, &pr) != 2) continue;
        if (pr != 0) pr = 1;

        if (n >= cap) {
            cap *= 2;
            PolicyPair* np = (PolicyPair*)realloc(pairs, sizeof(PolicyPair) * (size_t)cap);
            if (!np) break;
            pairs = np;
        }

        pairs[n].id = id;
        pairs[n].prune = (uint8_t)pr;
        n++;
    }

    fclose(f);

    if (n == 0) {
        free(pairs);
        printf("[BVH] policy: no entries loaded\n");
        return 0;
    }

    qsort(pairs, (size_t)n, sizeof(PolicyPair), policy_pair_cmp);

    bvh_apply_policy_rec(root, pairs, n);

#if YSU_POLICY_PHYSICAL_PRUNE
    // İstersen aç: subtree'leri gerçekten koparır (daha agresif)
    bvh_prune_subtrees_inplace(root);
#endif

    int pruned = bvh_count_pruned_rec(root);

    free(pairs);

    printf("[BVH] policy: loaded %d entries, marked %d pruned nodes\n", n, pruned);
    return pruned;
}

// ============================================================
// BVH HIT (near-first + PASS-2 policy-aware prune)
// ============================================================

// Projende render.c tarafının kullandığı intersection: HitRecord döndürür.
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

    // ✅ PASS-2: prune ise subtree 0 maliyet (ne visit, ne AABB)
    if (node->prune) return false;

    // visit
    g_bvh_node_visits++;
    ((bvh_node*)node)->visit_count++;

    // AABB
    if (!aabb_hit(&node->box, r, t_min, t_max)) return false;

    // Leaf
    if (node->count > 0) {
        bool hit_any = false;
        double closest = t_max;

        for (int i = 0; i < node->count; ++i) {
            HitRecord hr = sphere_intersect(
                spheres[node->start + i], *r,
                (float)t_min, (float)closest
            );
            if (hr.hit && hr.t < closest) {
                hit_any = true;
                closest = hr.t;
                *rec = hr;
            }
        }

        if (hit_any) ((bvh_node*)node)->useful_count++;
        return hit_any;
    }

    // Internal: near-first + second AABB skip (policy-aware)
    const bvh_node* L = node->left;
    const bvh_node* R = node->right;

    // ✅ pruned child varsa yok say
    double tL = (L && !L->prune) ? aabb_entry_tmin_no_count(&L->box, r) : 1e30;
    double tR = (R && !R->prune) ? aabb_entry_tmin_no_count(&R->box, r) : 1e30;

    const bvh_node* first  = (tL < tR) ? L : R;
    const bvh_node* second = (tL < tR) ? R : L;

    bool hit_any = false;
    double closest = t_max;
    HitRecord tmp;

    if (first && !first->prune && bvh_hit(first, spheres, r, t_min, closest, &tmp)) {
        hit_any = true;
        closest = tmp.t;
        *rec = tmp;
    }

    // ✅ second prune ise AABB testini bile yapma
    if (second && !second->prune && aabb_hit(&second->box, r, t_min, closest)) {
        if (bvh_hit(second, spheres, r, t_min, closest, &tmp)) {
            hit_any = true;
            *rec = tmp;
        }
    }

    if (hit_any) ((bvh_node*)node)->useful_count++;
    return hit_any;
}

// ============================================================
// CSV DUMP (node_id dahil)
// ============================================================

static void dump_rec(FILE* f, const bvh_node* n) {
    if (!n) return;
    fprintf(f, "%u,%d,%d,%u,%u,%u\n",
            n->depth, n->start, n->count,
            n->visit_count, n->useful_count,
            n->id);
    dump_rec(f, n->left);
    dump_rec(f, n->right);
}

void bvh_dump_stats(const char* path, const bvh_node* root) {
    FILE* f = fopen(path, "w");
    if (!f) return;
    fprintf(f, "depth,start,count,visits,useful,node_id\n");
    dump_rec(f, root);
    fclose(f);
}

// ============================================================
// FREE
// ============================================================

void bvh_free(bvh_node* node) {
    if (!node) return;
    bvh_free(node->left);
    bvh_free(node->right);
    free(node);
}
