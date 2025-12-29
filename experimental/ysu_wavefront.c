// ysu_wavefront.c
#include "ysu_wavefront.h"
#include <stdlib.h>
#include <string.h>

void ysu_queue_init(YSU_PathQueue* q, uint32_t capacity) {
    q->items = (YSU_Path*)calloc((size_t)capacity, sizeof(YSU_Path));
    q->count = 0;
    q->capacity = capacity;
}

void ysu_queue_free(YSU_PathQueue* q) {
    free(q->items);
    q->items = NULL;
    q->count = 0;
    q->capacity = 0;
}

void ysu_queue_clear(YSU_PathQueue* q) { q->count = 0; }

int ysu_queue_push(YSU_PathQueue* q, const YSU_Path* p) {
    if (q->count >= q->capacity) return 0;
    q->items[q->count++] = *p;
    return 1;
}

int ysu_wavefront_init(YSU_WavefrontState* st, uint32_t max_paths) {
    memset(st, 0, sizeof(*st));
    ysu_queue_init(&st->q_active, max_paths);
    ysu_queue_init(&st->q_next,   max_paths);
    st->hits = (YSU_SurfHit*)calloc((size_t)max_paths, sizeof(YSU_SurfHit));
    return st->hits != NULL;
}

void ysu_wavefront_free(YSU_WavefrontState* st) {
    ysu_queue_free(&st->q_active);
    ysu_queue_free(&st->q_next);
    free(st->hits);
    st->hits = NULL;
}

void ysu_wavefront_render(const YSU_WavefrontSettings* s,
                          YSU_WavefrontState* st,
                          YSU_IntersectCB intersect_cb,
                          YSU_ShadeCB shade_cb,
                          void* user)
{
    // This does not generate rays itself.
    // You will wire:
    //   - per-tile or per-thread primary ray generation into st->q_active
    //   - then call this function per tile / per batch

    for (uint32_t bounce = 0; bounce < s->max_depth; ++bounce) {
        uint32_t n = st->q_active.count;
        if (n == 0) break;

        // 1) Intersect (BVH / packets / etc)
        intersect_cb(st->q_active.items, n, st->hits, user);

        // 2) Shade / scatter (your materials + RR + throughput update)
        ysu_queue_clear(&st->q_next);
        shade_cb(st->q_active.items, st->hits, n, &st->q_next, user);

        // 3) Swap queues
        YSU_PathQueue tmp = st->q_active;
        st->q_active = st->q_next;
        st->q_next = tmp;
    }
}
