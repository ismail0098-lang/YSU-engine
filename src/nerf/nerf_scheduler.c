#include "nerf_scheduler.h"
#include <stdio.h>
#include <string.h>

int nerf_scheduler_init(NerfScheduleQueues* q, uint32_t batch_size){
    if(!q || batch_size == 0) return 0;
    memset(q, 0, sizeof(*q));
    if(!nerf_ray_batch_init(&q->gpu, batch_size)) return 0;
    if(!nerf_ray_batch_init(&q->cpu, batch_size)){
        nerf_ray_batch_free(&q->gpu);
        return 0;
    }
    return 1;
}

void nerf_scheduler_free(NerfScheduleQueues* q){
    if(!q) return;
    nerf_ray_batch_free(&q->gpu);
    nerf_ray_batch_free(&q->cpu);
    memset(q, 0, sizeof(*q));
}

void nerf_schedule_split(const NerfScheduleConfig* cfg,
                         const NerfRayBatch* frame_rays,
                         NerfScheduleQueues* out){
    if(!cfg || !frame_rays || !out) return;

    out->gpu.count = 0;
    out->cpu.count = 0;

    // Placeholder: even/odd split until foveation + cost model is added
    for(uint32_t i = 0; i < frame_rays->count; i++){
        const int to_gpu = ((i & 1u) == 0u);
        NerfRayBatch* dst = to_gpu ? &out->gpu : &out->cpu;
        if(dst->count >= dst->capacity){
            fprintf(stderr, "[NERF] scheduler: %s queue full (%u/%u), ray %u dropped\n",
                    to_gpu ? "GPU" : "CPU", dst->count, dst->capacity, i);
            continue;
        }

        uint32_t j = dst->count++;
        dst->pix[j] = frame_rays->pix[i];
        dst->ox[j] = frame_rays->ox[i];
        dst->oy[j] = frame_rays->oy[i];
        dst->oz[j] = frame_rays->oz[i];
        dst->dx[j] = frame_rays->dx[i];
        dst->dy[j] = frame_rays->dy[i];
        dst->dz[j] = frame_rays->dz[i];
        dst->tmin[j] = frame_rays->tmin[i];
        dst->tmax[j] = frame_rays->tmax[i];
    }
}
