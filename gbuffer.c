#include "gbuffer.h"

static YSU_GBuffer g_gb = {0};

void ysu_gbuffer_set_targets(YSU_GBuffer gb) {
    g_gb = gb;
}

// render.c bunu extern ile görecek
YSU_GBuffer ysu_gbuffer_get_targets(void);
YSU_GBuffer ysu_gbuffer_get_targets(void) { return g_gb; }
