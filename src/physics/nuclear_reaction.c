/* nuclear_reaction.c — Animated nuclear fission & fusion visualization.
 *
 * CPU-side: scripted keyframe animation of nucleon positions.
 * GPU-side: Vulkan compute pipeline evaluating Gaussian nucleon density
 *           on a 3D grid shared with the quantum raymarch renderer.
 *
 * Physics basis:
 *   Fission: liquid-drop model deformation (quadrupole β₂ → scission)
 *   Fusion:  Coulomb approach, quantum tunneling, compound nucleus
 *   Nuclear radius: R = 1.25 × A^(1/3) fm
 *   Nucleon Gaussian width: σ ≈ 0.55 fm
 */

#include "nuclear_reaction.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ═══════════════════════ RNG ═══════════════════════ */

static unsigned int nr_seed = 0;

static void nr_srand(unsigned int s) { nr_seed = s; }

static float nr_rand01(void) {
    nr_seed = nr_seed * 1103515245u + 12345u;
    return (float)((nr_seed >> 16) & 0x7FFF) / 32768.0f;
}

#if 0 /* reserved for future stochastic simulation */
static float nr_randn(void) {
    /* Box-Muller */
    float u1 = nr_rand01() + 1e-7f;
    float u2 = nr_rand01();
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2832f * u2);
}
#endif

/* ═══════════════════════ PHASE TIMING ═══════════════════════ */
/* Cumulative end-times for each phase, per reaction type. */

static const float PHASE_END[NR_REACTION_COUNT][7] = {
    /* U-235 fission: idle, approach, excite, deform, scission, separate, done */
    { 2.0f, 4.0f, 5.5f, 8.5f, 10.0f, 14.0f, 100.0f },
    /* D-T fusion */
    { 1.5f, 3.5f, 5.0f, 7.0f, 8.5f, 12.0f, 100.0f },
    /* D-D fusion */
    { 1.5f, 3.5f, 5.0f, 7.0f, 8.5f, 12.0f, 100.0f },
};

/* ═══════════════════════ NUCLEON PLACEMENT ═══════════════════════ */

static float nr_nuclear_radius(int A) {
    return NR_R0_FM * cbrtf((float)A);
}

/* Place nucleons uniformly in a sphere, protons first.
 * Returns number placed.  Appends to nr->nucleons starting at *offset. */
static int place_nucleons_sphere(NR_Nucleon *out, int max_out,
                                 int Z, int N_neutrons,
                                 float cx, float cy, float cz,
                                 int group_id) {
    int A = Z + N_neutrons;
    if (A > max_out) A = max_out;
    float R = nr_nuclear_radius(A);
    int placed = 0;

    for (int i = 0; i < A; i++) {
        int ok = 0;
        for (int attempt = 0; attempt < 200; attempt++) {
            /* Uniform random point in sphere */
            float u = nr_rand01();
            float ct = 2.0f * nr_rand01() - 1.0f;
            float st = sqrtf(1.0f - ct * ct);
            float phi = 6.2832f * nr_rand01();
            float r = R * cbrtf(u);

            float x = cx + r * st * cosf(phi);
            float y = cy + r * ct;
            float z = cz + r * st * sinf(phi);

            /* Check minimum separation */
            int collision = 0;
            for (int j = 0; j < placed; j++) {
                float dx = x - out[j].x;
                float dy = y - out[j].y;
                float dz = z - out[j].z;
                if (dx*dx + dy*dy + dz*dz < NR_MIN_SEP_FM * NR_MIN_SEP_FM) {
                    collision = 1; break;
                }
            }
            if (!collision) {
                NR_Nucleon *n = &out[placed];
                n->x = x;  n->y = y;  n->z = z;
                n->radius = 0.85f; /* proton rms charge radius [CODATA 2018: 0.841fm] */
                n->type = (i < Z) ? 1 : 0;
                n->group = group_id;
                n->pad0 = n->pad1 = 0.0f;
                placed++;
                ok = 1;
                break;
            }
        }
        if (!ok) {
            /* Force-place without separation check as fallback */
            float phi = 6.2832f * (float)i / (float)A;
            float ct = 1.0f - 2.0f * (float)i / (float)A;
            float st = sqrtf(1.0f - ct * ct);
            float r = R * cbrtf(nr_rand01());
            NR_Nucleon *n = &out[placed];
            n->x = cx + r * st * cosf(phi);
            n->y = cy + r * ct;
            n->z = cz + r * st * sinf(phi);
            n->radius = 0.85f;
            n->type = (i < Z) ? 1 : 0;
            n->group = group_id;
            n->pad0 = n->pad1 = 0.0f;
            placed++;
        }
    }
    return placed;
}

/* ═══════════════════════ REACTION SETUP ═══════════════════════ */

static void setup_fission_u235(NuclearReaction *nr) {
    nr_srand((unsigned int)time(NULL));
    nr->numNucleons = 0;

    /* U-235 nucleus: Z=92, N=143, A=235 */
    int count = place_nucleons_sphere(nr->nucleons, NR_MAX_NUCLEONS,
                                      92, 143, 0.0f, 0.0f, 0.0f, 0);
    nr->numNucleons = count;

    /* Save base positions for animation */
    memcpy(nr->base_pos, nr->nucleons, sizeof(NR_Nucleon) * count);

    float R = nr_nuclear_radius(235);
    nr->boxHalf = R * 2.5f;  /* Starts tight around nucleus */
}

static void setup_fusion_dt(NuclearReaction *nr) {
    nr_srand((unsigned int)time(NULL) + 1000);
    nr->numNucleons = 0;

    /* Deuterium (1p + 1n) on left, Tritium (1p + 2n) on right */
    float sep = 20.0f; /* initial separation in fm */

    int d_count = place_nucleons_sphere(nr->nucleons, NR_MAX_NUCLEONS,
                                        1, 1, -sep/2, 0.0f, 0.0f, 0);
    int t_count = place_nucleons_sphere(nr->nucleons + d_count,
                                        NR_MAX_NUCLEONS - d_count,
                                        1, 2, sep/2, 0.0f, 0.0f, 1);
    nr->numNucleons = d_count + t_count;
    memcpy(nr->base_pos, nr->nucleons, sizeof(NR_Nucleon) * nr->numNucleons);

    nr->boxHalf = sep * 0.8f;
}

static void setup_fusion_dd(NuclearReaction *nr) {
    nr_srand((unsigned int)time(NULL) + 2000);
    nr->numNucleons = 0;

    float sep = 18.0f;
    int d1 = place_nucleons_sphere(nr->nucleons, NR_MAX_NUCLEONS,
                                    1, 1, -sep/2, 0.0f, 0.0f, 0);
    int d2 = place_nucleons_sphere(nr->nucleons + d1, NR_MAX_NUCLEONS - d1,
                                    1, 1, sep/2, 0.0f, 0.0f, 1);
    nr->numNucleons = d1 + d2;
    memcpy(nr->base_pos, nr->nucleons, sizeof(NR_Nucleon) * nr->numNucleons);
    nr->boxHalf = sep * 0.8f;
}

void nuclear_reaction_setup(NuclearReaction *nr, NR_ReactionType type) {
    nr->reactionType = type;
    nr->phase = NR_PHASE_IDLE;
    nr->time = 0.0f;
    nr->phaseStart = 0.0f;
    nr->playing = 0;

    switch (type) {
        case NR_FISSION_U235: setup_fission_u235(nr); break;
        case NR_FUSION_DT:    setup_fusion_dt(nr);    break;
        case NR_FUSION_DD:    setup_fusion_dd(nr);    break;
        default: break;
    }
    printf("[NR] Setup: %s — %d nucleons, box=%.1f fm\n",
           nuclear_reaction_type_name(type), nr->numNucleons, nr->boxHalf);
}

/* ═══════════════════════ ANIMATION UPDATE ═══════════════════════ */

/* Smooth interpolation */
static float smoothstep(float edge0, float edge1, float x) {
    float t = (x - edge0) / (edge1 - edge0);
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    return t * t * (3.0f - 2.0f * t);
}

/* Get current phase from time */
static NR_Phase get_phase(NR_ReactionType type, float t) {
    for (int i = 0; i < 7; i++) {
        if (t < PHASE_END[type][i]) return (NR_Phase)i;
    }
    return NR_PHASE_DONE;
}

/* Phase-local [0,1] progress */
static float phase_progress(NR_ReactionType type, NR_Phase phase, float t) {
    float start = (phase > 0) ? PHASE_END[type][phase - 1] : 0.0f;
    float end   = PHASE_END[type][phase];
    float p = (t - start) / (end - start);
    if (p < 0.0f) p = 0.0f;
    if (p > 1.0f) p = 1.0f;
    return p;
}

/* Thermal jitter: small random displacement to make nucleons vibrate */
static void apply_jitter(NR_Nucleon *nucleons, int count, float amplitude, float time) {
    for (int i = 0; i < count; i++) {
        /* Deterministic per-nucleon jitter using index + time */
        float seed = (float)i * 7.31f + time * 15.0f;
        float jx = sinf(seed * 1.13f) * cosf(seed * 0.77f + 1.5f) * amplitude;
        float jy = sinf(seed * 0.89f + 2.3f) * cosf(seed * 1.31f) * amplitude;
        float jz = cosf(seed * 1.07f + 0.7f) * sinf(seed * 0.93f + 3.1f) * amplitude;
        nucleons[i].x += jx;
        nucleons[i].y += jy;
        nucleons[i].z += jz;
    }
}

/* ─── Fission animation ─── */
static void update_fission(NuclearReaction *nr) {
    float t = nr->time;
    NR_Phase phase = get_phase(nr->reactionType, t);
    nr->phase = phase;

    int N = nr->numNucleons;
    float R = nr_nuclear_radius(235);

    /* Start from base positions each frame */
    memcpy(nr->nucleons, nr->base_pos, sizeof(NR_Nucleon) * N);

    switch (phase) {
    case NR_PHASE_IDLE: {
        /* Static nucleus with gentle breathing mode oscillation */
        float breath = 1.0f + 0.02f * sinf(t * 3.0f);
        for (int i = 0; i < N; i++) {
            nr->nucleons[i].x *= breath;
            nr->nucleons[i].y *= breath;
            nr->nucleons[i].z *= breath;
        }
        apply_jitter(nr->nucleons, N, 0.15f, t);
        nr->boxHalf = R * 2.5f;
        break;
    }

    case NR_PHASE_APPROACH: {
        /* Incoming thermal neutron from +X direction */
        float p = phase_progress(nr->reactionType, phase, t);
        /* The nucleus already has 235 nucleons (92p + 143n).
         * Add an incoming neutron as an extra particle. */
        if (nr->numNucleons < 236) {
            /* Add the incoming neutron */
            NR_Nucleon *inc = &nr->nucleons[nr->numNucleons];
            inc->type = 0;  /* neutron */
            inc->group = 3; /* free */
            inc->radius = 0.85f;
            float approach_x = 25.0f * (1.0f - p) + R * (p);
            inc->x = approach_x;
            inc->y = 0.0f;
            inc->z = 0.0f;
            inc->pad0 = inc->pad1 = 0.0f;
            N = nr->numNucleons + 1;
        }
        apply_jitter(nr->nucleons, nr->numNucleons, 0.15f, t);
        nr->boxHalf = R * 3.5f;
        break;
    }

    case NR_PHASE_EXCITE: {
        /* Neutron absorbed — nucleus vibrates (excited U-236) */
        float p = phase_progress(nr->reactionType, phase, t);
        /* Quadrupole oscillation: stretch/compress along Y */
        float osc = sinf(p * 6.2832f * 3.0f) * 0.08f * (1.0f - p * 0.5f);
        for (int i = 0; i < N; i++) {
            nr->nucleons[i].y *= (1.0f + osc);
            nr->nucleons[i].x *= (1.0f - osc * 0.5f);
            nr->nucleons[i].z *= (1.0f - osc * 0.5f);
        }
        apply_jitter(nr->nucleons, N, 0.25f, t);
        nr->boxHalf = R * 2.8f;
        break;
    }

    case NR_PHASE_DEFORM: {
        /* Quadrupole deformation: sphere → prolate → peanut
         * β₂ increases from 0 to ~0.85 (highly deformed at scission)
         * β₃ (octupole) for mass-asymmetric fission
         * [Brosa et al., Phys. Rep. 197 (1990) 167: β₂~0.6-0.9 at scission;
         *  β₃~0.15-0.25 for asymmetric fission] */
        float p = phase_progress(nr->reactionType, phase, t);
        float beta2 = smoothstep(0.0f, 1.0f, p) * 0.85f;
        float beta3 = smoothstep(0.3f, 1.0f, p) * 0.20f;

        for (int i = 0; i < N; i++) {
            float y0 = nr->base_pos[i].y;
            float x0 = nr->base_pos[i].x;
            float z0 = nr->base_pos[i].z;

            /* Stretch along Y (fission axis) */
            float y_stretch = 1.0f + beta2 * 1.2f;
            /* Compress radially */
            float r_compress = 1.0f - beta2 * 0.35f;

            /* Octupole: top half gets slightly bigger */
            float asym = 1.0f + beta3 * (y0 > 0.0f ? 0.15f : -0.10f);

            float ny = y0 * y_stretch * asym;
            float nx = x0 * r_compress;
            float nz = z0 * r_compress;

            /* Neck constriction near equator when β₂ is large */
            if (beta2 > 0.3f) {
                float neck_width = R * (1.5f - beta2 * 1.0f);
                if (neck_width < 0.5f) neck_width = 0.5f;
                float abs_y = fabsf(ny);
                if (abs_y < neck_width) {
                    float constr = (1.0f - abs_y / neck_width) * (beta2 - 0.3f) * 1.2f;
                    if (constr > 0.8f) constr = 0.8f;
                    nx *= (1.0f - constr);
                    nz *= (1.0f - constr);
                }
            }

            nr->nucleons[i].x = nx;
            nr->nucleons[i].y = ny;
            nr->nucleons[i].z = nz;
        }
        apply_jitter(nr->nucleons, N, 0.2f, t);
        nr->boxHalf = R * 3.5f;
        break;
    }

    case NR_PHASE_SCISSION: {
        /* Neck breaks — assign nucleons to two groups + 3 free neutrons */
        float p = phase_progress(nr->reactionType, phase, t);

        /* First: apply max deformation (from end of DEFORM phase) */
        float beta2 = 0.85f;
        for (int i = 0; i < N; i++) {
            float y_stretch = 1.0f + beta2 * 1.2f;
            float r_compress = 1.0f - beta2 * 0.35f;
            float asym = 1.0f + 0.20f * (nr->base_pos[i].y > 0.0f ? 0.15f : -0.10f);

            nr->nucleons[i].y = nr->base_pos[i].y * y_stretch * asym;
            nr->nucleons[i].x = nr->base_pos[i].x * r_compress;
            nr->nucleons[i].z = nr->base_pos[i].z * r_compress;
        }

        /* Assign groups based on Y position: top → fragment 1, bottom → fragment 2 */
        /* Ba-141 (top, heavier): 56p + 85n = 141 */
        /* Kr-92 (bottom, lighter): 36p + 56n + remaining go to free neutrons */
        int n_free = 0;
        for (int i = 0; i < N; i++) {
            if (nr->nucleons[i].y > 0.0f) {
                nr->nucleons[i].group = 1; /* top fragment (Ba) */
            } else {
                nr->nucleons[i].group = 2; /* bottom fragment (Kr) */
            }
        }

        /* Pick 3 neutrons near the equator as "free neutrons" */
        float best_dist[3] = {1e9f, 1e9f, 1e9f};
        int   best_idx[3]  = {-1, -1, -1};
        for (int i = 0; i < N; i++) {
            if (nr->nucleons[i].type != 0) continue; /* protons can't be free neutrons */
            float dy = fabsf(nr->nucleons[i].y);
            for (int k = 0; k < 3; k++) {
                if (dy < best_dist[k]) {
                    /* Shift down */
                    for (int j = 2; j > k; j--) {
                        best_dist[j] = best_dist[j-1];
                        best_idx[j] = best_idx[j-1];
                    }
                    best_dist[k] = dy;
                    best_idx[k] = i;
                    break;
                }
            }
        }
        for (int k = 0; k < 3; k++) {
            if (best_idx[k] >= 0) {
                nr->nucleons[best_idx[k]].group = 3; /* free neutron */
                n_free++;
            }
        }

        /* Separation: fragments start to move apart */
        float sep = smoothstep(0.0f, 1.0f, p) * 5.0f;
        for (int i = 0; i < N; i++) {
            if (nr->nucleons[i].group == 1) nr->nucleons[i].y += sep;
            if (nr->nucleons[i].group == 2) nr->nucleons[i].y -= sep;
            if (nr->nucleons[i].group == 3) {
                /* Free neutrons fly outward radially */
                float angle = 6.2832f * (float)(i % 3) / 3.0f + 0.5f;
                nr->nucleons[i].x += sep * 2.0f * cosf(angle);
                nr->nucleons[i].z += sep * 2.0f * sinf(angle);
            }
        }
        apply_jitter(nr->nucleons, N, 0.15f, t);
        nr->boxHalf = R * 4.0f;
        break;
    }

    case NR_PHASE_SEPARATE: {
        /* Coulomb repulsion drives fragments apart */
        float p = phase_progress(nr->reactionType, phase, t);

        /* Apply max deformation as base */
        float beta2 = 0.85f;
        for (int i = 0; i < N; i++) {
            float y_stretch = 1.0f + beta2 * 1.2f;
            float r_compress = 1.0f - beta2 * 0.35f;
            float asym = 1.0f + 0.20f * (nr->base_pos[i].y > 0.0f ? 0.15f : -0.10f);

            nr->nucleons[i].y = nr->base_pos[i].y * y_stretch * asym;
            nr->nucleons[i].x = nr->base_pos[i].x * r_compress;
            nr->nucleons[i].z = nr->base_pos[i].z * r_compress;
        }

        /* Assign groups (same as scission) */
        for (int i = 0; i < N; i++) {
            if (nr->nucleons[i].y > 0.0f)
                nr->nucleons[i].group = 1;
            else
                nr->nucleons[i].group = 2;
        }
        /* Free neutrons: 3 closest to equator */
        float bd[3] = {1e9f, 1e9f, 1e9f};
        int   bi[3] = {-1, -1, -1};
        for (int i = 0; i < N; i++) {
            if (nr->nucleons[i].type != 0) continue;
            float dy = fabsf(nr->nucleons[i].y);
            for (int k = 0; k < 3; k++) {
                if (dy < bd[k]) {
                    for (int j = 2; j > k; j--) { bd[j] = bd[j-1]; bi[j] = bi[j-1]; }
                    bd[k] = dy; bi[k] = i; break;
                }
            }
        }
        for (int k = 0; k < 3; k++) if (bi[k] >= 0) nr->nucleons[bi[k]].group = 3;

        /* Coulomb separation: accelerating (v ∝ t → displacement ∝ t²) */
        float disp = 5.0f + smoothstep(0.0f, 1.0f, p) * 35.0f;
        float neutron_speed = 5.0f + p * 50.0f;

        for (int i = 0; i < N; i++) {
            if (nr->nucleons[i].group == 1) nr->nucleons[i].y += disp;
            if (nr->nucleons[i].group == 2) nr->nucleons[i].y -= disp * 0.8f;
            if (nr->nucleons[i].group == 3) {
                float angle = 6.2832f * (float)(i % 3) / 3.0f + 0.5f;
                nr->nucleons[i].x += neutron_speed * cosf(angle);
                nr->nucleons[i].z += neutron_speed * sinf(angle);
                nr->nucleons[i].y *= 0.3f; /* flatten toward equator */
            }
        }

        /* Fragments re-sphericalize as they separate */
        float relax = smoothstep(0.2f, 1.0f, p) * 0.5f;
        for (int i = 0; i < N; i++) {
            if (nr->nucleons[i].group == 3) continue;
            /* Lerp x,z toward rounder shape */
            float target_r_compress = 1.0f - 0.35f * (1.0f - relax);
            float actual_x = nr->base_pos[i].x * target_r_compress;
            float actual_z = nr->base_pos[i].z * target_r_compress;
            nr->nucleons[i].x = nr->nucleons[i].x * (1.0f - relax) + actual_x * relax;
            nr->nucleons[i].z = nr->nucleons[i].z * (1.0f - relax) + actual_z * relax;
        }

        apply_jitter(nr->nucleons, N, 0.12f, t);
        nr->boxHalf = 15.0f + disp * 1.2f;
        if (nr->boxHalf > 60.0f) nr->boxHalf = 60.0f;
        break;
    }

    case NR_PHASE_DONE:
        /* Freeze at final state */
        nr->playing = 0;
        break;

    default: break;
    }
}

/* ─── Fusion animation ─── */
static void update_fusion(NuclearReaction *nr) {
    float t = nr->time;
    NR_Phase phase = get_phase(nr->reactionType, t);
    nr->phase = phase;

    int N = nr->numNucleons;
    int is_dt = (nr->reactionType == NR_FUSION_DT);
    float init_sep = is_dt ? 20.0f : 18.0f;

    /* Identify group 0 (left, D) and group 1 (right, T or D₂) */
    memcpy(nr->nucleons, nr->base_pos, sizeof(NR_Nucleon) * N);

    switch (phase) {
    case NR_PHASE_IDLE: {
        /* Two nuclei at rest, slight thermal jitter */
        apply_jitter(nr->nucleons, N, 0.2f, t);
        nr->boxHalf = init_sep * 0.8f;
        break;
    }

    case NR_PHASE_APPROACH: {
        /* Nuclei accelerate toward each other (Coulomb decelerated but
         * kinetic energy overcomes barrier by end of phase) */
        float p = phase_progress(nr->reactionType, phase, t);
        float approach = smoothstep(0.0f, 1.0f, p);
        float dx = (init_sep / 2.0f) * (1.0f - approach);

        for (int i = 0; i < N; i++) {
            if (nr->nucleons[i].group == 0)
                nr->nucleons[i].x = nr->base_pos[i].x + (init_sep/2.0f - dx) - init_sep/2.0f + dx;
            else
                nr->nucleons[i].x = nr->base_pos[i].x - (init_sep/2.0f - dx) + init_sep/2.0f - dx;
        }
        /* Simpler: move group 0 right by approach, group 1 left by approach */
        float shift = (init_sep / 2.0f) * approach;
        for (int i = 0; i < N; i++) {
            /* Override with cleaner calculation */
            if (nr->base_pos[i].group == 0)
                nr->nucleons[i].x = nr->base_pos[i].x + shift;
            else
                nr->nucleons[i].x = nr->base_pos[i].x - shift;
        }
        apply_jitter(nr->nucleons, N, 0.2f, t);
        nr->boxHalf = init_sep * 0.7f * (1.0f - approach * 0.3f);
        break;
    }

    case NR_PHASE_EXCITE: {
        /* Touching / tunneling — nuclei are very close, start to deform */
        float p = phase_progress(nr->reactionType, phase, t);

        /* Move nuclei to contact point */
        for (int i = 0; i < N; i++) {
            if (nr->base_pos[i].group == 0)
                nr->nucleons[i].x = nr->base_pos[i].x + init_sep / 2.0f;
            else
                nr->nucleons[i].x = nr->base_pos[i].x - init_sep / 2.0f;
        }

        /* Slow merge: nucleons drift toward origin */
        float merge = smoothstep(0.0f, 1.0f, p) * 0.3f;
        for (int i = 0; i < N; i++) {
            nr->nucleons[i].x *= (1.0f - merge);
            nr->nucleons[i].y *= (1.0f - merge * 0.2f);
            nr->nucleons[i].z *= (1.0f - merge * 0.2f);
        }

        apply_jitter(nr->nucleons, N, 0.3f, t);
        nr->boxHalf = 8.0f;
        break;
    }

    case NR_PHASE_DEFORM: {
        /* Compound nucleus forms: everything collapses to center */
        float p = phase_progress(nr->reactionType, phase, t);

        float compound_R = nr_nuclear_radius(is_dt ? 5 : 4);

        for (int i = 0; i < N; i++) {
            /* Move all nucleons toward origin and compress into compound radius */
            float tx = nr->base_pos[i].x;
            if (nr->base_pos[i].group == 0) tx += init_sep / 2.0f;
            else tx -= init_sep / 2.0f;
            float ty = nr->base_pos[i].y;
            float tz = nr->base_pos[i].z;

            /* Lerp toward a compact sphere centered at origin */
            float merge = smoothstep(0.0f, 0.7f, p);
            float target_r = compound_R * 0.8f;

            /* Target: random position in compact sphere */
            float ti = (float)i * 2.399f; /* golden angle-like */
            float tct = cosf(ti * 0.773f);
            float tst = sinf(ti * 0.773f);
            float tphi = ti * 1.618f;
            float tr = target_r * cbrtf(((float)(i + 1)) / (float)N);

            float goal_x = tr * tst * cosf(tphi);
            float goal_y = tr * tct;
            float goal_z = tr * tst * sinf(tphi);

            nr->nucleons[i].x = tx * (1.0f - merge) + goal_x * merge;
            nr->nucleons[i].y = ty * (1.0f - merge) + goal_y * merge;
            nr->nucleons[i].z = tz * (1.0f - merge) + goal_z * merge;
        }

        /* Oscillation of compound nucleus */
        float osc = sinf(p * 6.2832f * 4.0f) * 0.05f * (1.0f - p);
        for (int i = 0; i < N; i++) {
            nr->nucleons[i].x *= (1.0f + osc);
            nr->nucleons[i].y *= (1.0f - osc * 0.5f);
        }

        apply_jitter(nr->nucleons, N, 0.25f, t);
        nr->boxHalf = 6.0f;
        break;
    }

    case NR_PHASE_SCISSION: {
        /* Products form: He-4 (or He-3) separates from neutron */
        float p = phase_progress(nr->reactionType, phase, t);

        float compound_R = nr_nuclear_radius(is_dt ? 5 : 4);
        int product_A = is_dt ? 4 : 3; /* He-4 or He-3 */

        /* Put all nucleons in compact form first */
        for (int i = 0; i < N; i++) {
            float ti = (float)i * 2.399f;
            float tct = cosf(ti * 0.773f);
            float tst = sinf(ti * 0.773f);
            float tphi = ti * 1.618f;
            float tr = compound_R * 0.8f * cbrtf(((float)(i + 1)) / (float)N);

            nr->nucleons[i].x = tr * tst * cosf(tphi);
            nr->nucleons[i].y = tr * tct;
            nr->nucleons[i].z = tr * tst * sinf(tphi);
        }

        /* Assign: first (product_A) nucleons → product (group 1)
         * last 1 nucleon → free neutron (group 3)     */
        for (int i = 0; i < N; i++) {
            if (i < product_A)
                nr->nucleons[i].group = 1; /* alpha / He-3 */
            else
                nr->nucleons[i].group = 3; /* free neutron */
        }

        /* Start separating */
        float sep = smoothstep(0.0f, 1.0f, p) * 4.0f;
        for (int i = 0; i < N; i++) {
            if (nr->nucleons[i].group == 3) {
                /* Neutron flies in +X direction (fastest, carries most KE) */
                nr->nucleons[i].x += sep * 3.0f;
            } else {
                /* Alpha recoils in -X (slower, heavier) */
                nr->nucleons[i].x -= sep * 0.5f;
            }
        }

        apply_jitter(nr->nucleons, N, 0.2f, t);
        nr->boxHalf = 8.0f + sep * 2.0f;
        break;
    }

    case NR_PHASE_SEPARATE: {
        /* Products fly apart with kinetic energy */
        float p = phase_progress(nr->reactionType, phase, t);

        float compound_R = nr_nuclear_radius(is_dt ? 5 : 4);
        int product_A = is_dt ? 4 : 3;

        for (int i = 0; i < N; i++) {
            float ti = (float)i * 2.399f;
            float tct = cosf(ti * 0.773f);
            float tst = sinf(ti * 0.773f);
            float tphi = ti * 1.618f;
            float tr = compound_R * 0.8f * cbrtf(((float)(i + 1)) / (float)N);

            nr->nucleons[i].x = tr * tst * cosf(tphi);
            nr->nucleons[i].y = tr * tct;
            nr->nucleons[i].z = tr * tst * sinf(tphi);

            if (i < product_A)
                nr->nucleons[i].group = 1;
            else
                nr->nucleons[i].group = 3;
        }

        /* Full separation */
        float disp = 4.0f + smoothstep(0.0f, 1.0f, p) * 30.0f;
        float n_speed = 4.0f + smoothstep(0.0f, 1.0f, p) * 45.0f;

        for (int i = 0; i < N; i++) {
            if (nr->nucleons[i].group == 3) {
                nr->nucleons[i].x += n_speed;
            } else {
                /* Alpha recoils opposite */
                nr->nucleons[i].x -= disp * 0.3f;
            }
        }

        apply_jitter(nr->nucleons, N, 0.1f, t);
        nr->boxHalf = 12.0f + disp * 1.0f;
        if (nr->boxHalf > 50.0f) nr->boxHalf = 50.0f;
        break;
    }

    case NR_PHASE_DONE:
        nr->playing = 0;
        break;

    default: break;
    }
}

void nuclear_reaction_update(NuclearReaction *nr, float dt) {
    if (!nr->playing) return;
    nr->time += dt;

    switch (nr->reactionType) {
    case NR_FISSION_U235:
        update_fission(nr);
        break;
    case NR_FUSION_DT:
    case NR_FUSION_DD:
        update_fusion(nr);
        break;
    default: break;
    }
}

/* ═══════════════════════ VULKAN HELPERS ═══════════════════════ */

static uint32_t nr_find_mem_type(VkPhysicalDevice phys, uint32_t type_bits,
                                 VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mem;
    vkGetPhysicalDeviceMemoryProperties(phys, &mem);
    for (uint32_t i = 0; i < mem.memoryTypeCount; i++) {
        if ((type_bits & (1u << i)) &&
            (mem.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    return UINT32_MAX;
}

static int nr_create_buffer(VkPhysicalDevice phys, VkDevice dev,
                            VkDeviceSize size, VkBufferUsageFlags usage,
                            VkMemoryPropertyFlags props,
                            VkBuffer *buf, VkDeviceMemory *mem) {
    VkBufferCreateInfo ci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size  = size;
    ci.usage = usage;
    if (vkCreateBuffer(dev, &ci, NULL, buf) != VK_SUCCESS) return -1;

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, *buf, &req);

    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = nr_find_mem_type(phys, req.memoryTypeBits, props);
    if (ai.memoryTypeIndex == UINT32_MAX) return -2;
    if (vkAllocateMemory(dev, &ai, NULL, mem) != VK_SUCCESS) return -3;
    vkBindBufferMemory(dev, *buf, *mem, 0);
    return 0;
}

static VkShaderModule nr_load_shader(VkDevice dev, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[NR] Cannot open shader: %s\n", path); return VK_NULL_HANDLE; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint32_t *code = (uint32_t *)malloc((size_t)sz);
    fread(code, 1, (size_t)sz, f);
    fclose(f);

    VkShaderModuleCreateInfo ci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = (size_t)sz;
    ci.pCode = code;
    VkShaderModule sm;
    VkResult r = vkCreateShaderModule(dev, &ci, NULL, &sm);
    free(code);
    return (r == VK_SUCCESS) ? sm : VK_NULL_HANDLE;
}

static void nr_submit_and_wait(VkDevice dev, VkQueue queue, VkCommandBuffer cmd) {
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);
}

/* ═══════════════════════ INIT / FREE ═══════════════════════ */

int nuclear_reaction_init(NuclearReaction *nr,
                          VkPhysicalDevice phys, VkDevice dev,
                          uint32_t queueFamilyIdx,
                          VkBuffer densityBuf, VkBuffer signedBuf,
                          int gridDim) {
    memset(nr, 0, sizeof(*nr));
    nr->device     = dev;
    nr->physDevice = phys;
    nr->densityBuf = densityBuf;
    nr->signedBuf  = signedBuf;
    nr->gridDim    = gridDim;
    nr->boxHalf    = 20.0f;

    /* ─── Nucleon SSBO (host-visible for CPU upload each frame) ─── */
    VkDeviceSize nucBufSize = sizeof(NR_Nucleon) * NR_MAX_NUCLEONS;
    if (nr_create_buffer(phys, dev, nucBufSize,
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         &nr->nucleonBuf, &nr->nucleonMem) != 0) {
        fprintf(stderr, "[NR] Failed to create nucleon buffer\n");
        return -1;
    }

    /* ─── Descriptor set layout: 3 storage buffers ─── */
    VkDescriptorSetLayoutBinding bindings[3] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
    };
    VkDescriptorSetLayoutCreateInfo dslci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount = 3;
    dslci.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(dev, &dslci, NULL, &nr->descLayout) != VK_SUCCESS) return -2;

    /* ─── Descriptor pool ─── */
    VkDescriptorPoolSize poolSizes[] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3}
    };
    VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = poolSizes;
    if (vkCreateDescriptorPool(dev, &dpci, NULL, &nr->descPool) != VK_SUCCESS) return -3;

    /* ─── Allocate descriptor set ─── */
    VkDescriptorSetAllocateInfo dsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool     = nr->descPool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &nr->descLayout;
    if (vkAllocateDescriptorSets(dev, &dsai, &nr->descSet) != VK_SUCCESS) return -4;

    /* ─── Write descriptors ─── */
    VkDeviceSize gridBufSize = (VkDeviceSize)gridDim * gridDim * gridDim * sizeof(float);

    VkDescriptorBufferInfo dbi[3] = {
        {densityBuf, 0, gridBufSize},
        {signedBuf,  0, gridBufSize},
        {nr->nucleonBuf, 0, nucBufSize},
    };
    VkWriteDescriptorSet writes[3];
    memset(writes, 0, sizeof(writes));
    for (int i = 0; i < 3; i++) {
        writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet          = nr->descSet;
        writes[i].dstBinding      = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &dbi[i];
    }
    vkUpdateDescriptorSets(dev, 3, writes, 0, NULL);

    /* ─── Pipeline layout (32-byte push constant) ─── */
    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 32};
    VkPipelineLayoutCreateInfo plci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &nr->descLayout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(dev, &plci, NULL, &nr->pipeLayout) != VK_SUCCESS) return -5;

    /* ─── Load shader ─── */
    nr->shader = nr_load_shader(dev, "shaders/nuclear_density.comp.spv");
    if (nr->shader == VK_NULL_HANDLE) return -6;

    /* ─── Compute pipeline ─── */
    VkComputePipelineCreateInfo cpci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = nr->shader;
    cpci.stage.pName  = "main";
    cpci.layout       = nr->pipeLayout;
    if (vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &nr->pipeline) != VK_SUCCESS)
        return -7;

    /* ─── Command pool + buffer ─── */
    VkCommandPoolCreateInfo cpi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.queueFamilyIndex = queueFamilyIdx;
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(dev, &cpi, NULL, &nr->cmdPool) != VK_SUCCESS) return -8;

    VkCommandBufferAllocateInfo cbai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool        = nr->cmdPool;
    cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(dev, &cbai, &nr->cmdBuf) != VK_SUCCESS) return -9;

    printf("[NR] Nuclear reaction system initialized (grid=%d)\n", gridDim);
    return 0;
}

/* ─── DISPATCH ─── */
typedef struct {
    int   gridDim;
    int   numNucleons;
    float boxHalf;
    float sigma;
    float densityScale;
    float energyGlow;
    int   pad0, pad1;
} NR_DensityPush;

int nuclear_reaction_dispatch(NuclearReaction *nr, VkQueue queue) {
    VkDevice dev = nr->device;

    /* Upload nucleon data to GPU */
    void *mapped = NULL;
    vkMapMemory(dev, nr->nucleonMem, 0, sizeof(NR_Nucleon) * nr->numNucleons, 0, &mapped);
    memcpy(mapped, nr->nucleons, sizeof(NR_Nucleon) * nr->numNucleons);
    vkUnmapMemory(dev, nr->nucleonMem);

    /* Record command buffer */
    vkResetCommandBuffer(nr->cmdBuf, 0);
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(nr->cmdBuf, &bi);

    vkCmdBindPipeline(nr->cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, nr->pipeline);
    vkCmdBindDescriptorSets(nr->cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            nr->pipeLayout, 0, 1, &nr->descSet, 0, NULL);

    /* Push constants */
    NR_DensityPush pc;
    pc.gridDim      = nr->gridDim;
    pc.numNucleons  = nr->numNucleons;
    pc.boxHalf      = nr->boxHalf;
    pc.sigma        = NR_SIGMA_FM;
    pc.densityScale = 1.0f;
    /* Energy glow during excite phase */
    pc.energyGlow   = (nr->phase == NR_PHASE_EXCITE) ? 1.0f : 0.0f;
    pc.pad0 = pc.pad1 = 0;

    vkCmdPushConstants(nr->cmdBuf, nr->pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    /* Dispatch: gridDim³ / (8³) workgroups */
    uint32_t groups = ((uint32_t)nr->gridDim + 7) / 8;
    vkCmdDispatch(nr->cmdBuf, groups, groups, groups);

    /* Barrier: density SSBO write → read (for subsequent raymarch) */
    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(nr->cmdBuf,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &mb, 0, NULL, 0, NULL);

    vkEndCommandBuffer(nr->cmdBuf);

    nr_submit_and_wait(dev, queue, nr->cmdBuf);
    return 0;
}

/* ═══════════════════════ ACCESSORS ═══════════════════════ */

float nuclear_reaction_get_box_half(const NuclearReaction *nr) {
    return nr->boxHalf;
}

const char *nuclear_reaction_phase_name(NR_Phase phase) {
    static const char *names[] = {
        "Idle", "Approach", "Excitation", "Deformation",
        "Scission", "Separation", "Complete"
    };
    return (phase >= 0 && phase < 7) ? names[phase] : "Unknown";
}

const char *nuclear_reaction_type_name(NR_ReactionType type) {
    static const char *names[] = {
        "U-235 Fission (n + ²³⁵U → ¹⁴¹Ba + ⁹²Kr + 3n)",
        "D-T Fusion (D + T → ⁴He + n + 17.6 MeV)",
        "D-D Fusion (D + D → ³He + n + 3.27 MeV)"
    };
    return (type >= 0 && type < NR_REACTION_COUNT) ? names[type] : "Unknown";
}

/* ═══════════════════════ CLEANUP ═══════════════════════ */

void nuclear_reaction_free(NuclearReaction *nr) {
    VkDevice dev = nr->device;
    if (!dev) return;

    vkDeviceWaitIdle(dev);

    if (nr->pipeline)   vkDestroyPipeline(dev, nr->pipeline, NULL);
    if (nr->pipeLayout) vkDestroyPipelineLayout(dev, nr->pipeLayout, NULL);
    if (nr->descPool)   vkDestroyDescriptorPool(dev, nr->descPool, NULL);
    if (nr->descLayout) vkDestroyDescriptorSetLayout(dev, nr->descLayout, NULL);
    if (nr->shader)     vkDestroyShaderModule(dev, nr->shader, NULL);

    if (nr->nucleonBuf) vkDestroyBuffer(dev, nr->nucleonBuf, NULL);
    if (nr->nucleonMem) vkFreeMemory(dev, nr->nucleonMem, NULL);

    if (nr->cmdPool)    vkDestroyCommandPool(dev, nr->cmdPool, NULL);

    memset(nr, 0, sizeof(*nr));
    printf("[NR] Nuclear reaction system freed\n");
}
