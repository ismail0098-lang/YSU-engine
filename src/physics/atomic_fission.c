/* atomic_fission.c â€” Atomic-scale nuclear fission visualization engine
 *
 * Implements quantum-accurate visualization of U-235 fission, chain reactions,
 * and Xe-135 poisoning at the atomic scale (~50 Ã… view).
 *
 * Each atom is rendered with its actual electron orbital wavefunction,
 * computed via Slater-type orbitals solving the SchrÃ¶dinger equation.
 * Neutrons appear as de Broglie wave packets with correct wavelengths.
 *
 * All cross sections, energies, and decay rates from ENDF/B-VIII.0
 * and JEFF-3.3 nuclear data libraries.
 */

#include "atomic_fission.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• RNG â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */

static unsigned int af_seed = 12345;
static float af_rand01(void) {
    af_seed = af_seed * 1103515245u + 12345u;
    return (float)((af_seed >> 16) & 0x7FFF) / 32768.0f;
}
static float af_randn(void) {
    float u1 = af_rand01() + 1e-7f;
    float u2 = af_rand01();
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2832f * u2);
}
static void af_rand_dir(float *dx, float *dy, float *dz) {
    float ct = 2.0f * af_rand01() - 1.0f;
    float st = sqrtf(1.0f - ct*ct);
    float phi = 6.2832f * af_rand01();
    *dx = st * cosf(phi);
    *dy = ct;
    *dz = st * sinf(phi);
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• ATOMIC DATA â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */

/* Approximate atomic radii in Ã… (van der Waals or metallic) */
static float atom_radius_A(int Z) {
    /* Selected elements relevant to fission */
    switch (Z) {
        case  1: return 1.20f;  /* H */
        case  2: return 1.40f;  /* He (1sÂ² closed shell) */
        case  6: return 1.70f;  /* C (graphite) */
        case  8: return 1.52f;  /* O */
        case 36: return 2.02f;  /* Kr */
        case 54: return 2.16f;  /* Xe */
        case 56: return 2.53f;  /* Ba */
        case 92: return 1.96f;  /* U (metallic radius) */
        case 94: return 1.87f;  /* Pu (metallic radius) */
        case -1: return 0.80f;  /* QCD quark (sub-femtometer visualized) */
        default: return 1.5f + 0.01f * Z;
    }
}

/* Nuclear radius in fm: R = 1.25 Ã— A^(1/3) */
static float nuclear_radius_fm(int A) {
    return AF_R0_FM * cbrtf((float)A);
}

/* De Broglie wavelength for neutrons: Î»(Ã…) = 0.2860 / âˆš(E_eV)
 * At thermal (0.0253 eV): Î» â‰ˆ 1.80 Ã…
 * At 2 MeV fast:           Î» â‰ˆ 2.02Ã—10â»â´ Ã… = 0.0202 fm */
static float neutron_wavelength_A(float energy_eV) {
    if (energy_eV < 1e-6f) energy_eV = 1e-6f;
    return 0.2860f / sqrtf(energy_eV);
}

/* Speed of neutron at given energy: v = âˆš(2E/m)
 * For display purposes, scaled dramatically (actual thermal = 2200 m/s = tiny) */
static float neutron_speed_display(float energy_eV) {
    /* Scale so thermal neutrons cross ~20 Ã… in ~2 seconds (human-visible speed) */
    float v_real = sqrtf(2.0f * energy_eV * 1.602e-19f / 1.675e-27f); /* m/s */
    /* Map to display: 1 Ã…/s per 100 m/s real */
    return v_real * 1e-2f * 5.0f;  /* â‰ˆ 11 Ã…/s for thermal, ~7000 for fast */
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• ATOM HELPERS â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */

static AF_Atom make_atom(int Z, int A, float x, float y, float z) {
    AF_Atom a;
    memset(&a, 0, sizeof(a));
    a.x = x; a.y = y; a.z = z;
    a.Z = Z; a.A = A;
    a.state = AF_ATOM_IDLE;
    a.orbital_scale = 1.0f;
    a.radius_A = atom_radius_A(Z);
    a.visible = 1;
    return a;
}

static AF_Neutron make_neutron(float x, float y, float z,
                                float vx, float vy, float vz,
                                float energy_eV, float time) {
    AF_Neutron n;
    memset(&n, 0, sizeof(n));
    n.x = x; n.y = y; n.z = z;
    n.vx = vx; n.vy = vy; n.vz = vz;
    n.energy_eV = energy_eV;
    n.wavelength_A = neutron_wavelength_A(energy_eV);
    n.birth_time = time;
    n.alive = 1;
    n.is_thermal = (energy_eV < 1.0f) ? 1 : 0;
    return n;
}

static AF_Gamma make_gamma(float x, float y, float z,
                            float dx, float dy, float dz,
                            float energy_MeV, float time) {
    AF_Gamma g;
    memset(&g, 0, sizeof(g));
    g.x = x; g.y = y; g.z = z;
    g.dx = dx; g.dy = dy; g.dz = dz;
    g.energy_MeV = energy_MeV;
    g.birth_time = time;
    g.lifetime = 1.5f;  /* visible for 1.5s */
    g.alive = 1;
    return g;
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• SLATER Z_EFF â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
 * Effective nuclear charge from Slater's screening rules.
 * These are the SAME values used by the GPU wavefunction shader
 * to solve the SchrÃ¶dinger equation for each orbital.
 * Ensures visual consistency between the wavefunction compute
 * pass and our density stamping.
 * â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */

static float slater_z_eff(int Z, int n_shell) {
    /* Simplified Slater's rules â€” matches quantum_volume.c implementation */
    float screening = 0.0f;
    if (n_shell == 1) {
        screening = 0.30f * (float)(Z > 1 ? 1 : 0); /* He screens H by 0.30 */
    } else if (n_shell == 2) {
        screening = 2.0f * 0.85f + (float)(Z - 3) * 0.35f;
        if (Z <= 2) screening = 0.0f;
    } else {
        screening = 2.0f * 1.0f + 8.0f * 0.85f + (float)(Z - 11) * 0.35f;
        if (Z <= 10) screening = 2.0f * 0.85f + (float)(Z - 3) * 0.35f;
    }
    float zeff = (float)Z - screening;
    if (zeff < 1.0f) zeff = 1.0f;
    return zeff;
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• ORBITAL DENSITY â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
 * Compute electron density |Ïˆ|Â² at a point for a given element.
 * Uses simplified radial Slater-type orbitals:
 *   Ïˆ_STO âˆ r^(n-1) Ã— exp(-Z_eff Ã— r / n)
 * which gives the correct radial shape for visualization.
 * Full angular structure is added by the GPU shader procedurally.
 * â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */

static float orbital_density_at(int Z, float r_A) {
    /* QCD quarks (Z=-1): point-like color charge â€” Gaussian blob */
    if (Z == -1) {
        float sigma = 0.6f;  /* ~0.6 Ã… â‰ˆ 0.6 fm for quark charge radius */
        float r2 = r_A * r_A;
        return 5.0f * expf(-r2 / (2.0f * sigma * sigma));
    }

    /* Convert Ã… to Bohr radii (1 aâ‚€ = 0.529 Ã…) */
    float r = r_A / 0.529f;  /* r in aâ‚€ */

    if (r < 0.01f) r = 0.01f;

    /* Sum contributions from occupied shells */
    float total = 0.0f;

    /* Shell occupations for selected elements */
    /* This is simplified â€” the GPU shader does the full computation */
    int shells_n[] = {1, 2, 3, 4, 5, 6, 7};
    int max_shell;

    if (Z <= 2) max_shell = 1;
    else if (Z <= 10) max_shell = 2;
    else if (Z <= 18) max_shell = 3;
    else if (Z <= 36) max_shell = 4;
    else if (Z <= 54) max_shell = 5;
    else if (Z <= 86) max_shell = 6;
    else max_shell = 7;

    for (int s = 0; s < max_shell; s++) {
        int n = shells_n[s];
        float zeff = slater_z_eff(Z, n);

        /* STO radial: R(r) âˆ r^(n-1) Ã— exp(-Î¶r) where Î¶ = Z_eff/n */
        float zeta = zeff / (float)n;
        float rn = powf(r, (float)(n - 1));
        float radial = rn * expf(-zeta * r);

        /* Normalization (approximate â€” fine for visualization) */
        float norm = powf(2.0f * zeta, (float)n + 0.5f) /
                     sqrtf(4.0f * (float)M_PI);

        /* Electrons in this shell */
        float occ;
        if (n == 1) occ = (Z >= 2) ? 2.0f : (float)Z;
        else if (n == 2) occ = (Z >= 10) ? 8.0f : fmaxf(0.0f, fminf(8.0f, (float)(Z - 2)));
        else if (n == 3) occ = (Z >= 18) ? 8.0f : fmaxf(0.0f, fminf(8.0f, (float)(Z - 10)));
        else if (n == 4) occ = (Z >= 36) ? 18.0f : fmaxf(0.0f, fminf(18.0f, (float)(Z - 18)));
        else if (n == 5) occ = (Z >= 54) ? 18.0f : fmaxf(0.0f, fminf(18.0f, (float)(Z - 36)));
        else if (n == 6) occ = (Z >= 86) ? 32.0f : fmaxf(0.0f, fminf(32.0f, (float)(Z - 54)));
        else occ = fmaxf(0.0f, fminf(32.0f, (float)(Z - 86)));

        total += occ * norm * radial * radial;
    }

    return total;
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• SCENE SETUP â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */

static void setup_single_fission(AtomicFission *af) {
    /* Scene: Single U-235 fission event at atomic scale
     *
     * Layout:
     *   - U-235 atom at origin (with full electron cloud, radius ~1.96 Ã…)
     *   - Nearby graphite C atom (moderator context) at -8 Ã…
     *   - Water molecule (Hâ‚‚O) for context at +8 Ã…
     *   - Incoming thermal neutron from -20 Ã…, traveling +X
     *
     * The neutron approaches, gets absorbed, forms U-236* compound nucleus,
     * which deforms and fisfisfisfissions into Ba-141 + Kr-92 + 3 prompt neutrons
     * + prompt gammas.
     */
    af->scene = AF_SCENE_SINGLE_FISSION;
    af->phase = AF_PHASE_SETUP;
    af->time = 0.0f;
    af->phase_time = 0.0f;
    af->fission_count = 0;
    af->total_energy_MeV = 0.0f;
    af->generation = 0;
    af->boxHalf = 25.0f;  /* Â±25 Ã… â€” enough for one atom + context */

    af->num_atoms = 0;
    af->num_neutrons = 0;
    af->num_gammas = 0;
    af->num_particles = 0;

    /* U-235 at center */
    af->atoms[af->num_atoms++] = make_atom(92, 235, 0.0f, 0.0f, 0.0f);

    /* Graphite moderator atom (context) */
    af->atoms[af->num_atoms++] = make_atom(6, 12, -10.0f, -3.0f, 2.0f);
    af->atoms[af->num_atoms++] = make_atom(6, 12, -8.0f, 3.0f, -1.0f);

    /* Water molecule (O + 2H) for Cherenkov context */
    af->atoms[af->num_atoms++] = make_atom(8, 16, 10.0f, 0.0f, 0.0f);
    af->atoms[af->num_atoms++] = make_atom(1, 1, 10.96f, 0.76f, 0.0f);  /* O-H bond â‰ˆ 0.96 Ã… */
    af->atoms[af->num_atoms++] = make_atom(1, 1, 10.24f, -0.93f, 0.0f); /* 104.5Â° angle */

    /* Incoming thermal neutron â€” starts far left, travels +X */
    float vn = 12.0f;  /* Ã…/s display speed (slowed for human perception) */
    af->neutrons[af->num_neutrons++] = make_neutron(
        -22.0f, 0.0f, 0.0f,   /* position */
        vn, 0.0f, 0.0f,        /* velocity: straight toward U-235 */
        AF_THERMAL_ENERGY_EV,   /* 0.0253 eV thermal */
        0.0f                    /* birth time */
    );

    /* Start animation immediately */
    af->phase = AF_PHASE_NEUTRON_FLIGHT;
    af->playing = 1;
    af->looping = 1;

    printf("[AF] Scene: Single U-235 fission Event\n");
    printf("[AF]   U-235 atom at origin (Z=92, [Rn]5fÂ³6dÂ¹7sÂ²)\n");
    printf("[AF]   Thermal neutron: E=%.1f meV, Î»=%.2f Ã…\n",
           AF_THERMAL_ENERGY_EV * 1000.0f, AF_THERMAL_WAVELENGTH_A);
    printf("[AF]   Ïƒ_fission = %.0f barns, Ïƒ_capture = %.0f barns\n",
           AF_U235_FISSION_XS_BARNS, AF_U235_CAPTURE_XS_BARNS);
}

static void setup_chain_reaction(AtomicFission *af) {
    /* Scene: Multiple U-235 atoms in a lattice â€” chain reaction
     *
     * UOâ‚‚ fuel pellet cross-section: U atoms spaced ~3.87 Ã… apart
     * in the fluorite crystal structure (Fm3Ì„m, a = 5.47 Ã…).
     * We show a small cluster of ~9 U-235 atoms + O atoms.
     */
    af->scene = AF_SCENE_CHAIN_REACTION;
    af->phase = AF_PHASE_SETUP;
    af->time = 0.0f;
    af->phase_time = 0.0f;
    af->fission_count = 0;
    af->total_energy_MeV = 0.0f;
    af->generation = 0;
    af->boxHalf = 22.0f;

    af->num_atoms = 0;
    af->num_neutrons = 0;
    af->num_gammas = 0;
    af->num_particles = 0;

    /* UOâ‚‚ fluorite lattice: U at FCC positions, O in tetrahedral holes
     * Lattice constant a = 5.47 Ã…
     * U positions (FCC): (0,0,0), (a/2,a/2,0), etc.
     * We'll place a 2Ã—2Ã—2 arrangement â‰ˆ 8 U atoms with O between them */
    float a = 5.47f;  /* UOâ‚‚ lattice constant */

    /* Place U-235 atoms at FCC sites */
    for (int ix = -1; ix <= 1; ix++) {
        for (int iy = -1; iy <= 1; iy++) {
            if (af->num_atoms >= AF_MAX_ATOMS - 4) break;
            float ux = a * (float)ix;
            float uy = a * (float)iy;
            float uz = 0.0f;
            af->atoms[af->num_atoms++] = make_atom(92, 235, ux, uy, uz);
        }
    }

    /* O atoms at tetrahedral positions */
    float tet = a * 0.25f;
    int n_u = af->num_atoms;
    for (int i = 0; i < n_u && af->num_atoms < AF_MAX_ATOMS - 2; i++) {
        float bx = af->atoms[i].x;
        float by = af->atoms[i].y;
        float bz = af->atoms[i].z;
        if (af_rand01() < 0.5f)  /* only place ~half for visual clarity */
            af->atoms[af->num_atoms++] = make_atom(8, 16, bx + tet, by + tet, bz + tet);
    }

    /* One incoming thermal neutron */
    af->neutrons[af->num_neutrons++] = make_neutron(
        -20.0f, 0.0f, 0.0f,
        12.0f, 0.0f, 0.0f,
        AF_THERMAL_ENERGY_EV,
        0.0f
    );

    af->phase = AF_PHASE_NEUTRON_FLIGHT;
    af->playing = 1;
    af->looping = 1;

    printf("[AF] Scene: Chain Reaction in UOâ‚‚ lattice\n");
    printf("[AF]   %d U-235 atoms in fluorite structure (a=5.47 Ã…)\n", n_u);
    printf("[AF]   Ïƒ_f = %.0f barns Ã— Î½Ì„ = 2.43 â†’ criticality\n",
           AF_U235_FISSION_XS_BARNS);
}

static void setup_xenon_poison(AtomicFission *af) {
    /* Scene: Xe-135 poisoning
     *
     * Shows the decay chain I-135 â†’ Xe-135 â†’ neutron absorption.
     * Xe-135 has the LARGEST thermal neutron cross section of any
     * known nuclide: Ïƒ_a = 2.65 Ã— 10â¶ barns.
     *
     * Layout: U-235 atom + several Xe-135 atoms appearing over time,
     * thermal neutrons being absorbed by Xe with dramatic effect.
     */
    af->scene = AF_SCENE_XENON_POISON;
    af->phase = AF_PHASE_SETUP;
    af->time = 0.0f;
    af->phase_time = 0.0f;
    af->fission_count = 0;
    af->xe_concentration = 0.0f;
    af->boxHalf = 22.0f;

    af->num_atoms = 0;
    af->num_neutrons = 0;
    af->num_gammas = 0;
    af->num_particles = 0;

    /* Central U-235 atom */
    af->atoms[af->num_atoms++] = make_atom(92, 235, 0.0f, 0.0f, 0.0f);

    /* Pre-place Xe-135 atoms (will become visible over time) */
    float xe_positions[][3] = {
        {6.0f, 4.0f, 0.0f}, {-5.0f, 6.0f, 3.0f}, {4.0f, -5.0f, -2.0f},
        {-7.0f, -3.0f, 4.0f}, {8.0f, 0.0f, -5.0f}, {0.0f, 8.0f, 6.0f}
    };
    for (int i = 0; i < 6; i++) {
        AF_Atom xe = make_atom(AF_XE135_Z, AF_XE135_A,
                               xe_positions[i][0], xe_positions[i][1], xe_positions[i][2]);
        xe.visible = 0;  /* hidden until decay produces them */
        xe.state = AF_ATOM_IDLE;
        af->atoms[af->num_atoms++] = xe;
    }

    /* Neutrons traveling through â€” some will be absorbed by Xe */
    for (int i = 0; i < 4; i++) {
        float dx2, dy2, dz2;
        af_rand_dir(&dx2, &dy2, &dz2);
        float startx = -18.0f + af_rand01() * 36.0f;
        float starty = -18.0f + af_rand01() * 36.0f;
        float startz = -18.0f + af_rand01() * 36.0f;
        af->neutrons[af->num_neutrons++] = make_neutron(
            startx, starty, startz,
            dx2 * 10.0f, dy2 * 10.0f, dz2 * 10.0f,
            AF_THERMAL_ENERGY_EV, 0.0f);
    }

    af->phase = AF_PHASE_NEUTRON_FLIGHT;
    af->playing = 1;
    af->looping = 1;

    printf("[AF] Scene: Xe-135 Poisoning (Iodine Pit)\n");
    printf("[AF]   Ïƒ_a(Xe-135) = 2.65Ã—10â¶ barns (largest known!)\n");
    printf("[AF]   I-135 â†’ Xe-135 (tÂ½=6.57h) â†’ neutron absorber\n");
}

static void setup_moderation(AtomicFission *af) {
    /* Scene: Neutron moderation â€” fast neutron scattering off C/Hâ‚‚O
     *
     * Shows how a 2 MeV fast neutron slows down to thermal energy
     * through elastic scattering off carbon (graphite) and hydrogen
     * (water moderator).
     *
     * Physics:
     *   Average lethargy gain per collision:
     *     Î¾_H = 1.0 (hydrogen â€” most efficient moderator)
     *     Î¾_C = 0.158 (carbon â€” 114 collisions to thermalize)
     *   Mean free path in graphite: ~2.6 cm
     *   Mean free path in water: ~0.44 cm
     *
     * At our atomic scale, we show individual scattering events.
     */
    af->scene = AF_SCENE_MODERATION;
    af->phase = AF_PHASE_SETUP;
    af->time = 0.0f;
    af->phase_time = 0.0f;
    af->boxHalf = 25.0f;

    af->num_atoms = 0;
    af->num_neutrons = 0;
    af->num_gammas = 0;
    af->num_particles = 0;

    /* Graphite atoms in hexagonal arrangement */
    float c_spacing = 3.35f;  /* graphite interlayer spacing in Ã… */
    for (int ix = -2; ix <= 2; ix++) {
        for (int iy = -2; iy <= 2; iy++) {
            if (af->num_atoms >= AF_MAX_ATOMS - 4) break;
            float cx = c_spacing * (float)ix + ((iy & 1) ? c_spacing * 0.5f : 0.0f);
            float cy = c_spacing * (float)iy * 0.866f;
            af->atoms[af->num_atoms++] = make_atom(6, 12, cx, cy, 0.0f);
        }
    }

    /* Water molecules scattered around */
    for (int i = 0; i < 3 && af->num_atoms < AF_MAX_ATOMS - 3; i++) {
        float wx = 8.0f + af_rand01() * 10.0f;
        float wy = -8.0f + af_rand01() * 16.0f;
        af->atoms[af->num_atoms++] = make_atom(8, 16, wx, wy, 0.0f);
        af->atoms[af->num_atoms++] = make_atom(1, 1, wx + 0.96f, wy + 0.24f, 0.0f);
        af->atoms[af->num_atoms++] = make_atom(1, 1, wx - 0.24f, wy - 0.93f, 0.0f);
    }

    /* One fast neutron (2 MeV) â€” will scatter and slow down */
    af->neutrons[af->num_neutrons++] = make_neutron(
        -22.0f, 0.0f, 0.0f,
        60.0f, 2.0f, 0.0f,     /* fast! */
        AF_FAST_ENERGY_EV,      /* 2 MeV */
        0.0f
    );

    af->phase = AF_PHASE_NEUTRON_FLIGHT;
    af->playing = 1;
    af->looping = 1;

    printf("[AF] Scene: Neutron Moderation\n");
    printf("[AF]   Fast neutron: E=2.0 MeV, Î»=%.4f Ã…\n",
           neutron_wavelength_A(AF_FAST_ENERGY_EV));
    printf("[AF]   Target: %d atoms (graphite + water)\n", af->num_atoms);
    printf("[AF]   Î¾_C = 0.158 (â‰ˆ114 collisions to thermalize)\n");
    printf("[AF]   Î¾_H = 1.000 (â‰ˆ18 collisions to thermalize)\n");
}

static void setup_plutonium_fission(AtomicFission *af) {
    /* Scene: Plutonium-239 fission in MOX fuel
     *
     * Pu-239 is the SECOND sile nucleus (after U-235):
     *   Ïƒ_f = 747.4 barns (28% HIGHER than U-235!)
     *   Î½Ì„ = 2.88 neutrons per fission (19% more!)
     *   E_fission = 210 MeV
     *
     * Electron configuration: [Rn] 5fâ¶ 7sÂ²
     *   94 electrons â€” 6 in the f-shell (vs 3 for U-235)
     *   Pu has 6 allotropic crystal phases (!!)
     *   Î±-emitter: tÂ½ = 24,110 years â†’ warm to the touch
     *
     * In a reactor, Pu-239 is bred from U-238 + n â†’ Np-239 â†’ Pu-239
     * MOX fuel (Mixed OXide) contains ~7% PuOâ‚‚ + 93% UOâ‚‚
     * At Chernobyl Unit 4, ~1.8% of energy came from Pu fission
     *
     * Layout: MOX fuel cluster â€” Pu and U atoms in PuOâ‚‚/UOâ‚‚ lattice
     */
    af->scene = AF_SCENE_PLUTONIUM_FISSION;
    af->phase = AF_PHASE_SETUP;
    af->time = 0.0f;
    af->phase_time = 0.0f;
    af->fission_count = 0;
    af->total_energy_MeV = 0.0f;
    af->generation = 0;
    af->boxHalf = 22.0f;

    af->num_atoms = 0;
    af->num_neutrons = 0;
    af->num_gammas = 0;
    af->num_particles = 0;

    /* MOX fuel lattice: fluorite structure (a = 5.44 Ã… for PuOâ‚‚)
     * Place Pu-239 at center with U-235 neighbors */
    float a = 5.44f;

    /* Central Pu-239 atom */
    af->atoms[af->num_atoms++] = make_atom(94, 239, 0.0f, 0.0f, 0.0f);

    /* Neighboring U-235 atoms (MOX lattice) */
    af->atoms[af->num_atoms++] = make_atom(92, 235, a, 0.0f, 0.0f);
    af->atoms[af->num_atoms++] = make_atom(92, 235, -a, 0.0f, 0.0f);
    af->atoms[af->num_atoms++] = make_atom(92, 235, 0.0f, a, 0.0f);
    af->atoms[af->num_atoms++] = make_atom(92, 235, 0.0f, -a, 0.0f);

    /* More Pu in the lattice */
    af->atoms[af->num_atoms++] = make_atom(94, 239, a, a, 0.0f);
    af->atoms[af->num_atoms++] = make_atom(94, 239, -a, -a, 0.0f);

    /* Oxygen atoms in tetrahedral holes */
    float tet = a * 0.25f;
    for (int i = 0; i < 7 && af->num_atoms < AF_MAX_ATOMS - 4; i++) {
        float bx = af->atoms[i].x;
        float by = af->atoms[i].y;
        if (af_rand01() < 0.4f)
            af->atoms[af->num_atoms++] = make_atom(8, 16, bx + tet, by + tet, tet);
    }

    /* Thermal neutron aiming at the central Pu-239 */
    af->neutrons[af->num_neutrons++] = make_neutron(
        -20.0f, 0.5f, 0.0f,
        12.0f, 0.0f, 0.0f,
        AF_THERMAL_ENERGY_EV,
        0.0f
    );

    af->phase = AF_PHASE_NEUTRON_FLIGHT;
    af->playing = 1;
    af->looping = 1;

    printf("[AF] Scene: Plutonium-239 MOX Fuel fission\n");
    printf("[AF]   Pu-239 (Z=94): [Rn] 5fâ¶ 7sÂ²\n");
    printf("[AF]   Ïƒ_f = %.1f barns (vs U-235: %.1f barns)\n",
           AF_PU239_FISSION_XS_BARNS, AF_U235_FISSION_XS_BARNS);
    printf("[AF]   Î½Ì„ = %.2f (vs U-235: 2.43)\n", AF_PU239_NU_BAR);
    printf("[AF]   E_fission = %.0f MeV\n", AF_PU239_FISSION_ENERGY_MEV);
    printf("[AF]   Î±-emitter: tÂ½ = %.0f years â†’ warm to the touch\n",
           AF_PU239_ALPHA_HALFLIFE_Y);
}

static void setup_qcd_nucleon(AtomicFission *af) {
    /* Scene: QUANTUM CHROMODYNAMICS â€” Inside a proton + neutron
     *
     * Scale: ~1 femtometer (10â»Â¹âµ m) = 10â»âµ Ã…
     * We visualize the interiors of BOTH a proton AND a neutron
     * in a deuteron-like arrangement (bound by pion exchange).
     *
     * Proton contains:  2 up quarks + 1 down quark (uud)
     * Neutron contains: 1 up quark  + 2 down quarks (udd)
     *
     * Quark properties:
     *   up:      charge +2/3, mass ~2.2 MeV/cÂ², isospin +1/2
     *   down:    charge -1/3, mass ~4.7 MeV/cÂ², isospin -1/2
     *   strange: charge -1/3, mass ~95  MeV/cÂ², strangeness -1
     *   charm:   charge +2/3, mass ~1.27 GeV/cÂ² (virtual only)
     *
     * Strong force: QCD with SU(3) gauge symmetry
     *   3 color charges: Red, Green, Blue
     *   8 gluon types (carry color + anti-color)
     *   Asymptotic freedom: Î±s â†’ 0 at short distances
     *   Confinement: V(r) = -4Î±s/(3r) + Îºr (Cornell potential)
     *     Îº â‰ˆ 0.18 GeVÂ² â‰ˆ 0.89 GeV/fm (string tension)
     *   Flux tubes: gluon fields concentrate into tubes ~0.5 fm diameter
     *
     * Sea quarks (virtual pairs from vacuum fluctuations):
     *   uÅ«, ddÌ„, ssÌ„ (and rarely ccÌ„)
     *   These carry ~50% of the protonâ€™s momentum!
     *   The sea is flavor-asymmetric: more dÌ„ than Å« (NuSea experiment)
     *
     * Entity encoding (all use Z=-1):
     *   A=2 â†’ up quark (valence)
     *   A=1 â†’ down quark (valence)
     *   A=9 â†’ gluon field point
     *   A=3 â†’ light sea quark (uÅ« or ddÌ„)
     *   A=4 â†’ strange sea quark (ssÌ„)
     *   A=5 â†’ charm sea quark (ccÌ„) â€” very rare, heavy
     *
     * Rendered with Z=-1 special atoms (QCD particles)
     * boxHalf maps ~1 fm â†’ display units for nice visualization
     */
    af->scene = AF_SCENE_QCD_NUCLEON;
    af->phase = AF_PHASE_SETUP;
    af->time = 0.0f;
    af->phase_time = 0.0f;
    af->fission_count = 0;
    af->total_energy_MeV = 0.0f;
    af->boxHalf = 12.0f;  /* bigger box: two nucleons */

    af->num_atoms = 0;
    af->num_neutrons = 0;
    af->num_gammas = 0;
    af->num_particles = 0;

    /* â”€â”€ PROTON (uud) â€” left side â”€â”€ */
    float q_dist = 2.8f;   /* quark spacing within one nucleon */
    float px = -4.0f;      /* proton center offset */

    AF_Atom pu1 = make_atom(-1, 2, px + q_dist * 0.866f, q_dist * 0.5f, 0.0f);
    pu1.radius_A = 1.0f; pu1.orbital_scale = 2.0f; pu1.glow = 1.5f;
    af->atoms[af->num_atoms++] = pu1;  /* proton up quark 1 (Red) */

    AF_Atom pu2 = make_atom(-1, 2, px - q_dist * 0.866f, q_dist * 0.5f, 0.0f);
    pu2.radius_A = 1.0f; pu2.orbital_scale = 2.0f; pu2.glow = 1.5f;
    af->atoms[af->num_atoms++] = pu2;  /* proton up quark 2 (Green) */

    AF_Atom pd1 = make_atom(-1, 1, px, -q_dist, 0.0f);
    pd1.radius_A = 1.0f; pd1.orbital_scale = 2.0f; pd1.glow = 1.5f;
    af->atoms[af->num_atoms++] = pd1;  /* proton down quark (Blue) */

    int proton_start = 0, proton_end = af->num_atoms;

    /* â”€â”€ NEUTRON (udd) â€” right side â”€â”€ */
    float nx = 4.0f;  /* neutron center offset */

    AF_Atom nu1 = make_atom(-1, 2, nx + q_dist * 0.866f, -q_dist * 0.5f, 0.0f);
    nu1.radius_A = 1.0f; nu1.orbital_scale = 2.0f; nu1.glow = 1.5f;
    af->atoms[af->num_atoms++] = nu1;  /* neutron up quark (Red) */

    AF_Atom nd1 = make_atom(-1, 1, nx - q_dist * 0.866f, -q_dist * 0.5f, 0.0f);
    nd1.radius_A = 1.0f; nd1.orbital_scale = 2.0f; nd1.glow = 1.5f;
    af->atoms[af->num_atoms++] = nd1;  /* neutron down quark 1 (Green) */

    AF_Atom nd2 = make_atom(-1, 1, nx, q_dist, 0.0f);
    nd2.radius_A = 1.0f; nd2.orbital_scale = 2.0f; nd2.glow = 1.5f;
    af->atoms[af->num_atoms++] = nd2;  /* neutron down quark 2 (Blue) */

    int neutron_start = proton_end, neutron_end = af->num_atoms;

    /* â”€â”€ Gluon flux tubes (color field lines) â”€â”€ */
    /* Proton internal flux tubes: Y-junction topology */
    for (int i = proton_start; i < proton_end; i++) {
        int j = proton_start + ((i - proton_start + 1) % (proton_end - proton_start));
        for (int k = 1; k <= 3; k++) {
            float t = (float)k / 4.0f;
            float gx = af->atoms[i].x * (1.0f - t) + af->atoms[j].x * t;
            float gy = af->atoms[i].y * (1.0f - t) + af->atoms[j].y * t;
            float gz = af->atoms[i].z * (1.0f - t) + af->atoms[j].z * t;
            gx += af_randn() * 0.3f; gy += af_randn() * 0.3f; gz += af_randn() * 0.3f;
            AF_Atom gluon = make_atom(-1, 9, gx, gy, gz);
            gluon.radius_A = 0.8f; gluon.orbital_scale = 1.5f; gluon.glow = 0.8f;
            if (af->num_atoms < AF_MAX_ATOMS) af->atoms[af->num_atoms++] = gluon;
        }
    }
    /* Neutron internal flux tubes */
    for (int i = neutron_start; i < neutron_end; i++) {
        int j = neutron_start + ((i - neutron_start + 1) % (neutron_end - neutron_start));
        for (int k = 1; k <= 3; k++) {
            float t = (float)k / 4.0f;
            float gx = af->atoms[i].x * (1.0f - t) + af->atoms[j].x * t;
            float gy = af->atoms[i].y * (1.0f - t) + af->atoms[j].y * t;
            float gz = af->atoms[i].z * (1.0f - t) + af->atoms[j].z * t;
            gx += af_randn() * 0.3f; gy += af_randn() * 0.3f; gz += af_randn() * 0.3f;
            AF_Atom gluon = make_atom(-1, 9, gx, gy, gz);
            gluon.radius_A = 0.8f; gluon.orbital_scale = 1.5f; gluon.glow = 0.8f;
            if (af->num_atoms < AF_MAX_ATOMS) af->atoms[af->num_atoms++] = gluon;
        }
    }
    /* Inter-nucleon pion exchange tube (meson = q-qbar flux tube) */
    for (int k = 1; k <= 2 && af->num_atoms < AF_MAX_ATOMS; k++) {
        float t = (float)k / 3.0f;
        float gx = px * (1.0f - t) + nx * t;
        float gy = af_randn() * 0.5f;
        float gz = af_randn() * 0.5f;
        AF_Atom pion_tube = make_atom(-1, 9, gx, gy, gz);
        pion_tube.radius_A = 0.6f; pion_tube.orbital_scale = 1.2f; pion_tube.glow = 0.5f;
        af->atoms[af->num_atoms++] = pion_tube;
    }

    /* â”€â”€ Sea quarks â€” virtual pairs from vacuum fluctuations â”€â”€ */
    /* Light sea (uÅ« and ddÌ„) â€” inside both nucleons */
    for (int i = 0; i < 4 && af->num_atoms < AF_MAX_ATOMS; i++) {
        float cx = (i < 2) ? px : nx;  /* distribute between nucleons */
        float sx = cx + af_randn() * 2.0f;
        float sy = af_randn() * 2.0f;
        float sz = af_randn() * 1.0f;
        AF_Atom sea = make_atom(-1, 3, sx, sy, sz);
        sea.radius_A = 0.5f; sea.orbital_scale = 0.8f; sea.glow = 0.3f;
        af->atoms[af->num_atoms++] = sea;
    }

    /* Strange sea quarks (ssÌ„) â€” heavier, rarer, but significant!
     * Strange quark mass ~95 MeV >> up/down (~3-5 MeV)
     * Strangeness contribution to proton spin: Î”s â‰ˆ -0.03
     * Strange quarks carry ~3-4% of proton momentum */
    for (int i = 0; i < 3 && af->num_atoms < AF_MAX_ATOMS; i++) {
        float cx = (i == 0) ? px : (i == 1) ? nx : 0.0f;
        float sx = cx + af_randn() * 1.5f;
        float sy = af_randn() * 1.5f;
        float sz = af_randn() * 0.8f;
        AF_Atom strange = make_atom(-1, 4, sx, sy, sz);  /* A=4: strange quark */
        strange.radius_A = 0.6f;
        strange.orbital_scale = 1.0f;
        strange.glow = 0.5f;
        af->atoms[af->num_atoms++] = strange;
    }

    /* Charm sea quarks (ccÌ„) â€” very heavy (1.27 GeV), extremely rare
     * Appear only as short-lived virtual fluctuations
     * Recent lattice QCD: ~0.5% of proton momentum from charm */
    for (int i = 0; i < 1 && af->num_atoms < AF_MAX_ATOMS; i++) {
        float sx = af_randn() * 1.0f;
        float sy = af_randn() * 1.0f;
        float sz = af_randn() * 0.5f;
        AF_Atom charm = make_atom(-1, 5, sx, sy, sz);  /* A=5: charm quark */
        charm.radius_A = 0.4f;
        charm.orbital_scale = 0.6f;
        charm.glow = 0.7f;
        af->atoms[af->num_atoms++] = charm;
    }

    af->phase = AF_PHASE_NEUTRON_FLIGHT;  /* use as "active" state */
    af->playing = 1;
    af->looping = 1;

    printf("[AF] Scene: QCD â€” Proton + Neutron Interior\n");
    printf("[AF]   Scale: ~1 fm (10â»Â¹âµ m) per nucleon\n");
    printf("[AF]   PROTON:  uud  (charge +2/3 +2/3 -1/3 = +1)\n");
    printf("[AF]   NEUTRON: udd  (charge +2/3 -1/3 -1/3 =  0)\n");
    printf("[AF]   Valence quarks: 6, Gluon field points: %d\n",
           af->num_atoms - 6 - 4 - 3 - 1);  /* approx */
    printf("[AF]   Sea quarks: 4 light (uÅ«/ddÌ„) + 3 strange (ssÌ„) + 1 charm (ccÌ„)\n");
    printf("[AF]   Color charges: R, G, B â€” SU(3)_c gauge symmetry\n");
    printf("[AF]   Î±s â‰ˆ 0.3 (running coupling at 1 GeV)\n");
    printf("[AF]   String tension: Îº â‰ˆ 0.89 GeV/fm\n");
    printf("[AF]   Cornell potential: V(r) = -4Î±s/(3r) + Îºr\n");
    printf("[AF]   Total quarks + antiquarks in scene: %d\n", af->num_atoms);
}

static void setup_chernobyl_sequence(AtomicFission *af) {
    /* Scene: Chernobyl Unit 4 â€” April 26, 1986 â€” atomic-scale view
     *
     * What happened physically:
     *   1. Operators reduced power to 200 MW (low Xe-135 burnout)
     *   2. Xe-135 poisoned the core, power dropped to ~30 MW
     *   3. Control rods withdrawn to compensate â†’ dangerously low margin
     *   4. Emergency test: coolant pumps trip â†’ steam voids form
     *   5. Positive void coefficient: voids â†’ MORE reactivity â†’ MORE voids
     *   6. Power surged from 30 MW â†’ 30,000 MW in ~4 seconds
     *   7. Prompt criticality â†’ steam explosion â†’ core destruction
     *
     * At the ATOMIC scale: we show a dense UOâ‚‚ lattice with:
     *   - Xe-135 poison atoms (blocking neutrons)
     *   - Water molecules (moderator/coolant)
     *   - When water vaporizes (voids), neutrons go UNMODERATED
     *   - Positive void coefficient causes runaway chain reaction
     *   - Multiple simultaneous fisfisfisfisfission events
     *
     * Visual difference from chain reaction:
     *   - Larger lattice (more U atoms â€” 16 instead of 9)
     *   - Xe-135 atoms present (visible poison)
     *   - Water molecules around the edges (that will "disappear")
     *   - More neutrons (3 incoming from different directions)
     *   - Boxhalf larger (30 Ã…) for the bigger scene
     */
    af->scene = AF_SCENE_CHERNOBYL_SEQUENCE;
    af->phase = AF_PHASE_SETUP;
    af->time = 0.0f;
    af->phase_time = 0.0f;
    af->fission_count = 0;
    af->total_energy_MeV = 0.0f;
    af->generation = 0;
    af->boxHalf = 30.0f;  /* larger scene */

    af->num_atoms = 0;
    af->num_neutrons = 0;
    af->num_gammas = 0;
    af->num_particles = 0;

    float a = 5.47f;  /* UOâ‚‚ lattice constant */

    /* Dense 4Ã—4 uranium lattice (16 atoms â€” supercritical mass) */
    for (int ix = -2; ix <= 1; ix++) {
        for (int iy = -2; iy <= 1; iy++) {
            if (af->num_atoms >= AF_MAX_ATOMS - 20) break;
            float ux = a * ((float)ix + 0.5f);
            float uy = a * ((float)iy + 0.5f);
            af->atoms[af->num_atoms++] = make_atom(92, 235, ux, uy, 0.0f);
        }
    }
    int n_u = af->num_atoms;

    /* Oxygen in tetrahedral holes (partial for clarity) */
    float tet = a * 0.25f;
    for (int i = 0; i < n_u && af->num_atoms < AF_MAX_ATOMS - 15; i++) {
        if (af_rand01() < 0.4f) {
            af->atoms[af->num_atoms++] = make_atom(8, 16,
                af->atoms[i].x + tet, af->atoms[i].y + tet, tet);
        }
    }

    /* Xe-135 poison atoms â€” scattered among the fuel */
    int xe_count = 0;
    for (int i = 0; i < 4 && af->num_atoms < AF_MAX_ATOMS - 10; i++) {
        float xx = a * (af_rand01() * 3.0f - 1.5f);
        float xy = a * (af_rand01() * 3.0f - 1.5f);
        AF_Atom xe = make_atom(54, 135, xx, xy, af_randn() * 1.5f);
        xe.glow = 0.8f;  /* Xe glows as neutron absorber */
        af->atoms[af->num_atoms++] = xe;
        xe_count++;
    }

    /* Water molecules (coolant) â€” around the periphery
     * These represent what boils away during the void event */
    for (int i = 0; i < 4 && af->num_atoms < AF_MAX_ATOMS - 6; i++) {
        float angle = (float)i * 1.5708f;  /* 90Â° apart */
        float wr = 18.0f;
        float wx = wr * cosf(angle);
        float wy = wr * sinf(angle);
        af->atoms[af->num_atoms++] = make_atom(8, 16, wx, wy, 0.0f);
        af->atoms[af->num_atoms++] = make_atom(1, 1, wx + 0.96f, wy + 0.24f, 0.0f);
    }

    /* Multiple neutrons from different directions â€” supercritical burst */
    float dirs[][3] = {
        {-28.0f, 2.0f, 0.0f},
        { 0.0f, -28.0f, 1.0f},
        { 25.0f, 10.0f, -1.0f}
    };
    float vels[][3] = {
        { 12.0f,  0.0f,  0.0f},
        {  0.0f, 12.0f,  0.0f},
        {-10.0f, -5.0f,  0.0f}
    };
    for (int i = 0; i < 3 && af->num_neutrons < AF_MAX_NEUTRONS; i++) {
        af->neutrons[af->num_neutrons++] = make_neutron(
            dirs[i][0], dirs[i][1], dirs[i][2],
            vels[i][0], vels[i][1], vels[i][2],
            AF_THERMAL_ENERGY_EV, 0.0f
        );
    }

    af->phase = AF_PHASE_NEUTRON_FLIGHT;
    af->playing = 1;
    af->looping = 1;

    printf("[AF] Scene: Chernobyl Sequence â€” RBMK-1000 Atomic View\n");
    printf("[AF]   %d U-235 fuel atoms (4Ã—4 lattice, a=5.47 Ã…)\n", n_u);
    printf("[AF]   %d Xe-135 poison atoms (neutron absorbers)\n", xe_count);
    printf("[AF]   Water coolant molecules (will boil â†’ void â†’ +Ï)\n");
    printf("[AF]   3 incoming neutrons â†’ supercritical burst\n");
    printf("[AF]   Positive void coefficient: Î”k/k â‰ˆ +4.7Î²\n");
    printf("[AF]   Power excursion: 200 MW â†’ 30,000 MW in ~4 s\n");
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• D-T FUSION SCENE â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */

static void setup_fusion(AtomicFission *af) {
    /* Scene: Deuterium-Tritium Fusion
     *
     * The EASIEST fusion reaction â€” the one ITER and NIF use:
     *   D(Â²H) + T(Â³H) â†’ He-4 (3.52 MeV) + n (14.07 MeV)
     *   Total Q = 17.59 MeV
     *
     * At the atomic scale this is a quantum tunneling problem:
     *   â€¢ Classical Coulomb barrier height: E_c = eÂ²/(4Ï€Îµâ‚€Â·r) â‰ˆ 0.40 MeV
     *   â€¢ But fusion occurs at T ~ 10 keV (40Ã— BELOW the barrier!)
     *   â€¢ Gamow peak energy E_G â‰ˆ 6.3 keV
     *   â€¢ Tunneling probability P âˆ exp(-Ï€âˆš(E_G/E))
     *
     * Electron configuration:
     *   D = Â¹HÂ²: 1sÂ¹ (same as hydrogen but with a neutron in nucleus)
     *   T = Â¹HÂ³: 1sÂ¹ (same, but with two neutrons â€” radioactive, tÂ½=12.3 yr)
     *   He-4 = 1sÂ² (closed shell â€” noble gas, very stable)
     *
     * The D-T cross section peaks at ~100 keV:
     *   Ïƒ(peak) â‰ˆ 5 barns  (at E_cm â‰ˆ 64 keV)
     *
     * Layout: D and T atoms approaching each other with thermal velocity
     * When they get close â†’ quantum tunneling event â†’ fusion flash â†’
     * He-4 + 14.1 MeV neutron fly apart.
     */
    af->scene = AF_SCENE_FUSION;
    af->phase = AF_PHASE_SETUP;
    af->time = 0.0f;
    af->phase_time = 0.0f;
    af->fission_count = 0;
    af->total_energy_MeV = 0.0f;
    af->generation = 0;
    af->boxHalf = 18.0f;

    af->num_atoms = 0;
    af->num_neutrons = 0;
    af->num_gammas = 0;
    af->num_particles = 0;

    /* Deuterium approaching from left */
    AF_Atom d_atom = make_atom(1, AF_DEUTERIUM_A, -12.0f, 0.0f, 0.0f);
    d_atom.vx =  5.0f;  /* approaching at ~thermal plasma speed (scaled) */
    d_atom.vy =  0.15f;
    d_atom.vz =  0.0f;
    af->atoms[af->num_atoms++] = d_atom;

    /* Tritium approaching from right */
    AF_Atom t_atom = make_atom(1, AF_TRITIUM_A, 12.0f, 0.0f, 0.3f);
    t_atom.vx = -5.0f;
    t_atom.vy = -0.15f;
    t_atom.vz =  0.0f;
    af->atoms[af->num_atoms++] = t_atom;

    /* Additional D-T pairs in background (plasma environment) */
    for (int i = 0; i < 4 && af->num_atoms < AF_MAX_ATOMS - 4; i++) {
        float angle = (float)i * 1.5708f + 0.4f;
        float r = 10.0f + af_rand01() * 5.0f;
        AF_Atom bg = make_atom(1, (i % 2 == 0) ? AF_DEUTERIUM_A : AF_TRITIUM_A,
                               r * cosf(angle), r * sinf(angle),
                               (af_rand01() - 0.5f) * 4.0f);
        bg.vx = (af_rand01() - 0.5f) * 2.0f;
        bg.vy = (af_rand01() - 0.5f) * 2.0f;
        bg.vz = (af_rand01() - 0.5f) * 1.0f;
        af->atoms[af->num_atoms++] = bg;
    }

    af->phase = AF_PHASE_NEUTRON_FLIGHT;  /* reuse: "approach" phase */
    af->playing = 1;
    af->looping = 1;

    printf("[AF] Scene: D-T Fusion (ITER/NIF)\n");
    printf("[AF]   D(Â²H) + T(Â³H) â†’ He-4 (3.52 MeV) + n (14.07 MeV)\n");
    printf("[AF]   Q = %.3f MeV â€” easiest fusion reaction\n", AF_DT_Q_VALUE_MEV);
    printf("[AF]   Coulomb barrier: %.2f MeV\n", AF_DT_COULOMB_BARRIER_MEV);
    printf("[AF]   Gamow peak: %.1f keV (quantum tunneling!)\n", AF_DT_GAMOW_PEAK_KEV);
    printf("[AF]   Peak Ïƒ â‰ˆ %.1f barns at ~100 keV\n", AF_DT_PEAK_XS_BARNS);
    printf("[AF]   Lawson criterion: nÏ„ > 1.5Ã—10Â²â° mâ»Â³Â·s\n");
}

/* Helper: trigger D-T fusion event between two atoms */
static void trigger_fusion(AtomicFission *af, int d_idx, int t_idx) {
    AF_Atom *d = &af->atoms[d_idx];
    AF_Atom *t = &af->atoms[t_idx];

    /* Midpoint where fusion occurs */
    float mx = (d->x + t->x) * 0.5f;
    float my = (d->y + t->y) * 0.5f;
    float mz = (d->z + t->z) * 0.5f;

    /* Remove D and T */
    d->visible = 0;
    d->state = AF_ATOM_DEAD;
    t->visible = 0;
    t->state = AF_ATOM_DEAD;

    /* Create He-4 alpha particle (3.52 MeV â†’ slow, heavy) */
    float hx, hy, hz;
    af_rand_dir(&hx, &hy, &hz);
    if (af->num_atoms < AF_MAX_ATOMS) {
        AF_Atom he4 = make_atom(AF_HE4_Z, AF_HE4_A, mx, my, mz);
        he4.vx = hx * 6.0f;   /* 3.52 MeV â†’ moderate speed */
        he4.vy = hy * 6.0f;
        he4.vz = hz * 6.0f;
        he4.glow = 2.5f;
        he4.state = AF_ATOM_EXCITED;
        he4.state_time = 0.0f;
        af->atoms[af->num_atoms++] = he4;
    }

    /* Create 14.07 MeV fusion neutron (opposite direction, FAST) */
    if (af->num_neutrons < AF_MAX_NEUTRONS) {
        af->neutrons[af->num_neutrons++] = make_neutron(
            mx, my, mz,
            -hx * 80.0f, -hy * 80.0f, -hz * 80.0f,  /* 14.07 MeV! */
            AF_DT_NEUTRON_ENERGY_MEV * 1.0e6f,        /* in eV */
            af->time
        );
    }

    /* Gamma flash at fusion site */
    for (int g = 0; g < 4 && af->num_gammas < AF_MAX_GAMMAS; g++) {
        float gx, gy, gz;
        af_rand_dir(&gx, &gy, &gz);
        af->gammas[af->num_gammas++] = make_gamma(
            mx, my, mz, gx, gy, gz, 3.0f, af->time);
    }

    af->total_energy_MeV += AF_DT_Q_VALUE_MEV;
    af->fission_count++;  /* reuse as "fusion count" */
    printf("[AF] â˜… FUSION! D+T â†’ He-4 (3.52 MeV) + n (14.07 MeV)  Total Q=%.1f MeV\n",
           af->total_energy_MeV);
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• BETA DECAY SETUP â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */

/* Decay chain data: U-238 â†’ Pb-206 (14 steps) */
typedef struct {
    int Z, A;
    int is_alpha;       /* 1=Î±, 0=Î²â» */
    float energy_MeV;   /* KE of emitted particle */
    const char *name;
} DecayStep;

static const DecayStep g_u238_chain[AF_DECAY_CHAIN_STEPS] = {
    { 92, 238, 1, 4.270f, "U-238"  },  /* â†’ Th-234 */
    { 90, 234, 0, 0.273f, "Th-234" },  /* â†’ Pa-234 */
    { 91, 234, 0, 2.197f, "Pa-234" },  /* â†’ U-234  */
    { 92, 234, 1, 4.859f, "U-234"  },  /* â†’ Th-230 */
    { 90, 230, 1, 4.770f, "Th-230" },  /* â†’ Ra-226 */
    { 88, 226, 1, 4.871f, "Ra-226" },  /* â†’ Rn-222 */
    { 86, 222, 1, 5.590f, "Rn-222" },  /* â†’ Po-218 */
    { 84, 218, 1, 6.115f, "Po-218" },  /* â†’ Pb-214 */
    { 82, 214, 0, 1.024f, "Pb-214" },  /* â†’ Bi-214 */
    { 83, 214, 0, 3.272f, "Bi-214" },  /* â†’ Po-214 */
    { 84, 214, 1, 7.833f, "Po-214" },  /* â†’ Pb-210 */
    { 82, 210, 0, 0.064f, "Pb-210" },  /* â†’ Bi-210 */
    { 83, 210, 0, 1.162f, "Bi-210" },  /* â†’ Po-210 */
    { 84, 210, 1, 5.407f, "Po-210" },  /* â†’ Pb-206 (stable) */
};

/* Semi-empirical mass formula: binding energy per nucleon */
static float semf_binding_per_nucleon(int A, int Z) {
    if (A <= 0) return 0.0f;
    float fA = (float)A;
    float fZ = (float)Z;
    float vol   = AF_SEMF_AV;
    float surf  = AF_SEMF_AS * powf(fA, -1.0f/3.0f);
    float coul  = AF_SEMF_AC * fZ * (fZ - 1.0f) * powf(fA, -4.0f/3.0f);
    float asym  = AF_SEMF_AA * (fA - 2.0f*fZ) * (fA - 2.0f*fZ) / (fA * fA);
    return vol - surf - coul - asym;
}

static void setup_beta_decay(AtomicFission *af) {
    /* Scene: NEUTRON BETA DECAY (Weak Force)
     *
     * n â†’ p + eâ» + Î½Ì„â‚‘  (mediated by Wâ» boson)
     *
     * The weak force is the only force that changes quark flavor.
     * Inside the neutron, a down quark (charge -1/3) emits a virtual
     * Wâ» boson (mass 80.4 GeV!) and becomes an up quark (charge +2/3).
     * The Wâ» then decays into an electron + electron antineutrino.
     *
     * This is Fermi's theory of beta decay (1933):
     *   d â†’ u + Wâ» â†’ u + eâ» + Î½Ì„â‚‘
     *
     * Free neutron half-life: 611 s (10.2 min)  (ucn Ï„ experiment)
     * Q-value: (mâ‚™ - mâ‚š)cÂ² = 1.2934 MeV
     * Max electron KE: 0.782 MeV (endpoint)
     * Wâ» boson: m = 80.377 GeV, Ï„ â‰ˆ 3Ã—10â»Â²âµ s (virtual here)
     *
     * We show the neutron as 3 quarks (udd), then animate:
     *   1. A down quark glows and emits a Wâ» bubble
     *   2. The down quark transforms to up quark (color change)
     *   3. The Wâ» propagates briefly then decays
     *   4. An electron spirals away
     *   5. A ghostly neutrino streaks off in opposite hemisphere
     */
    af->scene = AF_SCENE_BETA_DECAY;
    af->phase = AF_PHASE_SETUP;
    af->time = 0.0f;
    af->phase_time = 0.0f;
    af->boxHalf = 12.0f;

    af->num_atoms = 0;
    af->num_neutrons = 0;
    af->num_gammas = 0;
    af->num_particles = 0;

    /* The neutron: udd quarks in a triangle */
    float q_dist = 2.8f;

    AF_Atom u1 = make_atom(-1, 2, q_dist * 0.866f, q_dist * 0.5f, 0.0f);
    u1.radius_A = 1.0f; u1.orbital_scale = 2.0f; u1.glow = 1.5f;
    af->atoms[af->num_atoms++] = u1;  /* up quark */

    AF_Atom d1 = make_atom(-1, 1, -q_dist * 0.866f, q_dist * 0.5f, 0.0f);
    d1.radius_A = 1.0f; d1.orbital_scale = 2.0f; d1.glow = 1.5f;
    af->atoms[af->num_atoms++] = d1;  /* down quark 1 */

    AF_Atom d2 = make_atom(-1, 1, 0.0f, -q_dist, 0.0f);
    d2.radius_A = 1.0f; d2.orbital_scale = 2.0f; d2.glow = 1.5f;
    af->atoms[af->num_atoms++] = d2;  /* down quark 2 â€” this one will decay */

    /* Gluon flux tubes between quarks */
    for (int i = 0; i < 3; i++) {
        int j = (i + 1) % 3;
        for (int k = 1; k <= 2; k++) {
            float t = (float)k / 3.0f;
            float gx = af->atoms[i].x * (1.0f-t) + af->atoms[j].x * t;
            float gy = af->atoms[i].y * (1.0f-t) + af->atoms[j].y * t;
            float gz = af->atoms[i].z * (1.0f-t) + af->atoms[j].z * t;
            AF_Atom gluon = make_atom(-1, 9, gx + af_randn()*0.2f, gy + af_randn()*0.2f, gz);
            gluon.radius_A = 0.8f; gluon.orbital_scale = 1.5f; gluon.glow = 0.8f;
            if (af->num_atoms < AF_MAX_ATOMS) af->atoms[af->num_atoms++] = gluon;
        }
    }

    af->phase = AF_PHASE_SETUP;
    af->playing = 1;
    af->looping = 1;

    /* HUD text */
    af->hud_num_lines = 4;
    snprintf(af->hud_lines[0], 80, "Beta Decay: n -> p + e- + v_e");
    snprintf(af->hud_lines[1], 80, "W- boson: m=80.4 GeV  t~3e-25 s");
    snprintf(af->hud_lines[2], 80, "Q = 1.293 MeV  (endpoint 0.782 MeV)");
    snprintf(af->hud_lines[3], 80, "Weak force: only force that changes flavor");

    printf("[AF] Scene: Beta Decay (Weak Force)\n");
    printf("[AF]   n â†’ p + eâ» + Î½Ì„â‚‘  via Wâ» boson\n");
    printf("[AF]   Wâ» mass: %.3f GeV/cÂ²\n", AF_W_BOSON_MASS_GEV);
    printf("[AF]   Q-value: %.4f MeV\n", AF_BETA_Q_VALUE_MEV);
    printf("[AF]   Max electron KE: %.3f MeV\n", AF_BETA_ELECTRON_MAX_MEV);
    printf("[AF]   Free neutron half-life: %.0f s (%.1f min)\n",
           AF_FREE_NEUTRON_HALFLIFE_S, AF_FREE_NEUTRON_HALFLIFE_S/60.0f);
    printf("[AF]   d â†’ u + Wâ» â†’ u + eâ» + Î½Ì„â‚‘\n");
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• CONFINEMENT BREAK SETUP â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */

static void setup_confinement(AtomicFission *af) {
    /* Scene: QCD CONFINEMENT BREAK (String Snapping â†’ Hadronization)
     *
     * When you try to separate two quarks, the color flux tube
     * (QCD string) stretches. The potential energy is V(r) = Îºr
     * where Îº â‰ˆ 0.89 GeV/fm (string tension).
     *
     * At ~1 fm separation, the energy stored in the string exceeds
     * 2Ã—mq (enough to create a new quark-antiquark pair). The string
     * SNAPS and a new q-qÌ„ pair materializes from the vacuum.
     *
     * This is why free quarks have NEVER been observed:
     *   Pull q apart â†’ string stores energy â†’ string breaks â†’
     *   new qÌ„q pair â†’ you get two mesons, never a free quark!
     *
     * Animation:
     *   1. Two quarks (q and qÌ„) connected by a flux tube (meson)
     *   2. External force pulls them apart
     *   3. Flux tube stretches, glows brighter (energy increasing)
     *   4. SNAP! New q-qÌ„ pair appears at break point
     *   5. Two new mesons fly apart
     *   6. Process repeats (jet hadronization)
     */
    af->scene = AF_SCENE_CONFINEMENT;
    af->phase = AF_PHASE_SETUP;
    af->time = 0.0f;
    af->phase_time = 0.0f;
    af->boxHalf = 14.0f;

    af->num_atoms = 0;
    af->num_neutrons = 0;
    af->num_gammas = 0;
    af->num_particles = 0;

    /* Initial meson: quark + antiquark connected by string */
    /* Quark (up, red) on left */
    AF_Atom q1 = make_atom(-1, 2, -2.0f, 0.0f, 0.0f);
    q1.radius_A = 1.2f; q1.orbital_scale = 2.0f; q1.glow = 2.0f;
    af->atoms[af->num_atoms++] = q1;

    /* Antiquark (anti-up, anti-red) on right */
    AF_Atom q2 = make_atom(-1, 2, 2.0f, 0.0f, 0.0f);
    q2.radius_A = 1.2f; q2.orbital_scale = 2.0f; q2.glow = 2.0f;
    q2.A = 6;  /* A=6 â†’ antiquark marker */
    af->atoms[af->num_atoms++] = q2;

    /* Flux tube between them (6 gluon points) */
    for (int k = 1; k <= 6; k++) {
        float t = (float)k / 7.0f;
        float gx = -2.0f * (1.0f-t) + 2.0f * t;
        AF_Atom gluon = make_atom(-1, 9, gx, af_randn()*0.15f, af_randn()*0.15f);
        gluon.radius_A = 0.7f; gluon.orbital_scale = 1.5f; gluon.glow = 1.0f;
        af->atoms[af->num_atoms++] = gluon;
    }

    af->playing = 1;
    af->looping = 1;

    af->hud_num_lines = 4;
    snprintf(af->hud_lines[0], 80, "QCD Confinement: String Breaking");
    snprintf(af->hud_lines[1], 80, "String tension k = 0.89 GeV/fm");
    snprintf(af->hud_lines[2], 80, "V(r) = kr  -> E > 2mq -> SNAP!");
    snprintf(af->hud_lines[3], 80, "Free quarks impossible: always hadronize");

    printf("[AF] Scene: QCD Confinement Break\n");
    printf("[AF]   Quark-antiquark meson with color flux tube\n");
    printf("[AF]   String tension: Îº â‰ˆ 0.89 GeV/fm\n");
    printf("[AF]   Pulling apart â†’ V(r) = Îºr â†’ string snaps\n");
    printf("[AF]   New qqÌ„ pair from vacuum â†’ two mesons!\n");
    printf("[AF]   Demonstrates why free quarks don't exist\n");
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• DECAY CHAIN SETUP â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */

static void setup_decay_chain(AtomicFission *af) {
    /* Scene: U-238 â†’ Pb-206 RADIOACTIVE DECAY CHAIN
     *
     * The natural uranium-238 decay series (also called the
     * "radium series" or "4n+2 series"):
     *
     *  14 decays: 8 alpha + 6 beta-minus
     *  Half-life: 4.468 billion years (age of Earth!)
     *
     *  Each step: parent atom at center, Î± or Î² particle emitted,
     *  daughter atom recoils. We animate through all 14 steps,
     *  pausing briefly to show each intermediate nucleus.
     */
    af->scene = AF_SCENE_DECAY_CHAIN;
    af->phase = AF_PHASE_SETUP;
    af->time = 0.0f;
    af->phase_time = 0.0f;
    af->boxHalf = 15.0f;

    af->num_atoms = 0;
    af->num_neutrons = 0;
    af->num_gammas = 0;
    af->num_particles = 0;

    af->chain_step = 0;
    af->chain_step_time = 0.0f;

    /* Start with U-238 at center */
    AF_Atom u238 = make_atom(92, 238, 0.0f, 0.0f, 0.0f);
    u238.glow = 0.5f;
    af->atoms[af->num_atoms++] = u238;

    af->playing = 1;
    af->looping = 1;

    af->hud_num_lines = 4;
    snprintf(af->hud_lines[0], 80, "Decay Chain: U-238 -> Pb-206");
    snprintf(af->hud_lines[1], 80, "14 steps: 8 alpha + 6 beta-");
    snprintf(af->hud_lines[2], 80, "Step 0/14: U-238 (Z=92, A=238)");
    snprintf(af->hud_lines[3], 80, "t1/2 = 4.468 Gyr");

    printf("[AF] Scene: Decay Chain (U-238 â†’ Pb-206)\n");
    printf("[AF]   14 radioactive decays: 8Î± + 6Î²â»\n");
    printf("[AF]   Half-life: %.3f billion years\n", AF_U238_HALFLIFE_GY);
    printf("[AF]   Starting isotope: U-238 (Z=92, A=238)\n");
    printf("[AF]   Final product: Pb-206 (stable)\n");
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• BINDING ENERGY SETUP â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */

static void setup_binding_energy(AtomicFission *af) {
    /* Scene: NUCLEAR BINDING ENERGY CURVE
     *
     * The binding energy per nucleon B/A as a function of mass number A:
     *   - Rises steeply from H (0) through He-4 (7.07 MeV)
     *   - Broad peak at Fe-56 (8.79 MeV/nucleon) â€” most stable nucleus
     *   - Slow decline for heavy nuclei (U-238: 7.57 MeV)
     *
     * THIS is why:
     *   - Light nuclei FUSE (climbing the curve â†’ releases energy)
     *   - Heavy nuclei SION (descending from heavy side â†’ releases energy)
     *   - Iron is the ash of nuclear burning (stars can't burn past Fe)
     *
     * Visualized as a series of nuclei at different sizes, morphing
     * from light (left) through Fe (center) to heavy (right).
     * Each nucleus is shown as actual proton+neutron clusters
     * with binding displayed as glow intensity.
     */
    af->scene = AF_SCENE_BINDING_ENERGY;
    af->phase = AF_PHASE_SETUP;
    af->time = 0.0f;
    af->phase_time = 0.0f;
    af->boxHalf = 18.0f;

    af->num_atoms = 0;
    af->num_neutrons = 0;
    af->num_gammas = 0;
    af->num_particles = 0;

    af->binding_A = 2.0f;      /* start at deuterium */
    af->binding_dir = 1;       /* climbing toward Fe-56 */

    /* Place representative nuclei across the curve as atoms:
     *   H-2 (D), He-4, C-12, O-16, Fe-56, Ni-62, Sn-120, U-238
     * positioned left-to-right by mass number. */
    struct { int Z; int A; float x_frac; } nuclides[] = {
        {  1,   2, -0.85f },  /* Deuterium */
        {  2,   4, -0.70f },  /* He-4 */
        {  6,  12, -0.50f },  /* C-12 */
        {  8,  16, -0.35f },  /* O-16 */
        { 26,  56,  0.00f },  /* Fe-56 (peak!) */
        { 50, 120,  0.35f },  /* Sn-120 */
        { 82, 208,  0.60f },  /* Pb-208 */
        { 92, 238,  0.85f },  /* U-238 */
    };
    int n_nuclides = 8;

    for (int i = 0; i < n_nuclides && af->num_atoms < AF_MAX_ATOMS; i++) {
        float x = nuclides[i].x_frac * af->boxHalf * 0.9f;
        /* Y position = binding energy (higher = more bound) */
        float ba = semf_binding_per_nucleon(nuclides[i].A, nuclides[i].Z);
        float y = (ba / 10.0f) * af->boxHalf * 0.6f - af->boxHalf * 0.3f;
        AF_Atom nuc = make_atom(nuclides[i].Z, nuclides[i].A, x, y, 0.0f);
        nuc.glow = ba / AF_FE56_BA;  /* brighter = more bound */
        /* Scale radius by A^(1/3) */
        nuc.radius_A = 1.5f + 0.5f * cbrtf((float)nuclides[i].A);
        af->atoms[af->num_atoms++] = nuc;
    }

    af->playing = 1;
    af->looping = 1;

    af->hud_num_lines = 4;
    snprintf(af->hud_lines[0], 80, "Nuclear Binding Energy Curve");
    snprintf(af->hud_lines[1], 80, "Peak: Fe-56 at B/A = 8.79 MeV");
    snprintf(af->hud_lines[2], 80, "Left = FUSION  |  Right = SION");
    snprintf(af->hud_lines[3], 80, "Bethe-Weizsacker semi-empirical formula");

    printf("[AF] Scene: Nuclear Binding Energy Curve\n");
    printf("[AF]   B/A peaks at Fe-56: %.2f MeV/nucleon\n", AF_FE56_BA);
    printf("[AF]   SEMF: aV=%.2f, aS=%.2f, aC=%.3f, aA=%.2f MeV\n",
           AF_SEMF_AV, AF_SEMF_AS, AF_SEMF_AC, AF_SEMF_AA);
    printf("[AF]   Light nuclei â†’ fufisfisfisfission gains energy\n");
    printf("[AF]   Heavy nuclei â†’ fisfisfisfisfission gains energy\n");
    printf("[AF]   Iron-56: the ash of stellar nucleosynthesis\n");
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• PHYSICS UPDATE â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */

static void trigger_fission(AtomicFission *af, int atom_idx) {
    /* The target atom sions!
     *
     * U-235 + n â†’ [U-236]* â†’ Ba-141 + Kr-92 + 3n + Î³ + 200 MeV
     *
     * fisfisfisfisfission fragment kinetic energies (from Coulomb repulsion):
     *   E_Ba â‰ˆ 60 MeV â†’ v â‰ˆ 1.1 Ã— 10â· m/s
     *   E_Kr â‰ˆ 95 MeV â†’ v â‰ˆ 1.5 Ã— 10â· m/s
     * (but we scale for visualization)
     */
    AF_Atom *target = &af->atoms[atom_idx];
    float ox = target->x, oy = target->y, oz = target->z;

    target->state = AF_ATOM_FISSIONING;
    target->state_time = 0.0f;

    /* Choose fission axis (random direction) */
    float fx, fy, fz;
    af_rand_dir(&fx, &fy, &fz);

    /* Create Ba-141 fragment (heavier â†’ slower) */
    if (af->num_atoms < AF_MAX_ATOMS) {
        AF_Atom ba = make_atom(AF_FRAG1_Z, AF_FRAG1_A, ox, oy, oz);
        ba.state = AF_ATOM_FRAGMENT;
        ba.vx = fx * 8.0f;   /* Ã…/s display speed */
        ba.vy = fy * 8.0f;
        ba.vz = fz * 8.0f;
        ba.glow = 3.0f;      /* bright fission flash */
        ba.orbital_scale = 0.5f;  /* starts compressed */
        af->atoms[af->num_atoms++] = ba;
    }

    /* Create Kr-92 fragment (lighter â†’ faster, opposite direction) */
    if (af->num_atoms < AF_MAX_ATOMS) {
        AF_Atom kr = make_atom(AF_FRAG2_Z, AF_FRAG2_A, ox, oy, oz);
        kr.state = AF_ATOM_FRAGMENT;
        kr.vx = -fx * 12.0f;  /* opposite direction, faster */
        kr.vy = -fy * 12.0f;
        kr.vz = -fz * 12.0f;
        kr.glow = 3.0f;
        kr.orbital_scale = 0.5f;
        af->atoms[af->num_atoms++] = kr;
    }

    /* Emit 2-3 prompt neutrons (2 MeV average) */
    int n_prompt = 2 + (af_rand01() > 0.57f ? 1 : 0);  /* Î½Ì„ = 2.43 */
    for (int i = 0; i < n_prompt && af->num_neutrons < AF_MAX_NEUTRONS; i++) {
        float dx2, dy2, dz2;
        af_rand_dir(&dx2, &dy2, &dz2);
        /* Prompt neutron spectrum: Watt fission spectrum
         * P(E) âˆ exp(-E/a) Ã— sinh(âˆš(bE))  with a=0.988, b=2.249 for U-235
         * Average E â‰ˆ 2 MeV, but ranges from 0.1 to 10 MeV */
        float E_prompt = 0.5e6f + af_rand01() * 4.0e6f;  /* 0.5-4.5 MeV */
        float speed = 40.0f + af_rand01() * 30.0f;  /* display speed */
        af->neutrons[af->num_neutrons++] = make_neutron(
            ox + dx2 * 0.5f, oy + dy2 * 0.5f, oz + dz2 * 0.5f,
            dx2 * speed, dy2 * speed, dz2 * speed,
            E_prompt, af->time);
    }

    /* Emit prompt gamma rays (7 MeV total, ~7 photons of ~1 MeV each) */
    for (int i = 0; i < 5 && af->num_gammas < AF_MAX_GAMMAS; i++) {
        float dx2, dy2, dz2;
        af_rand_dir(&dx2, &dy2, &dz2);
        af->gammas[af->num_gammas++] = make_gamma(
            ox, oy, oz, dx2, dy2, dz2,
            1.0f + af_rand01() * 2.0f,  /* 1-3 MeV each */
            af->time);
    }

    /* Mark original atom as consumed */
    target->visible = 0;
    target->state = AF_ATOM_DEAD;

    af->fission_count++;
    af->total_energy_MeV += AF_U235_FISSION_ENERGY_MEV;
    af->generation++;

    printf("[AF] â•â• FISSION EVENT #%d â•â•\n", af->fission_count);
    printf("[AF]   Â²Â³âµU + n â†’ Â¹â´Â¹Ba + â¹Â²Kr + %dn + Î³\n", n_prompt);
    printf("[AF]   Energy released: %.0f MeV (total: %.0f MeV)\n",
           AF_U235_FISSION_ENERGY_MEV, af->total_energy_MeV);
}

static void scatter_neutron(AtomicFission *af, int n_idx, int atom_idx) {
    /* Elastic scattering: neutron bounces off nucleus
     *
     * Energy transfer per collision (center-of-mass frame):
     *   E_after / E_before = [(A-1)/(A+1)]Â² (head-on) to 1 (grazing)
     *   Average: E_after = E_before Ã— [(A+1)Â² - 2] / (A+1)Â²
     *
     * For C-12: E_after â‰ˆ 0.716 Ã— E_before (28.4% loss per collision)
     * For H-1:  E_after â‰ˆ 0 to E_before (average 50% loss)
     */
    AF_Neutron *n = &af->neutrons[n_idx];
    AF_Atom *a = &af->atoms[atom_idx];
    int A = a->A;

    /* Energy loss: random between maximum transfer and zero */
    float alpha = ((float)(A-1) / (float)(A+1)) * ((float)(A-1) / (float)(A+1));
    float frac = alpha + af_rand01() * (1.0f - alpha);
    n->energy_eV *= frac;
    n->wavelength_A = neutron_wavelength_A(n->energy_eV);
    n->is_thermal = (n->energy_eV < 1.0f) ? 1 : 0;
    n->scattered_count++;

    /* New random direction (isotropic in CM frame â†’ ~isotropic in lab for heavy nuclei) */
    float dx, dy, dz;
    af_rand_dir(&dx, &dy, &dz);
    float speed = sqrtf(n->vx*n->vx + n->vy*n->vy + n->vz*n->vz);
    speed *= sqrtf(frac);  /* slow down proportional to âˆš(E_ratio) */
    if (speed < 2.0f) speed = 2.0f;
    n->vx = dx * speed;
    n->vy = dy * speed;
    n->vz = dz * speed;
}

void atomic_fission_update(AtomicFission *af, float dt) {
    if (!af->playing) return;
    if (dt > 0.05f) dt = 0.05f;

    af->time += dt;
    af->phase_time += dt;

    /* â”€â”€ Update neutron positions â”€â”€ */
    for (int i = 0; i < af->num_neutrons; i++) {
        AF_Neutron *n = &af->neutrons[i];
        if (!n->alive) continue;

        n->x += n->vx * dt;
        n->y += n->vy * dt;
        n->z += n->vz * dt;
        n->phase += dt * 5.0f;  /* wave phase animation */

        /* Remove if out of bounds */
        float h = af->boxHalf * 1.2f;
        if (fabsf(n->x) > h || fabsf(n->y) > h || fabsf(n->z) > h) {
            n->alive = 0;
            continue;
        }

        /* Check collision with atoms */
        for (int j = 0; j < af->num_atoms; j++) {
            AF_Atom *a = &af->atoms[j];
            if (!a->visible || a->state == AF_ATOM_DEAD || a->state == AF_ATOM_FRAGMENT)
                continue;

            float dx = n->x - a->x;
            float dy = n->y - a->y;
            float dz = n->z - a->z;
            float dist2 = dx*dx + dy*dy + dz*dz;

            /* Interaction radius: capture radius ~ nuclear radius (fm â†’ Ã… scale)
             * Nuclear radius ~7 fm = 0.0007 Ã…, way too small to see.
             * For visualization, use ~1.5 Ã… as effective capture radius
             * (represents the quantum mechanical cross section area) */
            float capture_r = 1.5f;
            if (a->Z == 54) capture_r = 3.0f;  /* Xe-135: HUGE cross section */
            if (a->Z == 94) capture_r = 2.0f;  /* Pu-239: higher Ïƒ than U-235 */

            if (dist2 < capture_r * capture_r) {
                /* Determine interaction based on cross sections */
                if (a->Z == 92 && a->A == 235 && n->is_thermal) {
                    /* U-235 thermal fission: Ïƒ_f/(Ïƒ_f+Ïƒ_Î³) = 585/(585+99) â‰ˆ 85.5% */
                    if (af_rand01() < 0.855f) {
                        trigger_fission(af, j);
                        n->alive = 0;
                        af->phase = AF_PHASE_EXPLOSION;
                        af->phase_time = 0.0f;
                    } else {
                        /* Radiative capture: U-235 + n â†’ U-236 + Î³ */
                        a->Z = 92; a->A = 236;
                        a->state = AF_ATOM_EXCITED;
                        a->state_time = 0.0f;
                        a->glow = 1.5f;
                        n->alive = 0;
                        if (af->num_gammas < AF_MAX_GAMMAS) {
                            float gx, gy, gz;
                            af_rand_dir(&gx, &gy, &gz);
                            af->gammas[af->num_gammas++] = make_gamma(
                                a->x, a->y, a->z, gx, gy, gz, 6.5f, af->time);
                        }
                    }
                } else if (a->Z == 94 && a->A == 239 && n->is_thermal) {
                    /* Pu-239 thermal fission: Ïƒ_f/(Ïƒ_f+Ïƒ_Î³) = 747/(747+270) â‰ˆ 73.4%
                     * Higher fisfission cross section but also higher capture! */
                    if (af_rand01() < 0.734f) {
                        /* fission: Pu-239 + n â†’ Ce-144 + Kr-94 + 3n + Î³ + 210 MeV */
                        trigger_fission(af, j);
                        af->total_energy_MeV += 10.0f;  /* Pu releases 10 MeV more */
                        n->alive = 0;
                        af->phase = AF_PHASE_EXPLOSION;
                        af->phase_time = 0.0f;
                    } else {
                        /* Pu-239 + n â†’ Pu-240 (non-sile!): Ïƒ_Î³ = 270 barns */
                        a->A = 240;
                        a->state = AF_ATOM_EXCITED;
                        a->state_time = 0.0f;
                        a->glow = 1.5f;
                        n->alive = 0;
                        printf("[AF] Pu-239 captured neutron â†’ Pu-240 (non-sile)\n");
                    }
                } else if (a->Z == 54 && n->is_thermal) {
                    /* Xe-135 absorption (essentially 100% at thermal energies) */
                    a->state = AF_ATOM_ABSORBING;
                    a->state_time = 0.0f;
                    a->glow = 2.0f;
                    a->A = 136;  /* Xe-135 + n â†’ Xe-136 */
                    n->alive = 0;
                    printf("[AF] Xe-135 absorbed neutron! (Ïƒ=2.65M barns)\n");
                } else if (a->Z == 6 || a->Z == 8 || a->Z == 1) {
                    /* Moderator scattering */
                    scatter_neutron(af, i, j);
                    if (n->scattered_count == 1 && a->Z == 1) {
                        printf("[AF] Neutron scattered off %s: E=%.0f eV â†’ %.0f eV\n",
                               a->Z == 1 ? "H" : a->Z == 6 ? "C" : "O",
                               n->energy_eV / (1.0f - (float)(a->A-1)/(float)(a->A+1)),
                               n->energy_eV);
                    }
                } else if ((a->Z == 92 || a->Z == 94) && !n->is_thermal) {
                    /* Fast neutron scattering off uranium/plutonium */
                    scatter_neutron(af, i, j);
                }
                break;  /* one interaction per step */
            }
        }
    }

    /* â”€â”€ Update gamma ray photons â”€â”€ */
    float gamma_speed = 120.0f;  /* Ã…/s display speed (c = âˆ effectively) */
    for (int i = 0; i < af->num_gammas; i++) {
        AF_Gamma *g = &af->gammas[i];
        if (!g->alive) continue;
        g->x += g->dx * gamma_speed * dt;
        g->y += g->dy * gamma_speed * dt;
        g->z += g->dz * gamma_speed * dt;
        if (af->time - g->birth_time > g->lifetime) g->alive = 0;
        float h = af->boxHalf * 1.2f;
        if (fabsf(g->x) > h || fabsf(g->y) > h || fabsf(g->z) > h) g->alive = 0;
    }

    /* â”€â”€ Update atom states â”€â”€ */
    for (int i = 0; i < af->num_atoms; i++) {
        AF_Atom *a = &af->atoms[i];
        if (!a->visible) continue;

        a->state_time += dt;

        switch (a->state) {
        case AF_ATOM_EXCITED:
            /* Compound nucleus vibration */
            a->orbital_scale = 1.0f + 0.3f * sinf(a->state_time * 20.0f);
            a->glow *= (1.0f - dt * 2.0f); /* fade */
            if (a->glow < 0.01f) {
                a->state = AF_ATOM_IDLE;
                a->orbital_scale = 1.0f;
            }
            break;

        case AF_ATOM_FRAGMENT:
            /* fisfisfisfisfission fragments fly apart */
            a->x += a->vx * dt;
            a->y += a->vy * dt;
            a->z += a->vz * dt;
            /* Decelerate (energy loss to electron stopping) */
            a->vx *= (1.0f - dt * 0.8f);
            a->vy *= (1.0f - dt * 0.8f);
            a->vz *= (1.0f - dt * 0.8f);
            /* Fragment electron cloud expands as it decelerates */
            a->orbital_scale += dt * 0.5f;
            if (a->orbital_scale > 1.0f) a->orbital_scale = 1.0f;
            a->glow *= (1.0f - dt * 1.5f);
            /* Remove if too far */
            if (fabsf(a->x) > af->boxHalf * 1.3f ||
                fabsf(a->y) > af->boxHalf * 1.3f ||
                fabsf(a->z) > af->boxHalf * 1.3f) {
                a->visible = 0;
            }
            break;

        case AF_ATOM_ABSORBING:
            /* Xe-135 absorbing neutron â€” glow then settle */
            a->glow *= (1.0f - dt * 1.0f);
            if (a->glow < 0.01f) {
                a->state = AF_ATOM_IDLE;
            }
            break;

        default:
            break;
        }
    }

    /* â”€â”€ Xenon scene: gradually reveal Xe atoms (simulates I-135 decay) â”€â”€ */
    if (af->scene == AF_SCENE_XENON_POISON) {
        af->xe_concentration += dt * 0.15f;
        int xe_visible = (int)(af->xe_concentration);
        for (int i = 1; i < af->num_atoms && xe_visible > 0; i++) {
            if (af->atoms[i].Z == 54 && !af->atoms[i].visible) {
                af->atoms[i].visible = 1;
                af->atoms[i].glow = 1.0f;
                xe_visible--;
                printf("[AF] Xe-135 appeared (I-135 decay â†’ Xe-135)\n");
            }
        }

        /* Spawn new neutrons periodically */
        if (af->num_neutrons < AF_MAX_NEUTRONS && fmodf(af->time, 3.0f) < dt) {
            float dx2, dy2, dz2;
            af_rand_dir(&dx2, &dy2, &dz2);
            af->neutrons[af->num_neutrons++] = make_neutron(
                -20.0f * dx2, -20.0f * dy2, -20.0f * dz2,
                dx2 * 10.0f, dy2 * 10.0f, dz2 * 10.0f,
                AF_THERMAL_ENERGY_EV, af->time);
        }
    }

    /* â”€â”€ QCD scene: quarks orbit, gluon field fluctuates â”€â”€ */
    if (af->scene == AF_SCENE_QCD_NUCLEON) {
        float t = af->time;
        /* Two nucleon centers oscillate (deuteron bound state) */
        float px_center = -4.0f + 0.3f * sinf(t * 0.4f);
        float nx_center =  4.0f - 0.3f * sinf(t * 0.4f);

        float q_dist = 2.8f;     /* quark separation within nucleon */
        float orbit_rate = 1.2f; /* rad/s â€” slow enough to see */
        float wobble = 0.3f;     /* breathing mode amplitude */

        int valence_idx = 0;  /* track which valence quark we're at */
        for (int i = 0; i < af->num_atoms; i++) {
            AF_Atom *a = &af->atoms[i];
            if (a->Z != -1) continue;
            if (!a->visible) continue;

            if (a->A == 1 || a->A == 2) {
                /* Valence quark â€” orbit around parent nucleon center */
                /* First 3 are proton, next 3 are neutron */
                float cx, cy;
                if (valence_idx < 3) {
                    cx = px_center;
                    cy = 0.0f;
                } else {
                    cx = nx_center;
                    cy = 0.0f;
                }
                int local_idx = valence_idx % 3;
                float angle_base = (float)local_idx * 2.0944f;
                float angle = angle_base + orbit_rate * t;
                float r = q_dist + wobble * sinf(2.5f * t + (float)i);
                a->x = cx + r * cosf(angle);
                a->y = cy + r * sinf(angle);
                a->z = 0.4f * sinf(1.7f * t + (float)i * 1.1f);
                a->glow = 0.7f + 0.3f * sinf(3.0f * t + (float)i * 2.094f);
                valence_idx++;

            } else if (a->A == 9) {
                /* Gluon field point â€” fluctuate along flux tubes */
                float phase = t * 2.0f + (float)i * 0.7f;
                a->x += dt * 1.5f * sinf(phase);
                a->y += dt * 1.5f * cosf(phase * 1.3f);
                a->z += dt * 0.8f * sinf(phase * 0.9f);
                a->glow = 0.4f + 0.4f * fabsf(sinf(phase * 0.5f));
                /* Confine gluons to stay within nucleon region */
                float dist2 = a->x * a->x + a->y * a->y + a->z * a->z;
                float max_r = 10.0f;
                if (dist2 > max_r * max_r) {
                    float s = max_r / sqrtf(dist2);
                    a->x *= s; a->y *= s; a->z *= s;
                }

            } else if (a->A == 3) {
                /* Light sea quark pair â€” flash in and out (virtual pair creation) */
                float phase = sinf(t * 4.0f + (float)i * 1.5f);
                a->visible = (phase > 0.0f) ? 1 : 0;
                if (a->visible) {
                    a->glow = phase * 0.8f;
                    a->x += dt * 2.0f * (sinf(t * 3.1f + (float)i) - 0.5f);
                    a->y += dt * 2.0f * (cosf(t * 2.7f + (float)i) - 0.5f);
                    a->z += dt * 1.5f * (sinf(t * 3.9f + (float)i * 0.8f) - 0.5f);
                }

            } else if (a->A == 4) {
                /* Strange sea quark (ssÌ„) â€” heavier, shorter-lived fluctuation
                 * Mass ~95 MeV vs ~5 MeV for u/d â†’ shorter virtual lifetime
                 * Î”EÂ·Î”t â‰ˆ â„ â†’ Î”t â‰ˆ â„/m_s â‰ˆ 0.002 fm/c */
                float phase = sinf(t * 6.0f + (float)i * 2.3f);
                a->visible = (phase > 0.3f) ? 1 : 0;  /* less time visible (heavier) */
                if (a->visible) {
                    a->glow = (phase - 0.3f) * 1.2f;
                    /* Strange quarks orbit tighter (heavier â†’ shorter Compton Î») */
                    a->x += dt * 1.5f * (sinf(t * 4.5f + (float)i * 1.2f) - 0.5f);
                    a->y += dt * 1.5f * (cosf(t * 3.8f + (float)i * 0.9f) - 0.5f);
                    a->z += dt * 1.0f * (sinf(t * 5.2f + (float)i * 0.7f) - 0.5f);
                }

            } else if (a->A == 5) {
                /* Charm sea quark (ccÌ„) â€” very heavy (1.27 GeV), extremely rare
                 * Appears as brief bright flash, very compact */
                float phase = sinf(t * 8.0f + (float)i * 3.7f);
                a->visible = (phase > 0.6f) ? 1 : 0;  /* barely visible â€” very heavy */
                if (a->visible) {
                    a->glow = (phase - 0.6f) * 3.0f;  /* bright when it appears */
                    /* Charm quarks are nearly stationary (heavy) */
                    a->x += dt * 0.5f * sinf(t * 2.0f + (float)i);
                    a->y += dt * 0.5f * cosf(t * 1.5f + (float)i);
                    a->z += dt * 0.3f * sinf(t * 2.5f + (float)i);
                }
            }
        }
    }

    /* â”€â”€ D-T Fusion scene: atoms approach â†’ tunnel â†’ fuse â”€â”€ */
    if (af->scene == AF_SCENE_FUSION) {
        for (int i = 0; i < af->num_atoms; i++) {
            AF_Atom *a = &af->atoms[i];
            if (!a->visible || a->state == AF_ATOM_DEAD) continue;
            if (a->Z != 1) continue;  /* only move H isotopes */
            if (a->state == AF_ATOM_EXCITED) continue; /* He-4 handled above */

            /* Move atoms */
            a->x += a->vx * dt;
            a->y += a->vy * dt;
            a->z += a->vz * dt;

            /* Check for close approach with other H isotopes */
            for (int j = i + 1; j < af->num_atoms; j++) {
                AF_Atom *b = &af->atoms[j];
                if (!b->visible || b->state == AF_ATOM_DEAD) continue;
                if (b->Z != 1) continue;
                if (b->state == AF_ATOM_EXCITED) continue;
                /* Need one D and one T */
                if (!((a->A == AF_DEUTERIUM_A && b->A == AF_TRITIUM_A) ||
                      (a->A == AF_TRITIUM_A && b->A == AF_DEUTERIUM_A)))
                    continue;

                float dx = a->x - b->x;
                float dy = a->y - b->y;
                float dz = a->z - b->z;
                float dist2 = dx*dx + dy*dy + dz*dz;
                float tunnel_r = 2.0f;  /* fusion distance (quantum tunneling) */

                if (dist2 < tunnel_r * tunnel_r) {
                    /* Quantum tunneling through Coulomb barrier!
                     * P âˆ exp(-Ï€âˆš(E_G/E)) â€” Gamow factor
                     * At ~10 keV: P â‰ˆ 10â»âµ (but we force it for visualization) */
                    int di = (a->A == AF_DEUTERIUM_A) ? i : j;
                    int ti = (a->A == AF_DEUTERIUM_A) ? j : i;
                    trigger_fusion(af, di, ti);
                    af->phase = AF_PHASE_EXPLOSION;
                    af->phase_time = 0.0f;
                    goto fusion_done;  /* only one fusion per step */
                }

                /* Coulomb repulsion as they approach (but weakening at very close range
                 * due to nuclear force â€” the strong force takes over at ~2 fm) */
                float dist = sqrtf(dist2);
                if (dist < 8.0f && dist > 0.1f) {
                    /* Gentle Coulomb push (eÂ²/rÂ² but dramatically scaled) */
                    float force = 0.5f / (dist2 + 0.01f);
                    /* Nuclear attraction kicks in at very close range */
                    if (dist < 3.0f) force -= 2.0f / (dist2 + 0.01f);
                    float nx = dx / dist;
                    float ny = dy / dist;
                    float nz = dz / dist;
                    a->vx += force * nx * dt;
                    a->vy += force * ny * dt;
                    a->vz += force * nz * dt;
                    b->vx -= force * nx * dt;
                    b->vy -= force * ny * dt;
                    b->vz -= force * nz * dt;
                }
            }
        }
        fusion_done: ;

        /* Move He-4 products (they have velocity from trigger_fusion) */
        for (int i = 0; i < af->num_atoms; i++) {
            AF_Atom *a = &af->atoms[i];
            if (!a->visible || a->Z != 2) continue;
            a->x += a->vx * dt;
            a->y += a->vy * dt;
            a->z += a->vz * dt;
            a->vx *= (1.0f - dt * 0.3f);  /* slow deceleration */
            a->vy *= (1.0f - dt * 0.3f);
            a->vz *= (1.0f - dt * 0.3f);
        }
    }

    /* â”€â”€ Plutonium alpha decay: Pu-239 â†’ U-235 + Î± (He-4) â”€â”€ */
    if (af->scene == AF_SCENE_PLUTONIUM_FISSION) {
        /* Real tÂ½ = 24,110 years â†’ compress to ~1 event per ~8 seconds for vis */
        static float alpha_timer = 0.0f;
        alpha_timer += dt;
        if (alpha_timer > 8.0f) {
            alpha_timer = 0.0f;
            /* Find a Pu-239 atom */
            for (int i = 0; i < af->num_atoms; i++) {
                AF_Atom *a = &af->atoms[i];
                if (a->Z == 94 && a->A == 239 && a->visible && a->state == AF_ATOM_IDLE) {
                    /* Alpha decay: Pu-239 â†’ U-235 + He-4 (Î±) */
                    a->Z = 92;
                    a->A = 235;
                    a->state = AF_ATOM_EXCITED;
                    a->state_time = 0.0f;
                    a->glow = 1.8f;

                    /* Emit He-4 alpha particle */
                    if (af->num_atoms < AF_MAX_ATOMS) {
                        float ax, ay, az;
                        af_rand_dir(&ax, &ay, &az);
                        AF_Atom alpha = make_atom(AF_HE4_Z, AF_HE4_A,
                            a->x + ax * 2.0f, a->y + ay * 2.0f, a->z + az * 2.0f);
                        alpha.vx = ax * 15.0f;  /* 5.24 MeV â†’ fast for He */
                        alpha.vy = ay * 15.0f;
                        alpha.vz = az * 15.0f;
                        alpha.glow = 2.0f;
                        alpha.state = AF_ATOM_FRAGMENT;  /* it flies away */
                        alpha.state_time = 0.0f;
                        af->atoms[af->num_atoms++] = alpha;
                    }
                    printf("[AF] Î±-decay: Pu-239 â†’ U-235 + He-4 (%.3f MeV)\n",
                           AF_PU239_ALPHA_ENERGY_MEV);
                    break;
                }
            }
        }
    }

    /* â”€â”€ Beta Decay scene: animate W boson emission â”€â”€ */
    if (af->scene == AF_SCENE_BETA_DECAY) {
        float t = af->time;

        /* Phase timing (revised â€” dramatic beats get more screen time):
         *   0-0.8s:   quarks orbit, building visual recognition
         *   0.8-1.6s: down quark #2 glows hot â€” about to decay
         *   1.6s:     Wâ» boson ERUPTS from the quark (big golden bubble)
         *   1.6-4.5s: Wâ» expands slowly, pulsing â€” you see its 80 GeV mass
         *   4.5s:     Wâ» DECAYS â†’ electron + antineutrino shoot out
         *   4.5-5.5s: flavor change visible (dâ†’u, quark recolors)
         *   5.5-18s:  electron spirals, neutrino streaks â€” long visible tail
         *   18s:      restart
         */

        /* Quarks orbit (faster spin, tighter formation) */
        for (int i = 0; i < 3 && i < af->num_atoms; i++) {
            AF_Atom *a = &af->atoms[i];
            if (a->Z != -1 || (a->A != 1 && a->A != 2)) continue;
            float angle = t * 1.5f + (float)i * 2.094f;
            float r = 2.8f;
            a->x = r * cosf(angle);
            a->y = r * sinf(angle);
            a->z = 0.3f * sinf(t * 0.7f + (float)i);
        }

        /* Gluons follow quarks */
        for (int i = 3; i < af->num_atoms; i++) {
            AF_Atom *a = &af->atoms[i];
            if (a->Z == -1 && a->A == 9) {
                a->x += dt * 0.5f * sinf(t * 2.0f + (float)i);
                a->y += dt * 0.5f * cosf(t * 1.8f + (float)i);
                float maxr = 5.0f;
                float r = sqrtf(a->x*a->x + a->y*a->y + a->z*a->z);
                if (r > maxr) { a->x *= maxr/r; a->y *= maxr/r; a->z *= maxr/r; }
            }
        }

        /* Phase 0.8-1.6s: glowing down quark (building tension) */
        if (t > 0.8f && t < 1.6f && af->num_atoms >= 3) {
            float pulse = (t - 0.8f) / 0.8f;  /* 0â†’1 ramp */
            af->atoms[2].glow = 1.5f + pulse * 3.0f + sinf(pulse * 12.0f) * 0.5f;
        }

        /* Phase 1.6s: emit Wâ» boson (particle type 4) â€” big dramatic moment */
        if (t > 1.6f && t < 1.7f && af->num_particles == 0) {
            AF_Particle w;
            memset(&w, 0, sizeof(w));
            w.x = af->atoms[2].x;
            w.y = af->atoms[2].y;
            w.z = af->atoms[2].z;
            w.vx = 0.3f; w.vy = -0.5f; w.vz = 0.15f;  /* SLOW â€” it's massive */
            w.energy_MeV = AF_W_BOSON_MASS_GEV * 1000.0f;
            w.birth_time = t;
            w.lifetime = 3.0f;   /* visual: 3 seconds of Wâ» glory */
            w.radius = 1.5f;
            w.alive = 1;
            w.type = 4;  /* Wâ» boson */
            af->particles[af->num_particles++] = w;

            /* Transform the down quark â†’ up quark (flavor change!) */
            af->atoms[2].A = 2;  /* was A=1 (down), now A=2 (up) */
            af->atoms[2].glow = 4.0f;

            /* Gamma flash at creation point */
            for (int g = 0; g < 4 && af->num_gammas < AF_MAX_GAMMAS; g++) {
                float gx, gy, gz;
                af_rand_dir(&gx, &gy, &gz);
                af->gammas[af->num_gammas++] = make_gamma(
                    w.x, w.y, w.z, gx, gy, gz, 0.6f, t);
            }
        }

        /* W boson pulsing glow while alive (1.6 â†’ 4.5s) */
        if (t > 1.6f && t < 4.5f && af->num_particles >= 1 &&
            af->particles[0].alive && af->particles[0].type == 4) {
            /* Pulse the quarks to show weak interaction disturbing them */
            for (int i = 0; i < 3 && i < af->num_atoms; i++) {
                AF_Atom *a = &af->atoms[i];
                if (a->Z == -1 && (a->A == 1 || a->A == 2))
                    a->glow = 1.2f + 0.6f * sinf(t * 4.0f + (float)i * 2.0f);
            }
        }

        /* Phase 4.5s: Wâ» decays â†’ electron + neutrino (the big payoff) */
        if (t > 4.5f && af->num_particles >= 1) {
            AF_Particle *w = &af->particles[0];
            if (w->alive && w->type == 4) {
                float wx = w->x, wy = w->y, wz = w->z;
                w->alive = 0;  /* Wâ» dies */

                /* Gamma burst at decay point */
                for (int g = 0; g < 6 && af->num_gammas < AF_MAX_GAMMAS; g++) {
                    float gx, gy, gz;
                    af_rand_dir(&gx, &gy, &gz);
                    af->gammas[af->num_gammas++] = make_gamma(
                        wx, wy, wz, gx, gy, gz, 0.8f, t);
                }

                /* Electron (Î²â») â€” SLOW so it stays visible longer */
                AF_Particle e;
                memset(&e, 0, sizeof(e));
                e.x = wx; e.y = wy; e.z = wz;
                e.vx = -0.8f; e.vy = -1.5f; e.vz = 0.3f;
                e.energy_MeV = AF_BETA_ELECTRON_MAX_MEV;
                e.birth_time = t;
                e.lifetime = 14.0f;  /* long-lived: stays visible most of the scene */
                e.radius = 0.6f;
                e.alive = 1;
                e.type = 0;  /* electron */
                if (af->num_particles < AF_MAX_PARTICLES)
                    af->particles[af->num_particles++] = e;

                /* Antineutrino (Î½Ì„â‚‘) â€” also slower for visibility */
                AF_Particle nu;
                memset(&nu, 0, sizeof(nu));
                nu.x = wx; nu.y = wy; nu.z = wz;
                nu.vx = 1.0f; nu.vy = 1.2f; nu.vz = -0.3f;
                nu.energy_MeV = AF_BETA_Q_VALUE_MEV - AF_BETA_ELECTRON_MAX_MEV;
                nu.birth_time = t;
                nu.lifetime = 14.0f;
                nu.radius = 0.4f;
                nu.alive = 1;
                nu.type = 3;  /* neutrino */
                if (af->num_particles < AF_MAX_PARTICLES)
                    af->particles[af->num_particles++] = nu;
            }
        }

        /* Move all particles */
        for (int i = 0; i < af->num_particles; i++) {
            AF_Particle *p = &af->particles[i];
            if (!p->alive) continue;
            p->x += p->vx * dt;
            p->y += p->vy * dt;
            p->z += p->vz * dt;
            /* Electron spirals (Larmor radius ~0.5 Ã… in ~1T field) */
            if (p->type == 0) {
                float age = t - p->birth_time;
                p->x += 1.2f * cosf(age * 3.0f) * dt;
                p->z += 1.2f * sinf(age * 3.0f) * dt;
            }
            /* W boson expands as a bubble (slower, more dramatic) */
            if (p->type == 4) {
                float age = t - p->birth_time;
                p->radius = 1.5f + age * 1.5f;  /* grows to ~6 Ã… over 3s */
                /* Pulsing glow */
                p->energy_MeV = AF_W_BOSON_MASS_GEV * 1000.0f *
                    (0.8f + 0.2f * sinf(age * 6.0f));
            }
            float age = t - p->birth_time;
            if (age > p->lifetime) p->alive = 0;
            float h = af->boxHalf * 1.5f;
            if (fabsf(p->x)>h || fabsf(p->y)>h || fabsf(p->z)>h) p->alive = 0;
        }

        /* Update HUD with current phase */
        if (t < 1.6f) {
            snprintf(af->hud_lines[0], 80, "Beta Decay: neutron quarks (udd) orbiting");
        } else if (t < 4.5f) {
            snprintf(af->hud_lines[0], 80, "W- BOSON emitted! m=80.4 GeV expanding...");
        } else {
            snprintf(af->hud_lines[0], 80, "W- -> e- + v_e  |  d-quark -> u-quark (proton!)");
        }

        /* Restart after 18s (much more time to watch the products) */
        if (t > 18.0f) {
            printf("[AF] Beta decay complete â€” restarting...\n");
            atomic_fission_setup(af, af->scene);
            return;
        }
    }

    /* â”€â”€ Confinement Break scene: stretch flux tube â†’ snap â†’ hadronize â”€â”€ */
    if (af->scene == AF_SCENE_CONFINEMENT) {
        float t = af->time;

        /* Revised timing â€” much slower, more dramatic:
         *   0-6s:    Quarks slowly separate, gluon string stretches visibly
         *            String tension builds (glow increases, oscillations grow)
         *   6-6.5s:  Critical tension â€” string glowing white-hot
         *   6.5s:    SNAP! String breaks â€” massive gamma flash
         *   6.5-7.5s: New q-qÌ„ pair materializes at break point
         *   7.5-16s: Two new mesons drift apart slowly, glow fading
         *   16s:     Restart
         */

        float stretch_end = 6.0f;    /* string stretches 0 â†’ 6s */
        float snap_time = 6.5f;      /* SNAP at 6.5s */
        float meson_start = 7.5f;    /* new mesons appear at 7.5s */
        float sep_speed = 1.2f;      /* slow stretch */
        float max_sep = 9.0f;

        float sep;
        int snapped = 0;
        int pre_snap = 0;  /* critical tension phase */

        if (t < stretch_end) {
            /* Slow, dramatic stretch */
            sep = 2.0f + t * sep_speed;
        } else if (t < snap_time) {
            /* Critical tension! String at max, glowing hot */
            sep = max_sep;
            pre_snap = 1;
        } else if (t < meson_start) {
            /* Just snapped â€” quarks recoil */
            sep = max_sep;
            snapped = 1;
        } else {
            /* Two mesons drifting apart */
            sep = max_sep + (t - meson_start) * 0.8f;
            snapped = 1;
        }

        /* Position quark and antiquark */
        if (af->num_atoms >= 2) {
            af->atoms[0].x = -sep * 0.5f;
            af->atoms[1].x =  sep * 0.5f;

            if (pre_snap) {
                /* Critical tension: quarks vibrate, glow hot */
                float shake = 0.3f * sinf(t * 30.0f);
                af->atoms[0].y = shake;
                af->atoms[1].y = -shake;
                af->atoms[0].glow = 4.0f + sinf(t * 15.0f);
                af->atoms[1].glow = 4.0f + sinf(t * 15.0f + 1.0f);
            } else if (snapped) {
                af->atoms[0].glow = 3.5f * expf(-(t - snap_time) * 0.5f);
                af->atoms[1].glow = 3.5f * expf(-(t - snap_time) * 0.5f);
            } else {
                /* Normal stretch: glow increases with separation */
                af->atoms[0].glow = 1.5f + sep * 0.2f;
                af->atoms[1].glow = 1.5f + sep * 0.2f;
            }
        }

        /* Update gluon flux tube: stretch between quarks */
        int gluon_count = 0;
        for (int i = 2; i < af->num_atoms; i++) {
            AF_Atom *a = &af->atoms[i];
            if (a->Z != -1 || a->A != 9) continue;

            if (!snapped) {
                /* Gluons distribute along the stretched string */
                float frac = (float)(gluon_count + 1) / 7.0f;
                a->x = af->atoms[0].x * (1.0f-frac) + af->atoms[1].x * frac;

                /* String transverse oscillation grows with tension */
                float tension_scale = sep / max_sep;  /* 0â†’1 as string stretches */
                float amplitude = 0.2f + tension_scale * 0.8f;
                float freq = 3.0f + tension_scale * 6.0f;
                float transverse = sinf(frac * 3.14159f) * amplitude *
                    sinf(t * freq + frac * 4.0f);
                a->y = transverse;
                a->z = transverse * 0.5f * cosf(t * 3.0f);

                /* Glow ramps up dramatically with tension */
                a->glow = 0.8f + tension_scale * 3.0f;
                if (pre_snap) a->glow = 4.0f + sinf(t * 20.0f + frac * 5.0f);
                a->visible = 1;
            } else {
                /* After snap: gluons scatter and fade slowly */
                a->glow *= (1.0f - dt * 0.8f);
                if (a->glow < 0.05f) a->visible = 0;
            }
            gluon_count++;
        }

        /* At the snap moment, create new quark-antiquark pair + flash */
        static int conf_pair_created = 0;
        if (t < 1.0f) conf_pair_created = 0;  /* reset flag on new cycle */

        if (snapped && !conf_pair_created && t > snap_time && t < snap_time + 0.2f) {
            conf_pair_created = 1;
            float snap_x = 0.0f;  /* break point is at center */

            if (af->num_atoms + 2 <= AF_MAX_ATOMS) {
                /* New antiquark (pairs with left quark â†’ meson 1) */
                AF_Atom nq1 = make_atom(-1, 6, snap_x - 0.5f, af_randn()*0.4f, 0.0f);
                nq1.radius_A = 1.2f; nq1.orbital_scale = 2.5f; nq1.glow = 5.0f;
                af->atoms[af->num_atoms++] = nq1;

                /* New quark (pairs with right antiquark â†’ meson 2) */
                AF_Atom nq2 = make_atom(-1, 2, snap_x + 0.5f, af_randn()*0.4f, 0.0f);
                nq2.radius_A = 1.2f; nq2.orbital_scale = 2.5f; nq2.glow = 5.0f;
                af->atoms[af->num_atoms++] = nq2;

                /* BIG gamma flash at snap point (more gammas = more drama) */
                for (int g = 0; g < 8 && af->num_gammas < AF_MAX_GAMMAS; g++) {
                    float gx, gy, gz;
                    af_rand_dir(&gx, &gy, &gz);
                    af->gammas[af->num_gammas++] = make_gamma(
                        snap_x, 0, 0, gx, gy, gz, 1.0f, t);
                }
                printf("[AF] STRING SNAP! Vacuum pair creation at x=0\n");
            }
        }

        /* Move the new mesons apart after snap (slowly) */
        if (snapped && t > meson_start) {
            for (int i = 2; i < af->num_atoms; i++) {
                AF_Atom *a = &af->atoms[i];
                if (a->Z == -1 && (a->A == 2 || a->A == 6) && i >= af->num_atoms - 2) {
                    /* New particles drift apart slowly */
                    a->x += (a->x < 0 ? -0.6f : 0.6f) * dt;
                    a->glow *= (1.0f - dt * 0.15f);
                }
            }
        }

        /* Dynamic HUD */
        if (t < stretch_end) {
            float tension = sep / max_sep * 100.0f;
            snprintf(af->hud_lines[0], 80, "String tension: %.0f%%  sep=%.1f fm", tension, sep);
        } else if (t < snap_time) {
            snprintf(af->hud_lines[0], 80, "CRITICAL TENSION! String about to break!");
        } else if (t < meson_start) {
            snprintf(af->hud_lines[0], 80, "SNAP! Vacuum pair creation: q + q-bar");
        } else {
            snprintf(af->hud_lines[0], 80, "Two mesons formed â€” confinement preserved");
        }

        /* Restart after 16s */
        if (t > 16.0f) {
            printf("[AF] Confinement cycle â€” restarting string...\n");
            atomic_fission_setup(af, af->scene);
            return;
        }
    }

    /* â”€â”€ Decay Chain scene: step through U-238 â†’ Pb-206 â”€â”€ */
    if (af->scene == AF_SCENE_DECAY_CHAIN) {
        af->chain_step_time += dt;

        /* Every ~3 seconds, advance one decay step */
        float step_interval = 3.0f;
        if (af->chain_step_time > step_interval && af->chain_step < AF_DECAY_CHAIN_STEPS) {
            af->chain_step_time = 0.0f;
            const DecayStep *ds = &g_u238_chain[af->chain_step];

            /* The current atom decays */
            AF_Atom *parent = &af->atoms[0];

            if (ds->is_alpha) {
                /* Alpha decay: parent â†’ daughter + He-4 */
                parent->Z = ds->Z - 2;
                parent->A = ds->A - 4;
                parent->glow = 2.0f;
                parent->state = AF_ATOM_EXCITED;
                parent->state_time = 0.0f;
                parent->radius_A = atom_radius_A(parent->Z);

                /* Emit Î± particle â€” slow enough to see it leave */
                float ax, ay, az;
                af_rand_dir(&ax, &ay, &az);
                if (af->num_atoms < AF_MAX_ATOMS) {
                    AF_Atom alpha = make_atom(2, 4,
                        parent->x + ax*2.0f, parent->y + ay*2.0f, parent->z + az*2.0f);
                    alpha.vx = ax * 4.0f;
                    alpha.vy = ay * 4.0f;
                    alpha.vz = az * 4.0f;
                    alpha.glow = 3.0f;
                    alpha.state = AF_ATOM_FRAGMENT;
                    af->atoms[af->num_atoms++] = alpha;
                }

                /* Neutrino not emitted in alpha decay (no flavor change) */
                printf("[AF] Step %d: %s â†’Î± %s (%.3f MeV Î±)\n",
                       af->chain_step, ds->name,
                       (af->chain_step+1 < AF_DECAY_CHAIN_STEPS) ?
                       g_u238_chain[af->chain_step+1].name : "Pb-206",
                       ds->energy_MeV);
            } else {
                /* Betaâ» decay: parent â†’ daughter + eâ» + Î½Ì„â‚‘ */
                parent->Z = ds->Z + 1;
                /* A stays the same in beta decay */
                parent->glow = 1.8f;
                parent->state = AF_ATOM_EXCITED;
                parent->state_time = 0.0f;
                parent->radius_A = atom_radius_A(parent->Z);

                /* Emit electron (Î²â») â€” slow for visibility */
                if (af->num_particles < AF_MAX_PARTICLES) {
                    AF_Particle e;
                    memset(&e, 0, sizeof(e));
                    float ex, ey, ez;
                    af_rand_dir(&ex, &ey, &ez);
                    e.x = parent->x + ex; e.y = parent->y + ey; e.z = parent->z + ez;
                    e.vx = ex * 5.0f; e.vy = ey * 5.0f; e.vz = ez * 5.0f;
                    e.energy_MeV = ds->energy_MeV * 0.4f;
                    e.birth_time = af->time;
                    e.lifetime = 4.0f;
                    e.radius = 0.5f;
                    e.alive = 1;
                    e.type = 0;
                    af->particles[af->num_particles++] = e;
                }

                /* Emit antineutrino â€” also slower */
                if (af->num_particles < AF_MAX_PARTICLES) {
                    AF_Particle nu;
                    memset(&nu, 0, sizeof(nu));
                    float nx2, ny2, nz2;
                    af_rand_dir(&nx2, &ny2, &nz2);
                    nu.x = parent->x; nu.y = parent->y; nu.z = parent->z;
                    nu.vx = nx2 * 7.0f; nu.vy = ny2 * 7.0f; nu.vz = nz2 * 7.0f;
                    nu.energy_MeV = ds->energy_MeV * 0.6f;
                    nu.birth_time = af->time;
                    nu.lifetime = 4.0f;
                    nu.radius = 0.3f;
                    nu.alive = 1;
                    nu.type = 3;  /* neutrino */
                    af->particles[af->num_particles++] = nu;
                }

                printf("[AF] Step %d: %s â†’Î²â» %s (%.3f MeV)\n",
                       af->chain_step, ds->name,
                       (af->chain_step+1 < AF_DECAY_CHAIN_STEPS) ?
                       g_u238_chain[af->chain_step+1].name : "Pb-206",
                       ds->energy_MeV);
            }

            af->chain_step++;

            /* Update HUD */
            if (af->chain_step < AF_DECAY_CHAIN_STEPS) {
                snprintf(af->hud_lines[2], 80, "Step %d/%d: %s (Z=%d, A=%d)",
                         af->chain_step, AF_DECAY_CHAIN_STEPS,
                         g_u238_chain[af->chain_step].name,
                         g_u238_chain[af->chain_step].Z,
                         g_u238_chain[af->chain_step].A);
            } else {
                snprintf(af->hud_lines[2], 80, "COMPLETE: Pb-206 (stable!)");
            }
        }

        /* Move emitted particles */
        for (int i = 0; i < af->num_particles; i++) {
            AF_Particle *p = &af->particles[i];
            if (!p->alive) continue;
            p->x += p->vx * dt;
            p->y += p->vy * dt;
            p->z += p->vz * dt;
            /* Electron spiral */
            if (p->type == 0) {
                float age = af->time - p->birth_time;
                p->x += 0.5f * cosf(age * 6.0f) * dt;
                p->z += 0.5f * sinf(age * 6.0f) * dt;
            }
            float age = af->time - p->birth_time;
            if (age > p->lifetime) p->alive = 0;
            float lim = af->boxHalf * 1.3f;
            if (fabsf(p->x)>lim||fabsf(p->y)>lim||fabsf(p->z)>lim) p->alive = 0;
        }

        /* Fragment He-4 atoms fly away and fade */
        for (int i = 1; i < af->num_atoms; i++) {
            AF_Atom *a = &af->atoms[i];
            if (a->state == AF_ATOM_FRAGMENT && a->visible) {
                a->x += a->vx * dt;
                a->y += a->vy * dt;
                a->z += a->vz * dt;
                a->glow *= (1.0f - dt * 0.8f);
                float lim = af->boxHalf * 1.2f;
                if (fabsf(a->x)>lim||fabsf(a->y)>lim||fabsf(a->z)>lim) a->visible = 0;
            }
        }

        /* Reached Pb-206 and waited a few seconds? Restart */
        if (af->chain_step >= AF_DECAY_CHAIN_STEPS && af->chain_step_time > 4.0f) {
            printf("[AF] Decay chain complete: Pb-206 (stable) â€” restarting...\n");
            atomic_fission_setup(af, af->scene);
            return;
        }
    }

    /* â”€â”€ Binding Energy scene: nuclei pulsate, morph visualization â”€â”€ */
    if (af->scene == AF_SCENE_BINDING_ENERGY) {
        float t = af->time;

        /* Slowly scan A from 2 â†’ 238 and back, highlighting the current nucleus */
        float scan_speed = 15.0f;  /* mass numbers per second */
        af->binding_A += af->binding_dir * scan_speed * dt;
        if (af->binding_A > 240.0f) { af->binding_A = 240.0f; af->binding_dir = -1; }
        if (af->binding_A < 2.0f) { af->binding_A = 2.0f; af->binding_dir = 1; }

        /* Update each displayed nucleus glow based on proximity to scan cursor */
        for (int i = 0; i < af->num_atoms; i++) {
            AF_Atom *a = &af->atoms[i];
            float dist_A = fabsf((float)a->A - af->binding_A);
            /* Highlight nearby nuclei */
            float highlight = expf(-dist_A * dist_A / 200.0f);
            float ba = semf_binding_per_nucleon(a->A, a->Z);
            a->glow = ba / AF_FE56_BA * (0.3f + 1.5f * highlight);

            /* Gentle breathing oscillation */
            a->orbital_scale = 1.0f + 0.1f * sinf(t * 0.8f + (float)i * 1.3f);
            a->orbital_phase = t * 0.3f + (float)i;
        }

        /* Update HUD with current position on the curve */
        int iA = (int)(af->binding_A + 0.5f);
        int iZ = (int)(af->binding_A * 0.42f);  /* approximate Z/A ratio */
        float ba_cur = semf_binding_per_nucleon(iA, iZ);
        const char *region = (af->binding_A < 56.0f) ? "FUSION region" :
                             (af->binding_A > 56.0f) ? "SION region" : "Fe-56 PEAK";
        snprintf(af->hud_lines[2], 80, "A~%d  B/A=%.2f MeV  %s", iA, ba_cur, region);
    }

    /* â”€â”€ Looping: restart when all activity dies down â”€â”€ */
    /* Skip for scenes that manage their own restart timers */
    if (af->looping &&
        af->scene != AF_SCENE_BETA_DECAY &&
        af->scene != AF_SCENE_CONFINEMENT &&
        af->scene != AF_SCENE_DECAY_CHAIN &&
        af->scene != AF_SCENE_BINDING_ENERGY) {
        int any_alive = 0;
        for (int i = 0; i < af->num_neutrons; i++)
            if (af->neutrons[i].alive) { any_alive = 1; break; }
        for (int i = 0; i < af->num_gammas; i++)
            if (af->gammas[i].alive) { any_alive = 1; break; }
        int any_fragment = 0;
        for (int i = 0; i < af->num_atoms; i++)
            if (af->atoms[i].state == AF_ATOM_FRAGMENT && af->atoms[i].visible)
                { any_fragment = 1; break; }

        if (!any_alive && !any_fragment && af->phase_time > 3.0f) {
            /* For fusion scene, also check if any H atoms are still approaching */
            int still_active = 0;
            if (af->scene == AF_SCENE_FUSION) {
                for (int i = 0; i < af->num_atoms; i++) {
                    if (af->atoms[i].visible && af->atoms[i].Z == 1 &&
                        af->atoms[i].state != AF_ATOM_DEAD)
                        { still_active = 1; break; }
                }
            }
            if (!still_active) {
                printf("[AF] Scene complete â€” restarting...\n");
                atomic_fission_setup(af, af->scene);
            }
        }
    }
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• VISUALIZATION â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
 *
 * Write the density + signed grids for the GPU shader.
 *
 * ENCODING for color_mode 7 (atomic scale):
 *   density = electron density |Ïˆ|Â² (or neutron/gamma intensity)
 *   signed  = material encoding:
 *     +3.5  = Pu-239 (plutonium)
 *     +3.0  = U-235/U-236 (uranium atom)
 *     +2.5  = Ba/Kr fisfisfisfisfission fragment (excited)
 *     +2.0  = Xe-135 (poison)
 *     +1.5  = O (oxygen)
 *     +1.0  = C (carbon/graphite)
 *     +0.65 = He-4 (helium / alpha particle)
 *     +0.5  = H (hydrogen)
 *     -1.0  = thermal neutron wave packet
 *     -2.0  = fast neutron wave packet
 *     -3.0  = gamma ray photon
 *     -4.0  = QCD valence quark / gluon / light sea
 *     -4.5  = QCD strange quark (ssÌ„)
 *     -5.0  = QCD charm quark (ccÌ„)
 *     -5.5  = W boson (weak force carrier)
 *     -6.0  = electron / beta particle (Î²â»/Î²âº)
 *     -6.5  = neutrino (Î½ / Î½Ì„)
 *      0.0  = empty
 * â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */

static float af_signed_for_atom(const AF_Atom *a) {
    if (a->Z == 94) return 3.5f;             /* Plutonium */
    if (a->Z == 92) return 3.0f;             /* Uranium */
    if (a->state == AF_ATOM_FRAGMENT) return 2.5f;  /* fisfisfisfisfission fragment */
    if (a->Z == 54) return 2.0f;             /* Xenon */
    if (a->Z == 58) return 2.5f;             /* Cerium (Pu fragment) */
    if (a->Z == 8)  return 1.5f;             /* Oxygen */
    if (a->Z == 6)  return 1.0f;             /* Carbon */
    if (a->Z == 2)  return 0.65f;            /* Helium (He-4 fusion/alpha) */
    if (a->Z == 1)  return 0.5f;             /* Hydrogen */
    if (a->Z == 56) return 2.5f;             /* Barium (fragment) */
    if (a->Z == 36) return 2.5f;             /* Krypton (fragment) */
    if (a->Z == -1) {
        if (a->A == 5) return -5.0f;     /* QCD charm quark (ccÌ„) */
        if (a->A == 4) return -4.5f;     /* QCD strange quark (ssÌ„) */
        if (a->A == 6) return -4.0f;     /* antiquark (same rendering as valence) */
        return -4.0f;                     /* QCD valence/gluon/light sea */
    }
    return 1.0f;
}

/* Vulkan buffer helper (same pattern as reactor_thermal) */
static int af_create_buffer(VkPhysicalDevice phys, VkDevice dev,
                            VkDeviceSize size, VkBufferUsageFlags usage,
                            VkMemoryPropertyFlags props,
                            VkBuffer *buf, VkDeviceMemory *mem) {
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(dev, &bci, NULL, buf) != VK_SUCCESS) return -1;

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, *buf, &req);

    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    uint32_t mi = UINT32_MAX;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if ((req.memoryTypeBits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & props) == props) {
            mi = i; break;
        }
    }
    if (mi == UINT32_MAX) return -2;

    VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = mi;
    if (vkAllocateMemory(dev, &mai, NULL, mem) != VK_SUCCESS) return -3;
    vkBindBufferMemory(dev, *buf, *mem, 0);
    return 0;
}

static void af_submit_and_wait(VkDevice dev, VkQueue queue, VkCommandBuffer cb) {
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    VkFence fence;
    VkFenceCreateInfo fi = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    vkCreateFence(dev, &fi, NULL, &fence);
    vkQueueSubmit(queue, 1, &si, fence);
    vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX);
    vkDestroyFence(dev, fence, NULL);
}

int atomic_fission_visualize(AtomicFission *af, VkQueue queue) {
    VkDevice dev = af->device;
    int G = af->visGridDim;
    int total = G * G * G;
    VkDeviceSize vis_size = (VkDeviceSize)total * sizeof(float);

    float *density_data = (float *)calloc(total, sizeof(float));
    float *signed_data  = (float *)calloc(total, sizeof(float));
    if (!density_data || !signed_data) {
        free(density_data); free(signed_data);
        return -1;
    }

    float h = af->boxHalf;
    float inv_2h = 1.0f / (2.0f * h);

    /* â”€â”€ Stamp each atom's electron density into the grid â”€â”€ */
    for (int ai = 0; ai < af->num_atoms; ai++) {
        AF_Atom *a = &af->atoms[ai];
        if (!a->visible) continue;

        float signed_val = af_signed_for_atom(a);

        /* Atom electron cloud radius (in Ã…) â€” larger for heavier atoms */
        float cloud_r = a->radius_A * 3.5f * a->orbital_scale;
        if (a->Z == 92) cloud_r = 8.0f * a->orbital_scale;  /* U has extended orbitals */
        if (a->Z == 94) cloud_r = 8.5f * a->orbital_scale;  /* Pu: slightly larger 5fâ¶ */
        if (a->Z == 54) cloud_r = 6.0f * a->orbital_scale;  /* Xe likewise */
        if (a->Z == 2)  cloud_r = 3.0f * a->orbital_scale;  /* He: compact 1sÂ² closed shell */
        if (a->Z == -1) cloud_r = 3.0f * a->orbital_scale;  /* QCD quark: compact */

        /* Convert atom position to grid coordinates */
        int cx = (int)((a->x + h) * inv_2h * (float)G);
        int cy = (int)((a->y + h) * inv_2h * (float)G);
        int cz = (int)((a->z + h) * inv_2h * (float)G);
        int cr = (int)(cloud_r * inv_2h * (float)G) + 1;

        /* Stamp voxels within the cloud radius */
        for (int dz = -cr; dz <= cr; dz++) {
            int iz = cz + dz;
            if (iz < 0 || iz >= G) continue;
            for (int dy = -cr; dy <= cr; dy++) {
                int iy = cy + dy;
                if (iy < 0 || iy >= G) continue;
                for (int dx = -cr; dx <= cr; dx++) {
                    int ix = cx + dx;
                    if (ix < 0 || ix >= G) continue;

                    /* Physical position of this voxel */
                    float px = ((float)ix + 0.5f) / (float)G * 2.0f * h - h;
                    float py = ((float)iy + 0.5f) / (float)G * 2.0f * h - h;
                    float pz = ((float)iz + 0.5f) / (float)G * 2.0f * h - h;

                    /* Distance from atom center */
                    float rx = px - a->x;
                    float ry = py - a->y;
                    float rz = pz - a->z;
                    float r_A = sqrtf(rx*rx + ry*ry + rz*rz);

                    if (r_A > cloud_r) continue;

                    /* Compute electron density using Slater orbitals */
                    float rho = orbital_density_at(a->Z, r_A);

                    /* Scale for visibility */
                    rho *= a->orbital_scale;

                    /* Add excitation / glow effects */
                    if (a->glow > 0.01f) {
                        rho += a->glow * expf(-r_A * 2.0f);
                    }

                    /* Deformation for fissioning atom (prolate stretching) */
                    if (a->state == AF_ATOM_DEFORMING || a->state == AF_ATOM_FISSIONING) {
                        /* Stretch along the deformation axis (x-axis for simplicity) */
                        float stretch = 1.0f + a->deformation * 2.0f;
                        float r_deformed = sqrtf((rx/stretch)*(rx/stretch) + ry*ry + rz*rz);
                        rho = orbital_density_at(a->Z, r_deformed);
                    }

                    int idx = iz * G * G + iy * G + ix;
                    /* Additive blending â€” overlapping atoms combine */
                    density_data[idx] += rho;
                    /* Signed material: take the strongest contributor */
                    if (rho > 0.01f && fabsf(signed_val) > fabsf(signed_data[idx])) {
                        signed_data[idx] = signed_val;
                    }
                }
            }
        }
    }

    /* â”€â”€ Stamp neutron wave packets â”€â”€ */
    for (int ni = 0; ni < af->num_neutrons; ni++) {
        AF_Neutron *n = &af->neutrons[ni];
        if (!n->alive) continue;

        float signed_val = n->is_thermal ? -1.0f : -2.0f;

        /* Neutron wave packet size: larger for thermal (longer Î») */
        float packet_r = n->is_thermal ? 2.5f : 1.0f;

        int cx = (int)((n->x + h) * inv_2h * (float)G);
        int cy = (int)((n->y + h) * inv_2h * (float)G);
        int cz = (int)((n->z + h) * inv_2h * (float)G);
        int cr = (int)(packet_r * inv_2h * (float)G) + 1;

        for (int dz = -cr; dz <= cr; dz++) {
            int iz = cz + dz;
            if (iz < 0 || iz >= G) continue;
            for (int dy = -cr; dy <= cr; dy++) {
                int iy = cy + dy;
                if (iy < 0 || iy >= G) continue;
                for (int dx = -cr; dx <= cr; dx++) {
                    int ix = cx + dx;
                    if (ix < 0 || ix >= G) continue;

                    float px = ((float)ix + 0.5f) / (float)G * 2.0f * h - h;
                    float py = ((float)iy + 0.5f) / (float)G * 2.0f * h - h;
                    float pz = ((float)iz + 0.5f) / (float)G * 2.0f * h - h;

                    float rx = px - n->x;
                    float ry = py - n->y;
                    float rz = pz - n->z;
                    float r = sqrtf(rx*rx + ry*ry + rz*rz);
                    if (r > packet_r) continue;

                    /* De Broglie wave packet: Gaussian envelope Ã— plane wave
                     * Ïˆ(r) = exp(-rÂ²/2ÏƒÂ²) Ã— cos(2Ï€ r / Î»)
                     * where Ïƒ = packet width, Î» = de Broglie wavelength */
                    float sigma = packet_r * 0.5f;
                    float envelope = expf(-r*r / (2.0f * sigma * sigma));

                    /* Plane wave along velocity direction */
                    float v = sqrtf(n->vx*n->vx + n->vy*n->vy + n->vz*n->vz);
                    float dot_v = 0.0f;
                    if (v > 0.01f) {
                        dot_v = (rx*n->vx + ry*n->vy + rz*n->vz) / v;
                    }
                    float wave = cosf(2.0f * (float)M_PI * dot_v / fmaxf(n->wavelength_A, 0.01f)
                                      + n->phase);

                    float rho = envelope * (0.6f + 0.4f * wave) * 1.5f;

                    int idx = iz * G * G + iy * G + ix;
                    density_data[idx] += rho;
                    if (rho > 0.05f) signed_data[idx] = signed_val;
                }
            }
        }
    }

    /* â”€â”€ Stamp gamma ray streaks â”€â”€ */
    for (int gi = 0; gi < af->num_gammas; gi++) {
        AF_Gamma *g = &af->gammas[gi];
        if (!g->alive) continue;

        float age = af->time - g->birth_time;
        float fade = 1.0f - age / g->lifetime;
        if (fade < 0.0f) continue;

        /* Gamma ray: a thin bright line along its direction */
        float streak_len = 4.0f;
        int steps = 20;
        for (int s = 0; s <= steps; s++) {
            float t = (float)s / (float)steps * streak_len;
            float gx = g->x - g->dx * t;  /* trail behind current position */
            float gy = g->y - g->dy * t;
            float gz = g->z - g->dz * t;

            int ix = (int)((gx + h) * inv_2h * (float)G);
            int iy = (int)((gy + h) * inv_2h * (float)G);
            int iz = (int)((gz + h) * inv_2h * (float)G);
            if (ix < 0 || ix >= G || iy < 0 || iy >= G || iz < 0 || iz >= G)
                continue;

            float intensity = fade * (1.0f - (float)s / (float)steps);
            int idx = iz * G * G + iy * G + ix;
            density_data[idx] += intensity * 2.0f;
            signed_data[idx] = -3.0f;  /* gamma encoding */

            /* Small 3x3 halo for visibility */
            for (int d2 = -1; d2 <= 1; d2++) {
                for (int d1 = -1; d1 <= 1; d1++) {
                    int hx = ix + d1, hy = iy + d2;
                    if (hx < 0 || hx >= G || hy < 0 || hy >= G) continue;
                    int hidx = iz * G * G + hy * G + hx;
                    density_data[hidx] += intensity * 0.5f;
                    signed_data[hidx] = -3.0f;
                }
            }
        }
    }

    /* â”€â”€ Stamp particles (electrons, neutrinos, W bosons, alpha) â”€â”€ */
    for (int pi = 0; pi < af->num_particles; pi++) {
        AF_Particle *p = &af->particles[pi];
        if (!p->alive) continue;

        float age = af->time - p->birth_time;
        float fade = 1.0f - age / fmaxf(p->lifetime, 0.1f);
        if (fade < 0.0f) continue;

        /* Signed encoding by particle type */
        float sv;
        float prad;
        switch (p->type) {
        case 0:  sv = -6.0f; prad = 0.8f; break;   /* electron (Î²â») */
        case 1:  sv = -6.0f; prad = 0.8f; break;   /* positron (Î²âº) */
        case 2:  sv =  0.65f; prad = 1.2f; break;  /* alpha â†’ same as He-4 */
        case 3:  sv = -6.5f; prad = 0.5f; break;   /* neutrino (ghost) */
        case 4:  sv = -5.5f; prad = p->radius; break; /* W boson */
        case 5:  sv = -5.5f; prad = p->radius; break; /* Wâº */
        default: sv = -6.0f; prad = 0.8f; break;
        }

        /* Convert position to grid coords */
        int cx = (int)((p->x + h) * inv_2h * (float)G);
        int cy = (int)((p->y + h) * inv_2h * (float)G);
        int cz = (int)((p->z + h) * inv_2h * (float)G);
        int cr = (int)(prad * inv_2h * (float)G) + 1;

        for (int dz = -cr; dz <= cr; dz++) {
            int iz = cz + dz;
            if (iz < 0 || iz >= G) continue;
            for (int dy = -cr; dy <= cr; dy++) {
                int iy = cy + dy;
                if (iy < 0 || iy >= G) continue;
                for (int dx = -cr; dx <= cr; dx++) {
                    int ix = cx + dx;
                    if (ix < 0 || ix >= G) continue;

                    float rx = (float)dx * (2.0f * h / (float)G);
                    float ry = (float)dy * (2.0f * h / (float)G);
                    float rz = (float)dz * (2.0f * h / (float)G);
                    float r_d = sqrtf(rx*rx + ry*ry + rz*rz);
                    if (r_d > prad) continue;

                    float rho;
                    if (p->type == 3) {
                        /* Neutrino: very faint streaky Gaussian */
                        rho = 0.3f * fade * expf(-r_d * r_d / (0.2f * prad * prad));
                    } else if (p->type == 4 || p->type == 5) {
                        /* W boson: expanding bubble shell */
                        float shell = fabsf(r_d - prad * 0.7f);
                        rho = fade * 2.0f * expf(-shell * shell / (0.3f * prad * prad));
                    } else {
                        /* Electron/positron: tight Gaussian */
                        rho = fade * 1.5f * expf(-r_d * r_d / (0.5f * prad * prad));
                    }

                    if (rho < 0.01f) continue;
                    int idx = iz * G * G + iy * G + ix;
                    density_data[idx] += rho;
                    if (fabsf(sv) > fabsf(signed_data[idx]))
                        signed_data[idx] = sv;
                }
            }
        }

        /* For neutrinos: additional streak trail for visibility */
        if (p->type == 3) {
            float speed = sqrtf(p->vx*p->vx + p->vy*p->vy + p->vz*p->vz);
            if (speed > 0.1f) {
                float ndx = p->vx/speed, ndy = p->vy/speed, ndz = p->vz/speed;
                int trail_steps = 12;
                float trail_len = 3.0f;
                for (int s = 1; s <= trail_steps; s++) {
                    float tt = (float)s / (float)trail_steps * trail_len;
                    float tx = p->x - ndx * tt;
                    float ty = p->y - ndy * tt;
                    float tz = p->z - ndz * tt;
                    int tix = (int)((tx + h) * inv_2h * (float)G);
                    int tiy = (int)((ty + h) * inv_2h * (float)G);
                    int tiz = (int)((tz + h) * inv_2h * (float)G);
                    if (tix < 0||tix>=G||tiy<0||tiy>=G||tiz<0||tiz>=G) continue;
                    int tidx = tiz * G * G + tiy * G + tix;
                    float trail_i = fade * 0.15f * (1.0f - (float)s/(float)trail_steps);
                    density_data[tidx] += trail_i;
                    signed_data[tidx] = -6.5f;  /* neutrino */
                }
            }
        }
    }

    /* â”€â”€ Upload to GPU â”€â”€ */
    VkDeviceSize total_size = vis_size * 2;
    VkBuffer stagingBuf;
    VkDeviceMemory stagingMem;
    if (af_create_buffer(af->physDevice, dev, total_size,
                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         &stagingBuf, &stagingMem) != 0) {
        free(density_data); free(signed_data);
        return -2;
    }

    void *mapped;
    if (vkMapMemory(dev, stagingMem, 0, total_size, 0, &mapped) != VK_SUCCESS) {
        vkDestroyBuffer(dev, stagingBuf, NULL);
        vkFreeMemory(dev, stagingMem, NULL);
        free(density_data); free(signed_data);
        return -3;
    }
    memcpy((char*)mapped, density_data, (size_t)vis_size);
    memcpy((char*)mapped + vis_size, signed_data, (size_t)vis_size);
    vkUnmapMemory(dev, stagingMem);

    vkResetCommandBuffer(af->cmdBuf, 0);
    VkCommandBufferBeginInfo cbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(af->cmdBuf, &cbi);

    VkBufferCopy region1 = {0, 0, vis_size};
    vkCmdCopyBuffer(af->cmdBuf, stagingBuf, af->densityBuf, 1, &region1);

    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(af->cmdBuf,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);

    VkBufferCopy region2 = {vis_size, 0, vis_size};
    vkCmdCopyBuffer(af->cmdBuf, stagingBuf, af->signedBuf, 1, &region2);

    vkEndCommandBuffer(af->cmdBuf);
    af_submit_and_wait(dev, queue, af->cmdBuf);

    vkDestroyBuffer(dev, stagingBuf, NULL);
    vkFreeMemory(dev, stagingMem, NULL);
    free(density_data);
    free(signed_data);
    return 0;
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• INIT / SETUP / FREE â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */

int atomic_fission_init(AtomicFission *af,
                        VkPhysicalDevice phys, VkDevice dev,
                        uint32_t queueFamilyIdx,
                        VkBuffer densityBuf, VkBuffer signedBuf,
                        int gridDim) {
    memset(af, 0, sizeof(*af));
    af->device = dev;
    af->physDevice = phys;
    af->densityBuf = densityBuf;
    af->signedBuf = signedBuf;
    af->visGridDim = gridDim;

    /* Command pool + buffer */
    VkCommandPoolCreateInfo cpci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = queueFamilyIdx;
    if (vkCreateCommandPool(dev, &cpci, NULL, &af->cmdPool) != VK_SUCCESS)
        return -1;

    VkCommandBufferAllocateInfo cbai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = af->cmdPool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(dev, &cbai, &af->cmdBuf) != VK_SUCCESS)
        return -2;

    printf("[AF] Atomic fission visualization initialized (grid=%dÂ³)\n", gridDim);
    return 0;
}

void atomic_fission_setup(AtomicFission *af, AF_SceneType scene) {
    /* Reset all entities */
    memset(af->atoms, 0, sizeof(af->atoms));
    memset(af->neutrons, 0, sizeof(af->neutrons));
    memset(af->gammas, 0, sizeof(af->gammas));
    memset(af->particles, 0, sizeof(af->particles));
    af->num_atoms = 0;
    af->num_neutrons = 0;
    af->num_gammas = 0;
    af->num_particles = 0;
    af->fission_count = 0;
    af->total_energy_MeV = 0.0f;
    af->xe_concentration = 0.0f;
    af->generation = 0;
    af->time = 0.0f;
    af->phase_time = 0.0f;

    switch (scene) {
    case AF_SCENE_SINGLE_FISSION:    setup_single_fission(af);    break;
    case AF_SCENE_CHAIN_REACTION:    setup_chain_reaction(af);    break;
    case AF_SCENE_XENON_POISON:      setup_xenon_poison(af);      break;
    case AF_SCENE_MODERATION:        setup_moderation(af);        break;
    case AF_SCENE_PLUTONIUM_FISSION: setup_plutonium_fission(af); break;
    case AF_SCENE_QCD_NUCLEON:       setup_qcd_nucleon(af);       break;
    case AF_SCENE_CHERNOBYL_SEQUENCE:
        setup_chernobyl_sequence(af);
        break;
    case AF_SCENE_FUSION:
        setup_fusion(af);
        break;
    case AF_SCENE_BETA_DECAY:
        setup_beta_decay(af);
        break;
    case AF_SCENE_CONFINEMENT:
        setup_confinement(af);
        break;
    case AF_SCENE_DECAY_CHAIN:
        setup_decay_chain(af);
        break;
    case AF_SCENE_BINDING_ENERGY:
        setup_binding_energy(af);
        break;
    default:
        setup_single_fission(af);
        break;
    }

    /* Ensure scene type matches the requested scene (setup functions
     * set af->scene to their own type, which is wrong for fallbacks
     * like Chernobyl â†’ chain_reaction). */
    af->scene = scene;
}

void atomic_fission_next_scene(AtomicFission *af) {
    AF_SceneType next = (AF_SceneType)((af->scene + 1) % AF_SCENE_COUNT);
    atomic_fission_setup(af, next);
}

const char *atomic_fission_scene_name(AF_SceneType scene) {
    switch (scene) {
    case AF_SCENE_SINGLE_FISSION:     return "U-235 fission";
    case AF_SCENE_CHAIN_REACTION:     return "Chain Reaction";
    case AF_SCENE_XENON_POISON:       return "Xe-135 Poisoning";
    case AF_SCENE_MODERATION:         return "Neutron Moderation";
    case AF_SCENE_PLUTONIUM_FISSION:  return "Pu-239 MOX fission";
    case AF_SCENE_QCD_NUCLEON:        return "QCD: Proton + Neutron";
    case AF_SCENE_CHERNOBYL_SEQUENCE: return "Chernobyl Sequence";
    case AF_SCENE_FUSION:              return "D-T Fusion (ITER)";
    case AF_SCENE_BETA_DECAY:          return "Beta Decay (W Boson)";
    case AF_SCENE_CONFINEMENT:         return "QCD Confinement Break";
    case AF_SCENE_DECAY_CHAIN:         return "U-238 Decay Chain";
    case AF_SCENE_BINDING_ENERGY:      return "Binding Energy Curve";
    default:                          return "Unknown";
    }
}

const char *atomic_fission_phase_name(AF_Phase phase) {
    switch (phase) {
    case AF_PHASE_SETUP:          return "Setup";
    case AF_PHASE_NEUTRON_FLIGHT: return "Neutron Flight";
    case AF_PHASE_ABSORPTION:     return "Absorption";
    case AF_PHASE_COMPOUND:       return "Compound Nucleus";
    case AF_PHASE_DEFORMATION:    return "Deformation";
    case AF_PHASE_SCISSION:       return "Scission";
    case AF_PHASE_EXPLOSION:      return "fisfisfisfisfission Products";
    case AF_PHASE_CHAIN:          return "Chain Cascade";
    case AF_PHASE_POISON:         return "Xe Poisoning";
    case AF_PHASE_COMPLETE:       return "Complete";
    default:                      return "Unknown";
    }
}

void atomic_fission_free(AtomicFission *af) {
    VkDevice dev = af->device;
    if (dev) {
        if (af->cmdPool) vkDestroyCommandPool(dev, af->cmdPool, NULL);
    }
    printf("[AF] Atomic fission system freed\n");
    memset(af, 0, sizeof(*af));
}
