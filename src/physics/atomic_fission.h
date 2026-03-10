/* atomic_fission.h — Atomic-scale nuclear fission visualization
 *
 * This module simulates nuclear fission, chain reactions, and Xe-135
 * poisoning at the ATOMIC SCALE with quantum-accurate electron orbitals.
 *
 * Unlike the macro-scale reactor_thermal module (14m × 14m RBMK core),
 * this operates at ~50 Å (5 nm) scale, showing individual atoms:
 *
 *   • U-235 with full [Rn]5f³6d¹7s² electron orbital wavefunctions
 *   • Incoming neutrons as de Broglie wave packets (λ_th ≈ 1.8 Å)
 *   • U-236* compound nucleus excitation + liquid-drop deformation
 *   • Fission: Ba-141 + Kr-92 + 3n + 200 MeV (or other products)
 *   • Prompt gamma-ray photons (visualized as directed light pulses)
 *   • Xe-135 atoms with 5p⁶ electron clouds absorbing neutrons
 *   • Graphite moderator C atoms scattering/slowing neutrons
 *   • H₂O molecules showing Cherenkov radiation
 *   • Delayed neutron emission from fission products
 *
 * Physics references:
 *   - Nuclear Data: ENDF/B-VIII.0  (cross sections)
 *   - Wavefunctions: Slater-type orbitals with Clementi-Raimondi Z_eff
 *   - Fission yields: JEFF-3.3 cumulative yield data
 *   - Nuclear radius: R = 1.25 × A^(1/3) fm
 *   - De Broglie: λ = h/√(2mE) = 0.286/√(E_eV) Å for neutrons
 *
 * Integration: Writes density/signed grids for quantum_raymarch.comp.
 * Uses the existing wavefunction compute shader for accurate orbital
 * rendering of each atom.
 */

#ifndef ATOMIC_FISSION_H
#define ATOMIC_FISSION_H

#ifdef _WIN32
  #define VK_USE_PLATFORM_WIN32_KHR
#endif
#include <vulkan/vulkan.h>

/* ═══════════════════════ NUCLEAR PHYSICS CONSTANTS ═══════════════════════ */
/*
 * Sources for nuclear data constants in this file:
 *
 *   [ENDF]   ENDF/B-VIII.0  — Evaluated Nuclear Data File, NNDC/BNL (2018)
 *            https://www.nndc.bnl.gov/endf-b8.0/
 *   [JEFF]   JEFF-3.3       — Joint Evaluated Fission and Fusion File, NEA (2017)
 *   [CODATA] CODATA 2018    — NIST Fundamental Physical Constants
 *            https://physics.nist.gov/cuu/Constants/
 *   [PDG]    Particle Data Group — Review of Particle Physics (2022)
 *            https://pdg.lbl.gov/
 *   [KATRIN] KATRIN Collaboration, Nature Physics 18 (2022) 160–166
 *   [KRANE]  K.S. Krane, "Introductory Nuclear Physics" (Wiley, 1988)
 *   [NRL]    NRL Plasma Formulary (Naval Research Laboratory, 2019)
 *   [ITER]   ITER Physics Basis, Nucl. Fusion 39 (1999) 2137
 *   [NNDC]   National Nuclear Data Center, Brookhaven National Laboratory
 *            https://www.nndc.bnl.gov/
 */

/* Fission energy release (MeV) — [ENDF] U-235 thermal fission energy partition */
#define AF_U235_FISSION_ENERGY_MEV     200.0f
#define AF_U235_KE_FRAGMENTS_MEV       165.0f   /* kinetic energy of fragments */
#define AF_U235_PROMPT_GAMMA_MEV        7.0f    /* prompt gamma rays */
#define AF_U235_PROMPT_NEUTRON_KE_MEV   5.0f    /* kinetic energy of prompt neutrons */
#define AF_U235_DELAYED_ENERGY_MEV      23.0f   /* beta/gamma from fission products */

/* Plutonium-239 nuclear data — [ENDF] ENDF/B-VIII.0, MAT 9437 */
#define AF_PU239_FISSION_ENERGY_MEV    210.0f   /* slightly higher than U-235 */
#define AF_PU239_FISSION_XS_BARNS      747.4f   /* σ_f for thermal neutrons */
#define AF_PU239_CAPTURE_XS_BARNS      270.3f   /* σ_γ (radiative capture) */
#define AF_PU239_TOTAL_XS_BARNS       1025.0f   /* σ_total */
#define AF_PU239_NU_BAR                  2.88f   /* ν̄ = mean neutrons per fission */
#define AF_PU239_ALPHA_HALFLIFE_Y    24110.0f   /* α-decay half-life (years) */
/* Pu-239 fission products (most probable) */
#define AF_PU_FRAG1_Z   58    /* Cerium-144  */
#define AF_PU_FRAG1_A  144
#define AF_PU_FRAG2_Z   36    /* Krypton-94  */
#define AF_PU_FRAG2_A   94

/* Thermal neutron cross sections (barns) — [ENDF] ENDF/B-VIII.0, evaluated at 0.0253 eV */
#define AF_U235_FISSION_XS_BARNS      585.1f    /* σ_f for thermal neutrons  [ENDF MAT 9228] */
#define AF_U235_CAPTURE_XS_BARNS       98.7f    /* σ_γ (radiative capture)   [ENDF MAT 9228] */
#define AF_U235_TOTAL_XS_BARNS        694.0f    /* σ_total                   [ENDF MAT 9228] */
#define AF_XE135_ABSORB_XS_BARNS  2650000.0f    /* 2.65 × 10⁶ barns !!!     [ENDF MAT 5431] */
#define AF_B10_ABSORB_XS_BARNS       3840.0f    /* B-10 thermal absorption   [ENDF MAT 525]  */
#define AF_C12_SCATTER_XS_BARNS        4.74f    /* graphite scattering       [ENDF MAT 600]  */
#define AF_H2O_SCATTER_XS_BARNS       49.2f     /* water scattering          [ENDF]           */

/* Nuclear radii (fm) — [KRANE] §3.1 */
#define AF_R0_FM                        1.25f    /* R = r₀ × A^(1/3) */

/* De Broglie wavelength: λ(Å) = 0.2860 / √(E_eV) for neutrons — [CODATA] */
#define AF_NEUTRON_MASS_EV        939.565e6f     /* neutron rest mass in eV/c²   [CODATA 2018] */
#define AF_HBAR_C_EV_FM           197.327f       /* ℏc in MeV·fm                 [CODATA 2018] */
#define AF_THERMAL_ENERGY_EV         0.0253f     /* kT at 20°C = 25.3 meV        [CODATA]      */
#define AF_THERMAL_WAVELENGTH_A      1.8f        /* λ_th ≈ 1.8 Å */
#define AF_FAST_ENERGY_EV        2000000.0f      /* 2 MeV fast neutrons */
#define AF_FAST_WAVELENGTH_FM         20.2f      /* λ_fast ≈ 2.0×10⁻¹⁴ m */

/* Fission product data — most probable split for U-235 thermal fission */
/* [JEFF] JEFF-3.3 cumulative yields: highest yield pair is A≈95 + A≈140 */
#define AF_FRAG1_Z   56    /* Barium-141 */
#define AF_FRAG1_A  141
#define AF_FRAG2_Z   36    /* Krypton-92 */
#define AF_FRAG2_A   92
#define AF_PROMPT_NEUTRONS  3   /* average ν̄ = 2.43, most probable = 2-3 */

/* I-135 / Xe-135 decay chain — [NNDC] NuDat 3.0 evaluated half-lives */
#define AF_I135_HALFLIFE_S      23652.0f  /* 6.57 hours */
#define AF_XE135_HALFLIFE_S     32904.0f  /* 9.14 hours */
#define AF_XE135_Z  54
#define AF_XE135_A  135

/* ═══════════════ D-T FUSION (ITER/NIF) NUCLEAR DATA ═══════════════ */
/* Sources: [ITER] ITER Physics Basis, Nucl. Fusion 39 (1999) 2137     */
/*          [NRL]  NRL Plasma Formulary, §4 (2019)                     */
/*          [ENDF] ENDF/B-VIII.0 d(t,n)⁴He reaction                    */
/* Deuterium + Tritium → He-4 (3.517 MeV) + n (14.069 MeV)       */
/*   Total Q = 17.586 MeV — the easiest fusion reaction          */
/*   Coulomb barrier ≈ 0.40 MeV (for D-T)                        */
/*   Gamow peak energy ≈ 6.3 keV at T = 10 keV                  */
/*   Peak cross section ≈ 5 barns at ~100 keV                    */
/*   Lawson criterion: n·τ_E > 1.5×10²⁰ m⁻³·s at 10 keV        */
#define AF_DT_Q_VALUE_MEV          17.586f  /* total Q-value        [NRL]  */
#define AF_DT_HE4_ENERGY_MEV        3.517f  /* α-particle KE        [NRL]  */
#define AF_DT_NEUTRON_ENERGY_MEV    14.069f  /* fusion neutron KE    [NRL]  */
#define AF_DT_COULOMB_BARRIER_MEV    0.40f   /* classical barrier    [NRL]  */
#define AF_DT_GAMOW_PEAK_KEV         6.3f   /* Gamow window peak    [NRL]  */
#define AF_DT_PEAK_XS_BARNS          5.0f   /* σ at ~100 keV        [ENDF] */
#define AF_DEUTERIUM_A                  2    /* D = ¹H² */
#define AF_TRITIUM_A                    3    /* T = ¹H³ */
#define AF_HE4_Z                        2    /* Helium-4 */
#define AF_HE4_A                        4

/* Pu-239 alpha decay: Pu-239 → U-235 + He-4 (α) — [NNDC] NuDat 3.0 */
#define AF_PU239_ALPHA_ENERGY_MEV    5.244f  /* α-particle KE */

/* ═══════════════ BETA DECAY / WEAK FORCE ═══════════════ */
/* Sources: [PDG] Particle Data Group, Review of Particle Physics (2022) */
/*          [KATRIN] KATRIN Collaboration, Nature Physics 18 (2022) 160  */
/*          [CODATA] CODATA 2018 recommended values                     */
/* n → p + e⁻ + ν̄ₑ  (via W⁻ boson)                      */
/*   Free neutron half-life: 611 s (10.2 min)             */
/*   W⁻ boson mass: 80.377 GeV/c²                         */
/*   W⁻ lifetime: τ = ℏ/Γ ≈ 3.16 × 10⁻²⁵ s              */
/*   Q-value: (mₙ - mₚ)c² = 1.2934 MeV                   */
/*   Max electron KE: 0.782 MeV (endpoint)                */
/*   Fermi coupling: G_F/(ℏc)³ = 1.166 × 10⁻⁵ GeV⁻²     */
#define AF_BETA_Q_VALUE_MEV          1.2934f  /* [CODATA]  */
#define AF_BETA_ELECTRON_MAX_MEV     0.782f   /* [CODATA]  */
#define AF_W_BOSON_MASS_GEV         80.377f   /* [PDG]     */
#define AF_W_BOSON_LIFETIME_S        3.16e-25f/* [PDG]     */
#define AF_FREE_NEUTRON_HALFLIFE_S   611.0f   /* [PDG]     */
#define AF_ELECTRON_MASS_MEV         0.511f   /* [CODATA]  */
#define AF_NEUTRINO_MASS_EV          0.06f    /* [KATRIN] upper bound */

/* ═══════════════ DECAY CHAIN: U-238 → Pb-206 ═══════════════ */
/* [NNDC] NuDat 3.0, NNDC/BNL — evaluated decay energies       */
/* 14-step decay chain (8α + 6β⁻):                            */
/*   U-238 →α Th-234 →β Pa-234 →β U-234 →α Th-230 →α        */
/*   Ra-226 →α Rn-222 →α Po-218 →α Pb-214 →β Bi-214 →β      */
/*   Po-214 →α Pb-210 →β Bi-210 →β Po-210 →α Pb-206 (stable) */
#define AF_DECAY_CHAIN_STEPS        14
#define AF_U238_HALFLIFE_GY          4.468f  /* billion years  [NNDC] */

/* ═══════════════ BINDING ENERGY / NUCLEON ═══════════════ */
/* Semi-empirical mass formula (Bethe-Weizsäcker):          */
/* [KRANE] K.S. Krane, "Introductory Nuclear Physics" (1988), Table 3.3 */
/*   B/A = aV - aS·A⁻¹/³ - aC·Z²·A⁻⁴/³ - aA·(A-2Z)²/A²  */
/*   aV=15.56 MeV, aS=17.23 MeV, aC=0.7 MeV, aA=23.29 MeV*/
/*   Peak: Fe-56 at B/A ≈ 8.79 MeV                          */
#define AF_SEMF_AV   15.56f   /* volume term           [KRANE] */
#define AF_SEMF_AS   17.23f   /* surface term          [KRANE] */
#define AF_SEMF_AC    0.697f  /* Coulomb term          [KRANE] */
#define AF_SEMF_AA   23.29f   /* asymmetry term        [KRANE] */
#define AF_FE56_BA    8.79f   /* Fe-56 B/A peak        [NNDC]  */

/* ═══════════════════════ SCENE DEFINITIONS ═══════════════════════ */

#define AF_MAX_ATOMS      64    /* max visible atoms */
#define AF_MAX_NEUTRONS   64    /* max neutron wave packets */
#define AF_MAX_GAMMAS     48    /* max gamma ray photon */
#define AF_MAX_PARTICLES  32    /* beta/alpha/neutrino/W boson particles */

/* Atom state machine */
typedef enum {
    AF_ATOM_IDLE = 0,       /* ground state, quantum orbitals visible */
    AF_ATOM_EXCITED,        /* just absorbed neutron → compound nucleus */
    AF_ATOM_DEFORMING,      /* liquid-drop deformation (β₂ parameter) */
    AF_ATOM_FISSIONING,     /* scission — splitting into fragments */
    AF_ATOM_FRAGMENT,       /* post-fission fragment flying apart */
    AF_ATOM_DECAYING,       /* radioactive decay (β⁻ emission) */
    AF_ATOM_ABSORBING,      /* Xe-135 absorbing neutron → Xe-136 */
    AF_ATOM_DEAD            /* consumed / off-screen */
} AF_AtomState;

/* A single atom with quantum state */
typedef struct {
    /* Position in Ångströms (1 Å = 10⁻¹⁰ m) */
    float x, y, z;
    float vx, vy, vz;      /* velocity (Å/s) for fragment motion */

    /* Nuclear identity */
    int   Z;                /* atomic number (protons) */
    int   A;                /* mass number (protons + neutrons) */

    /* Quantum state */
    AF_AtomState state;
    float state_time;       /* time in current state (seconds) */
    float excitation_MeV;   /* excitation energy above ground state */
    float deformation;      /* quadrupole deformation β₂ (0=sphere, ~0.6=scission) */

    /* Electron orbital parameters */
    float orbital_scale;    /* 1.0 = normal, increases during excitation */
    float orbital_phase;    /* animation phase for time-dependent ψ */

    /* Rendering */
    float radius_A;         /* atomic radius in Ångströms */
    float glow;             /* extra emission (fission flash, Cherenkov, etc.) */
    int   visible;
} AF_Atom;

/* Neutron wave packet */
typedef struct {
    float x, y, z;          /* position (Å) */
    float vx, vy, vz;       /* velocity (Å/s) */
    float energy_eV;        /* kinetic energy */
    float wavelength_A;     /* de Broglie wavelength λ = h/p */
    float phase;            /* wave phase for animation */
    float birth_time;
    int   alive;
    int   is_thermal;       /* 0 = fast (just born), 1 = thermalized */
    int   scattered_count;  /* number of scattering events */
} AF_Neutron;

/* Gamma ray photon */
typedef struct {
    float x, y, z;
    float dx, dy, dz;       /* unit direction */
    float energy_MeV;
    float birth_time;
    float lifetime;          /* visual lifetime (seconds) */
    int   alive;
} AF_Gamma;

/* Generic particle (electron, positron, alpha, neutrino, W boson) */
typedef struct {
    float x, y, z;
    float vx, vy, vz;
    float energy_MeV;
    float birth_time;
    float lifetime;          /* visual lifetime (seconds) */
    float radius;            /* display radius in Å */
    int   alive;
    int   type;              /* 0=β⁻(e⁻), 1=β⁺(e⁺), 2=α, 3=ν(neutrino), 4=W⁻, 5=W⁺ */
} AF_Particle;

/* Overall scene type */
typedef enum {
    AF_SCENE_SINGLE_FISSION = 0,  /* Watch one U-235 fission event close up */
    AF_SCENE_CHAIN_REACTION,       /* Multiple U-235 atoms → cascade */
    AF_SCENE_XENON_POISON,         /* Xe-135 buildup and neutron absorption */
    AF_SCENE_MODERATION,           /* Neutron slowing in graphite/water */
    AF_SCENE_PLUTONIUM_FISSION,    /* Pu-239 MOX fuel fission */
    AF_SCENE_QCD_NUCLEON,          /* QCD: inside a nucleon — quarks & gluons */
    AF_SCENE_CHERNOBYL_SEQUENCE,   /* Full accident: void→prompt critical→explosion */
    AF_SCENE_FUSION,               /* D+T → He-4 + n : quantum tunneling through Coulomb barrier */
    AF_SCENE_BETA_DECAY,           /* n → p + e⁻ + ν̄ₑ via W⁻ boson (weak force) */
    AF_SCENE_CONFINEMENT,          /* Quark confinement break → string snap → hadronization */
    AF_SCENE_DECAY_CHAIN,          /* U-238 → Pb-206 : 14-step α/β chain */
    AF_SCENE_BINDING_ENERGY,       /* Nuclear binding energy curve (Fe-56 peak) */
    AF_SCENE_COUNT
} AF_SceneType;

/* Animation phase (scene-level) */
typedef enum {
    AF_PHASE_SETUP = 0,      /* atoms placed, waiting */
    AF_PHASE_NEUTRON_FLIGHT,  /* neutron traveling toward target */
    AF_PHASE_ABSORPTION,      /* neutron enters nucleus */
    AF_PHASE_COMPOUND,        /* compound nucleus vibration */
    AF_PHASE_DEFORMATION,     /* liquid-drop elongation */
    AF_PHASE_SCISSION,        /* neck breaks → two fragments */
    AF_PHASE_EXPLOSION,       /* fragments + neutrons fly apart */
    AF_PHASE_CHAIN,           /* daughter neutrons hit more atoms */
    AF_PHASE_POISON,          /* Xe-135 absorbing neutrons */
    AF_PHASE_COMPLETE         /* animation done → loop or pause */
} AF_Phase;

/* ═══════════════════════ MAIN STATE ═══════════════════════ */

typedef struct {
    /* Scene control */
    AF_SceneType  scene;
    AF_Phase      phase;
    float         time;           /* total elapsed time (s) */
    float         phase_time;     /* time in current phase (s) */
    int           playing;        /* 0 = paused */
    int           looping;        /* 1 = auto-restart when done */

    /* Entities */
    AF_Atom       atoms[AF_MAX_ATOMS];
    int           num_atoms;
    AF_Neutron    neutrons[AF_MAX_NEUTRONS];
    int           num_neutrons;
    AF_Gamma      gammas[AF_MAX_GAMMAS];
    int           num_gammas;
    AF_Particle   particles[AF_MAX_PARTICLES];
    int           num_particles;

    /* Physics statistics */
    int           fission_count;
    float         total_energy_MeV;      /* accumulated fission energy */
    float         xe_concentration;      /* relative Xe-135 level */
    int           generation;            /* fission generation counter */

    /* Rendering parameters */
    float         boxHalf;               /* scene extent in Å */
    int           visGridDim;            /* must match QuantumVis gridDim */

    /* Decay chain progress */
    int           chain_step;            /* current step in U-238 decay chain (0-13) */
    float         chain_step_time;       /* time since last decay */

    /* Binding energy morphing */
    float         binding_A;             /* current mass number being displayed */
    int           binding_dir;           /* +1 climbing to Fe-56, -1 descending */

    /* HUD overlay text (up to 8 lines × 80 chars, rasterized separately) */
    char          hud_lines[8][80];
    int           hud_num_lines;

    /* Vulkan (shared with QuantumVis) */
    VkDevice          device;
    VkPhysicalDevice  physDevice;
    VkBuffer          densityBuf;        /* shared SSBO */
    VkBuffer          signedBuf;         /* shared SSBO */
    VkCommandPool     cmdPool;
    VkCommandBuffer   cmdBuf;
} AtomicFission;

/* ═══════════════════════ API ═══════════════════════ */

/* Initialize. densityBuf/signedBuf are shared SSBOs from QuantumVis.
 * Returns 0 on success. */
int  atomic_fission_init(AtomicFission *af,
                         VkPhysicalDevice phys, VkDevice dev,
                         uint32_t queueFamilyIdx,
                         VkBuffer densityBuf, VkBuffer signedBuf,
                         int gridDim);

/* Set up a specific scene (places atoms, resets state). */
void atomic_fission_setup(AtomicFission *af, AF_SceneType scene);

/* Advance physics by dt seconds. */
void atomic_fission_update(AtomicFission *af, float dt);

/* Rasterize all atoms/neutrons/gammas into the density+signed grids
 * and upload to GPU. Color mode 7 in the shader reads these. */
int  atomic_fission_visualize(AtomicFission *af, VkQueue queue);

/* Cycle to next scene. */
void atomic_fission_next_scene(AtomicFission *af);

/* Accessors */
const char *atomic_fission_scene_name(AF_SceneType scene);
const char *atomic_fission_phase_name(AF_Phase phase);

/* Free owned resources. */
void atomic_fission_free(AtomicFission *af);

#endif /* ATOMIC_FISSION_H */
