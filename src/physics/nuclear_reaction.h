/* nuclear_reaction.h — Animated nuclear fission & fusion visualization
 *
 * GPU-accelerated nucleon density evaluation on a 3D grid, rendered by the
 * existing quantum_raymarch.comp volume raycaster.
 *
 * Physics modelled:
 *   - Nuclear radius: R = r₀ × A^(1/3), r₀ = 1.25 fm
 *   - Nucleons rendered as Gaussian metaballs (σ ≈ 0.6 fm)
 *   - Fission: liquid-drop deformation → scission → Coulomb repulsion
 *   - Fusion: Coulomb approach → tunneling → compound nucleus → products
 *   - Proton/neutron coloring via signed density channel
 *
 * Integration: writes to the same density/signed SSBOs used by
 * quantum_raymarch.comp, so the existing raymarch pipeline renders it.
 */

#ifndef NUCLEAR_REACTION_H
#define NUCLEAR_REACTION_H

#ifdef _WIN32
  #define VK_USE_PLATFORM_WIN32_KHR
#endif
#include <vulkan/vulkan.h>

/* ─────────── Constants ─────────── */
#define NR_MAX_NUCLEONS     300      /* U-236 = 236 + incoming neutron + products */
#define NR_SIGMA_FM         0.55f    /* Gaussian width per nucleon (fm)
                                      * [Hofstadter, Rev. Mod. Phys. 28 (1956) 214;
                                      *  proton rms charge radius ~0.87fm → σ≈0.55fm] */
#define NR_R0_FM            1.25f    /* Nuclear radius constant r₀ (fm) in R=r₀·A^(1/3)
                                      * [Krane, Introductory Nuclear Physics (1988) §3.1] */
#define NR_MIN_SEP_FM       1.4f     /* Minimum nucleon separation (fm)
                                      * [Reid, Ann. Phys. 50 (1968) 411 — hard-core range] */

/* ─────────── Nucleon (CPU + GPU, 32-byte aligned) ─────────── */
typedef struct {
    float x, y, z;       /* position in femtometers            */
    float radius;         /* display radius (fm)                */
    int   type;           /* 0 = neutron, 1 = proton            */
    int   group;          /* 0 = parent, 1 = frag1, 2 = frag2, 3+ = free */
    float pad0, pad1;     /* pad to 32 bytes for std430         */
} NR_Nucleon;

/* ─────────── Reaction types ─────────── */
typedef enum {
    NR_FISSION_U235 = 0, /* ²³⁵U + n → ¹⁴¹Ba + ⁹²Kr + 3n + 200 MeV       */
    NR_FUSION_DT    = 1, /* D + T → ⁴He + n + 17.6 MeV                     */
    NR_FUSION_DD    = 2, /* D + D → ³He + n + 3.27 MeV                     */
    NR_REACTION_COUNT
} NR_ReactionType;

/* ─────────── Animation phases ─────────── */
typedef enum {
    NR_PHASE_IDLE = 0,    /* static nucleus / nuclei — thermal jitter       */
    NR_PHASE_APPROACH,    /* incoming particle / nuclei approaching          */
    NR_PHASE_EXCITE,      /* absorption / tunneling — glow effect            */
    NR_PHASE_DEFORM,      /* nucleus deforms (fission) or merges (fusion)    */
    NR_PHASE_SCISSION,    /* neck breaks / products form                     */
    NR_PHASE_SEPARATE,    /* fragments fly apart                             */
    NR_PHASE_DONE         /* animation complete, static final state          */
} NR_Phase;

/* ─────────── Phase timing (seconds, per reaction type) ─────────── */
/* Index: [reaction][phase] — cumulative end times */
/* Defined in nuclear_reaction.c */

/* ─────────── Main state ─────────── */
typedef struct {
    /* Particle data */
    NR_Nucleon  nucleons[NR_MAX_NUCLEONS];
    NR_Nucleon  base_pos[NR_MAX_NUCLEONS]; /* rest-frame positions for animation */
    int         numNucleons;

    /* Reaction state */
    NR_ReactionType reactionType;
    NR_Phase        phase;
    float           time;              /* animation clock (seconds)  */
    float           phaseStart;        /* time when current phase started */
    int             playing;           /* 0 = paused, 1 = playing */

    /* Scene extents */
    float           boxHalf;           /* current box half-extent (fm) — changes during animation */

    /* Vulkan resources (owns nucleon buffer + density pipeline) */
    VkDevice            device;
    VkPhysicalDevice    physDevice;

    VkPipeline          pipeline;
    VkPipelineLayout    pipeLayout;
    VkDescriptorSetLayout descLayout;
    VkDescriptorPool    descPool;
    VkDescriptorSet     descSet;
    VkShaderModule      shader;

    VkBuffer            nucleonBuf;
    VkDeviceMemory      nucleonMem;

    VkCommandPool       cmdPool;
    VkCommandBuffer     cmdBuf;

    /* Shared references (owned by QuantumVis — not freed here) */
    VkBuffer            densityBuf;
    VkBuffer            signedBuf;
    int                 gridDim;
} NuclearReaction;

/* ─────────── API ─────────── */

/* Initialize nuclear reaction system.
 * density/signedBuf are shared SSBOs from QuantumVis.
 * Returns 0 on success. */
int  nuclear_reaction_init(NuclearReaction *nr,
                           VkPhysicalDevice phys, VkDevice dev,
                           uint32_t queueFamilyIdx,
                           VkBuffer densityBuf, VkBuffer signedBuf,
                           int gridDim);

/* Set up a specific reaction (places initial nucleons). */
void nuclear_reaction_setup(NuclearReaction *nr, NR_ReactionType type);

/* Advance animation by dt seconds. Updates nucleon positions on CPU. */
void nuclear_reaction_update(NuclearReaction *nr, float dt);

/* Upload nucleons to GPU and dispatch nuclear_density.comp.
 * Writes to the shared density/signed SSBOs. */
int  nuclear_reaction_dispatch(NuclearReaction *nr, VkQueue queue);

/* Accessors */
float       nuclear_reaction_get_box_half(const NuclearReaction *nr);
const char *nuclear_reaction_phase_name(NR_Phase phase);
const char *nuclear_reaction_type_name(NR_ReactionType type);

/* Free Vulkan resources owned by this module. */
void nuclear_reaction_free(NuclearReaction *nr);

#endif /* NUCLEAR_REACTION_H */
