/* reactor_thermal.h — Reactor thermodynamics for Chernobyl RBMK-1000 simulation
 *
 * Physics modelled:
 *   - Heat generation from fission: q̇ = Σ_f × φ × E_f (200 MeV/fission)
 *   - 3D heat diffusion: ∂T/∂t = α∇²T + q̇/(ρ·c_p)    [Fourier's law]
 *   - Coolant flow & boiling: enthalpy-based two-phase model
 *   - Thermal expansion: ε = α_L × ΔT
 *   - Steam pressure buildup: Clausius-Clapeyron + ideal gas
 *   - Thermodynamic laws: 1st law energy conservation, 2nd law entropy
 *
 * Materials: UO₂ fuel, Zircaloy-4 cladding, light water coolant,
 *            graphite moderator, B₄C control rods
 *
 * Integration: Writes temperature data to the density SSBO shared with
 * quantum_raymarch.comp (color mode 5 = thermal visualization).
 * GPU-accelerated heat diffusion via thermal_diffusion.comp shader.
 *
 * RBMK-1000 reference data:
 *   - Thermal power: 3200 MWt (nominal)
 *   - Core: ~11.8 m diameter, 7 m height
 *   - 1661 fuel channels in graphite moderator
 *   - Coolant: light water at 7 MPa, Tin=270°C, Tout=284°C
 *   - Positive void coefficient of reactivity (critical for accident)
 */

#ifndef REACTOR_THERMAL_H
#define REACTOR_THERMAL_H

#ifdef _WIN32
  #define VK_USE_PLATFORM_WIN32_KHR
#endif
#include <vulkan/vulkan.h>

/* ═══════════════════════ PHYSICAL CONSTANTS ═══════════════════════ */

/* Energy per fission (joules) — 200 MeV
 * [Chadwick et al., Nucl. Data Sheets 112 (2011) — ENDF/B-VII.1] */
#define RT_FISSION_ENERGY_J     3.204e-11f

/* Boltzmann constant (J/K) [CODATA 2018: exact since 2019 SI redefinition] */
#define RT_BOLTZMANN            1.38064852e-23f

/* Universal gas constant (J/(mol·K)) [CODATA 2018: R = 8.314462618 J/(mol·K)] */
#define RT_GAS_CONSTANT         8.314462f

/* Water molecular mass (kg/mol) [IUPAC: M(H₂O) = 18.01528 g/mol] */
#define RT_H2O_MOLAR_MASS      0.018015f

/* Stefan-Boltzmann constant (W/(m²·K⁴)) [CODATA 2018: σ = 5.670374419e-8] */
#define RT_STEFAN_BOLTZMANN     5.670374e-8f

/* Absolute zero offset (K) */
#define RT_CELSIUS_OFFSET       273.15f

/* ═══════════════════════ RBMK OPERATING PARAMETERS ═══════════════
 * All values from INSAG-7, The Chernobyl Accident (1992), IAEA Safety
 * Series No. 75-INSAG-7; and RBMK-1000 design documentation.
 * T_sat from IAPWS-IF97: T_sat(7 MPa) = 285.83°C ≈ 559 K            */

#define RT_OPERATING_PRESSURE   7.0e6f   /* 7 MPa coolant pressure [INSAG-7]  */
#define RT_INLET_TEMP_K         543.15f  /* 270°C inlet coolant   [INSAG-7]   */
#define RT_OUTLET_TEMP_K        557.15f  /* 284°C outlet coolant  [INSAG-7]   */
#define RT_SATURATION_TEMP_K    559.0f   /* T_sat at 7 MPa ≈ 286°C [IAPWS-IF97] */
#define RT_NOMINAL_POWER_MW     3200.0f  /* 3200 MWt full power   [INSAG-7]   */
#define RT_NUM_FUEL_CHANNELS    1661     /* RBMK-1000 fuel channels          */
#define RT_CORE_RADIUS_M        5.9f     /* ~11.8 m diameter                 */
#define RT_CORE_HEIGHT_M        7.0f     /* Active core height               */
#define RT_CHANNEL_PITCH_M      0.25f    /* 25 cm lattice pitch              */

/* ═══════════════════════ DELAYED NEUTRON PARAMETERS ═════════════ */
/* 6-group delayed neutron data for U-235 thermal fission.
 * [Keepin, Wimett & Zeigler, Phys. Rev. 107 (1957) 1044;
 *  Keepin, Physics of Nuclear Kinetics (1965), Table 4-5]
 * β_total = Σβ_i = 0.0065  (≈ 0.65 %)
 * C_i = precursor concentration (normalised to steady-state at P₀)
 * λ_i = decay constant (s⁻¹)
 * β_i = fractional yield of group i                                    */
#define RT_NUM_DELAYED_GROUPS   6
#define RT_BETA_TOTAL           0.0065f  /* total delayed neutron fraction    */
#define RT_LAMBDA_PROMPT        1.0e-4f  /* prompt neutron generation time (s)*/

/* ═══════════════════════ XENON-135 / IODINE-135 POISONING ═══════ */
/* Xe-135 has the largest thermal neutron absorption cross-section
 * of ANY nuclide: σ_a = 2.65 × 10⁶ barns.  During normal operation
 * it reaches equilibrium, but after a power reduction I-135 keeps
 * decaying into Xe-135 while the burn-out rate drops → Xe builds up
 * → the "iodine pit".  Operators must withdraw rods to compensate.
 * If Xe then burns off rapidly → huge positive reactivity → the
 * exact sequence that caused the Chernobyl disaster on 26 Apr 1986. */

#define RT_LAMBDA_I135     2.87e-5f   /* I-135 decay constant  (s⁻¹), t½=6.7h [NNDC/ENSDF] */
#define RT_LAMBDA_XE135    2.09e-5f   /* Xe-135 decay constant (s⁻¹), t½=9.2h [NNDC/ENSDF] */
#define RT_GAMMA_I135      0.061f     /* I-135 fission yield (per fission) [ENDF/B-VIII.0]  */
#define RT_GAMMA_XE135     0.003f     /* Xe-135 direct fission yield [ENDF/B-VIII.0]        */
#define RT_SIGMA_XE135     2.65e-18f  /* Xe-135 micro σ_a (cm²) = 2.65 Mb [ENDF/B-VIII.0]  */
#define RT_SIGMA_F_U235    5.82e-22f  /* U-235 fission σ_f (cm²) at thermal [ENDF/B-VIII.0] */
#define RT_XE_WORTH_FULL   0.030f     /* Xe-135 reactivity worth at equilibrium at full
                                        power: ~3.0% Δk/k (≈4.6$)
                                        [Duderstadt & Hamilton, Nuclear Reactor Analysis (1976)] */

/* ═══════════════════════ MATERIAL TYPES ═══════════════════════════ */

typedef enum {
    RT_MAT_VOID = 0,       /* Empty / steam-only region               */
    RT_MAT_FUEL_UO2,       /* Uranium dioxide fuel pellet              */
    RT_MAT_CLAD_ZR4,       /* Zircaloy-4 cladding                     */
    RT_MAT_COOLANT_H2O,    /* Light water coolant (liquid or two-phase)*/
    RT_MAT_MODERATOR_C,    /* Graphite moderator                      */
    RT_MAT_CONTROL_B4C,    /* Boron carbide control rod                */
    RT_MAT_CONTROL_GRAPH,  /* Graphite displacer tip of control rod    */
    RT_MAT_XENON,          /* Xe-135 poisoned region (visual only)     */
    RT_MAT_COUNT
} RT_Material;

/* ═══════════════════════ CONTROL ROD SYSTEM ═════════════════════ */
/* RBMK-1000: 211 control rods in the core.
 * Each rod is a 7-metre B₄C absorber with a ~1.2 m graphite displacer
 * tip at the bottom.  When rods are fully withdrawn, the graphite tip
 * sits inside the active core displacing water → positive reactivity.
 * On SCRAM (AZ-5), rods descend at ~0.4 m/s — the graphite tip enters
 * FIRST and briefly adds more positive reactivity (the fatal design flaw). */

#define RT_MAX_RODS          211
#define RT_ROD_LENGTH_M      7.0f     /* B₄C absorber length (m) [INSAG-7]   */
#define RT_GRAPHITE_TIP_M    1.25f    /* Graphite displacer tip (m) [INSAG-7] */
#define RT_ROD_TRAVEL_M      7.0f     /* Full travel distance (m) [INSAG-7]   */
#define RT_ROD_SPEED_MPS     0.4f     /* Rod insertion speed (m/s) [INSAG-7]  */
#define RT_ROD_SPEED_WITHDRAW 0.13f   /* Rod withdrawal speed (m/s)     */

typedef struct {
    float x, y;            /* Rod position in core (m, center of rod)  */
    float insertion;       /* 0.0 = fully withdrawn, 1.0 = fully inserted */
    float target;          /* Target insertion (for auto-drive)        */
    int   group;           /* Rod group: 0=manual, 1=auto, 2=emergency*/
    int   active;          /* 1 = rod is present in this slot          */
} RT_ControlRod;

/* ═══════════════════════ NEUTRON PARTICLE VIS ═══════════════════ */
/* Visible neutron "wave packets" for visualization.  Each neutron
 * is a small quantum wave packet that travels through the core,
 * gets moderated by graphite, absorbed by fuel/Xe/B₄C, or escapes. */

#define RT_MAX_NEUTRONS      4096

typedef enum {
    RT_NEUTRON_FAST = 0,   /* Fast neutron (just born from fission)    */
    RT_NEUTRON_THERMAL,    /* Thermalized neutron (moderated)          */
    RT_NEUTRON_ABSORBED,   /* Just absorbed (visual flash, then die)   */
} RT_NeutronState;

typedef struct {
    float x, y, z;         /* Position (m)                             */
    float vx, vy, vz;      /* Velocity direction (normalised × speed)  */
    float px, py, pz;      /* Previous position for streak trail       */
    float energy;          /* Kinetic energy (eV): fast~2MeV, thermal~0.025 */
    float age;             /* Time alive (s) — for visual fade         */
    RT_NeutronState state;
    int   alive;           /* 1 = active, 0 = dead/unused              */
} RT_Neutron;

/* ═══════════════════════ PER-CELL STATE ═════════════════════════ */

typedef struct {
    float temperature;     /* Temperature (K)                          */
    float pressure;        /* Local pressure (Pa)                      */
    float density;         /* Material density (kg/m³)                 */
    float void_fraction;   /* Steam void fraction 0..1 (coolant only)  */
    float heat_source;     /* Volumetric heat generation (W/m³)        */
    float enthalpy;        /* Specific enthalpy (J/kg) for coolant     */
    float entropy;         /* Specific entropy (J/(kg·K))              */
    float thermal_strain;  /* Volumetric thermal strain (dimensionless)*/
    int   material;        /* RT_Material enum                         */
    int   pad[3];          /* Align to 48 bytes for GPU                */
} RT_ThermalCell;

/* ═══════════════════════ COOLANT CHANNEL ═════════════════════════ */

typedef struct {
    float inlet_temp;      /* K — inlet temperature                    */
    float outlet_temp;     /* K — outlet temperature                   */
    float mass_flow_rate;  /* kg/s — coolant mass flow                 */
    float pressure;        /* Pa — channel pressure                    */
    float steam_quality;   /* x = 0..1 (0=liquid, 1=vapor)            */
    float void_fraction;   /* Average void fraction along channel      */
    float heat_absorbed;   /* W — total heat absorbed by channel       */
    float boiling_onset_z; /* Height (m) where boiling begins          */
} RT_CoolantChannel;

/* ═══════════════════════ GLOBAL REACTOR STATE ═══════════════════ */

typedef struct {
    /* 3D temperature grid (gridDim³) — stored contiguously for GPU */
    int   gridDim;
    float boxHalf;           /* Physical half-extent of simulated region (m) */
    float *temperature;      /* gridDim³ temperatures (K), CPU-side         */
    float *heat_source;      /* gridDim³ volumetric heat generation (W/m³)  */
    int   *material_id;      /* gridDim³ material IDs                       */
    float *pressure_field;   /* gridDim³ pressures (Pa)                     */
    float *void_fraction;    /* gridDim³ void fractions                     */

    /* Coolant channels (simplified 1D per-channel model) */
    int              n_channels;
    RT_CoolantChannel *channels;

    /* Global thermodynamic state */
    float total_power_w;     /* Total fission power (W)                 */
    float avg_fuel_temp;     /* Volume-averaged fuel temperature (K)    */
    float max_fuel_temp;     /* Peak fuel temperature (K)               */
    float avg_coolant_temp;  /* Average coolant temperature (K)         */
    float max_coolant_temp;  /* Peak coolant temperature (K)            */
    float system_pressure;   /* System-wide pressure (Pa)               */
    float total_steam_mass;  /* Total steam mass (kg)                   */
    float total_entropy;     /* Total entropy (J/K)                     */
    float energy_balance;    /* Q_gen - Q_removed (W) — 1st law check  */

    /* Zircaloy oxidation tracking (Chernobyl-critical) */
    float zr_oxidation_frac; /* 0..1 fraction of cladding oxidized      */
    float hydrogen_mass;     /* kg of H₂ generated from Zr+H₂O         */

    /* Simulation parameters */
    float time;              /* Simulation time (s)                      */
    float dt;                /* Time step (s)                            */
    float power_fraction;    /* Fraction of nominal power (0..1+)        */
    int   scram_active;      /* 1 = control rods inserting               */
    float scram_time;        /* Time since scram initiated (s)           */

    /* Void coefficient (RBMK positive!) */
    float void_coefficient;  /* Δk/Δα_void (positive means +reactivity) */
    float reactivity;        /* Current excess reactivity Δk/k          */
    float operator_rod_rho;  /* Manual rod reactivity offset (Δk/k).    */
    float initial_power_frac; /* Power at init (for prompt feedback ref)  */

    /* Baseline operating point (set at init so all ρ = 0 at startup) */
    float baseline_void;     /* Average void fraction at init            */
    float baseline_fuel_temp;/* Average fuel temperature at init         */

    /* 6-group delayed neutron precursor concentrations.
     * Normalised so that at steady-state P₀:
     *   C_i = (β_i / (Λ · λ_i)) · P_frac                             */
    float precursor[RT_NUM_DELAYED_GROUPS];
    float neutron_pop;        /* Normalised neutron population (≡ power_fraction at equilibrium) */

    /* ── Xenon-135 / Iodine-135 fission product poisoning ── */
    float iodine_135;         /* I-135 concentration (atoms/cm³)         */
    float xenon_135;          /* Xe-135 concentration (atoms/cm³)        */
    float xe_reactivity;      /* Reactivity from Xe-135 (Δk/k)          */
    float xe_eq;              /* Equilibrium Xe at current power         */
    float xe_baseline;        /* Xe conc at init (reference for ρ=0)     */
    float neutron_flux;       /* Average thermal neutron flux (n/cm²/s)  */

    /* Vulkan resources */
    VkDevice            device;
    VkPhysicalDevice    physDevice;

    VkPipeline          diffusionPipeline;
    VkPipelineLayout    diffusionPipeLayout;
    VkDescriptorSetLayout diffusionDescLayout;
    VkDescriptorPool    diffusionDescPool;
    VkDescriptorSet     diffusionDescSet;
    VkShaderModule      diffusionShader;

    VkBuffer            tempBuf;       /* GPU temperature grid         */
    VkDeviceMemory      tempMem;
    VkBuffer            tempBuf2;      /* GPU ping-pong buffer         */
    VkDeviceMemory      tempMem2;
    VkBuffer            heatSrcBuf;    /* GPU heat source grid         */
    VkDeviceMemory      heatSrcMem;
    VkBuffer            matBuf;        /* GPU material ID grid         */
    VkDeviceMemory      matMem;

    VkCommandPool       cmdPool;
    VkCommandBuffer     cmdBuf;

    /* Shared references (owned by QuantumVis) */
    VkBuffer            densityBuf;    /* For visualization output     */
    VkBuffer            signedBuf;     /* For phase/heat coloring      */
    int                 visGridDim;    /* Visualization grid dimension */

    /* ── Per-rod control rod tracking ── */
    RT_ControlRod       rods[RT_MAX_RODS];
    int                 n_rods;        /* Actual number of rods placed */
    int                 rods_scram;    /* 1 = AZ-5 SCRAM active        */
    float               rod_scram_t;   /* Time since AZ-5 initiated    */

    /* ── Visible neutron particles ── */
    RT_Neutron          neutrons[RT_MAX_NEUTRONS];
    int                 n_neutrons_alive;
    float               neutron_spawn_accum;  /* fractional spawn accumulator */

    /* ── Xenon-135 spatial distribution ── */
    float              *xe_density;    /* gridDim³ — local Xe-135 concentration */
} ReactorThermal;

/* ═══════════════════════ MATERIAL PROPERTIES API ═══════════════ */

/* Temperature-dependent thermal conductivity (W/(m·K)) */
float rt_thermal_conductivity(RT_Material mat, float T_kelvin);

/* Temperature-dependent specific heat capacity (J/(kg·K)) */
float rt_specific_heat(RT_Material mat, float T_kelvin);

/* Temperature-dependent density (kg/m³) */
float rt_density(RT_Material mat, float T_kelvin);

/* Linear thermal expansion coefficient (1/K) */
float rt_expansion_coeff(RT_Material mat, float T_kelvin);

/* Thermal diffusivity α = k/(ρ·c_p) (m²/s) */
float rt_diffusivity(RT_Material mat, float T_kelvin);

/* Melting point (K) */
float rt_melting_point(RT_Material mat);

/* ═══════════════════════ STEAM / WATER PROPERTIES ══════════════ */

/* Saturation temperature at given pressure (K) — Antoine equation */
float rt_saturation_temp(float pressure_pa);

/* Saturation pressure at given temperature (Pa) */
float rt_saturation_pressure(float T_kelvin);

/* Latent heat of vaporization at given pressure (J/kg) */
float rt_latent_heat(float pressure_pa);

/* Specific enthalpy of saturated liquid (J/kg) */
float rt_enthalpy_liquid(float pressure_pa);

/* Specific enthalpy of saturated vapor (J/kg) */
float rt_enthalpy_vapor(float pressure_pa);

/* Void fraction from steam quality and pressure */
float rt_void_fraction(float quality, float pressure_pa);

/* ═══════════════════════ ZIRCALOY OXIDATION ═══════════════════ */

/* Baker-Just correlation: oxidation rate of Zr + 2H₂O → ZrO₂ + 2H₂
 * Returns oxide thickness growth rate (m/s) */
float rt_zr_oxidation_rate(float T_kelvin);

/* Hydrogen generation rate from Zr oxidation (kg/s per m² of cladding) */
float rt_hydrogen_rate(float T_kelvin);

/* ═══════════════════════ MAIN API ═══════════════════════════════ */

/* Initialize the reactor thermal system.
 * Sets up RBMK-1000 geometry, material map, initial temperatures.
 * density/signedBuf are shared SSBOs from QuantumVis for visualization.
 * Returns 0 on success. */
int reactor_thermal_init(ReactorThermal *rt,
                         VkPhysicalDevice phys, VkDevice dev,
                         uint32_t queueFamilyIdx,
                         VkBuffer densityBuf, VkBuffer signedBuf,
                         int visGridDim);

/* Set reactor power level (fraction of 3200 MWt nominal). */
void reactor_thermal_set_power(ReactorThermal *rt, float fraction);

/* Trigger reactor SCRAM (emergency shutdown). */
void reactor_thermal_scram(ReactorThermal *rt);

/* Advance thermal simulation by one time step.
 * Performs: heat generation → diffusion → coolant flow → boiling →
 *           pressure update → thermal expansion → energy balance.
 * Returns 0 on success. */
int reactor_thermal_update(ReactorThermal *rt, float dt);

/* Upload temperature field to GPU and run thermal_diffusion.comp
 * for nIterations Jacobi iterations on the GPU.
 * Returns 0 on success. */
int reactor_thermal_gpu_diffuse(ReactorThermal *rt, VkQueue queue, int nIterations);

/* Write temperature field to the visualization density SSBO
 * for rendering via quantum_raymarch.comp color mode 5.
 * Returns 0 on success. */
int reactor_thermal_visualize(ReactorThermal *rt, VkQueue queue);

/* Get human-readable status string. */
const char *reactor_thermal_status(const ReactorThermal *rt);

/* Free all resources. */
void reactor_thermal_free(ReactorThermal *rt);

/* ═══════════════════ CONTROL ROD API ═══════════════════════════ */

/* Trigger AZ-5 emergency SCRAM — all rods drive to full insertion. */
void reactor_thermal_az5(ReactorThermal *rt);

/* Withdraw/insert operator rod group (group 0) by delta (0..1).
 * Positive delta = withdraw, negative = insert. */
void reactor_thermal_rods_adjust(ReactorThermal *rt, float delta);

/* Get average rod insertion for a group (-1 = all). */
float reactor_thermal_rod_avg(const ReactorThermal *rt, int group);

#endif /* REACTOR_THERMAL_H */
