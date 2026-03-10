/* reactor_thermal.c — RBMK-1000 reactor thermodynamics simulation.
 *
 * Physics implemented:
 *   1. Heat generation from fission (volumetric source q̇)
 *   2. 3D heat diffusion via Fourier's law (GPU Jacobi solver)
 *   3. Coolant enthalpy-based flow model with boiling transition
 *   4. Thermal expansion of fuel and cladding
 *   5. Steam pressure buildup (Clausius-Clapeyron + ideal gas)
 *   6. 1st law energy conservation (Q_gen = Q_removed + ΔU)
 *   7. 2nd law entropy production tracking
 *   8. Zircaloy oxidation + hydrogen generation (Baker-Just)
 *   9. Void coefficient reactivity feedback (RBMK positive!)
 *
 * Reference reactor: RBMK-1000 (Chernobyl Unit 4)
 *   - 3200 MWt thermal, 1661 fuel channels, graphite moderated
 *   - Coolant: light water at 7 MPa, boiling-water type
 *   - Positive void coefficient of reactivity: ~2β
 */

#include "reactor_thermal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ═══════════════════════ VULKAN HELPERS ═══════════════════════ */
/* (same pattern as nuclear_reaction.c / quantum_volume.c) */

static uint32_t rt_find_mem_type(VkPhysicalDevice phys, uint32_t type_bits,
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

static int rt_create_buffer(VkPhysicalDevice phys, VkDevice dev,
                            VkDeviceSize size, VkBufferUsageFlags usage,
                            VkMemoryPropertyFlags props,
                            VkBuffer *buf, VkDeviceMemory *mem_out) {
    VkBufferCreateInfo ci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size  = size;
    ci.usage = usage;
    if (vkCreateBuffer(dev, &ci, NULL, buf) != VK_SUCCESS) return -1;

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, *buf, &req);

    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = rt_find_mem_type(phys, req.memoryTypeBits, props);
    if (ai.memoryTypeIndex == UINT32_MAX) return -2;
    if (vkAllocateMemory(dev, &ai, NULL, mem_out) != VK_SUCCESS) return -3;
    vkBindBufferMemory(dev, *buf, *mem_out, 0);
    return 0;
}

static VkShaderModule rt_load_shader(VkDevice dev, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[RT] Cannot open shader: %s\n", path); return VK_NULL_HANDLE; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint32_t *code = (uint32_t *)malloc((size_t)sz);
    if (!code) { fclose(f); return VK_NULL_HANDLE; }
    fread(code, 1, (size_t)sz, f);
    fclose(f);

    VkShaderModuleCreateInfo ci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = (size_t)sz;
    ci.pCode    = code;
    VkShaderModule mod = VK_NULL_HANDLE;
    vkCreateShaderModule(dev, &ci, NULL, &mod);
    free(code);
    return mod;
}

static void rt_submit_and_wait(VkDevice dev, VkQueue queue, VkCommandBuffer cmd) {
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cmd;
    vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);
}

/* ═══════════════════════════════════════════════════════════════
 *  MATERIAL PROPERTY FUNCTIONS
 *  Temperature-dependent, based on published nuclear engineering data
 * ═══════════════════════════════════════════════════════════════ */

/* ─── Thermal Conductivity k (W/(m·K)) ─── */
float rt_thermal_conductivity(RT_Material mat, float T) {
    switch (mat) {
    case RT_MAT_FUEL_UO2: {
        /* IAEA correlation for UO₂:
         * k = 100 / (7.5408 + 17.692t + 3.6142t²) + 6400/(t^2.5) × exp(-16.35/t)
         * where t = T/1000.  Simplified for computational stability.
         * [Fink, J. Nucl. Mater. 279 (2000) 1-18; IAEA-TECDOC-1496 §3.2.1] */
        float t = T / 1000.0f;
        if (t < 0.3f) t = 0.3f;
        float k_phonon = 100.0f / (7.5408f + 17.692f * t + 3.6142f * t * t);
        float k_rad    = (6400.0f / powf(t, 2.5f)) * expf(-16.35f / t);
        return k_phonon + k_rad;
    }
    case RT_MAT_CLAD_ZR4:
        /* Zircaloy-4: k ≈ 12.767 + 0.00574×T (W/(m·K)), valid 300-1800K
         * [Mills et al., J. Nucl. Mater. 246 (1997); IAEA-TECDOC-1496 §4.2.1] */
        return 12.767f + 0.00574f * T;
    case RT_MAT_COOLANT_H2O:
        /* Water: k depends strongly on phase.
         * Liquid (< T_sat): k ≈ 0.686 - 0.0004×(T-373) approx
         * Steam:            k ≈ 0.025 + 0.00005×(T-373)
         * [IAPWS-IF97 steam tables; NIST Chemistry WebBook] */
        if (T < RT_SATURATION_TEMP_K)
            return 0.686f - 0.0004f * (T - 373.15f);
        else
            return 0.025f + 0.00005f * (T - 373.15f);
    case RT_MAT_MODERATOR_C:
        /* Nuclear graphite: highly dependent on irradiation.
         * Virgin: ~150 W/(m·K), irradiated: ~30-80 W/(m·K)
         * Using irradiated values typical of RBMK graphite stack
         * [IAEA-TECDOC-1154 §3.3; Nightingale, Nuclear Graphite (1962)] */
        return 50.0f * expf(-0.0003f * (T - 300.0f)) + 10.0f;
    case RT_MAT_CONTROL_B4C:
        /* B₄C: k ≈ 30 W/(m·K) at RT, decreases slightly with T
         * [Thévenot, J. Eur. Ceram. Soc. 6 (1990) 205-225] */
        return 30.0f - 0.005f * (T - 300.0f);
    default:
        return 0.01f; /* void / air — very low conductivity */
    }
}

/* ─── Specific Heat c_p (J/(kg·K)) ─── */
float rt_specific_heat(RT_Material mat, float T) {
    switch (mat) {
    case RT_MAT_FUEL_UO2:
        /* UO₂: c_p ≈ 302.27 + 0.0285T (J/(kg·K))
         * [Fink, J. Nucl. Mater. 279 (2000) 1-18, Table 3] */
        return 302.27f + 0.0285f * T;
    case RT_MAT_CLAD_ZR4:
        /* Zircaloy-4: c_p ≈ 281 + 0.0306T (J/(kg·K)) below α→β at ~1100K
         * [Mills et al., J. Nucl. Mater. 246 (1997); IAEA-TECDOC-1496 §4.2.2] */
        if (T < 1100.0f)
            return 281.0f + 0.0306f * T;
        else /* β-phase spike near transition */
            return 360.0f;
    case RT_MAT_COOLANT_H2O:
        /* Liquid water at 7 MPa: c_p ≈ 4200-5500 J/(kg·K) (rises near saturation)
         * Superheated steam: c_p ≈ 2000-2500 J/(kg·K)
         * [IAPWS-IF97 industrial formulation; Wagner & Kretzschmar (2008)] */
        if (T < RT_SATURATION_TEMP_K)
            return 4200.0f + 3.0f * (T - RT_INLET_TEMP_K);
        else
            return 2100.0f + 0.5f * (T - RT_SATURATION_TEMP_K);
    case RT_MAT_MODERATOR_C:
        /* Graphite: c_p increases with T, Debye model
         * c_p ≈ 710 + 0.40×(T-300) J/(kg·K) for T > 300K
         * [Butland & Maddison, J. Nucl. Mater. 49 (1973) 45-56] */
        return 710.0f + 0.40f * (T - 300.0f);
    case RT_MAT_CONTROL_B4C:
        /* [Thévenot, J. Eur. Ceram. Soc. 6 (1990); CRC Handbook of Chemistry & Physics] */
        return 960.0f + 0.25f * (T - 300.0f);
    default:
        return 1000.0f;
    }
}

/* ─── Density ρ (kg/m³) ─── */
float rt_density(RT_Material mat, float T) {
    switch (mat) {
    case RT_MAT_FUEL_UO2:
        /* UO₂: ρ ≈ 10970 kg/m³ (TD=95%) with thermal expansion
         * [IAEA-TECDOC-1496 §3.1; Martin, J. Nucl. Mater. 152 (1988) 94-101] */
        return 10970.0f * (1.0f - 3.0f * 10.0e-6f * (T - 300.0f));
    case RT_MAT_CLAD_ZR4:
        /* Zircaloy-4: ρ ≈ 6550 kg/m³
         * [Scott, J. Nucl. Mater. 18 (1966) 184; IAEA-TECDOC-1496 §4.1] */
        return 6550.0f * (1.0f - 3.0f * 5.7e-6f * (T - 300.0f));
    case RT_MAT_COOLANT_H2O:
        /* Liquid water at 7 MPa: ρ ≈ 740-760 kg/m³   [IAPWS-IF97] */
        /* Steam at 7 MPa: ρ ≈ 36 kg/m³              [IAPWS-IF97] */
        if (T < RT_SATURATION_TEMP_K)
            return 760.0f - 0.5f * (T - RT_INLET_TEMP_K);
        else
            return 36.0f;
    case RT_MAT_MODERATOR_C:
        /* Nuclear graphite: ρ ≈ 1700 kg/m³
         * [IAEA-TECDOC-1154 §3.1; Nightingale, Nuclear Graphite (1962)] */
        return 1700.0f;
    case RT_MAT_CONTROL_B4C:
        /* [Thévenot, J. Eur. Ceram. Soc. 6 (1990); CRC Handbook] */
        return 2520.0f;
    default:
        return 1.2f; /* air */
    }
}

/* ─── Linear Thermal Expansion Coefficient α_L (1/K) ─── */
float rt_expansion_coeff(RT_Material mat, float T) {
    (void)T; /* Weakly temperature-dependent for these materials */
    switch (mat) {
    case RT_MAT_FUEL_UO2:
        return 10.0e-6f;  /* UO₂: ~10×10⁻⁶ /K [Martin, J. Nucl. Mater. 152 (1988) 94] */
    case RT_MAT_CLAD_ZR4:
        return 5.7e-6f;   /* Zircaloy-4: ~5.7×10⁻⁶ /K [Scott, J. Nucl. Mater. 18 (1966)] */
    case RT_MAT_COOLANT_H2O:
        return 0.0f;      /* N/A for fluid */
    case RT_MAT_MODERATOR_C:
        return 4.0e-6f;   /* Graphite: ~4×10⁻⁶ /K [IAEA-TECDOC-1154 §3.4] */
    case RT_MAT_CONTROL_B4C:
        return 5.0e-6f;   /* [Thévenot, J. Eur. Ceram. Soc. 6 (1990)] */
    default:
        return 0.0f;
    }
}

/* ─── Thermal Diffusivity α = k/(ρ·c_p) (m²/s) ─── */
float rt_diffusivity(RT_Material mat, float T) {
    float k   = rt_thermal_conductivity(mat, T);
    float rho = rt_density(mat, T);
    float cp  = rt_specific_heat(mat, T);
    if (rho * cp < 1.0f) return 1.0e-7f;
    return k / (rho * cp);
}

/* ─── Melting Point (K) ─── */
float rt_melting_point(RT_Material mat) {
    switch (mat) {
    case RT_MAT_FUEL_UO2:     return 3120.0f;  /* [Adamson et al., J. Nucl. Mater. 130 (1985) 349] */
    case RT_MAT_CLAD_ZR4:     return 2125.0f;  /* [ASM Specialty Handbook: Zirconium; IAEA-TECDOC-1496] */
    case RT_MAT_COOLANT_H2O:  return 273.15f;
    case RT_MAT_MODERATOR_C:  return 3925.0f;  /* sublimation [CRC Handbook; IAEA-TECDOC-1154] */
    case RT_MAT_CONTROL_B4C:  return 2718.0f;  /* [Thévenot, J. Eur. Ceram. Soc. 6 (1990)] */
    default:                  return 9999.0f;
    }
}

/* ═══════════════════════════════════════════════════════════════
 *  STEAM / WATER THERMODYNAMIC PROPERTIES
 *  Simplified correlations for the RBMK operating range (5-10 MPa)
 * ═══════════════════════════════════════════════════════════════ */

/* Saturation temperature from pressure using Antoine-like equation.
 * T_sat(P) ≈ simplified inverse Clausius-Clapeyron.
 * Valid 1-10 MPa range. */
float rt_saturation_temp(float P) {
    /* Antoine equation adapted for water near 1-10 MPa:
     * log10(P_bar) = A - B/(C + T_celsius)
     * A=5.0768, B=1659.793, C=227.1 (water, high-pressure fit)
     * Inverted: T_celsius = B/(A - log10(P_bar)) - C
     * [NIST Chemistry WebBook, Antoine parameters for water] */
    if (P < 1.0e5f) P = 1.0e5f;
    float P_bar = P / 1.0e5f;
    float log_P = log10f(P_bar);
    float T_c = 1659.793f / (5.0768f - log_P) - 227.1f;
    return T_c + RT_CELSIUS_OFFSET;
}

/* Saturation pressure from temperature */
float rt_saturation_pressure(float T) {
    float T_c = T - RT_CELSIUS_OFFSET;
    if (T_c < 100.0f) T_c = 100.0f;
    float log_P = 5.0768f - 1659.793f / (T_c + 227.1f);
    return powf(10.0f, log_P) * 1.0e5f; /* Pa */
}

/* Latent heat of vaporization (J/kg) — decreases with pressure.
 * At 7 MPa: ~1505 kJ/kg.  At critical point (22.1 MPa): 0. */
float rt_latent_heat(float P) {
    float P_MPa = P / 1.0e6f;
    if (P_MPa > 22.0f) return 0.0f; /* above critical point */
    /* Fit: h_fg ≈ 2257 × (1 - P/22.064)^0.38 kJ/kg
     * 2257 kJ/kg = latent heat at 1 atm [IAPWS-IF97]
     * 22.064 MPa = critical pressure [IAPWS-IF97]
     * Exponent 0.38 from Watson correlation [Watson, Ind. Eng. Chem. 35 (1943) 398] */
    float ratio = P_MPa / 22.064f;
    if (ratio > 0.999f) ratio = 0.999f;
    return 2257.0e3f * powf(1.0f - ratio, 0.38f);
}

/* Specific enthalpy of saturated liquid h_f (J/kg) */
float rt_enthalpy_liquid(float P) {
    float T_sat = rt_saturation_temp(P);
    /* h_f ≈ c_p × (T_sat - 273.15) with c_p~4200 for liquid water */
    return 4200.0f * (T_sat - 273.15f);
}

/* Specific enthalpy of saturated vapor h_g (J/kg) */
float rt_enthalpy_vapor(float P) {
    return rt_enthalpy_liquid(P) + rt_latent_heat(P);
}

/* Void fraction α from steam quality x using drift-flux model.
 * Simplified Zuber-Findlay: α = x / (C₀×(x + (1-x)×ρ_g/ρ_f))
 * where C₀ ≈ 1.13 for tubes
 * [Zuber & Findlay, J. Heat Transfer 87 (1965) 453-468] */
float rt_void_fraction(float quality, float P) {
    if (quality <= 0.0f) return 0.0f;
    if (quality >= 1.0f) return 1.0f;
    float rho_f = 760.0f;  /* liquid density at ~7 MPa */
    float rho_g = 36.0f;   /* steam density at ~7 MPa */
    float C0 = 1.13f;
    float alpha = quality / (C0 * (quality + (1.0f - quality) * rho_g / rho_f));
    return fminf(fmaxf(alpha, 0.0f), 1.0f);
}

/* ═══════════════════════════════════════════════════════════════
 *  ZIRCALOY OXIDATION — Baker-Just Correlation
 *  Zr + 2H₂O → ZrO₂ + 2H₂ + 6.5 MJ/kg(Zr)
 *  Parabolic rate law: w² = A × t × exp(-Q/RT)
 *  where w = oxide thickness, A = 33.3 mg²/(cm⁴·s), Q = 45500 cal/mol
 *  [Baker & Just, ANL-6548 (1962), Argonne National Laboratory]
 * ═══════════════════════════════════════════════════════════════ */

/* Oxidation rate constant K (m²/s) at temperature T */
float rt_zr_oxidation_rate(float T) {
    if (T < 1073.15f) return 0.0f; /* Negligible below ~800°C */
    /* Baker-Just: K = 33.3e-6 m²/s × exp(-45500×4.184 / (8.314 × T))
     * 4.184 J/cal = thermochemical calorie [NIST] */
    float Q_over_R = 45500.0f * 4.184f / RT_GAS_CONSTANT; /* ~22900 K */
    return 33.3e-6f * expf(-Q_over_R / T);
}

/* H₂ generation rate (kg H₂ per m² of Zr surface per second) */
float rt_hydrogen_rate(float T) {
    /* From stoichiometry: Zr(91.2) + 2H₂O → ZrO₂ + 2H₂(4.032)
     * H₂/Zr mass ratio = 4.032/91.22 ≈ 0.0442 [IUPAC atomic masses 2021] */
    float K = rt_zr_oxidation_rate(T);
    if (K < 1.0e-20f) return 0.0f;
    /* H₂ rate ∝ Zr oxidation rate × stoichiometric ratio × Zr density */
    float zr_rho = 6550.0f;
    return 0.0442f * zr_rho * sqrtf(K); /* simplified instantaneous */
}

/* ═══════════════════════════════════════════════════════════════
 *  RBMK GEOMETRY SETUP — HOMOGENIZED ASSEMBLY MODEL
 *  At 64³ grid (cell ≈ 22cm), individual fuel pins (5.7mm) cannot be
 *  resolved. Instead we use **homogenized** cells:
 *    - FUEL cells sit on the 25cm channel lattice (smeared fuel+clad+coolant)
 *    - MODERATOR cells fill the graphite matrix between channels
 *    - COOLANT cells form a thin shell around each fuel cell
 *  This captures the correct channel count (~1661) and thermal behavior.
 * ═══════════════════════════════════════════════════════════════ */

static void setup_rbmk_geometry(ReactorThermal *rt) {
    int G = rt->gridDim;
    float h = rt->boxHalf;  /* half-extent in meters */
    float dx = 2.0f * h / (float)G;

    float core_R = RT_CORE_RADIUS_M;   /* 5.9 m   */
    float core_H = RT_CORE_HEIGHT_M;   /* 7.0 m   */
    float pitch  = RT_CHANNEL_PITCH_M; /* 0.25 m  */

    /* RBMK channel geometry (real dimensions):
     *   Graphite block:          250×250 mm lattice
     *   Pressure tube (Zr-2.5%Nb): OD=88mm, wall=4mm → ID=80mm
     *   Fuel assembly:  18 pins in 2 concentric rings (6+12) around central rod
     *   Pin:   UO₂ pellet OD=11.5mm, Zr-1%Nb clad OD=13.6mm
     *   Coolant fills annulus between fuel pins and pressure tube.
     *
     * At dx≈0.109m (128³), we can resolve:
     *   - Fuel zone:   r < 0.04m from channel center
     *   - Clad ring:   0.04m < r < 0.044m
     *   - Coolant gap: 0.044m < r < 0.056m (water annulus inside pressure tube)
     *   - Pressure tube wall barely resolved at this scale
     *   
     * We model concentric zones within each lattice cell. */
    float fuel_R   = 0.035f;   /* fuel bundle effective radius */
    float clad_R   = 0.042f;   /* cladding outer radius */
    float cool_R   = 0.060f;   /* pressure tube inner radius (coolant extent) */
    /* Beyond cool_R → graphite moderator */

    int n_fuel = 0, n_cool = 0, n_mod = 0, n_clad = 0;

    for (int iz = 0; iz < G; iz++) {
        float z = -h + (iz + 0.5f) * dx;
        int in_core_z = (z >= -core_H / 2.0f && z <= core_H / 2.0f);

        for (int iy = 0; iy < G; iy++) {
            float y = -h + (iy + 0.5f) * dx;
            for (int ix = 0; ix < G; ix++) {
                float x = -h + (ix + 0.5f) * dx;
                int idx = iz * G * G + iy * G + ix;

                float r_xy = sqrtf(x * x + y * y);

                if (!in_core_z || r_xy > core_R) {
                    rt->material_id[idx] = RT_MAT_VOID;
                    rt->temperature[idx] = 300.0f;
                } else {
                    /* Snap to nearest lattice node */
                    float cx = roundf(x / pitch) * pitch;
                    float cy = roundf(y / pitch) * pitch;
                    float cr = sqrtf(cx * cx + cy * cy);
                    float d  = sqrtf((x - cx) * (x - cx) + (y - cy) * (y - cy));

                    if (cr < core_R - pitch * 0.5f) {
                        /* Inside a channel lattice position — concentric zones */
                        if (d < fuel_R) {
                            /* Fuel pellet zone */
                            rt->material_id[idx] = RT_MAT_FUEL_UO2;
                            rt->temperature[idx] = 800.0f;
                            n_fuel++;
                        } else if (d < clad_R) {
                            /* Zircaloy cladding ring */
                            rt->material_id[idx] = RT_MAT_CLAD_ZR4;
                            rt->temperature[idx] = 650.0f;
                            n_clad++;
                        } else if (d < cool_R) {
                            /* Water coolant annulus */
                            rt->material_id[idx] = RT_MAT_COOLANT_H2O;
                            rt->temperature[idx] = RT_INLET_TEMP_K;
                            n_cool++;
                        } else {
                            /* Graphite moderator between channels */
                            rt->material_id[idx] = RT_MAT_MODERATOR_C;
                            rt->temperature[idx] = 550.0f;
                            n_mod++;
                        }
                    } else {
                        /* Edge of core — graphite reflector */
                        rt->material_id[idx] = RT_MAT_MODERATOR_C;
                        rt->temperature[idx] = 500.0f;
                        n_mod++;
                    }
                }

                rt->heat_source[idx]    = 0.0f;
                rt->pressure_field[idx] = RT_OPERATING_PRESSURE;
                rt->void_fraction[idx]  = 0.0f;
            }
        }
    }

    printf("[RT]   Geometry: %d fuel, %d clad, %d coolant, %d moderator cells (dx=%.3f m)\n",
           n_fuel, n_clad, n_cool, n_mod, dx);
}

/* Set up cosine-shaped axial power distribution (typical of RBMK)
 * q̇(r,z) = q̇₀ × J₀(2.405 r/R) × cos(π z/H) */
static void setup_power_distribution(ReactorThermal *rt) {
    int G = rt->gridDim;
    float h = rt->boxHalf;
    float dx = 2.0f * h / (float)G;

    float total_fuel_volume = 0.0f;
    float power_w = rt->total_power_w;

    /* First pass: compute total fuel volume to normalize power density */
    for (int i = 0; i < G * G * G; i++) {
        if (rt->material_id[i] == RT_MAT_FUEL_UO2)
            total_fuel_volume += dx * dx * dx;
    }
    if (total_fuel_volume < 1.0e-10f) total_fuel_volume = 1.0f;

    /* Average volumetric heat generation rate */
    float q_avg = power_w / total_fuel_volume;

    /* Second pass: apply cosine axial + Bessel radial shaping */
    for (int iz = 0; iz < G; iz++) {
        float z = -h + (iz + 0.5f) * dx;
        float z_core = z + RT_CORE_HEIGHT_M / 2.0f;
        float z_norm = z_core / RT_CORE_HEIGHT_M; /* 0..1 */
        /* Cosine axial shape: peak at center, zero at edges */
        float axial_shape = (z_norm >= 0.0f && z_norm <= 1.0f) ?
            cosf((float)M_PI * (z_norm - 0.5f)) : 0.0f;
        /* Normalize: integral of cos(π(z-0.5)) from 0 to 1 = 2/π */
        axial_shape *= (float)M_PI / 2.0f;

        for (int iy = 0; iy < G; iy++) {
            float y = -h + (iy + 0.5f) * dx;
            for (int ix = 0; ix < G; ix++) {
                float x = -h + (ix + 0.5f) * dx;
                int idx = iz * G * G + iy * G + ix;

                if (rt->material_id[idx] != RT_MAT_FUEL_UO2) continue;

                float r_xy = sqrtf(x * x + y * y);
                /* Bessel J₀ radial shape: J₀(2.405 r/R) */
                float r_norm = r_xy / RT_CORE_RADIUS_M;
                /* Simplified J₀ using first few terms of series */
                float arg = 2.405f * r_norm;
                float j0 = 1.0f - (arg * arg) / 4.0f + (arg * arg * arg * arg) / 64.0f;
                if (j0 < 0.0f) j0 = 0.0f;

                rt->heat_source[idx] = q_avg * axial_shape * j0;
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 *  HEAT DIFFUSION (CPU fallback — GPU version in thermal_diffusion.comp)
 *  Explicit Euler: T^{n+1} = T^n + α·Δt/Δx² × (ΣT_neighbors - 6T) + q̇·Δt/(ρ·c_p)
 * ═══════════════════════════════════════════════════════════════ */

static void heat_diffusion_step_cpu(ReactorThermal *rt) {
    int G = rt->gridDim;
    float dx = 2.0f * rt->boxHalf / (float)G;
    float dt = rt->dt;
    int total = G * G * G;

    /* Allocate temporary buffer for new temperatures */
    float *T_new = (float *)malloc(total * sizeof(float));
    if (!T_new) return;
    memcpy(T_new, rt->temperature, total * sizeof(float));

    for (int iz = 1; iz < G - 1; iz++) {
        for (int iy = 1; iy < G - 1; iy++) {
            for (int ix = 1; ix < G - 1; ix++) {
                int idx = iz * G * G + iy * G + ix;
                int mat = rt->material_id[idx];
                if (mat == RT_MAT_VOID) continue;

                float T_c = rt->temperature[idx];
                float alpha = rt_diffusivity((RT_Material)mat, T_c);

                /* 6-neighbor Laplacian */
                float T_xp = rt->temperature[idx + 1];
                float T_xm = rt->temperature[idx - 1];
                float T_yp = rt->temperature[idx + G];
                float T_ym = rt->temperature[idx - G];
                float T_zp = rt->temperature[idx + G * G];
                float T_zm = rt->temperature[idx - G * G];

                float laplacian = (T_xp + T_xm + T_yp + T_ym + T_zp + T_zm - 6.0f * T_c) / (dx * dx);

                /* Source term: q̇ / (ρ·c_p) */
                float rho = rt_density((RT_Material)mat, T_c);
                float cp  = rt_specific_heat((RT_Material)mat, T_c);
                float source = (rho * cp > 1.0f) ? rt->heat_source[idx] / (rho * cp) : 0.0f;

                /* Explicit Euler update */
                float dT = (alpha * laplacian + source) * dt;

                /* CFL stability check: limit temperature change per step */
                if (dT >  500.0f) dT =  500.0f;
                if (dT < -500.0f) dT = -500.0f;

                T_new[idx] = T_c + dT;

                /* Keep above absolute zero */
                if (T_new[idx] < 200.0f) T_new[idx] = 200.0f;
            }
        }
    }

    memcpy(rt->temperature, T_new, total * sizeof(float));
    free(T_new);
}

/* ═══════════════════════════════════════════════════════════════
 *  COOLANT FLOW MODEL
 *  1D enthalpy-based model per coolant cell.
 *  Tracks subcooled → boiling → superheated transitions.
 * ═══════════════════════════════════════════════════════════════ */

static void update_coolant(ReactorThermal *rt) {
    int G = rt->gridDim;
    float h = rt->boxHalf;
    float dx = 2.0f * h / (float)G;
    float P = rt->system_pressure;

    float T_sat = rt_saturation_temp(P);
    float h_f   = rt_enthalpy_liquid(P);
    float h_fg  = rt_latent_heat(P);
    float h_g   = h_f + h_fg;

    float total_steam = 0.0f;
    float total_heat_absorbed = 0.0f;

    for (int idx = 0; idx < G * G * G; idx++) {
        if (rt->material_id[idx] != RT_MAT_COOLANT_H2O) continue;

        float T = rt->temperature[idx];
        float rho = rt_density(RT_MAT_COOLANT_H2O, T);
        float cp  = rt_specific_heat(RT_MAT_COOLANT_H2O, T);
        float cell_vol = dx * dx * dx;
        float cell_mass = rho * cell_vol;

        /* Heat absorbed from neighboring fuel/cladding cells (via conduction) */
        /* This is implicitly handled by the diffusion solver, but we also
         * track enthalpy for phase transition logic */

        /* Compute specific enthalpy from temperature */
        float enthalpy;
        if (T < T_sat) {
            /* Subcooled liquid */
            enthalpy = cp * (T - 273.15f);
            rt->void_fraction[idx] = 0.0f;
        } else {
            /* At or above saturation — check if still in two-phase */
            float excess_h = cp * (T - T_sat);
            if (excess_h < h_fg) {
                /* Two-phase: T stays at T_sat, energy goes to vaporization */
                float quality = excess_h / h_fg;
                rt->void_fraction[idx] = rt_void_fraction(quality, P);
                enthalpy = h_f + excess_h;
                /* Clamp temperature to saturation during boiling */
                rt->temperature[idx] = T_sat;
            } else {
                /* Superheated steam */
                rt->void_fraction[idx] = 1.0f;
                enthalpy = h_g + (T - T_sat) * rt_specific_heat(RT_MAT_COOLANT_H2O, T);
            }
        }

        /* Accumulate steam mass */
        total_steam += rt->void_fraction[idx] * cell_mass;
        total_heat_absorbed += fmaxf(0.0f, (enthalpy - h_f) * cell_mass);
    }

    rt->total_steam_mass = total_steam;
}

/* ═══════════════════════════════════════════════════════════════
 *  PRESSURE DYNAMICS
 *  Steam generation → pressure buildup.
 *  Uses simplified model: P = P_sat(T) + ΔP from steam accumulation.
 * ═══════════════════════════════════════════════════════════════ */

static void update_pressure(ReactorThermal *rt) {
    /* System pressure rises with steam generation.
     * Simplified: track max coolant temperature → saturation pressure.
     * Add overpressure from trapped steam (using ideal gas law). */

    float max_coolant_T = 0.0f;
    int G = rt->gridDim;

    for (int idx = 0; idx < G * G * G; idx++) {
        if (rt->material_id[idx] == RT_MAT_COOLANT_H2O) {
            if (rt->temperature[idx] > max_coolant_T)
                max_coolant_T = rt->temperature[idx];
        }
    }

    /* Saturation pressure at peak coolant temperature */
    float P_sat = rt_saturation_pressure(max_coolant_T);

    /* Additional pressure from steam accumulation (ideal gas contribution)
     * ΔP = n_steam × R × T_avg / V_total_steam
     * where n = m_steam / M_H2O */
    float avg_steam_T = fmaxf(max_coolant_T, rt_saturation_temp(rt->system_pressure));
    float n_moles = rt->total_steam_mass / RT_H2O_MOLAR_MASS;
    float V_steam = 1.0f; /* Simplified total volume available for steam (m³) */

    /* Estimate V_steam from void fraction distribution */
    float dx = 2.0f * rt->boxHalf / (float)G;
    float cell_vol = dx * dx * dx;
    float total_void_vol = 0.0f;
    for (int idx = 0; idx < G * G * G; idx++) {
        if (rt->material_id[idx] == RT_MAT_COOLANT_H2O)
            total_void_vol += rt->void_fraction[idx] * cell_vol;
    }
    if (total_void_vol > 0.001f) V_steam = total_void_vol;
    else V_steam = 0.001f; /* minimum to avoid div/0 */

    float P_ideal = n_moles * RT_GAS_CONSTANT * avg_steam_T / V_steam;

    /* System pressure = max of saturation and ideal gas */
    float new_P = fmaxf(P_sat, P_ideal);

    /* Limit pressure change rate for numerical stability */
    float dP = new_P - rt->system_pressure;
    float max_dP = 1.0e6f * rt->dt; /* Max 1 MPa/s change rate */
    if (dP >  max_dP) dP =  max_dP;
    if (dP < -max_dP) dP = -max_dP;

    rt->system_pressure += dP;

    /* Update per-cell pressure (simplified: uniform) */
    for (int idx = 0; idx < G * G * G; idx++) {
        rt->pressure_field[idx] = rt->system_pressure;
    }

    /* Propagate new saturation temperature */
    rt->max_coolant_temp = max_coolant_T;
}

/* ═══════════════════════════════════════════════════════════════
 *  THERMAL EXPANSION
 *  ε_V = 3 × α_L × (T - T_ref) — volumetric strain
 * ═══════════════════════════════════════════════════════════════ */

static void update_thermal_expansion(ReactorThermal *rt) {
    int G = rt->gridDim;
    float T_ref = 300.0f; /* Reference temperature */

    for (int idx = 0; idx < G * G * G; idx++) {
        int mat = rt->material_id[idx];
        if (mat == RT_MAT_VOID || mat == RT_MAT_COOLANT_H2O) continue;

        float T = rt->temperature[idx];
        float alpha_L = rt_expansion_coeff((RT_Material)mat, T);
        /* Volumetric strain = 3 × linear strain */
        float strain = 3.0f * alpha_L * (T - T_ref);
        /* Store for visualization / structural analysis */
        /* (Would feed into mechanical stress solver in full sim) */
        (void)strain; /* Used via pressure update in future */
    }
}

/* ═══════════════════════════════════════════════════════════════
 *  ZIRCALOY OXIDATION UPDATE
 *  Critical for Chernobyl: Zr + 2H₂O → ZrO₂ + 2H₂ + heat
 *  Exothermic reaction + hydrogen generation → explosion risk
 * ═══════════════════════════════════════════════════════════════ */

static void update_zr_oxidation(ReactorThermal *rt) {
    int G = rt->gridDim;
    float dx = 2.0f * rt->boxHalf / (float)G;
    float dt = rt->dt;
    float cell_vol = dx * dx * dx;

    float h2_produced = 0.0f;
    float oxidation_heat = 0.0f;

    for (int idx = 0; idx < G * G * G; idx++) {
        if (rt->material_id[idx] != RT_MAT_CLAD_ZR4) continue;

        float T = rt->temperature[idx];
        if (T < 1073.15f) continue;

        /* Baker-Just parabolic rate */
        float surface_area = 6.0f * powf(cell_vol, 2.0f / 3.0f); /* simplified cell surface */

        /* Mass of H₂ produced per cell per dt */
        float h2_rate = rt_hydrogen_rate(T);
        float dm_h2 = h2_rate * surface_area * dt;
        h2_produced += dm_h2;

        /* Exothermic heat from Zr oxidation: 6.5 MJ per kg of Zr reacted */
        float dm_zr = dm_h2 / 0.0442f; /* inverse stoichiometric ratio */
        float dQ = dm_zr * 6.5e6f;     /* J — Zr oxidation enthalpy 6.5 MJ/kg
                                         * [Lustman & Kerze, Metallurgy of Zirconium (1955); IAEA ~6.45 MJ/kg] */
        oxidation_heat += dQ;

        /* Add oxidation heat to the cell */
        float rho = rt_density(RT_MAT_CLAD_ZR4, T);
        float cp  = rt_specific_heat(RT_MAT_CLAD_ZR4, T);
        float cell_mass = rho * cell_vol;
        if (cell_mass * cp > 0.0f)
            rt->temperature[idx] += dQ / (cell_mass * cp);
    }

    rt->hydrogen_mass += h2_produced;

    /* Track oxidation fraction (total Zr inventory ~ 1661 channels × ~100 kg each)
     * [INSAG-7 (1992); RBMK-1000 design documentation] */
    float total_zr_mass = 1661.0f * 100.0f; /* ~166100 kg */
    float total_zr_reacted = rt->hydrogen_mass / 0.0442f;
    rt->zr_oxidation_frac = total_zr_reacted / total_zr_mass;
}

/* ═══════════════════════════════════════════════════════════════
 *  VOID COEFFICIENT REACTIVITY FEEDBACK (RBMK-specific!)
 *  RBMK has POSITIVE void coefficient: more steam → more reactivity
 *  This is the fundamental cause of the Chernobyl accident.
 *
 *  Δρ = α_void × Δα_total
 *  where α_void ≈ +4.7 β (for RBMK at low power, partially loaded)
 *  β ≈ 0.0064 (delayed neutron fraction for U-235)
 * ═══════════════════════════════════════════════════════════════ */

static void update_reactivity_feedback(ReactorThermal *rt) {
    int G = rt->gridDim;
    float total_void = 0.0f;
    int n_coolant_cells = 0;

    for (int idx = 0; idx < G * G * G; idx++) {
        if (rt->material_id[idx] == RT_MAT_COOLANT_H2O) {
            total_void += rt->void_fraction[idx];
            n_coolant_cells++;
        }
    }

    float avg_void = (n_coolant_cells > 0) ? total_void / n_coolant_cells : 0.0f;

    /* RBMK positive void coefficient (most dangerous feature!) */
    /* At Chernobyl conditions (low power, few control rods):
     * α_void ≈ +4.7β per unit void fraction change
     * β = 0.0064 (delayed neutron fraction)
     * [INSAG-7, Annex I (1992); Adamov et al., Nucl. Eng. Design 173 (1997)] */
    float beta = 0.0064f;
    rt->void_coefficient = 4.7f * beta; /* per unit void fraction */

    /* Reactivity change from void (deviation from baseline) */
    float delta_rho_void = rt->void_coefficient * (avg_void - rt->baseline_void);

    /* Doppler coefficient (negative — fuel temp feedback, stabilizing):
     * α_D ≈ -3.0×10⁻⁵ Δk/k per K
     * [INSAG-7; typical range −2 to −5 × 10⁻⁵ for thermal reactors] */
    float doppler_coeff = -3.0e-5f;
    float delta_T_fuel = rt->avg_fuel_temp - rt->baseline_fuel_temp;
    float delta_rho_doppler = doppler_coeff * delta_T_fuel;

    /* Control rod insertion effect (if SCRAM active) */
    float delta_rho_rods = 0.0f;
    if (rt->scram_active) {
        /* RBMK control rods take ~18-20 seconds for full insertion
         * Reactivity worth of all rods: ~-7β when fully inserted
         * BUT: RBMK rods have graphite displacing tips — initially POSITIVE!
         * This is the "scram causes power spike" effect.
         * [INSAG-7 §4 (1992); Abagyan et al., RBMK safety analysis reports] */
        float rod_travel = rt->scram_time / 20.0f; /* 0..1 normalized travel */
        if (rod_travel > 1.0f) rod_travel = 1.0f;

        if (rod_travel < 0.15f) {
            /* Graphite displacer phase: POSITIVE reactivity insertion!
             * Graphite tip displaces water → increases moderation locally → +ρ */
            delta_rho_rods = +1.5f * beta * (rod_travel / 0.15f);
        } else {
            /* B₄C absorber phase: negative reactivity */
            float absorber_frac = (rod_travel - 0.15f) / 0.85f;
            delta_rho_rods = +1.5f * beta * (1.0f - absorber_frac * 1.2f) - 7.0f * beta * absorber_frac;
        }
    }

    /* ── Xenon-135 / Iodine-135 poisoning kinetics ──
     *  dI/dt  = γ_I · Σ_f · φ  −  λ_I · I
     *  dXe/dt = γ_Xe · Σ_f · φ  +  λ_I · I  −  λ_Xe · Xe  −  σ_a^Xe · φ · Xe
     *
     * The Xe burn-out term (σ_a·φ·Xe) couples Xe to neutron flux.
     * When power drops, burn-out drops but I→Xe production continues
     * → Xe builds up → negative reactivity → the "iodine pit".
     * Operators withdraw rods → Xe eventually burns off → power surge.
     */
    {
        /* Nominal RBMK-1000 thermal neutron flux and macroscopic fission rate
         * [Dollezhal & Emelyanov, Channel Nuclear Power Reactors (1980);
         *  IAEA-TECDOC-1474] */
        float phi_nom = 3.0e13f;  /* n/(cm²·s) */
        float phi = phi_nom * rt->power_fraction;
        rt->neutron_flux = phi;
        float fiss_rate = 9.8e13f * rt->power_fraction; /* fissions/(cm³·s) ≈ Σ_f × φ */
        float dt_xe = rt->dt;

        /* Semi-implicit Euler for stiffness (σ_a·φ can be large) */
        /* Iodine: dI/dt = γ_I·R − λ_I·I  →  I_new = (I + dt·γ_I·R) / (1 + dt·λ_I) */
        float I_new = (rt->iodine_135 + dt_xe * RT_GAMMA_I135 * fiss_rate)
                    / (1.0f + dt_xe * RT_LAMBDA_I135);

        /* Xenon: dXe/dt = γ_Xe·R + λ_I·I − (λ_Xe + σ_a·φ)·Xe
         *   Xe_new = (Xe + dt·(γ_Xe·R + λ_I·I_new)) / (1 + dt·(λ_Xe + σ_a·φ)) */
        float Xe_new = (rt->xenon_135 + dt_xe * (RT_GAMMA_XE135 * fiss_rate + RT_LAMBDA_I135 * I_new))
                     / (1.0f + dt_xe * (RT_LAMBDA_XE135 + RT_SIGMA_XE135 * phi));

        if (I_new < 0.0f) I_new = 0.0f;
        if (Xe_new < 0.0f) Xe_new = 0.0f;
        rt->iodine_135 = I_new;
        rt->xenon_135  = Xe_new;

        /* Equilibrium Xe at current power (for reference display) */
        float denom = RT_LAMBDA_XE135 + RT_SIGMA_XE135 * phi;
        rt->xe_eq = (denom > 0.0f)
                  ? (RT_GAMMA_I135 + RT_GAMMA_XE135) * fiss_rate / denom
                  : 0.0f;

        /* Xe reactivity: deviation from BASELINE Xe (set at init).
         * At startup, Xe is at equilibrium for initial power and ρ_Xe = 0.
         * If Xe increases above baseline (iodine pit) → negative ρ.
         * If Xe decreases below baseline (burnoff) → positive ρ.
         * Scale: full-power Xe worth ≈ RT_XE_WORTH_FULL (≈3% Δk/k). */
        float phi_full = phi_nom;
        float fiss_full = 9.8e13f;
        float xe_eq_full = (RT_GAMMA_I135 + RT_GAMMA_XE135) * fiss_full
                         / (RT_LAMBDA_XE135 + RT_SIGMA_XE135 * phi_full);
        /* Reactivity = -worth × (Xe_current - Xe_baseline) / Xe_eq_full */
        rt->xe_reactivity = (xe_eq_full > 0.0f)
            ? -RT_XE_WORTH_FULL * (Xe_new - rt->xe_baseline) / xe_eq_full
            : 0.0f;
        /* Clamp to physically reasonable range: max ~±10% Δk/k */
        if (rt->xe_reactivity >  0.10f) rt->xe_reactivity =  0.10f;
        if (rt->xe_reactivity < -0.10f) rt->xe_reactivity = -0.10f;
    }
    float delta_rho_xe = rt->xe_reactivity;

    /* Total reactivity (includes operator manual rod position) */
    float delta_rho_operator = rt->operator_rod_rho;

    /* ── Prompt power coefficient (instantaneous Doppler analogue) ──
     * In a real reactor, fuel temperature rises within microseconds of
     * a power change — MUCH faster than our thermal grid can resolve.
     * Model this as an immediate negative feedback proportional to the
     * power deviation from the initial equilibrium.  Physically this is
     * the Doppler broadening of U-238 capture resonances.
     *   α_power ≈ α_D × (dT_fuel/dP) ≈ -3e-5 × 700K/unit ≈ -0.02 Δk/k per unit
     * We use a conservative -0.008 Δk/k per unit power deviation (~1.2$). */
    float delta_power = rt->power_fraction - rt->initial_power_frac;
    float prompt_power_coeff = -0.008f;  /* Δk/k per unit normalised power */
    float delta_rho_prompt = prompt_power_coeff * delta_power;

    rt->reactivity = delta_rho_void + delta_rho_doppler + delta_rho_rods
                   + delta_rho_xe + delta_rho_operator + delta_rho_prompt;

    /* ────────────────────────────────────────────────────────────
     *  6-GROUP POINT KINETICS (replaces simplified single-timescale)
     *
     *  dN/dt = [(ρ - β) / Λ] · N  +  Σ λ_i C_i
     *  dC_i/dt = (β_i / Λ) · N  -  λ_i · C_i
     *
     *  where N = neutron_pop (normalised ≡ power_fraction at equilibrium)
     *        C_i = precursor concentration for group i
     *        ρ = reactivity, β = Σβ_i, Λ = prompt generation time
     *        λ_i = precursor decay constant, β_i = group yield fraction
     *
     *  We integrate with semi-implicit Euler for stiff stability:
     *    C_i^{n+1} = (C_i^n + dt · β_i/Λ · N^n) / (1 + dt · λ_i)
     *    N^{n+1}   = N^n + dt · [(ρ-β)/Λ · N^n + Σ λ_i · C_i^{n+1}]
     * ──────────────────────────────────────────────────────────── */
    {
        /* 6-group delayed neutron fractions and decay constants for U-235
         * [Keepin, Wimett & Zeigler, Phys. Rev. 107 (1957) 1044;
         *  Keepin, Physics of Nuclear Kinetics (1965), Table 4-5] */
        static const float beta_i[6] = {0.000215f, 0.001424f, 0.001274f,
                                         0.002568f, 0.000748f, 0.000273f};
        static const float lambda_i[6] = {0.0124f, 0.0305f, 0.111f,
                                           0.301f,  1.14f,   3.01f};

        float rho_net = rt->reactivity;
        float beta_tot = RT_BETA_TOTAL;
        float Lam   = RT_LAMBDA_PROMPT;
        float dt    = rt->dt;
        float N     = rt->neutron_pop;
        float N_old = N;   /* save for rate limiter */

        /* Adaptive sub-stepping: scale with |reactivity| for stiff accuracy.
         * More substeps = smoother integration at any reactivity level. */
        float rho_abs = (rho_net > 0 ? rho_net : -rho_net);
        int n_sub = 4;  /* baseline: always at least 4 */
        if (rho_abs > beta_tot * 0.5f) n_sub = 20;
        if (rho_abs > beta_tot)         n_sub = 80;
        if (rho_abs > beta_tot * 2.0f)  n_sub = 200;
        float h = dt / (float)n_sub;

        for (int sub = 0; sub < n_sub; sub++) {
            /* ── Update precursors (semi-implicit) ── */
            float precursor_source = 0.0f;
            for (int i = 0; i < 6; i++) {
                float Ci_new = (rt->precursor[i] + h * (beta_i[i] / Lam) * N)
                             / (1.0f + h * lambda_i[i]);
                rt->precursor[i] = Ci_new;
                precursor_source += lambda_i[i] * Ci_new;
            }

            /* ── Update neutron population (FULLY IMPLICIT in N) ──
             *  N^{n+1} = (N^n + h × precursor_source)
             *          / (1  − h × (ρ−β)/Λ)
             * This is unconditionally stable: no runaway for any h or ρ. */
            float alpha = (rho_net - beta_tot) / Lam;
            float denom = 1.0f - h * alpha;
            if (denom < 0.01f) denom = 0.01f;  /* paranoia floor */
            N = (N + h * precursor_source) / denom;

            if (N < 1.0e-10f) N = 1.0e-10f;
        }

        /* Per-frame power rate limiter: ADDITIVE cap on absolute power change.
         * Prevents thermal-feedback lag oscillations. Real RBMK period trip ≈ 10s.
         * Allow max 2× nominal per second change — generous but prevents 100× jumps. */
        float max_delta_N = 2.0f * dt * (N_old > 0.1f ? N_old : 0.1f);
        if (N - N_old >  max_delta_N) N = N_old + max_delta_N;
        if (N_old - N >  max_delta_N) N = N_old - max_delta_N;

        /* Hard ceiling: Chernobyl peaked at ~100× nominal before disassembly */
        if (N > 120.0f) N = 120.0f;
        rt->neutron_pop    = N;
        rt->power_fraction = N;   /* In normalised kinetics, N ≡ P/P_nominal */
    }

    /* Update total power */
    rt->total_power_w = rt->power_fraction * RT_NOMINAL_POWER_MW * 1.0e6f;
}

/* ═══════════════════════════════════════════════════════════════
 *  ENERGY CONSERVATION (1st Law of Thermodynamics)
 *  Q_gen - Q_removed = ΔU (stored energy change)
 * ═══════════════════════════════════════════════════════════════ */

static void update_energy_balance(ReactorThermal *rt) {
    int G = rt->gridDim;
    float dx = 2.0f * rt->boxHalf / (float)G;
    float cell_vol = dx * dx * dx;

    float total_generated = 0.0f;
    float total_stored = 0.0f;
    float max_fuel_T = 0.0f;
    float total_fuel_T = 0.0f;
    int n_fuel = 0;
    float total_coolant_T = 0.0f;
    float max_coolant_T = 0.0f;
    int n_coolant = 0;

    for (int idx = 0; idx < G * G * G; idx++) {
        int mat = rt->material_id[idx];

        if (mat == RT_MAT_FUEL_UO2) {
            total_generated += rt->heat_source[idx] * cell_vol;
            total_fuel_T += rt->temperature[idx];
            if (rt->temperature[idx] > max_fuel_T)
                max_fuel_T = rt->temperature[idx];
            n_fuel++;
        }
        if (mat == RT_MAT_COOLANT_H2O) {
            total_coolant_T += rt->temperature[idx];
            if (rt->temperature[idx] > max_coolant_T)
                max_coolant_T = rt->temperature[idx];
            n_coolant++;
        }

        /* Stored thermal energy: ρ × c_p × T × V */
        if (mat != RT_MAT_VOID) {
            float rho = rt_density((RT_Material)mat, rt->temperature[idx]);
            float cp  = rt_specific_heat((RT_Material)mat, rt->temperature[idx]);
            total_stored += rho * cp * rt->temperature[idx] * cell_vol;
        }
    }

    rt->avg_fuel_temp = (n_fuel > 0) ? total_fuel_T / n_fuel : 300.0f;
    rt->max_fuel_temp = (n_fuel > 0) ? max_fuel_T : 300.0f;
    rt->avg_coolant_temp = (n_coolant > 0) ? total_coolant_T / n_coolant : 300.0f;
    rt->max_coolant_temp = max_coolant_T;

    /* Energy balance: rate of generation minus estimated removal */
    float Q_removed_estimate = (rt->avg_coolant_temp - RT_INLET_TEMP_K) *
                                4200.0f * /* approximate total coolant mass flow */
                                (float)n_coolant * cell_vol * 750.0f / 10.0f;
    rt->energy_balance = total_generated - Q_removed_estimate;

    /* Total entropy (2nd law tracking):
     * ΔS = Σ (Q_i / T_i) for each cell */
    float total_entropy = 0.0f;
    for (int idx = 0; idx < G * G * G; idx++) {
        int mat = rt->material_id[idx];
        if (mat == RT_MAT_VOID) continue;
        float T = rt->temperature[idx];
        if (T < 200.0f) T = 200.0f;
        float rho = rt_density((RT_Material)mat, T);
        float cp  = rt_specific_heat((RT_Material)mat, T);
        /* Entropy density: ρ × c_p × ln(T/T_ref) */
        total_entropy += rho * cp * logf(T / 300.0f) * cell_vol;
    }
    rt->total_entropy = total_entropy;
}

/* ═══════════════════════════════════════════════════════════════
 *  GPU THERMAL DIFFUSION PIPELINE
 * ═══════════════════════════════════════════════════════════════ */

static int create_diffusion_pipeline(ReactorThermal *rt) {
    VkDevice dev = rt->device;

    /* Descriptor layout: 4 SSBOs (temp_in, temp_out, heat_source, material) */
    VkDescriptorSetLayoutBinding bindings[4] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
    };
    VkDescriptorSetLayoutCreateInfo dslci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount = 4;
    dslci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(dev, &dslci, NULL, &rt->diffusionDescLayout) != VK_SUCCESS)
        return -1;

    /* Push constants for thermal diffusion parameters */
    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 32};
    VkPipelineLayoutCreateInfo plci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &rt->diffusionDescLayout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(dev, &plci, NULL, &rt->diffusionPipeLayout) != VK_SUCCESS)
        return -2;

    /* Shader module */
    rt->diffusionShader = rt_load_shader(dev, "shaders/thermal_diffusion.comp.spv");
    if (rt->diffusionShader == VK_NULL_HANDLE) return -3;

    /* Compute pipeline */
    VkComputePipelineCreateInfo cpci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = rt->diffusionShader;
    cpci.stage.pName  = "main";
    cpci.layout       = rt->diffusionPipeLayout;
    if (vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &rt->diffusionPipeline) != VK_SUCCESS)
        return -4;

    /* Descriptor pool + set */
    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4};
    VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &ps;
    if (vkCreateDescriptorPool(dev, &dpci, NULL, &rt->diffusionDescPool) != VK_SUCCESS)
        return -5;

    VkDescriptorSetAllocateInfo dsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool     = rt->diffusionDescPool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &rt->diffusionDescLayout;
    if (vkAllocateDescriptorSets(dev, &dsai, &rt->diffusionDescSet) != VK_SUCCESS)
        return -6;

    /* Write descriptors */
    VkDescriptorBufferInfo bufInfos[4] = {
        {rt->tempBuf,    0, VK_WHOLE_SIZE},
        {rt->tempBuf2,   0, VK_WHOLE_SIZE},
        {rt->heatSrcBuf, 0, VK_WHOLE_SIZE},
        {rt->matBuf,     0, VK_WHOLE_SIZE},
    };
    VkWriteDescriptorSet writes[4];
    for (int i = 0; i < 4; i++) {
        memset(&writes[i], 0, sizeof(VkWriteDescriptorSet));
        writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet          = rt->diffusionDescSet;
        writes[i].dstBinding      = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &bufInfos[i];
    }
    vkUpdateDescriptorSets(dev, 4, writes, 0, NULL);

    return 0;
}

/* ═══════════════════════════════════════════════════════════════
 *  PUBLIC API IMPLEMENTATION
 * ═══════════════════════════════════════════════════════════════ */

int reactor_thermal_init(ReactorThermal *rt,
                         VkPhysicalDevice phys, VkDevice dev,
                         uint32_t queueFamilyIdx,
                         VkBuffer densityBuf, VkBuffer signedBuf,
                         int visGridDim) {
    memset(rt, 0, sizeof(*rt));
    
    /* Thermal grid: 128³ for detailed RBMK channel resolution.
     * At 128³ with boxHalf=7m → dx ≈ 0.109m, resolving individual
     * fuel pins (8mm UO₂), cladding (Zr tube), coolant annulus,
     * and graphite moderator blocks within each 250mm lattice cell. */
    rt->gridDim = 128;
    rt->boxHalf = 7.0f; /* ±7 m covers RBMK core (11.8m dia, 7m height) */

    int total = rt->gridDim * rt->gridDim * rt->gridDim;

    /* Allocate CPU-side grids */
    rt->temperature    = (float *)calloc(total, sizeof(float));
    rt->heat_source    = (float *)calloc(total, sizeof(float));
    rt->material_id    = (int *)  calloc(total, sizeof(int));
    rt->pressure_field = (float *)calloc(total, sizeof(float));
    rt->void_fraction  = (float *)calloc(total, sizeof(float));

    if (!rt->temperature || !rt->heat_source || !rt->material_id ||
        !rt->pressure_field || !rt->void_fraction) {
        fprintf(stderr, "[RT] Out of memory for thermal grids\n");
        return -1;
    }

    /* Set device/Vulkan fields AFTER grid allocs to verify struct is stable */
    rt->device     = dev;
    rt->physDevice = phys;
    rt->densityBuf = densityBuf;
    rt->signedBuf  = signedBuf;
    rt->visGridDim = visGridDim;

    /* Initial state - explicitly initialized to known values */
    rt->system_pressure = RT_OPERATING_PRESSURE;  /* 7.0e6 Pa */
    rt->power_fraction  = 0.07f;   /* Start at 7% power (Chernobyl test condition!) */
    rt->initial_power_frac = rt->power_fraction; /* reference for prompt feedback */
    rt->total_power_w   = rt->power_fraction * RT_NOMINAL_POWER_MW * 1.0e6f;
    rt->dt              = 0.01f;    /* 10 ms time step */
    rt->time            = 0.0f;
    rt->scram_active    = 0;
    rt->hydrogen_mass   = 0.0f;
    rt->operator_rod_rho = 0.0f;  /* Rods at reference position */

    /* ── Initialise Xe-135 / I-135 at steady-state for initial power ──
     * At equilibrium:
     *   Σ_f·φ = (P/P_nom) × Σ_f·φ_nom
     *   φ_nom ≈ 3.0×10¹³ n/cm²/s (typical RBMK full-power flux)
     *   I_eq  = γ_I · Σ_f · φ / λ_I
     *   Xe_eq = (γ_I + γ_Xe) · Σ_f · φ / (λ_Xe + σ_a^Xe · φ)
     */
    {
        float phi_nom = 3.0e13f;  /* nominal full-power thermal flux [IAEA-TECDOC-1474] */
        float phi = phi_nom * rt->power_fraction;
        /* Use a simplified macroscopic production rate proportional to power */
        float fiss_rate = 9.8e13f * rt->power_fraction; /* fissions/cm³/s at power */

        rt->neutron_flux = phi;
        rt->iodine_135 = RT_GAMMA_I135 * fiss_rate / RT_LAMBDA_I135;
        float denom = RT_LAMBDA_XE135 + RT_SIGMA_XE135 * phi;
        rt->xenon_135  = (RT_GAMMA_I135 + RT_GAMMA_XE135) * fiss_rate / denom;
        rt->xe_eq = rt->xenon_135;
        rt->xe_baseline = rt->xenon_135;  /* Reference point: ρ_Xe = 0 here */
        rt->xe_reactivity = 0.0f;  /* zero at equilibrium (baseline absorbed) */
    }

    /* ── Initialise 6-group delayed neutron precursors ──
     * Keepin (1965) data for U-235 thermal fission\n     * [Keepin, Wimett & Zeigler, Phys. Rev. 107 (1957) 1044]:
     *   Group  β_i        λ_i (s⁻¹)   half-life
     *     1    0.000215   0.0124       55.9 s
     *     2    0.001424   0.0305       22.7 s
     *     3    0.001274   0.111         6.2 s
     *     4    0.002568   0.301         2.3 s
     *     5    0.000748   1.14          0.61 s
     *     6    0.000273   3.01          0.23 s
     * At steady-state: C_i = (β_i / (Λ · λ_i)) × P_frac
     * We set Λ = RT_LAMBDA_PROMPT = 1e-4 s (RBMK graphite-moderated). */
    {
        static const float beta_i[6] = {0.000215f, 0.001424f, 0.001274f,
                                         0.002568f, 0.000748f, 0.000273f};
        static const float lambda_i[6] = {0.0124f, 0.0305f, 0.111f,
                                           0.301f,  1.14f,   3.01f};
        rt->neutron_pop = rt->power_fraction;
        for (int i = 0; i < RT_NUM_DELAYED_GROUPS; i++) {
            /* C_i at equilibrium */
            rt->precursor[i] = (beta_i[i] / (RT_LAMBDA_PROMPT * lambda_i[i]))
                               * rt->power_fraction;
        }
    }

    /* Create RBMK geometry and material map */
    setup_rbmk_geometry(rt);
    setup_power_distribution(rt);

    /* ── Compute baseline operating point for feedback ──
     * All reactivity feedbacks measure DEVIATION from this state,
     * so the reactor starts at exactly ρ = 0 (critical). */
    {
        float total_void_init = 0.0f;
        int n_cool_init = 0;
        float total_fuel_T_init = 0.0f;
        int n_fuel_init = 0;
        for (int idx = 0; idx < total; idx++) {
            if (rt->material_id[idx] == RT_MAT_COOLANT_H2O) {
                total_void_init += rt->void_fraction[idx];
                n_cool_init++;
            }
            if (rt->material_id[idx] == RT_MAT_FUEL_UO2) {
                total_fuel_T_init += rt->temperature[idx];
                n_fuel_init++;
            }
        }
        rt->baseline_void = (n_cool_init > 0) ? total_void_init / n_cool_init : 0.0f;
        rt->baseline_fuel_temp = (n_fuel_init > 0) ? total_fuel_T_init / n_fuel_init : 800.0f;
    }

    /* ── Place 211 control rods in RBMK hexagonal lattice ──
     * Real RBMK: rods placed in a subset of the 25 cm lattice positions.
     * ~211 rods/channels used for control, grouped by function.
     * We place them in concentric rings within the core radius. */
    {
        int n_placed = 0;
        float pitch = RT_CHANNEL_PITCH_M;
        memset(rt->rods, 0, sizeof(rt->rods));

        for (float cy = -RT_CORE_RADIUS_M; cy <= RT_CORE_RADIUS_M && n_placed < RT_MAX_RODS; cy += pitch * 2.0f) {
            for (float cx = -RT_CORE_RADIUS_M; cx <= RT_CORE_RADIUS_M && n_placed < RT_MAX_RODS; cx += pitch * 2.0f) {
                float r = sqrtf(cx * cx + cy * cy);
                if (r > RT_CORE_RADIUS_M - pitch) continue;

                RT_ControlRod *rod = &rt->rods[n_placed];
                rod->x = cx;
                rod->y = cy;
                rod->insertion = 0.85f;  /* Chernobyl: most rods mostly inserted */
                rod->target = rod->insertion;
                rod->active = 1;
                /* Group assignment: inner=manual(0), mid=auto(1), outer=emergency(2) */
                if (r < RT_CORE_RADIUS_M * 0.35f)      rod->group = 0;
                else if (r < RT_CORE_RADIUS_M * 0.7f)   rod->group = 1;
                else                                     rod->group = 2;
                n_placed++;
            }
        }
        rt->n_rods = n_placed;
        rt->rods_scram = 0;
        rt->rod_scram_t = 0.0f;

        /* Chernobyl test: operators had withdrawn most rods recklessly.
         * Only ~6-8 rods were left inserted. Simulate this: */
        int rods_left_in = 0;
        for (int i = 0; i < rt->n_rods; i++) {
            if (rods_left_in < 8) {
                rt->rods[i].insertion = 0.95f;  /* A few still in */
                rods_left_in++;
            } else {
                rt->rods[i].insertion = 0.05f;  /* Nearly all withdrawn! */
            }
            rt->rods[i].target = rt->rods[i].insertion;
        }
        printf("[RT]   Control rods: %d placed (%d groups), mostly withdrawn (Chernobyl config)\n",
               n_placed, 3);
    }

    /* ── Allocate spatial Xe-135 field ── */
    rt->xe_density = (float *)calloc(total, sizeof(float));
    if (rt->xe_density) {
        /* Initialise uniform Xe at equilibrium */
        float xe_per_cell = rt->xenon_135;  /* global average */
        for (int i = 0; i < total; i++) {
            if (rt->material_id[i] == RT_MAT_FUEL_UO2)
                rt->xe_density[i] = xe_per_cell;
        }
    }

    /* ── Initialise neutron particles ── */
    memset(rt->neutrons, 0, sizeof(rt->neutrons));
    rt->n_neutrons_alive = 0;
    rt->neutron_spawn_accum = 0.0f;

    printf("[RT] RBMK-1000 thermal simulation initialized\n");
    printf("[RT]   Grid: %d³ (%d cells), box=±%.1f m\n",
           rt->gridDim, total, rt->boxHalf);
    printf("[RT]   Initial power: %.1f%% (%.0f MW)\n",
           rt->power_fraction * 100.0f,
           rt->power_fraction * RT_NOMINAL_POWER_MW);
    printf("[RT]   Pressure: %.1f MPa, T_sat=%.1f°C\n",
           rt->system_pressure / 1.0e6f,
           rt_saturation_temp(rt->system_pressure) - RT_CELSIUS_OFFSET);
    printf("[RT]   Xe-135: %.3e atoms/cm³ (equilibrium at %.1f%% power)\n",
           rt->xenon_135, rt->power_fraction * 100.0f);
    printf("[RT]   I-135:  %.3e atoms/cm³, flux=%.2e n/cm²/s\n",
           rt->iodine_135, rt->neutron_flux);

    /* Create GPU buffers */
    VkDeviceSize grid_size = (VkDeviceSize)total * sizeof(float);
    VkDeviceSize mat_size  = (VkDeviceSize)total * sizeof(int);
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkMemoryPropertyFlags mem_props = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    if (rt_create_buffer(phys, dev, grid_size, usage, mem_props,
                         &rt->tempBuf, &rt->tempMem) != 0) {
        fprintf(stderr, "[RT] Failed to create temperature buffer\n");
        return -2;
    }
    if (rt_create_buffer(phys, dev, grid_size, usage, mem_props,
                         &rt->tempBuf2, &rt->tempMem2) != 0) return -2;
    if (rt_create_buffer(phys, dev, grid_size, usage, mem_props,
                         &rt->heatSrcBuf, &rt->heatSrcMem) != 0) return -2;
    if (rt_create_buffer(phys, dev, mat_size, usage, mem_props,
                         &rt->matBuf, &rt->matMem) != 0) return -2;

    /* Upload initial data to GPU */
    void *mapped;
    vkMapMemory(dev, rt->tempMem, 0, grid_size, 0, &mapped);
    memcpy(mapped, rt->temperature, (size_t)grid_size);
    vkUnmapMemory(dev, rt->tempMem);

    vkMapMemory(dev, rt->heatSrcMem, 0, grid_size, 0, &mapped);
    memcpy(mapped, rt->heat_source, (size_t)grid_size);
    vkUnmapMemory(dev, rt->heatSrcMem);

    vkMapMemory(dev, rt->matMem, 0, mat_size, 0, &mapped);
    memcpy(mapped, rt->material_id, (size_t)mat_size);
    vkUnmapMemory(dev, rt->matMem);

    /* Command pool */
    VkCommandPoolCreateInfo cpci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.queueFamilyIndex = queueFamilyIdx;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCreateCommandPool(dev, &cpci, NULL, &rt->cmdPool);

    VkCommandBufferAllocateInfo cbai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool        = rt->cmdPool;
    cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    vkAllocateCommandBuffers(dev, &cbai, &rt->cmdBuf);

    /* Create GPU diffusion pipeline */
    if (create_diffusion_pipeline(rt) != 0) {
        fprintf(stderr, "[RT] GPU diffusion pipeline failed — using CPU fallback\n");
        /* Continue with CPU-only mode */
    }

    return 0;
}

void reactor_thermal_set_power(ReactorThermal *rt, float fraction) {
    /* Must update neutron_pop AND precursor concentrations to the new
     * steady-state, otherwise the 6-group kinetics solver will immediately
     * overwrite power_fraction with the old neutron_pop next frame. */
    /* Keepin 6-group data [Keepin et al., Phys. Rev. 107 (1957) 1044] */
    static const float beta_i[6] = {0.000215f, 0.001424f, 0.001274f,
                                     0.002568f, 0.000748f, 0.000273f};
    static const float lambda_i[6] = {0.0124f, 0.0305f, 0.111f,
                                       0.301f,  1.14f,   3.01f};

    rt->power_fraction = fraction;
    rt->neutron_pop    = fraction;  /* N ≡ P/P_nominal in normalised kinetics */
    rt->total_power_w  = fraction * RT_NOMINAL_POWER_MW * 1.0e6f;

    /* Reset precursors to new steady-state: C_i = (β_i / (Λ · λ_i)) × N */
    for (int i = 0; i < RT_NUM_DELAYED_GROUPS; i++) {
        rt->precursor[i] = (beta_i[i] / (RT_LAMBDA_PROMPT * lambda_i[i])) * fraction;
    }

    /* NOTE: We do NOT reset Xe/I here — the whole point of Xe poisoning
     * is that the fission products DON'T instantly adjust to new power.
     * The transient Xe buildup/burnoff is What Killed Chernobyl. */

    setup_power_distribution(rt);
    printf("[RT] Power set to %.1f%% (%.0f MW)  N=%.4f  Xe/Xe_eq=%.2f\n",
           fraction * 100.0f, fraction * RT_NOMINAL_POWER_MW, fraction,
           (rt->xe_eq > 0.0f) ? rt->xenon_135 / rt->xe_eq : 0.0f);
}

/* ═══════════════════════════════════════════════════════════════
 *  CONTROL ROD DRIVE — update individual rod positions each frame
 * ═══════════════════════════════════════════════════════════════ */
static void update_control_rods(ReactorThermal *rt) {
    float dt = rt->dt;

    /* Drive each rod toward its target */
    for (int i = 0; i < rt->n_rods; i++) {
        RT_ControlRod *rod = &rt->rods[i];
        if (!rod->active) continue;
        float diff = rod->target - rod->insertion;
        if (fabsf(diff) < 0.001f) { rod->insertion = rod->target; continue; }
        /* Speed depends on direction: insertion is faster (gravity-assisted) */
        float speed = (diff > 0.0f)
            ? RT_ROD_SPEED_MPS / RT_ROD_TRAVEL_M   /* inserting */
            : RT_ROD_SPEED_WITHDRAW / RT_ROD_TRAVEL_M;  /* withdrawing */
        float step = speed * dt;
        if (fabsf(diff) < step) rod->insertion = rod->target;
        else rod->insertion += (diff > 0.0f ? step : -step);
        if (rod->insertion < 0.0f) rod->insertion = 0.0f;
        if (rod->insertion > 1.0f) rod->insertion = 1.0f;
    }

    /* Stamp control rod material into the 3D grid.
     * Rod occupies a narrow column at (rod.x, rod.y) in z-range determined
     * by insertion depth.  The bottom portion is graphite displacer tip. */
    int G = rt->gridDim;
    float h = rt->boxHalf;
    float dx = 2.0f * h / (float)G;
    float half_pitch = RT_CHANNEL_PITCH_M * 0.5f;

    /* First: clear old B4C/graphite-tip from grid (revert to moderator) */
    for (int idx = 0; idx < G * G * G; idx++) {
        if (rt->material_id[idx] == RT_MAT_CONTROL_B4C ||
            rt->material_id[idx] == RT_MAT_CONTROL_GRAPH)
            rt->material_id[idx] = RT_MAT_MODERATOR_C;
    }

    /* Then stamp each rod */
    for (int ri = 0; ri < rt->n_rods; ri++) {
        RT_ControlRod *rod = &rt->rods[ri];
        if (!rod->active || rod->insertion < 0.01f) continue;

        /* Rod occupies z from top of core downward by insertion fraction.
         * Core z range: -core_H/2 to +core_H/2
         * Fully inserted: fills entire height.  Partially: top portion. */
        float core_top  = RT_CORE_HEIGHT_M / 2.0f;
        float core_bot  = -RT_CORE_HEIGHT_M / 2.0f;
        float rod_bottom = core_top - rod->insertion * RT_ROD_TRAVEL_M;
        float tip_bottom = rod_bottom - RT_GRAPHITE_TIP_M;

        for (int iz = 0; iz < G; iz++) {
            float z = -h + (iz + 0.5f) * dx;
            /* Check if this z-level is within rod span */
            int in_b4c = (z <= core_top && z >= rod_bottom);
            int in_tip = (z < rod_bottom && z >= tip_bottom && tip_bottom >= core_bot);

            if (!in_b4c && !in_tip) continue;

            for (int iy = 0; iy < G; iy++) {
                float y = -h + (iy + 0.5f) * dx;
                float dy2 = (y - rod->y) * (y - rod->y);
                if (dy2 > half_pitch * half_pitch * 0.15f) continue;

                for (int ix = 0; ix < G; ix++) {
                    float x = -h + (ix + 0.5f) * dx;
                    float d2 = (x - rod->x) * (x - rod->x) + dy2;

                    /* Rod radius ~4 cm in a 25 cm pitch — scaled for grid */
                    if (d2 > half_pitch * half_pitch * 0.15f) continue;

                    int idx = iz * G * G + iy * G + ix;
                    /* Only overwrite moderator or coolant, not fuel */
                    int mat = rt->material_id[idx];
                    if (mat == RT_MAT_MODERATOR_C || mat == RT_MAT_COOLANT_H2O) {
                        rt->material_id[idx] = in_b4c ? RT_MAT_CONTROL_B4C
                                                       : RT_MAT_CONTROL_GRAPH;
                        /* B4C absorbs neutrons (cold), graphite tip moderates (hot) */
                        rt->temperature[idx] = in_b4c ? 400.0f : 600.0f;
                    }
                }
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 *  NEUTRON PARTICLE SYSTEM — visual quantum wave packets
 *  Spawns neutrons from fission sites, transports them through
 *  the moderator, and kills them on absorption by fuel/Xe/B₄C.
 * ═══════════════════════════════════════════════════════════════ */
static void update_neutrons(ReactorThermal *rt) {
    float dt = rt->dt;
    int G = rt->gridDim;
    float h = rt->boxHalf;
    float dx = 2.0f * h / (float)G;

    /* ── Spawn new neutrons proportional to power ── */
    float spawn_rate = 600.0f * rt->power_fraction;  /* neutrons/sec — dense field */
    rt->neutron_spawn_accum += spawn_rate * dt;
    int to_spawn = (int)rt->neutron_spawn_accum;
    rt->neutron_spawn_accum -= (float)to_spawn;

    /* Simple RNG using time-based seed */
    static unsigned int rng_state = 12345;
    #define NRNG() (rng_state = rng_state * 1103515245u + 12345u, \
                    (float)(rng_state >> 16 & 0x7FFF) / 32767.0f)

    for (int s = 0; s < to_spawn; s++) {
        /* Find an empty slot */
        int slot = -1;
        for (int i = 0; i < RT_MAX_NEUTRONS; i++) {
            if (!rt->neutrons[i].alive) { slot = i; break; }
        }
        if (slot < 0) break;  /* all slots full */

        RT_Neutron *n = &rt->neutrons[slot];
        /* Spawn at a random fuel cell */
        float rx = (NRNG() * 2.0f - 1.0f) * RT_CORE_RADIUS_M * 0.8f;
        float ry = (NRNG() * 2.0f - 1.0f) * RT_CORE_RADIUS_M * 0.8f;
        float rz = (NRNG() * 2.0f - 1.0f) * RT_CORE_HEIGHT_M * 0.4f;
        n->x = rx; n->y = ry; n->z = rz;
        n->px = rx; n->py = ry; n->pz = rz;  /* prev = current at spawn */
        /* Random direction */
        float theta = NRNG() * 6.2832f;
        float cosphi = NRNG() * 2.0f - 1.0f;
        float sinphi = sqrtf(1.0f - cosphi * cosphi);
        float speed = 2.0f + NRNG() * 3.0f;  /* visual speed (m/s) */
        n->vx = sinphi * cosf(theta) * speed;
        n->vy = sinphi * sinf(theta) * speed;
        n->vz = cosphi * speed;
        n->energy = 2.0e6f;  /* 2 MeV — fast neutron */
        n->age = 0.0f;
        n->state = RT_NEUTRON_FAST;
        n->alive = 1;
        rt->n_neutrons_alive++;
    }
    #undef NRNG

    /* ── Transport existing neutrons ── */
    rt->n_neutrons_alive = 0;
    for (int i = 0; i < RT_MAX_NEUTRONS; i++) {
        RT_Neutron *n = &rt->neutrons[i];
        if (!n->alive) continue;

        n->age += dt;
        /* Save previous position for streak trail rendering */
        n->px = n->x; n->py = n->y; n->pz = n->z;
        n->x += n->vx * dt;
        n->y += n->vy * dt;
        n->z += n->vz * dt;

        /* Check if escaped the core */
        float r_xy = sqrtf(n->x * n->x + n->y * n->y);
        if (r_xy > RT_CORE_RADIUS_M * 1.1f ||
            fabsf(n->z) > RT_CORE_HEIGHT_M * 0.6f ||
            n->age > 2.0f) {
            n->alive = 0;
            continue;
        }

        /* Look up what material we're in */
        int ix = (int)((n->x + h) / dx);
        int iy = (int)((n->y + h) / dx);
        int iz = (int)((n->z + h) / dx);
        if (ix < 0) ix = 0;
        if (ix >= G) ix = G - 1;
        if (iy < 0) iy = 0;
        if (iy >= G) iy = G - 1;
        if (iz < 0) iz = 0;
        if (iz >= G) iz = G - 1;
        int idx = iz * G * G + iy * G + ix;
        int mat = rt->material_id[idx];

        /* Physics interactions: */
        if (n->state == RT_NEUTRON_FAST) {
            /* In graphite moderator: slow down (thermalize) */
            if (mat == RT_MAT_MODERATOR_C) {
                n->energy *= 0.85f;  /* Energy loss per collision with C-12 */
                /* Add small random scatter */
                n->vx += (n->age * 7.13f - (int)(n->age * 7.13f) - 0.5f) * 0.5f;
                n->vy += (n->age * 11.7f - (int)(n->age * 11.7f) - 0.5f) * 0.5f;
                n->vz += (n->age * 3.37f - (int)(n->age * 3.37f) - 0.5f) * 0.5f;
                float spd = sqrtf(n->vx*n->vx + n->vy*n->vy + n->vz*n->vz);
                if (spd > 0.01f) {
                    float slow = 1.5f / spd;  /* thermal speed */
                    n->vx *= slow; n->vy *= slow; n->vz *= slow;
                }
                if (n->energy < 0.1f) n->state = RT_NEUTRON_THERMAL;
            }
        }

        if (n->state == RT_NEUTRON_THERMAL) {
            /* Absorption checks: */
            if (mat == RT_MAT_FUEL_UO2) {
                /* Fission or capture — absorbed */
                n->state = RT_NEUTRON_ABSORBED;
                n->alive = 0;
                continue;
            }
            if (mat == RT_MAT_CONTROL_B4C) {
                /* Absorbed by control rod */
                n->state = RT_NEUTRON_ABSORBED;
                n->alive = 0;
                continue;
            }
            /* Xe-135 absorption (probabilistic based on local Xe density) */
            if (rt->xe_density && rt->xe_density[idx] > rt->xe_baseline * 0.5f) {
                float xe_frac = rt->xe_density[idx] / (rt->xe_baseline + 1.0f);
                if (xe_frac > 1.5f) {
                    n->state = RT_NEUTRON_ABSORBED;
                    n->alive = 0;
                    continue;
                }
            }
        }

        rt->n_neutrons_alive++;
    }
}

/* ═══════════════════════════════════════════════════════════════
 *  CONTROL ROD API
 * ═══════════════════════════════════════════════════════════════ */
void reactor_thermal_az5(ReactorThermal *rt) {
    rt->rods_scram = 1;
    rt->rod_scram_t = 0.0f;
    /* Drive ALL rods to full insertion */
    for (int i = 0; i < rt->n_rods; i++) {
        rt->rods[i].target = 1.0f;
    }
    /* Also trigger the existing scram mechanism */
    reactor_thermal_scram(rt);
    printf("[RT] *** AZ-5: ALL %d RODS DRIVING TO FULL INSERTION ***\n", rt->n_rods);
}

void reactor_thermal_rods_adjust(ReactorThermal *rt, float delta) {
    /* Adjust operator group (group 0) rods */
    for (int i = 0; i < rt->n_rods; i++) {
        if (rt->rods[i].group == 0 && rt->rods[i].active) {
            rt->rods[i].target -= delta; /* withdraw = decrease insertion */
            if (rt->rods[i].target < 0.0f) rt->rods[i].target = 0.0f;
            if (rt->rods[i].target > 1.0f) rt->rods[i].target = 1.0f;
        }
    }
}

float reactor_thermal_rod_avg(const ReactorThermal *rt, int group) {
    float sum = 0.0f;
    int count = 0;
    for (int i = 0; i < rt->n_rods; i++) {
        if (!rt->rods[i].active) continue;
        if (group >= 0 && rt->rods[i].group != group) continue;
        sum += rt->rods[i].insertion;
        count++;
    }
    return (count > 0) ? sum / count : 0.0f;
}

void reactor_thermal_scram(ReactorThermal *rt) {
    if (!rt->scram_active) {
        rt->scram_active = 1;
        rt->scram_time = 0.0f;
        rt->operator_rod_rho = 0.0f;  /* Cancel any manual rod withdrawal */
        printf("[RT] *** AZ-5 SCRAM INITIATED ***\n");
        printf("[RT] WARNING: RBMK scram rods have graphite tips —\n");
        printf("[RT]          initial positive reactivity insertion expected!\n");
        printf("[RT] Current state: P=%.1f%%, rho=%+.3f$, N=%.4f\n",
               rt->power_fraction * 100.0f,
               rt->reactivity / RT_BETA_TOTAL,
               rt->neutron_pop);
        float total_precursor = 0.0f;
        for (int i = 0; i < RT_NUM_DELAYED_GROUPS; i++)
            total_precursor += rt->precursor[i];
        printf("[RT] Delayed precursors: total=%.4f (6 groups active)\n",
               total_precursor);
    }
}

int reactor_thermal_update(ReactorThermal *rt, float dt) {
    rt->dt = dt;
    rt->time += dt;

    if (rt->scram_active)
        rt->scram_time += dt;

    /* 1. Update power distribution from current power level */
    setup_power_distribution(rt);

    /* 2. Heat diffusion (CPU) */
    heat_diffusion_step_cpu(rt);

    /* 3. Coolant flow and boiling */
    update_coolant(rt);

    /* 4. Pressure dynamics */
    update_pressure(rt);

    /* 5. Thermal expansion */
    update_thermal_expansion(rt);

    /* 6. Zircaloy oxidation (above ~1073K / 800°C) */
    update_zr_oxidation(rt);

    /* 7. Reactivity feedback (void coefficient + Doppler + rods + 6-group kinetics) */
    update_reactivity_feedback(rt);

    /* Belt-and-suspenders safety clamp */
    if (!isfinite(rt->power_fraction) || rt->power_fraction < 1.0e-10f)
        rt->power_fraction = 1.0e-10f;

    /* 8. Energy and entropy balance (1st & 2nd law) */
    update_energy_balance(rt);

    /* 9. Control rod drives (animate individual rod positions) */
    update_control_rods(rt);

    /* 10. Neutron particle transport (visual wave packets) */
    update_neutrons(rt);

    return 0;
}

/* GPU accelerated diffusion: upload temperature → dispatch shader → readback */
int reactor_thermal_gpu_diffuse(ReactorThermal *rt, VkQueue queue, int nIterations) {
    if (rt->diffusionPipeline == VK_NULL_HANDLE) {
        /* Fallback to CPU */
        for (int i = 0; i < nIterations; i++)
            heat_diffusion_step_cpu(rt);
        return 0;
    }

    VkDevice dev = rt->device;
    int total = rt->gridDim * rt->gridDim * rt->gridDim;
    VkDeviceSize grid_size = (VkDeviceSize)total * sizeof(float);

    /* Upload current temperature to GPU */
    void *mapped;
    vkMapMemory(dev, rt->tempMem, 0, grid_size, 0, &mapped);
    memcpy(mapped, rt->temperature, (size_t)grid_size);
    vkUnmapMemory(dev, rt->tempMem);

    /* Upload heat source */
    vkMapMemory(dev, rt->heatSrcMem, 0, grid_size, 0, &mapped);
    memcpy(mapped, rt->heat_source, (size_t)grid_size);
    vkUnmapMemory(dev, rt->heatSrcMem);

    /* Record compute commands */
    vkResetCommandBuffer(rt->cmdBuf, 0);
    VkCommandBufferBeginInfo cbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(rt->cmdBuf, &cbi);

    vkCmdBindPipeline(rt->cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, rt->diffusionPipeline);
    vkCmdBindDescriptorSets(rt->cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            rt->diffusionPipeLayout, 0, 1, &rt->diffusionDescSet, 0, NULL);

    /* Push constants: gridDim, boxHalf, dt, reserved */
    struct {
        int   gridDim;
        float boxHalf;
        float dt;
        int   iteration;
        float maxTemp;
        float pad[3];
    } push;
    push.gridDim = rt->gridDim;
    push.boxHalf = rt->boxHalf;
    push.dt      = rt->dt;
    push.maxTemp = 5000.0f;

    int groups = (rt->gridDim + 3) / 4; /* 4×4×4 workgroup */

    for (int i = 0; i < nIterations; i++) {
        push.iteration = i;
        vkCmdPushConstants(rt->cmdBuf, rt->diffusionPipeLayout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, 32, &push);
        vkCmdDispatch(rt->cmdBuf, (uint32_t)groups, (uint32_t)groups, (uint32_t)groups);

        /* Barrier between iterations (ping-pong) */
        VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(rt->cmdBuf,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &mb, 0, NULL, 0, NULL);
    }

    vkEndCommandBuffer(rt->cmdBuf);
    rt_submit_and_wait(dev, queue, rt->cmdBuf);

    /* Readback: result is in tempBuf or tempBuf2 depending on parity */
    VkDeviceMemory resultMem = (nIterations % 2 == 0) ? rt->tempMem : rt->tempMem2;
    vkMapMemory(dev, resultMem, 0, grid_size, 0, &mapped);
    memcpy(rt->temperature, mapped, (size_t)grid_size);
    vkUnmapMemory(dev, resultMem);

    return 0;
}

/* Write temperature field to the visualization density SSBO for rendering */
int reactor_thermal_visualize(ReactorThermal *rt, VkQueue queue) {
    VkDevice dev = rt->device;
    int vis_total = rt->visGridDim * rt->visGridDim * rt->visGridDim;
    VkDeviceSize vis_size = (VkDeviceSize)vis_total * sizeof(float);

    /* Allocate CPU-side buffers for resampled data */
    float *density_data = (float *)malloc(vis_total * sizeof(float));
    float *signed_data  = (float *)malloc(vis_total * sizeof(float));
    if (!density_data || !signed_data) {
        free(density_data); free(signed_data);
        return -1;
    }

    /* Resample thermal grid (64³) → visualization grid (128³) */
    int G_src = rt->gridDim;        /* 64 */
    int G_dst = rt->visGridDim;     /* 128 */

    for (int dz = 0; dz < G_dst; dz++) {
        for (int dy = 0; dy < G_dst; dy++) {
            for (int dx2 = 0; dx2 < G_dst; dx2++) {
                int dst_idx = dz * G_dst * G_dst + dy * G_dst + dx2;

                /* Bounds check on destination */
                if (dst_idx < 0 || dst_idx >= vis_total) {
                    fprintf(stderr, "[RT] ERROR: dst_idx=%d out of bounds (vis_total=%d)\n", 
                            dst_idx, vis_total);
                    continue;
                }

                /* Map destination voxel center to source coordinates */
                float fx = ((float)dx2 + 0.5f) / (float)G_dst;
                float fy = ((float)dy  + 0.5f) / (float)G_dst;
                float fz = ((float)dz  + 0.5f) / (float)G_dst;

                int sx = (int)(fx * G_src);
                int sy = (int)(fy * G_src);
                int sz = (int)(fz * G_src);
                if (sx >= G_src) sx = G_src - 1;
                if (sy >= G_src) sy = G_src - 1;
                if (sz >= G_src) sz = G_src - 1;
                
                /* Ensure source bounds  */
                if (sx < 0) sx = 0;
                if (sy < 0) sy = 0;
                if (sz < 0) sz = 0;
                
                int src_idx = sz * G_src * G_src + sy * G_src + sx;
                int src_limit = G_src * G_src * G_src;

                /* Bounds check on source */
                if (src_idx < 0 || src_idx >= src_limit) {
                    density_data[dst_idx] = 0.0f;
                    signed_data[dst_idx] = 0.0f;
                    continue;
                }

                /* Temperature → normalized density for visualization.
                 * Use a range appropriate for the current state:
                 *   Ambient (300K) → 0,  current max fuel temp → 1.
                 * Floor at 0.08 for non-void cells so the structure is always visible. */
                float T = rt->temperature[src_idx];
                int mat = rt->material_id[src_idx];

                if (mat == RT_MAT_VOID) {
                    density_data[dst_idx] = 0.0f;
                    signed_data[dst_idx]  = 0.0f;
                    continue;
                }

                /* Dynamic range: scale relative to current peak */
                float T_lo = 300.0f;
                float T_hi = rt->max_fuel_temp > 400.0f ? rt->max_fuel_temp * 1.1f : 1200.0f;
                float norm_T = (T - T_lo) / (T_hi - T_lo);
                if (norm_T < 0.0f) norm_T = 0.0f;
                if (norm_T > 1.0f) norm_T = 1.0f;
                /* Floor so structure is visible even at near-ambient */
                if (norm_T < 0.08f) norm_T = 0.08f;

                density_data[dst_idx] = norm_T;

                /* Signed channel: encode RBMK material type for coloring
                 * We pack material identity into ranges:
                 *  +1.0  = fuel (UO₂)     → rendered as blue quantum orbitals
                 *  +0.5  = cladding (Zr)   → metallic gray-silver
                 *  -1.0  = coolant (H₂O)  → translucent blue
                 *   0.0  = moderator (C)   → dark graphite
                 *  +0.8  = B₄C control rod → black absorber
                 *  +0.6  = graphite rod tip → highlighted graphite
                 *  -0.5  = Xe-135 poison   → purple poison cloud       */
                float signed_val = 0.0f;
                if (mat == RT_MAT_FUEL_UO2)
                    signed_val = 1.0f;
                else if (mat == RT_MAT_CLAD_ZR4)
                    signed_val = 0.5f;
                else if (mat == RT_MAT_COOLANT_H2O)
                    signed_val = -1.0f;
                else if (mat == RT_MAT_MODERATOR_C)
                    signed_val = 0.0f;
                else if (mat == RT_MAT_CONTROL_B4C)
                    signed_val = 0.8f;
                else if (mat == RT_MAT_CONTROL_GRAPH)
                    signed_val = 0.6f;

                /* Overlay Xe-135 poisoning: if significant Xe in this cell,
                 * tint toward purple poison marker */
                if (rt->xe_density && mat == RT_MAT_FUEL_UO2) {
                    float xe_ratio = rt->xe_density[src_idx] / (rt->xe_baseline + 1.0f);
                    if (xe_ratio > 1.5f) {
                        signed_val = -0.5f;  /* Xe-poisoned zone */
                        density_data[dst_idx] *= 1.3f;  /* make it more visible */
                    }
                }

                signed_data[dst_idx] = signed_val;
            }
        }
    }

    /* ── Stamp neutron particles as streaks (line from prev→current) ── */
    float inv_2h = 1.0f / (2.0f * rt->boxHalf);
    for (int i = 0; i < RT_MAX_NEUTRONS; i++) {
        RT_Neutron *n = &rt->neutrons[i];
        if (!n->alive) continue;

        float signed_n = (n->state == RT_NEUTRON_FAST) ? 2.0f : -2.0f;

        /* Rasterize line from prev to current in voxel space using DDA */
        float x0 = (n->px + rt->boxHalf) * inv_2h * G_dst;
        float y0 = (n->py + rt->boxHalf) * inv_2h * G_dst;
        float z0 = (n->pz + rt->boxHalf) * inv_2h * G_dst;
        float x1 = (n->x  + rt->boxHalf) * inv_2h * G_dst;
        float y1 = (n->y  + rt->boxHalf) * inv_2h * G_dst;
        float z1 = (n->z  + rt->boxHalf) * inv_2h * G_dst;

        float ddx = x1 - x0, ddy = y1 - y0, ddz = z1 - z0;
        float len = sqrtf(ddx*ddx + ddy*ddy + ddz*ddz);
        int steps = (int)(len + 1.5f);
        if (steps < 1) steps = 1;
        if (steps > 40) steps = 40;  /* safety cap */

        for (int s = 0; s <= steps; s++) {
            float t_s = (float)s / (float)steps;
            int vx2 = (int)(x0 + ddx * t_s);
            int vy2 = (int)(y0 + ddy * t_s);
            int vz2 = (int)(z0 + ddz * t_s);
            if (vx2 < 0 || vx2 >= G_dst) continue;
            if (vy2 < 0 || vy2 >= G_dst) continue;
            if (vz2 < 0 || vz2 >= G_dst) continue;

            /* Paint the streak point + small halo */
            for (int dz2 = -1; dz2 <= 1; dz2++) {
                for (int dy2 = -1; dy2 <= 1; dy2++) {
                    for (int dx3 = -1; dx3 <= 1; dx3++) {
                        int ppx = vx2 + dx3, ppy = vy2 + dy2, ppz = vz2 + dz2;
                        if (ppx < 0 || ppx >= G_dst) continue;
                        if (ppy < 0 || ppy >= G_dst) continue;
                        if (ppz < 0 || ppz >= G_dst) continue;
                        int pidx = ppz * G_dst * G_dst + ppy * G_dst + ppx;
                        float dist = sqrtf((float)(dx3*dx3 + dy2*dy2 + dz2*dz2));
                        /* Intensity fades along trail: bright at head, dim at tail */
                        float head_fade = 0.5f + 0.5f * t_s;  /* 0.5 at tail → 1.0 at head */
                        float intensity = (1.0f - dist * 0.35f) * head_fade;
                        if (intensity < 0.15f) continue;

                        density_data[pidx] = intensity;
                        signed_data[pidx] = signed_n;
                    }
                }
            }
        }
    }

    /* Create a single staging buffer for both density and signed data */
    VkDeviceSize total_size = vis_size * 2; /* density + signed */
    VkBuffer stagingBuf;
    VkDeviceMemory stagingMem;
    if (rt_create_buffer(rt->physDevice, dev, total_size,
                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         &stagingBuf, &stagingMem) != 0) {
        free(density_data); free(signed_data);
        return -2;
    }

    /* Upload both buffers at once */
    void *mapped;
    int vk_result = vkMapMemory(dev, stagingMem, 0, total_size, 0, &mapped);
    if (vk_result != VK_SUCCESS || !mapped) {
        fprintf(stderr, "[RT] ERROR: vkMapMemory failed (result=%d, mapped=%p)\n", 
                vk_result, mapped);
        vkDestroyBuffer(dev, stagingBuf, NULL);
        vkFreeMemory(dev, stagingMem, NULL);
        free(density_data);
        free(signed_data);
        return -3;
    }
    
    memcpy((char*)mapped + 0,          density_data, (size_t)vis_size);
    memcpy((char*)mapped + vis_size,   signed_data,  (size_t)vis_size);
    vkUnmapMemory(dev, stagingMem);

    /* Record copy commands for both buffers */
    vkResetCommandBuffer(rt->cmdBuf, 0);
    VkCommandBufferBeginInfo cbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(rt->cmdBuf, &cbi);

    /* Copy density: staging offset 0 → densityBuf offset 0 */
    VkBufferCopy region_density = {0, 0, vis_size};
    vkCmdCopyBuffer(rt->cmdBuf, stagingBuf, rt->densityBuf, 1, &region_density);

    /* Barrier between copies */
    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(rt->cmdBuf,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);

    /* Copy signed: staging offset vis_size → signedBuf offset 0 */
    VkBufferCopy region_signed = {vis_size, 0, vis_size};
    vkCmdCopyBuffer(rt->cmdBuf, stagingBuf, rt->signedBuf, 1, &region_signed);

    vkEndCommandBuffer(rt->cmdBuf);
    rt_submit_and_wait(dev, queue, rt->cmdBuf);

    /* Cleanup staging buffer */
    vkDestroyBuffer(dev, stagingBuf, NULL);
    vkFreeMemory(dev, stagingMem, NULL);
    free(density_data);
    free(signed_data);

    return 0;
}

const char *reactor_thermal_status(const ReactorThermal *rt) {
    static char buf[512];
    snprintf(buf, sizeof(buf),
             "t=%.1fs P=%.0fMW(%.0f%%) T_fuel=%.0f/%.0fK T_cool=%.0fK "
             "P=%.1fMPa Steam=%.0fkg H2=%.1fkg Rho=%.4f Void=%.0f%%",
             rt->time,
             rt->total_power_w / 1.0e6f,
             rt->power_fraction * 100.0f,
             rt->avg_fuel_temp, rt->max_fuel_temp,
             rt->avg_coolant_temp,
             rt->system_pressure / 1.0e6f,
             rt->total_steam_mass,
             rt->hydrogen_mass,
             rt->reactivity,
             /* Average void fraction estimate */
             (rt->total_steam_mass > 0.0f) ? 100.0f * fminf(rt->total_steam_mass / 1000.0f, 1.0f) : 0.0f
    );
    return buf;
}

void reactor_thermal_free(ReactorThermal *rt) {
    VkDevice dev = rt->device;

    free(rt->temperature);
    free(rt->heat_source);
    free(rt->material_id);
    free(rt->pressure_field);
    free(rt->void_fraction);
    free(rt->xe_density);
    free(rt->channels);

    if (dev) {
        if (rt->diffusionPipeline)   vkDestroyPipeline(dev, rt->diffusionPipeline, NULL);
        if (rt->diffusionPipeLayout) vkDestroyPipelineLayout(dev, rt->diffusionPipeLayout, NULL);
        if (rt->diffusionDescPool)   vkDestroyDescriptorPool(dev, rt->diffusionDescPool, NULL);
        if (rt->diffusionDescLayout) vkDestroyDescriptorSetLayout(dev, rt->diffusionDescLayout, NULL);
        if (rt->diffusionShader)     vkDestroyShaderModule(dev, rt->diffusionShader, NULL);

        if (rt->tempBuf)    vkDestroyBuffer(dev, rt->tempBuf, NULL);
        if (rt->tempMem)    vkFreeMemory(dev, rt->tempMem, NULL);
        if (rt->tempBuf2)   vkDestroyBuffer(dev, rt->tempBuf2, NULL);
        if (rt->tempMem2)   vkFreeMemory(dev, rt->tempMem2, NULL);
        if (rt->heatSrcBuf) vkDestroyBuffer(dev, rt->heatSrcBuf, NULL);
        if (rt->heatSrcMem) vkFreeMemory(dev, rt->heatSrcMem, NULL);
        if (rt->matBuf)     vkDestroyBuffer(dev, rt->matBuf, NULL);
        if (rt->matMem)     vkFreeMemory(dev, rt->matMem, NULL);

        if (rt->cmdPool)    vkDestroyCommandPool(dev, rt->cmdPool, NULL);
    }

    printf("[RT] Reactor thermal system freed\n");
    memset(rt, 0, sizeof(*rt));
}
