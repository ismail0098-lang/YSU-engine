# YSU ENGINE - 10 FEATURE ENHANCEMENT ROADMAP

## COMPLETED (1-5)

### [1/10] STOCHASTIC SAMPLING 
**What**: Per-pixel random jitter on ray directions
**File**: shaders/tri.comp (main() function)
**Implementation**: 
```glsl
uint rng_state = wang_hash(uint(pix.x + pix.y * 73856093) ^ uint(pc.frame));
float jitter_x = (rand01(rng_state) - 0.5) * jitter_scale;
float jitter_y = (rand01(rng_state) - 0.5) * jitter_scale;
float u = (float(pix.x) + 0.5 + jitter_x) / float(pc.W);
```
**Impact**: Rays now vary per frame, enabling noise for denoising to work
**Test**: Set YSU_GPU_FRAMES=8 to accumulate variance

---

### [2/10] TEMPORAL FILTERING 
**What**: Multi-frame accumulation (EMA or running average)
**File**: shaders/tri.comp (end of main())
**Implementation**:
```glsl
// EMA: final = mix(prev, curr, alpha)
// Running avg: final = mix(prev, curr, 1/(frame+1))
final_accum = mix(prev_accum, col, pc.alpha);
```
**Impact**: Smooth results across frames, enables convergence
**Test**: Set YSU_GPU_FRAMES>1 to see accumulation

---

### [3/10] ADVANCED TONE MAPPING 
**What**: ACES tone mapping + sRGB color space conversion
**File**: shaders/tonemap.comp
**Implementation**:
```glsl
vec3 tonemap_aces(vec3 x){
 const float a=2.51, b=0.03, c=2.43, d=0.59, e=0.14;
 return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0.0, 1.0);
}
// + sRGB conversion for proper color space
```
**Impact**: Industry-standard tone mapping + correct color space
**Test**: Visual quality improvement in final output

---

### [4/10] ADAPTIVE SAMPLING 
**What**: Per-pixel jitter scaled by local variance
**File**: shaders/tri.comp (main() function)
**Implementation**:
```glsl
// Calculate neighborhood variance
float variance = ...; // 3x3 neighborhood
float jitter_scale = 1.0 + min(variance * 5.0, 1.0);
// Use jitter_scale in stochastic sampling
```
**Impact**: High-variance regions get more samples
**Test**: Edges and details converge faster

---

### [5/10] MATERIAL VARIANTS 
**What**: Basic PBR-inspired shading (diffuse + fresnel)
**File**: shaders/tri.comp (color calculation)
**Implementation**:
```glsl
float lambert = max(dot(nHit, -light_dir), 0.0);
float fresnel = pow(1.0 - dot(nHit, -rd), 4.0) * 0.5 + 0.5;
vec3 col = base_color * (lambert * factor + ambient) + fresnel * spec;
```
**Impact**: Geometry now has realistic shading instead of flat grayscale
**Test**: Renders show material variation with lighting

---

## REMAINING (6-10)

### [6/10] COLOR MANAGEMENT
**What**: Full sRGB<->linear conversions throughout pipeline
**Status**: PARTIALLY DONE (sRGB in tonemap.comp)
**Remaining**: Apply linear input, maintain linear working space
**Priority**: Medium

---

### [7/10] GPU BVH BUILDING
**What**: Parallel BVH construction on GPU
**Status**: CPU-only currently
**Approach**: Add compute shader for SAH partitioning
**Impact**: Faster mesh loading (currently CPU bottleneck)
**Priority**: Low (offscreen, not realtime)

---

### [8/10] INTERACTIVE GPU RENDERING
**What**: Real-time GPU viewport in editor
**Status**: GPU rendering works, editor exists separately
**Approach**: Integrate GPU output into raylib viewport
**Impact**: Live preview while editing meshes
**Priority**: Medium (UX improvement)

---

### [9/10] ANTI-ALIASING
**What**: Proper pixel filter (Box, Triangle, or Blackman-Harris)
**Status**: Currently implicit box filter
**Approach**: Add configurable filter kernel in output
**Implementation**: ConvolvePixel with filter kernel
**Priority**: Low (stochastic sampling already helps)

---

### [10/10] SHADER VARIANTS
**What**: Specialize shaders for material types
**Status**: Basic PBR now in main shader
**Approach**: Compile multiple specialized shaders (metal, dielectric, emissive)
**Impact**: Better performance + accuracy
**Priority**: Medium

---

## BUILD & TEST

All 5 completed features compiled successfully:
```bash
cd shaders
glslc tri.comp -o tri.comp.spv 
glslc tonemap.comp -o tonemap.comp.spv 
```

Test configuration:
```bash
set YSU_GPU_W=320
set YSU_GPU_H=180
set YSU_GPU_OBJ=TestSubjects/3M.obj
set YSU_GPU_FRAMES=8 # See temporal accumulation
set YSU_NEURAL_DENOISE=1 # Denoiser works with variance
gpu_demo.exe
```

---

## NEXT STEPS

1. **Immediate**: Test denoiser effectiveness with stochastic samples
2. **Near-term**: Add #8 (Interactive viewport) for workflow
3. **Future**: #6, #9, #10 for quality polish
4. **Optional**: #7 (GPU BVH) for large mesh loading

---

**Status**: 5/10 Features Complete, Shaders Compiling, Ready for Testing
