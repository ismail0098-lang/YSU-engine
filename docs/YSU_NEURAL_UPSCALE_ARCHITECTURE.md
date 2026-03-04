# YSU Neural Upscale — DLSS-class Temporal Super-Resolution Architecture

> Engine-agnostic, pure Vulkan compute, no proprietary SDK.
> Target: 1080p internal → 4K output. Dynamic resolution scaling supported.

---

## 0. Terminology & Constants

| Symbol | Meaning |
|--------|---------|
| `W_lo`, `H_lo` | Internal (low-res) render dimensions |
| `W_hi`, `H_hi` | Display (high-res) output dimensions |
| `r` | Upscale ratio = `W_hi / W_lo` (typically 2.0 for "Performance", 1.5 for "Quality") |
| `J_n` | Sub-pixel jitter offset for frame n — Halton(2,3) in [−0.5, +0.5]² |
| `MV(x,y)` | Per-pixel screen-space motion vector (in low-res pixel coords) |
| `D(x,y)` | Linear depth at pixel |
| `C_n` | Current LDR/HDR color buffer (low-res) |
| `H_{n-1}` | History buffer (high-res) — accumulated previous output |
| `O_n` | Output buffer (high-res) — final upscaled frame |

---

## 1. Architectural Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ FRAME N PIPELINE │
│ │
│ ┌─────────┐ ┌───────────┐ ┌──────────────┐ │
│ │ G-Buffer │──│ Low-Res │──│ Motion Vector│ │
│ │ + Depth │ │ Raytracer │ │ Generation │ │
│ └─────────┘ └───────────┘ └──────┬───────┘ │
│ │ │ │ │
│ ▼ ▼ ▼ │
│ ┌─────────────────────────────────────────────────┐ │
│ │ PASS 1: Temporal Reprojection (Compute) │ │
│ │ │ │
│ │ • Reproject history H_{n-1} using MV │ │
│ │ • Neighborhood color clamping (AABB / variance) │ │
│ │ • Disocclusion detection via depth + MV length │ │
│ │ • Output: reprojected_color, confidence_mask │ │
│ └─────────────────────┬───────────────────────────┘ │
│ │ │
│ ▼ │
│ ┌─────────────────────────────────────────────────┐ │
│ │ PASS 2: Neural Super-Resolution (Compute) │ │
│ │ │ │
│ │ Input tensor (per output pixel): │ │
│ │ • Bilinear-sampled low-res color (3ch) │ │
│ │ • Reprojected history color (3ch) │ │
│ │ • Flow/MV at output res (2ch) │ │
│ │ • Depth (1ch) │ │
│ │ • Jitter offset (2ch) │ │
│ │ • Confidence/disocclusion mask (1ch) │ │
│ │ ───────────────────────────────── │ │
│ │ Total: 12 channels │ │
│ │ │ │
│ │ Architecture: Lightweight hybrid CNN │ │
│ │ Encoder → 3×3 depthwise-sep convolutions │ │
│ │ Bottleneck → channel attention (SE block) │ │
│ │ Decoder → sub-pixel shuffle (ESPCN-style) │ │
│ │ Skip → direct bilinear upsample bypass │ │
│ │ │ │
│ │ Output: upscaled color (3ch) at W_hi × H_hi │ │
│ └─────────────────────┬───────────────────────────┘ │
│ │ │
│ ▼ │
│ ┌─────────────────────────────────────────────────┐ │
│ │ PASS 3: Sharpening + History Update │ │
│ │ │ │
│ │ • Adaptive CAS (contrast-adaptive sharpening) │ │
│ │ • Write output to display target │ │
│ │ • Copy output → history buffer H_n │ │
│ └─────────────────────────────────────────────────┘ │
│ │
│ FALLBACK PATH (no neural weights): │
│ Pass 1 reprojection → Catmull-Rom upsample + TAA blend │
│ (No neural inference, ~0.3 ms at 4K on mid-range GPU) │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Mathematical Formulation — Temporal Reprojection

### 2.1 Sub-pixel jitter pattern (Halton sequence)

For frame index `n`, the jitter in clip-space is:

```
J_n = ( Halton(n, base=2) − 0.5, Halton(n, base=3) − 0.5 )
```

This is added to the projection matrix *before* rasterization / ray generation:

```
P_jittered = T(2·J_n.x / W_lo, 2·J_n.y / H_lo, 0) · P
```

Where `T(tx, ty, 0)` is a translation in NDC that shifts by sub-pixel amounts.

### 2.2 Motion vector reprojection

Given pixel `p = (x, y)` in the current low-res frame with motion vector `MV(p)`:

```
p_prev = p + MV(p) (screen-space, low-res coordinates)
```

To sample the high-res history buffer:

```
p_hi = p_prev · r + 0.5 · (r − 1) (upscale to high-res pixel coords)
```

The reprojected history sample is obtained via **bicubic Catmull-Rom** filtering of `H_{n-1}` at `p_hi`:

```
H_reproj(p) = CatmullRom( H_{n-1}, p_hi )
```

### 2.3 Neighborhood clamping (variance-based AABB)

Compute the local color statistics in a 3×3 neighborhood of current frame `C_n` around pixel `p`:

```
μ = (1/9) · Σ_{q ∈ N₃(p)} C_n(q)
σ² = (1/9) · Σ_{q ∈ N₃(p)} (C_n(q) − μ)²

AABB_min = μ − γ · σ
AABB_max = μ + γ · σ
```

Where `γ ∈ [1.0, 1.25]` is a tunable clamp extent (1.0 = aggressive, removes ghosting but adds flicker; 1.25 = relaxed, smoother but may ghost).

The clamped history is:

```
H_clamped(p) = clamp( H_reproj(p), AABB_min, AABB_max )
```

This operates in **YCoCg** color space for perceptual correctness:

```
Y = 0.25·R + 0.50·G + 0.25·B
Co = 0.50·R − 0.50·B
Cg = −0.25·R + 0.50·G − 0.25·B
```

### 2.4 Disocclusion detection

A pixel is disoccluded (no valid history) when:

1. **Depth discontinuity:** `|D(p) − D(p_prev)| / D(p) > τ_depth` where `τ_depth = 0.05`
2. **Motion vector divergence:** the MV magnitude exceeds a threshold OR the MV of the reprojected pixel disagrees with forward MV by more than 1 pixel
3. **Out-of-screen:** `p_prev` falls outside `[0, W_lo) × [0, H_lo)`

The confidence mask `α(p) ∈ [0, 1]`:

```
α(p) = 1.0 if none triggered
α(p) = max(0, 1 − |Δ_depth|/τ) if depth mismatch
α(p) = 0.0 if out-of-screen
```

### 2.5 Temporal blend (fallback / pre-neural)

For the non-neural path, the blend weight is:

```
w = lerp(0.05, 0.90, 1 − α(p))

O_n(p) = (1 − w) · H_clamped(p) + w · Upsample(C_n, p)
```

Where `w = 0.05` means 95% history retention (steady state), and `w = 0.90` means 90% current frame (disoccluded).

---

## 3. Neural Network Architecture

### 3.1 Overview: Lightweight Temporal Super-Resolution CNN

The network is deliberately small for real-time inference at 4K output (~2ms target on RTX 3060).

**Input:** 12-channel feature map at `W_lo × H_lo` resolution
- Channels 0–2: Current frame color (linear HDR, YCoCg)
- Channels 3–5: Reprojected & clamped history color (YCoCg)
- Channels 6–7: Motion vector (normalized to [−1, 1])
- Channel 8: Linear depth (log-normalized)
- Channels 9–10: Jitter offset (normalized)
- Channel 11: Confidence/disocclusion mask

**Output:** 3-channel color at `W_hi × H_hi` (residual added to bilinear upsample)

### 3.2 Network Topology

```
Layer 0 — Input Projection
 Conv2D 3×3, 12 → 32 channels, ReLU
 (depthwise-separable: 12→12 DW 3×3, then 12→32 PW 1×1)

Layer 1 — Feature Extraction Block A
 Conv2D 3×3, 32 → 32, ReLU (depthwise-sep)
 Conv2D 3×3, 32 → 32, ReLU (depthwise-sep)
 + Residual skip

Layer 2 — Feature Extraction Block B
 Conv2D 3×3, 32 → 32, ReLU (depthwise-sep)
 Conv2D 3×3, 32 → 32, ReLU (depthwise-sep)
 + Residual skip

Layer 3 — Channel Attention (SE Block)
 GlobalAvgPool → 32
 FC 32 → 8, ReLU
 FC 8 → 32, Sigmoid
 Scale features element-wise

Layer 4 — Sub-pixel Reconstruction
 Conv2D 3×3, 32 → 3·r², no activation
 PixelShuffle(r) → 3 channels at W_hi × H_hi

Layer 5 — Residual Addition
 Output = PixelShuffle_output + BilinearUpsample(current_color)
```

**Parameter count (r=2):**
- Layer 0: 12×12×9 + 12×32 = 1680
- Layer 1: 2×(32×9 + 32×32) = 2624
- Layer 2: same = 2624
- Layer 3: 32×8 + 8×32 = 512
- Layer 4: 32×(3×4)×9 = 3456
- **Total ≈ 10,896 parameters (43.6 KB at FP32, 21.8 KB at FP16)**

### 3.3 Why this architecture

- **Depthwise-separable convolutions:** 8–9× fewer FLOPs than standard convolutions. At ~11K params the entire weight set fits in L1/L2 cache or shared memory.
- **SE block:** Allows the network to selectively weight history vs. current frame channels based on global image statistics (e.g., high motion → lower history weight).
- **Sub-pixel shuffle (ESPCN):** Avoids checkerboard artifacts from transposed convolutions. Each low-res pixel produces `r²` sub-pixels via channel rearrangement.
- **Residual learning:** The network only learns the *difference* from bilinear upsample, making training converge faster and preventing catastrophic quality degradation if weights are corrupted.

### 3.4 Activation functions

All hidden activations use **ReLU** (trivial in compute shaders: `max(0, x)`). The final layer has **no activation** since we learn a signed residual in linear color space.

---

## 4. Training Methodology

### 4.1 Dataset generation

Generate paired data from the engine itself:

1. **Render reference at 4K** with high SPP (≥256) and no jitter → ground truth `G`.
2. **Render at 1080p** with jitter pattern `J_n` — this is the input `C_n`.
3. **Render at 1080p** with jitter pattern `J_{n-1}` for previous frame (with known camera delta) → used to generate motion vectors and history.
4. **Record per-pixel:** color, depth, motion vectors, object ID (for disocclusion truth).

**Dataset size:** ~50K frame-pairs across diverse scenes (outdoor, indoor, high-motion, static, specular, volumetric).

### 4.2 Jitter patterns for training

Use the same Halton(2,3) sequence used at runtime. For each training pair, randomly select a pair of consecutive Halton indices to simulate temporal jitter diversity.

Augmentation:
- Random horizontal/vertical flip
- Random 90° rotation
- Random exposure scaling (×0.5 to ×2.0 in linear space)
- Random crop to 64×64 low-res patches (256×256 at 4K)

### 4.3 Loss functions

**Primary loss — Charbonnier (smooth L1):**

```
L_char = sqrt( ||O − G||² + ε² ), ε = 1e-3
```

Better than L1 at preserving sharp edges; better than L2 at avoiding over-smoothing.

**Perceptual loss (LPIPS-like, using a frozen feature extractor):**

```
L_perc = Σ_l w_l · || φ_l(O) − φ_l(G) ||₁
```

Where `φ_l` are features from layers {conv1, conv2, conv3} of a small VGG-style classifier. Weights `w_l = {0.1, 0.1, 0.05}`.

**Temporal consistency loss:**

```
L_temp = || Warp(O_n, MV) − O_{n-1} ||₁ · mask_valid
```

Penalizes temporal flicker by enforcing that the warped current output matches the previous output in non-disoccluded regions.

**Total loss:**

```
L = L_char + 0.1 · L_perc + 0.05 · L_temp
```

### 4.4 Training schedule

- Optimizer: AdamW, lr=2e-4, weight decay=1e-4
- Cosine annealing to 1e-6 over 300 epochs
- Batch size: 16 patches (64×64 low-res)
- Mixed precision training (FP16 forward, FP32 accumulation)
- Gradient clipping: max norm 1.0

### 4.5 Weight export

Export trained weights as a flat binary blob:

```
[header: 16 bytes — magic, version, param_count, flags]
[weights: N × sizeof(float16)]
```

Runtime loads this blob into a `VkBuffer` with `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT`.

---

## 5. Vulkan Integration

### 5.1 Resource layout

| Resource | Format | Size | Usage |
|----------|--------|------|-------|
| `color_lo` | `R16G16B16A16_SFLOAT` | W_lo × H_lo | Sampled + Storage |
| `depth_lo` | `R32_SFLOAT` | W_lo × H_lo | Sampled |
| `motion_vec` | `R16G16_SFLOAT` | W_lo × H_lo | Sampled |
| `history_hi` | `R16G16B16A16_SFLOAT` | W_hi × H_hi | Sampled + Storage |
| `output_hi` | `R16G16B16A16_SFLOAT` | W_hi × H_hi | Storage |
| `reproj_hi` | `R16G16B16A16_SFLOAT` | W_hi × H_hi | Storage (intermediate) |
| `confidence` | `R16_SFLOAT` | W_hi × H_hi | Storage (intermediate) |
| `weights_buf` | — | ~22 KB | Storage (read-only) |

### 5.2 Descriptor set layout

```
Set 0 (Temporal Reprojection):
 binding 0: combined image sampler — color_lo
 binding 1: combined image sampler — depth_lo
 binding 2: combined image sampler — motion_vec
 binding 3: combined image sampler — history_hi
 binding 4: storage image — reproj_hi (output)
 binding 5: storage image — confidence (output)

Set 1 (Neural Upscale):
 binding 0: combined image sampler — color_lo
 binding 1: storage image — reproj_hi (input)
 binding 2: storage image — confidence (input)
 binding 3: combined image sampler — depth_lo
 binding 4: combined image sampler — motion_vec
 binding 5: storage buffer — weights_buf
 binding 6: storage image — output_hi

Set 2 (Sharpen + History Copy):
 binding 0: storage image — output_hi (input)
 binding 1: storage image — history_hi (output, for next frame)
 binding 2: storage image — display_target
```

### 5.3 Compute dispatch

```
Pass 1 (Reprojection):
 Workgroup: 16×16
 Dispatch: ceil(W_hi/16) × ceil(H_hi/16) × 1

 ── Image barrier: reproj_hi, confidence → SHADER_READ ──

Pass 2 (Neural Upscale):
 Workgroup: 16×16
 Dispatch: ceil(W_hi/16) × ceil(H_hi/16) × 1

 ── Image barrier: output_hi → SHADER_READ ──

Pass 3 (Sharpen + History Write):
 Workgroup: 16×16
 Dispatch: ceil(W_hi/16) × ceil(H_hi/16) × 1

 ── Image barrier: history_hi → SHADER_READ (for next frame) ──
```

### 5.4 Memory barriers between passes

```c
// After Pass 1, before Pass 2:
VkImageMemoryBarrier barriers[] = {
 { // reproj_hi
 .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
 .newLayout = VK_IMAGE_LAYOUT_GENERAL,
 .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
 .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
 },
 { // confidence
 .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
 .newLayout = VK_IMAGE_LAYOUT_GENERAL,
 .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
 .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
 },
};
vkCmdPipelineBarrier(cmd,
 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
 0, 0, NULL, 0, NULL, 2, barriers);
```

### 5.5 Async compute

The upscale pipeline can run on a dedicated **async compute queue** overlapping with the next frame's geometry/rasterization work:

```
Frame N:
 Graphics Queue: [G-buffer render N] ────────────────── [G-buffer render N+1]
 Compute Queue: [Upscale N (Pass 1→2→3)] ──
 ↑
 semaphore signal from graphics
```

Use `VkSemaphore` to synchronize: graphics queue signals after low-res render completes, compute queue waits on that semaphore before starting upscale.

---

## 6. Performance Optimization Strategies

### 6.1 FP16 throughout

All intermediate textures and compute shader arithmetic use `float16_t` / `f16vec4`. The weight buffer stores FP16. This:
- Halves memory bandwidth (dominant bottleneck at 4K)
- Enables 2× throughput on FP16 ALUs (all modern GPUs since GCN/Pascal)
- Fits more data in L2 cache

Enable `VK_KHR_shader_float16_int8` and use `#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require` in GLSL.

### 6.2 Shared memory tiling for convolutions

Each 16×16 workgroup loads a (16+2)×(16+2) = 18×18 tile into shared memory (`shared f16vec4 tile[18][18]`). This eliminates redundant global memory reads for 3×3 convolution kernels.

Memory per workgroup: `18 × 18 × 8 bytes = 2592 bytes` — well within 48 KB limit.

### 6.3 Fused passes

Pass 2 (neural upscale) fuses all convolution layers into a single dispatch. Each thread computes one output pixel by:
1. Loading the 12-channel input patch (3×3 neighborhood → 12×9 = 108 values)
2. Running all conv layers in registers
3. Writing the 3-channel output

This avoids writing/reading intermediate feature maps to global memory.

### 6.4 Cooperative matrices (tensor core equivalent)

On GPUs supporting `VK_KHR_cooperative_matrix` (Turing+, RDNA3+):
- Pack convolution weights as 16×16 cooperative matrix tiles
- Execute `OpCooperativeMatrixMulAdd` for the 1×1 pointwise convolutions
- 4–8× throughput improvement for the FC layers in SE block

Fallback: standard dot-product loop when cooperative matrices unavailable.

### 6.5 Dynamic resolution scaling integration

The upscale ratio `r` can change per-frame:
- At init, allocate `history_hi` and `output_hi` at maximum display resolution
- Push constant `r`, `W_lo`, `H_lo` change per frame
- Weight set supports continuous `r` via the PixelShuffle layer reinterpretation:
 - For `r=2`: 32 → 12 channels, shuffle 2×2
 - For `r=1.5`: render at `ceil(W_hi/1.5)`, pad, shuffle 2×2, crop
 - For `r=1.33` (Quality): same strategy

---

## 7. Ghosting & Disocclusion Artifact Mitigation

### 7.1 Source of ghosting

Ghosting occurs when stale history data persists in regions that should show new content. Primary causes:
1. **Disoccluded regions:** areas revealed by camera/object motion that have no valid history
2. **Specular surfaces:** reflections change faster than geometry motion vectors capture
3. **Semitransparent geometry:** particles, foliage with alpha — MV represents opaque surface only

### 7.2 Mitigation strategies

**7.2.1 Variance-based AABB clamping (§2.3)**
The YCoCg AABB clamp is the first line of defense. By constraining reprojected history to lie within the local color distribution of the current frame, stale colors are rejected.

**7.2.2 Per-pixel blend weight modulation**
The confidence mask `α(p)` from §2.4 drives the temporal blend weight. In regions with:
- Low confidence: more current frame, less history → eliminates ghost immediately but increases noise
- High confidence: standard 95% history blend → maximally stable

**7.2.3 Depth-aware rejection**
Compare warped depth `D_warp = D(p + MV(p))_{n-1}` with current depth `D(p)_n`:
```
reject = (|D_warp − D_n| / D_n) > 0.05
```
This catches camera-revealed disocclusions that pure MV analysis misses (e.g., thin objects).

**7.2.4 Stencil / object-ID rejection**
If the engine provides per-pixel object IDs (mesh ID in G-buffer), reject history when the ID changes:
```
reject = (ID_current(p) ≠ ID_warped(p_prev))
```
This is the most reliable disocclusion signal but requires engine-side support.

**7.2.5 Luminance-based rectification (for specular)**
For specular highlights that move faster than geometry:
```
L_curr = Luminance(C_n(p))
L_hist = Luminance(H_reproj(p))
if |L_curr − L_hist| / max(L_curr, L_hist, 0.001) > 0.5:
 α(p) *= 0.5 // reduce history confidence for this pixel
```

**7.2.6 Neural safety net**
The trained network learns to detect and compensate for residual ghosting artifacts that heuristic methods miss. The disocclusion mask input channel (§3.1, channel 11) gives the network explicit signal about where history is unreliable.

### 7.3 Debug visualization

Expose env var `YSU_UPSCALE_DEBUG`:
- `=1`: Visualize confidence mask as heatmap overlay
- `=2`: Show rejected pixels in red
- `=3`: Show motion vector magnitude as flow visualization
- `=4`: Split-screen: left=upscaled, right=bilinear (quality comparison)

---

## 8. Integration Checklist

1. [ ] Modify render pipeline to produce jittered low-res color + depth + motion vectors
2. [ ] Create Vulkan resources (images, buffers, descriptor sets)
3. [ ] Compile GLSL compute shaders to SPIR-V
4. [ ] Load neural network weights into GPU buffer
5. [ ] Record 3-pass command buffer per frame
6. [ ] Ping-pong history buffers between frames
7. [ ] Add dynamic resolution scaling control (env var or runtime API)
8. [ ] Train network on engine-generated datasets
9. [ ] Validate temporal stability with motion stress tests
10. [ ] Profile and optimize for async compute overlap

See implementation files:
- `ysu_upscale.h` — public API
- `ysu_upscale.c` — Vulkan resource management and dispatch
- `shaders/upscale_reproj.comp` — temporal reprojection compute shader
- `shaders/upscale_neural.comp` — neural upscale compute shader
- `shaders/upscale_sharpen.comp` — sharpening + history update compute shader
