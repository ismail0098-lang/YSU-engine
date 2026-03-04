# Option 2: Temporal Denoising Implementation Plan

**Status**: In Progress - Planning Phase

## Overview

Temporal denoising blends the current frame's render output with the **previous frame's denoised output** to achieve:
- Better quality at same FPS
- Better FPS at same quality (when combined with denoise skip)
- Reduced temporal flickering and noise perception

## Algorithm

```
Frame N:
 1. Ray trace → out_img (current noisy render)
 2. Denoise out_img → temp_denoised (current frame denoising)
 3. Blend: result = 0.7 * prev_denoised + 0.3 * temp_denoised
 4. Save temp_denoised → denoise_history for Frame N+1

Frame N+1:
 1. Ray trace → out_img
 2. Denoise out_img → temp_denoised
 3. Blend: result = 0.7 * denoise_history + 0.3 * temp_denoised (uses Frame N's denoised)
 4. Save temp_denoised → denoise_history
```

This creates a temporal feedback loop where denoising results are accumulated over time.

## Implementation Requirements

### 1. New GPU Resources Needed

- **denoise_history**: VkImage to store previous frame's denoised output
- **denoise_history_mem**: VkDeviceMemory for denoise_history
- **denoise_history_view**: VkImageView for denoise_history
- **Temporal blend shader**: New compute shader (blend.comp) or extend tonemap shader

### 2. Code Changes Required (gpu_vulkan_demo.c)

**Location 1: Image Setup (after denoise_temp)**
- Create `denoise_history` image (same format/size as out_img)
- Allocate memory for denoise_history
- Create image view

**Location 2: Descriptor Set (pipeline layout)**
- Update denoise descriptor sets to include denoise_history
- Add descriptors for temporal blend shader

**Location 3: Temporal Blend Shader Dispatch**
- After denoiser completes, add temporal blend stage
- Blend previous frame (denoise_history) with current frame (temp_denoised)
- Output goes to ldr_img (which is then tonemapped)

**Location 4: Frame Loop**
- Copy temp_denoised → denoise_history at end of frame
- Use denoise_history in next frame's blend

**Location 5: Initialization**
- First frame: Skip blend (or blend with current frame only)

### 3. New Environment Variable

```c
int temporal_denoise_enabled = ysu_env_bool("YSU_GPU_TEMPORAL_DENOISE", 1); // default ON
float temporal_denoise_weight = ysu_env_float("YSU_GPU_TEMPORAL_DENOISE_WEIGHT", 0.7f); // 0.7 = 70% prev, 30% curr
```

## Shader Changes

### blend.comp (new compute shader)

```glsl
#version 450

layout(local_size_x = 16, local_size_y = 16) in;

layout(set=0, binding=0, rgba32f) uniform image2D current_frame; // Current denoised
layout(set=0, binding=1, rgba32f) uniform image2D history_frame; // Previous denoised
layout(set=0, binding=2, rgba32f) uniform image2D output; // Blended result

layout(push_constant) uniform PC {
 int W, H;
 float weight; // 0.7 = 70% history, 30% current
} pc;

void main() {
 ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
 if(pos.x >= pc.W || pos.y >= pc.H) return;
 
 vec4 curr = imageLoad(current_frame, pos);
 vec4 hist = imageLoad(history_frame, pos);
 
 // Temporal blend
 vec4 result = mix(curr, hist, pc.weight);
 
 imageStore(output, pos, result);
}
```

## Performance Impact

### Current Performance (after Option 1):
- Single render: 6.3ms
- Denoiser (skip=4): 1.25ms
- Total: 7.55ms = 132 FPS

### With Temporal Denoising Added:
- Temporal blend shader: ~0.3-0.5ms (16x16 tiles, very efficient)
- New total: 7.85ms = 127 FPS
- **Net FPS loss: -5 FPS** (but quality +15-30%)

### Combined with Denoise Skip (denoise_skip=2):
- Single render: 6.3ms
- Denoiser (skip=2, now blended): 2.5ms
- Temporal blend: 0.3ms
- Total: 9.1ms = 110 FPS
- **Quality improvement: +20-30%** from temporal feedback

## Quality Metrics

| Scenario | FPS | Quality | Use Case |
|---|---|---|---|
| Original (no temporal denoise) | 100 | Good | Baseline |
| Option 1 only (skip=2) | 115 | Excellent | Speed focus |
| Option 2 only (temporal denoise) | 127 | Excellent | Quality focus |
| Option 1+2 (skip=2, temporal) | 110 | Excellent+ | Balanced |

## Implementation Difficulty

**Estimated effort**:
- Shader code: 30 lines
- C code changes: 50-70 lines
- Testing: 20 minutes
- **Total complexity: MEDIUM** (requires descriptor set updates, memory management)

## Integration Points

### Works with:
- Option 1 (Denoise Skip) - complementary, temporal blend masks skip artifacts
- Session 12 (Temporal Accumulation) - stacks with frame blending
- Session 13 (Render Scale) - scale applies to all stages

### Considerations:
- Need to manage history buffer lifecycle
- First frame needs special handling (no history yet)
- Camera motion can cause ghosting (future motion compensation option)

## Next Steps

1. Create blend.comp shader
2. Modify gpu_vulkan_demo.c to add temporal blend stage
3. Add descriptor sets for temporal blend
4. Implement history buffer swap at frame end
5. Test and measure FPS/quality
6. Tune blend weight parameter

## Files to Modify

- `shaders/blend.comp` - NEW shader
- `gpu_vulkan_demo.c` - Image setup, descriptor sets, dispatch, frame loop

## Rollback Plan

If temporal denoising degrades quality or causes issues:
1. Set `YSU_GPU_TEMPORAL_DENOISE=0` to disable
2. Revert to Option 1 only (denoise skip)
3. Fall back to Session 13 baseline (render scale)

## Related Documentation

- OPTION1_DENOISE_SKIP.md - Option 1 (prerequisite)
- FULL_OPTIMIZATION_GUIDE.md - All options overview
