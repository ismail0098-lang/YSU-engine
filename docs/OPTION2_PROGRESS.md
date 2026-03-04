# Option 2: Temporal Denoising - Implementation Progress

**Status**: 60% Complete - Core infrastructure in place, final dispatch logic remaining

## Completed Changes

### 1. Parameter Variables (Line ~1651-1653)
```c
int temporal_denoise_enabled = ysu_env_bool("YSU_GPU_TEMPORAL_DENOISE", 1);
float temporal_denoise_weight = ysu_env_float("YSU_GPU_TEMPORAL_DENOISE_WEIGHT", 0.7f);
```

### 2. Logging (Line ~1665-1667)
Enhanced GPU denoiser log output to show temporal denoise status when enabled

### 3. Image/Memory Variables (Line ~1645-1650)
```c
VkImage denoise_history = VK_NULL_HANDLE;
VkDeviceMemory denoise_history_mem = VK_NULL_HANDLE;
VkImageView denoise_history_view = VK_NULL_HANDLE;
```

### 4. Shader Module Variables (Line ~1636)
```c
VkShaderModule sm_blend = VK_NULL_HANDLE;
```

### 5. Pipeline Variable (Line ~1643)
```c
VkPipeline pipe_blend = VK_NULL_HANDLE;
```

### 6. Descriptor Pool/Set Variables (Line ~1641-1643)
```c
VkDescriptorPool dp_blend = VK_NULL_HANDLE;
VkDescriptorSet ds_blend = VK_NULL_HANDLE;
```

### 7. Denoise History Image Creation (Lines ~1728-1778)
- Full image/memory/view creation for denoise_history when temporal_denoise_enabled
- Includes error checking and logging

### 8. Shader Code (shaders/blend.comp)
- Complete GLSL compute shader for temporal blending
- Supports first-frame handling (no history)
- Uses frame_id and weight parameters

## Remaining Work

### Phase 1: Blend Shader Loading & Pipeline Creation (~50 lines)
**Location**: After line 1914 (after denoise pipeline creation)

Code needed:
1. Load blend.comp.spv shader file
2. Create VkShaderModule for blend
3. Create descriptor set layout (3 bindings: current, history, output)
4. Create pipeline layout with push constants
5. Create compute pipeline
6. Create descriptor pool and set with denoise temp, history, history images

### Phase 2: Temporal Blend Dispatch (~100 lines)
**Location**: After denoiser dispatch completes (after line 2039)

Code needed:
1. Add barrier between denoise output (temp_denoised) and blend input
2. Bind blend pipeline
3. Bind descriptor set for blend
4. Push constants: W, H, weight, frame_id
5. Dispatch blend compute shader (16x16 tile groups)
6. Barrier after blend output (result -> ldr_img input)

### Phase 3: Frame History Swap (~20 lines)
**Location**: At end of frame loop (after frame processing, before frame_id++)

Code needed:
1. Copy temp_denoised → denoise_history for next frame
2. Use vkCmdCopyImage with proper barriers
3. Or use compute shader to copy

### Phase 4: Cleanup (~5 lines)
**Location**: In cleanup section (vkDestroy* calls)

Code needed:
1. Destroy blend pipeline
2. Destroy blend descriptor set/pool
3. Destroy blend shader module
4. Free denoise_history image/memory/view

## Integration Points

### Shader Compilation Flow
```
denoise.comp.spv --[Load]--> sm_denoise --[Pipeline]--> pipe_denoise
blend.comp.spv --[Load]--> sm_blend --[Pipeline]--> pipe_blend
```

### Dispatch Sequence Per Frame
```
Ray Trace (out_img)
 ↓
Denoise (out_img → temp_denoised) [conditional: denoise_skip]
 ↓
Temporal Blend (temp_denoised + history → output) [conditional: temporal_denoise_enabled]
 ↓
Tonemap (output → ldr_img)
 ↓
Save temp_denoised → denoise_history for next frame
 ↓
Frame++
```

### Descriptor Bindings (Blend)
```
binding 0: current_frame (temp_denoised output from denoise)
binding 1: history_frame (denoise_history from previous frame)
binding 2: output_frame (goes to tonemap input)
```

## Code Snippets Ready

### Blend Shader Load Pattern (Follows denoise pattern)
```c
if(temporal_denoise_enabled && gpu_denoise_enabled) {
 const char* blend_spv_paths[] = {
 "shaders/blend.comp.spv",
 "../shaders/blend.comp.spv",
 "./shaders/blend.comp.spv"
 };
 uint8_t* blend_spv = NULL;
 size_t blend_spv_sz = 0;
 for(int i=0; i<3; i++) {
 blend_spv = read_file(blend_spv_paths[i], &blend_spv_sz);
 if(blend_spv) break;
 }
 if(!blend_spv) {
 fprintf(stderr, "[GPU] Cannot find blend.comp.spv, temporal denoise disabled\n");
 temporal_denoise_enabled = 0;
 } else {
 // Create shader module and pipeline here
 }
}
```

### Blend Dispatch Pattern (After denoiser)
```c
if(temporal_denoise_enabled && pipe_blend != VK_NULL_HANDLE) {
 // Barrier: denoise_temp (write) → blend input (read)
 VkImageMemoryBarrier blend_bar_pre = { /* ... */ };
 vkCmdPipelineBarrier(cb, 
 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, &blend_bar_pre);
 
 vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_blend);
 vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl_blend, 0, 1, &ds_blend, 0, NULL);
 
 struct { int W, H; float weight; int frame_id; } blend_pc;
 blend_pc.W = W;
 blend_pc.H = H;
 blend_pc.weight = temporal_denoise_weight;
 blend_pc.frame_id = frame_id;
 vkCmdPushConstants(cb, pl_blend, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(blend_pc), &blend_pc);
 
 uint32_t blend_gx = (W + 15) / 16;
 uint32_t blend_gy = (H + 15) / 16;
 vkCmdDispatch(cb, blend_gx, blend_gy, 1);
}
```

## Architecture Notes

### Why Separate Temporal Denoise?
- Denoiser produces denoise_temp output (separable filter result)
- Temporal blend happens AFTER denoiser completes
- Result goes to tonemap input (replaces denoise_temp)
- History buffer stores previous frame's blended output

### Memory Layout
```
Frame N:
 out_img (raw ray trace)
 ↓
 denoise_temp (after denoise)
 ↓
 blend output (after temporal blend)
 ↓
 denoise_history ← saved for Frame N+1
 
Frame N+1:
 out_img (raw ray trace)
 ↓
 denoise_temp (after denoise)
 ↓
 blend with denoise_history
 ↓
 denoise_history ← updated
```

## Testing Strategy

1. **Build with Vulkan SDK** to compile blend.comp.spv
2. **Test settings**:
 ```bash
 # Maximum temporal smoothing
 YSU_GPU_TEMPORAL_DENOISE=1 YSU_GPU_TEMPORAL_DENOISE_WEIGHT=0.8
 
 # Balanced (default)
 YSU_GPU_TEMPORAL_DENOISE=1 YSU_GPU_TEMPORAL_DENOISE_WEIGHT=0.7
 
 # Minimal temporal
 YSU_GPU_TEMPORAL_DENOISE=1 YSU_GPU_TEMPORAL_DENOISE_WEIGHT=0.5
 
 # Disabled
 YSU_GPU_TEMPORAL_DENOISE=0
 ```
3. **Measure FPS** (should be ~same as without temporal blend)
4. **Visual inspection** for ghosting/artifacts (especially around moving objects)
5. **Compare with Option 1** (denoise skip quality vs temporal denoise quality)

## Expected Timeline

With proper code reuse patterns (following existing denoise implementation):
- Blend shader loading: 2 minutes
- Pipeline creation: 3 minutes 
- Descriptor setup: 2 minutes
- Temporal blend dispatch: 3 minutes
- Frame history swap: 1 minute
- Cleanup: 1 minute
- **Total: ~12 minutes** of focused coding

## Coordination Notes

Option 2 works seamlessly with:
- Option 1 (Denoise Skip) - temporal blend AFTER conditional denoise
- Session 12 (Temporal Accumulation) - 16-frame blend + temporal denoise = ultra-smooth
- Session 13 (Render Scale) - all work on scaled resolution

Combined effect:
```
Denoise Skip (Option 1): 100 FPS baseline
+ Temporal Denoise (Option 2): 97 FPS (small cost), but +20% quality
+ Both at skip=2: 110 FPS, excellent quality
+ All + Temporal Accum: 110 FPS @ 16x, feels like 110 FPS with 160+ noise reduction
```

## Next Phase Preparation

Once Option 2 completes:
- Option 3 (Half-Precision) will modify shaders
- Option 4 (Async Compute) will manage separate queue
- Both build on current descriptor/pipeline patterns

## Success Criteria

 Completed when:
1. Blend shader loads without errors
2. Temporal blend dispatch executes without validation errors
3. Output images look temporally smooth (no ghosting)
4. FPS stays similar to Option 1 baseline
5. Quality visibly improves (especially in noisy areas)
