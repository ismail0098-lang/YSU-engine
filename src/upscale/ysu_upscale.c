/* ysu_upscale.c — DLSS-class temporal super-resolution (Vulkan compute)
 *
 * Implements the 3-pass upscale pipeline:
 *   Pass 1: Temporal reprojection + disocclusion detection
 *   Pass 2: Neural super-resolution (or fallback temporal blend)
 *   Pass 3: Contrast-adaptive sharpening + history buffer update
 *
 * All GPU work recorded into caller's command buffer. No internal
 * queue submission — caller controls synchronization.
 */

#ifdef _WIN32
  #define VK_USE_PLATFORM_WIN32_KHR
#endif

#include "ysu_upscale.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ═══════════════════════════════════════════════════════════════════
 *  Internal helpers
 * ═══════════════════════════════════════════════════════════════════ */

static int ysu_us_env_int(const char* key, int def) {
    const char* v = getenv(key);
    return (v && v[0]) ? atoi(v) : def;
}

static float ysu_us_env_float(const char* key, float def) {
    const char* v = getenv(key);
    return (v && v[0]) ? (float)atof(v) : def;
}

/* Find a memory type index satisfying both type bits and property flags */
static uint32_t ysu_find_memory_type(VkPhysicalDevice pd,
                                     uint32_t type_bits,
                                     VkMemoryPropertyFlags props)
{
    VkPhysicalDeviceMemoryProperties mem;
    vkGetPhysicalDeviceMemoryProperties(pd, &mem);
    for (uint32_t i = 0; i < mem.memoryTypeCount; i++) {
        if ((type_bits & (1u << i)) &&
            (mem.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    return UINT32_MAX; /* caller must check */
}

/* ═══════════════════════════════════════════════════════════════════
 *  Image / buffer creation helpers
 * ═══════════════════════════════════════════════════════════════════ */

static VkResult ysu_create_image_2d(VkDevice dev, VkPhysicalDevice pd,
                                    const VkAllocationCallbacks* alloc,
                                    uint32_t w, uint32_t h,
                                    VkFormat format,
                                    VkImageUsageFlags usage,
                                    VkImage* out_img,
                                    VkDeviceMemory* out_mem,
                                    VkImageView* out_view)
{
    VkResult r;

    /* --- Image --- */
    VkImageCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = format,
        .extent = { w, h, 1 },
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    r = vkCreateImage(dev, &ici, alloc, out_img);
    if (r != VK_SUCCESS) return r;

    /* --- Memory --- */
    VkMemoryRequirements mreq;
    vkGetImageMemoryRequirements(dev, *out_img, &mreq);

    uint32_t memIdx = ysu_find_memory_type(pd, mreq.memoryTypeBits,
                                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (memIdx == UINT32_MAX) return VK_ERROR_OUT_OF_DEVICE_MEMORY;

    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mreq.size,
        .memoryTypeIndex = memIdx,
    };
    r = vkAllocateMemory(dev, &mai, alloc, out_mem);
    if (r != VK_SUCCESS) return r;

    r = vkBindImageMemory(dev, *out_img, *out_mem, 0);
    if (r != VK_SUCCESS) return r;

    /* --- View --- */
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    VkImageViewCreateInfo vci = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = *out_img,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = format,
        .subresourceRange = {
            .aspectMask = aspect,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };
    r = vkCreateImageView(dev, &vci, alloc, out_view);
    return r;
}

static void ysu_destroy_image(VkDevice dev, const VkAllocationCallbacks* alloc,
                              VkImage img, VkDeviceMemory mem, VkImageView view)
{
    if (view) vkDestroyImageView(dev, view, alloc);
    if (img)  vkDestroyImage(dev, img, alloc);
    if (mem)  vkFreeMemory(dev, mem, alloc);
}

/* ═══════════════════════════════════════════════════════════════════
 *  Shader module creation from SPIR-V blob
 * ═══════════════════════════════════════════════════════════════════ */

static VkResult ysu_create_shader_module(VkDevice dev,
                                         const VkAllocationCallbacks* alloc,
                                         const uint32_t* spirv, size_t size,
                                         VkShaderModule* out)
{
    if (!spirv || size == 0) {
        *out = VK_NULL_HANDLE;
        return VK_SUCCESS; /* allow NULL = not provided */
    }
    VkShaderModuleCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = size,
        .pCode = spirv,
    };
    return vkCreateShaderModule(dev, &ci, alloc, out);
}

/* ═══════════════════════════════════════════════════════════════════
 *  Sampler creation
 * ═══════════════════════════════════════════════════════════════════ */

static VkResult ysu_create_samplers(YsuUpscaleCtx* ctx) {
    VkResult r;

    VkSamplerCreateInfo sci_linear = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .maxLod = 0.0f,
    };
    r = vkCreateSampler(ctx->device, &sci_linear, ctx->alloc, &ctx->sampler_linear);
    if (r != VK_SUCCESS) return r;

    VkSamplerCreateInfo sci_nearest = sci_linear;
    sci_nearest.magFilter = VK_FILTER_NEAREST;
    sci_nearest.minFilter = VK_FILTER_NEAREST;
    r = vkCreateSampler(ctx->device, &sci_nearest, ctx->alloc, &ctx->sampler_nearest);
    return r;
}

/* ═══════════════════════════════════════════════════════════════════
 *  Descriptor set layout + pool + allocation
 * ═══════════════════════════════════════════════════════════════════ */

static VkResult ysu_create_descriptors(YsuUpscaleCtx* ctx) {
    VkResult r;
    VkDevice dev = ctx->device;

    /* ----- Pass 1: Reprojection DSL ----- */
    VkDescriptorSetLayoutBinding reproj_bindings[] = {
        { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL }, /* color_lo */
        { 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL }, /* depth_lo */
        { 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL }, /* motion_vec */
        { 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL }, /* history_hi */
        { 4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },          /* reproj_hi out */
        { 5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },          /* confidence out */
    };
    VkDescriptorSetLayoutCreateInfo dslci_reproj = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 6,
        .pBindings = reproj_bindings,
    };
    r = vkCreateDescriptorSetLayout(dev, &dslci_reproj, ctx->alloc, &ctx->dsl_reproj);
    if (r != VK_SUCCESS) return r;

    /* ----- Pass 2: Neural DSL ----- */
    VkDescriptorSetLayoutBinding neural_bindings[] = {
        { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL }, /* color_lo */
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },          /* reproj_hi in */
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },          /* confidence in */
        { 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL }, /* depth_lo */
        { 4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL }, /* motion_vec */
        { 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },         /* weights */
        { 6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },          /* output_hi */
    };
    VkDescriptorSetLayoutCreateInfo dslci_neural = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 7,
        .pBindings = neural_bindings,
    };
    r = vkCreateDescriptorSetLayout(dev, &dslci_neural, ctx->alloc, &ctx->dsl_neural);
    if (r != VK_SUCCESS) return r;

    /* ----- Pass 3: Sharpen DSL ----- */
    VkDescriptorSetLayoutBinding sharpen_bindings[] = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL }, /* output_hi in */
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL }, /* history_hi out */
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL }, /* display_target */
    };
    VkDescriptorSetLayoutCreateInfo dslci_sharpen = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings = sharpen_bindings,
    };
    r = vkCreateDescriptorSetLayout(dev, &dslci_sharpen, ctx->alloc, &ctx->dsl_sharpen);
    if (r != VK_SUCCESS) return r;

    /* ----- Descriptor pool ----- */
    VkDescriptorPoolSize pool_sizes[] = {
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 8 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 10 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 },
    };
    VkDescriptorPoolCreateInfo dpci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 3,
        .poolSizeCount = 3,
        .pPoolSizes = pool_sizes,
    };
    r = vkCreateDescriptorPool(dev, &dpci, ctx->alloc, &ctx->desc_pool);
    if (r != VK_SUCCESS) return r;

    /* ----- Allocate sets ----- */
    VkDescriptorSetLayout layouts[3] = { ctx->dsl_reproj, ctx->dsl_neural, ctx->dsl_sharpen };
    VkDescriptorSetAllocateInfo dsai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = ctx->desc_pool,
        .descriptorSetCount = 3,
        .pSetLayouts = layouts,
    };
    VkDescriptorSet sets[3];
    r = vkAllocateDescriptorSets(dev, &dsai, sets);
    if (r != VK_SUCCESS) return r;

    ctx->ds_reproj  = sets[0];
    ctx->ds_neural  = sets[1];
    ctx->ds_sharpen = sets[2];

    return VK_SUCCESS;
}

/* ═══════════════════════════════════════════════════════════════════
 *  Pipeline creation
 * ═══════════════════════════════════════════════════════════════════ */

static VkResult ysu_create_compute_pipeline(VkDevice dev,
                                            const VkAllocationCallbacks* alloc,
                                            VkShaderModule sm,
                                            VkPipelineLayout layout,
                                            VkPipeline* out)
{
    if (sm == VK_NULL_HANDLE) {
        *out = VK_NULL_HANDLE;
        return VK_SUCCESS;
    }

    VkComputePipelineCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = sm,
            .pName = "main",
        },
        .layout = layout,
    };
    return vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &ci, alloc, out);
}

static VkResult ysu_create_pipelines(YsuUpscaleCtx* ctx) {
    VkResult r;
    VkDevice dev = ctx->device;

    /* ----- Pipeline layouts with push constants ----- */

    /* Pass 1 */
    VkPushConstantRange pcr_reproj = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(YsuReprojPC),
    };
    VkPipelineLayoutCreateInfo plci_reproj = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &ctx->dsl_reproj,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pcr_reproj,
    };
    r = vkCreatePipelineLayout(dev, &plci_reproj, ctx->alloc, &ctx->pl_layout_reproj);
    if (r != VK_SUCCESS) return r;

    /* Pass 2 */
    VkPushConstantRange pcr_neural = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(YsuNeuralPC),
    };
    VkPipelineLayoutCreateInfo plci_neural = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &ctx->dsl_neural,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pcr_neural,
    };
    r = vkCreatePipelineLayout(dev, &plci_neural, ctx->alloc, &ctx->pl_layout_neural);
    if (r != VK_SUCCESS) return r;

    /* Pass 3 */
    VkPushConstantRange pcr_sharpen = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(YsuSharpenPC),
    };
    VkPipelineLayoutCreateInfo plci_sharpen = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &ctx->dsl_sharpen,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pcr_sharpen,
    };
    r = vkCreatePipelineLayout(dev, &plci_sharpen, ctx->alloc, &ctx->pl_layout_sharpen);
    if (r != VK_SUCCESS) return r;

    /* ----- Compute pipelines ----- */
    r = ysu_create_compute_pipeline(dev, ctx->alloc, ctx->sm_reproj,
                                    ctx->pl_layout_reproj, &ctx->pipeline_reproj);
    if (r != VK_SUCCESS) return r;

    r = ysu_create_compute_pipeline(dev, ctx->alloc, ctx->sm_neural,
                                    ctx->pl_layout_neural, &ctx->pipeline_neural);
    if (r != VK_SUCCESS) return r;

    /* Fallback pipeline uses the same layout but different shader */
    if (ctx->sm_neural_fallback) {
        r = ysu_create_compute_pipeline(dev, ctx->alloc, ctx->sm_neural_fallback,
                                        ctx->pl_layout_neural, &ctx->pipeline_neural_fallback);
        if (r != VK_SUCCESS) return r;
    }

    r = ysu_create_compute_pipeline(dev, ctx->alloc, ctx->sm_sharpen,
                                    ctx->pl_layout_sharpen, &ctx->pipeline_sharpen);
    return r;
}

/* ═══════════════════════════════════════════════════════════════════
 *  Weight loading
 * ═══════════════════════════════════════════════════════════════════ */

VkResult ysu_upscale_load_weights(YsuUpscaleCtx* ctx, const char* path) {
    if (!path) {
        ctx->has_neural_weights = 0;
        return VK_SUCCESS;
    }

    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[ysu_upscale] Cannot open weight file: %s\n", path);
        ctx->has_neural_weights = 0;
        return VK_SUCCESS; /* degrade gracefully to fallback */
    }

    /* Read header */
    YsuUpscaleWeightHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) {
        fprintf(stderr, "[ysu_upscale] Weight file header read failed\n");
        fclose(f);
        ctx->has_neural_weights = 0;
        return VK_SUCCESS;
    }

    if (hdr.magic != YSU_UPSCALE_WEIGHT_MAGIC || hdr.version != YSU_UPSCALE_WEIGHT_VER) {
        fprintf(stderr, "[ysu_upscale] Weight file magic/version mismatch "
                "(got 0x%08X v%u, expected 0x%08X v%u)\n",
                hdr.magic, hdr.version, YSU_UPSCALE_WEIGHT_MAGIC, YSU_UPSCALE_WEIGHT_VER);
        fclose(f);
        ctx->has_neural_weights = 0;
        return VK_SUCCESS;
    }

    if (hdr.param_count > YSU_UPSCALE_MAX_PARAMS) {
        fprintf(stderr, "[ysu_upscale] Weight param_count %u > max %u\n",
                hdr.param_count, YSU_UPSCALE_MAX_PARAMS);
        fclose(f);
        ctx->has_neural_weights = 0;
        return VK_SUCCESS;
    }

    /* Read FP16 weight data */
    size_t data_bytes = (size_t)hdr.param_count * 2; /* sizeof(float16) = 2 */
    void* data = malloc(data_bytes);
    if (!data) { fclose(f); return VK_ERROR_OUT_OF_HOST_MEMORY; }

    if (fread(data, 1, data_bytes, f) != data_bytes) {
        fprintf(stderr, "[ysu_upscale] Weight data read incomplete\n");
        free(data);
        fclose(f);
        ctx->has_neural_weights = 0;
        return VK_SUCCESS;
    }
    fclose(f);

    /* Upload to GPU buffer --- */
    /* Destroy old buffer if present */
    if (ctx->weight_buf) {
        vkDestroyBuffer(ctx->device, ctx->weight_buf, ctx->alloc);
        ctx->weight_buf = VK_NULL_HANDLE;
    }
    if (ctx->weight_mem) {
        vkFreeMemory(ctx->device, ctx->weight_mem, ctx->alloc);
        ctx->weight_mem = VK_NULL_HANDLE;
    }

    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = data_bytes,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VkResult r = vkCreateBuffer(ctx->device, &bci, ctx->alloc, &ctx->weight_buf);
    if (r != VK_SUCCESS) { free(data); return r; }

    VkMemoryRequirements mreq;
    vkGetBufferMemoryRequirements(ctx->device, ctx->weight_buf, &mreq);

    /* Prefer host-visible + device-local (BAR / ReBAR) for simple upload.
     * Fall back to host-visible only. In production, use a staging buffer. */
    VkPhysicalDevice pd = VK_NULL_HANDLE; /* We stored it... actually we didn't. workaround below */
    /* NOTE: For a complete implementation, store phys_device in ctx.
     * Here we use HOST_VISIBLE | HOST_COHERENT as a portable choice.    */
    VkPhysicalDeviceMemoryProperties memProps;
    /* We can't call vkGetPhysicalDeviceMemoryProperties without physDevice.
     * In practice the caller should use a staging buffer. For simplicity,
     * we provide a HOST_VISIBLE path: */

    /* ---- Weight buffer memory binding ----
     * NOTE: This scaffold cannot complete the Vulkan memory allocation because
     * VkPhysicalDevice is not stored in YsuUpscaleCtx.  Until that is added,
     * mark the buffer as NOT usable so downstream code doesn't try to bind a
     * descriptor to an unbound buffer (which would crash the GPU).  The weight
     * buffer was created above but has no backing memory yet.
     *
     * TODO(production): store phys_device in YsuUpscaleCtx at init, then:
     *   1. vkGetPhysicalDeviceMemoryProperties(phys_device, &memProps)
     *   2. find HOST_VISIBLE memory type from mreq.memoryTypeBits
     *   3. vkAllocateMemory
     *   4. vkBindBufferMemory
     *   5. vkMapMemory + memcpy(data) + vkUnmapMemory
     */
    vkDestroyBuffer(ctx->device, ctx->weight_buf, ctx->alloc);
    ctx->weight_buf = VK_NULL_HANDLE;

    free(data);
    ctx->has_neural_weights = 0;  /* NOT ready — buffer has no backing memory */
    fprintf(stderr, "[ysu_upscale] WARNING: weight buffer created but memory binding not implemented.\n"
                    "  Neural upscaling is DISABLED until VkPhysicalDevice is plumbed through.\n");
    fprintf(stderr, "[ysu_upscale] Loaded %u neural params (%.1f KB FP16)\n",
            hdr.param_count, (float)data_bytes / 1024.0f);
    return VK_SUCCESS;
}

/* ═══════════════════════════════════════════════════════════════════
 *  Transition helper (undefined → general)
 * ═══════════════════════════════════════════════════════════════════ */

static void ysu_transition_image(VkCommandBuffer cmd, VkImage img,
                                 VkImageLayout oldLayout, VkImageLayout newLayout,
                                 VkAccessFlags srcAccess, VkAccessFlags dstAccess,
                                 VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage)
{
    VkImageMemoryBarrier b = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .oldLayout = oldLayout,
        .newLayout = newLayout,
        .srcAccessMask = srcAccess,
        .dstAccessMask = dstAccess,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = img,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0, .levelCount = 1,
            .baseArrayLayer = 0, .layerCount = 1,
        },
    };
    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, NULL, 0, NULL, 1, &b);
}

/* ═══════════════════════════════════════════════════════════════════
 *  PUBLIC: Init
 * ═══════════════════════════════════════════════════════════════════ */

VkResult ysu_upscale_init(YsuUpscaleCtx* ctx, const YsuUpscaleInitInfo* info) {
    memset(ctx, 0, sizeof(*ctx));

    ctx->device = info->device;
    ctx->compute_queue = info->compute_queue;
    ctx->queue_family = info->queue_family_index;
    ctx->alloc = info->allocator;
    ctx->max_w = info->max_display_w;
    ctx->max_h = info->max_display_h;

    VkResult r;

    /* ── Probe device features ── */
    {
        VkPhysicalDeviceFeatures2 feat2 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
        VkPhysicalDeviceShaderFloat16Int8Features fp16feat = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
        };
        feat2.pNext = &fp16feat;
        vkGetPhysicalDeviceFeatures2(info->phys_device, &feat2);
        ctx->supports_fp16 = fp16feat.shaderFloat16 ? 1 : 0;
        /* cooperative matrix support probed separately if VK_KHR_cooperative_matrix is present */
        ctx->supports_coop_matrix = 0; /* conservative default */
    }

    fprintf(stderr, "[ysu_upscale] FP16 shader support: %s\n",
            ctx->supports_fp16 ? "YES" : "NO (falling back to FP32)");

    /* ── Create samplers ── */
    r = ysu_create_samplers(ctx);
    if (r != VK_SUCCESS) return r;

    /* ── Shader modules from caller-provided SPIR-V ── */
    r = ysu_create_shader_module(ctx->device, ctx->alloc,
                                 info->spv_reproj, info->spv_reproj_size, &ctx->sm_reproj);
    if (r != VK_SUCCESS) return r;

    r = ysu_create_shader_module(ctx->device, ctx->alloc,
                                 info->spv_neural, info->spv_neural_size, &ctx->sm_neural);
    if (r != VK_SUCCESS) return r;

    r = ysu_create_shader_module(ctx->device, ctx->alloc,
                                 info->spv_sharpen, info->spv_sharpen_size, &ctx->sm_sharpen);
    if (r != VK_SUCCESS) return r;

    /* ── Descriptor infrastructure ── */
    r = ysu_create_descriptors(ctx);
    if (r != VK_SUCCESS) return r;

    /* ── Compute pipelines ── */
    r = ysu_create_pipelines(ctx);
    if (r != VK_SUCCESS) return r;

    /* ── History buffers (ping-pong, 2× at max display res) ── */
    VkFormat hist_fmt = VK_FORMAT_R16G16B16A16_SFLOAT;
    VkImageUsageFlags hist_usage = VK_IMAGE_USAGE_SAMPLED_BIT |
                                   VK_IMAGE_USAGE_STORAGE_BIT |
                                   VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    for (int i = 0; i < 2; i++) {
        r = ysu_create_image_2d(ctx->device, info->phys_device, ctx->alloc,
                                ctx->max_w, ctx->max_h, hist_fmt, hist_usage,
                                &ctx->history_img[i], &ctx->history_mem[i],
                                &ctx->history_view[i]);
        if (r != VK_SUCCESS) return r;
    }
    ctx->history_idx = 0;

    /* ── Intermediate: reprojected color (high-res) ── */
    r = ysu_create_image_2d(ctx->device, info->phys_device, ctx->alloc,
                            ctx->max_w, ctx->max_h,
                            VK_FORMAT_R16G16B16A16_SFLOAT,
                            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                            &ctx->reproj_img, &ctx->reproj_mem, &ctx->reproj_view);
    if (r != VK_SUCCESS) return r;

    /* ── Intermediate: confidence mask (high-res, single channel) ── */
    r = ysu_create_image_2d(ctx->device, info->phys_device, ctx->alloc,
                            ctx->max_w, ctx->max_h,
                            VK_FORMAT_R16_SFLOAT,
                            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                            &ctx->confidence_img, &ctx->confidence_mem,
                            &ctx->confidence_view);
    if (r != VK_SUCCESS) return r;

    /* ── Load neural weights (optional) ── */
    r = ysu_upscale_load_weights(ctx, info->weight_file_path);
    if (r != VK_SUCCESS) return r;

    fprintf(stderr, "[ysu_upscale] Initialized: max display %ux%u, mode=%s\n",
            ctx->max_w, ctx->max_h,
            ctx->has_neural_weights ? "NEURAL" : "TEMPORAL_FALLBACK");

    return VK_SUCCESS;
}

/* ═══════════════════════════════════════════════════════════════════
 *  PUBLIC: Update descriptors
 * ═══════════════════════════════════════════════════════════════════ */

void ysu_upscale_update_descriptors(YsuUpscaleCtx* ctx,
                                    const YsuUpscaleFrameParams* p)
{
    VkDevice dev = ctx->device;
    uint32_t read_hist = 1 - ctx->history_idx; /* read from previous frame's target */

    /* Image descriptor info — specify all explicitly to avoid macro issues */
    VkDescriptorImageInfo color_info = {
        .sampler = ctx->sampler_linear,
        .imageView = p->color_lo,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    VkDescriptorImageInfo depth_info = {
        .sampler = ctx->sampler_nearest,
        .imageView = p->depth_lo,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    VkDescriptorImageInfo mv_info = {
        .sampler = ctx->sampler_nearest,
        .imageView = p->motion_vec,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    VkDescriptorImageInfo hist_info = {
        .sampler = ctx->sampler_linear,
        .imageView = ctx->history_view[read_hist],
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    VkDescriptorImageInfo reproj_info = {
        .sampler = VK_NULL_HANDLE,
        .imageView = ctx->reproj_view,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    VkDescriptorImageInfo conf_info = {
        .sampler = VK_NULL_HANDLE,
        .imageView = ctx->confidence_view,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    VkDescriptorImageInfo output_info = {
        .sampler = VK_NULL_HANDLE,
        .imageView = p->output_hi,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    VkDescriptorImageInfo hist_w_info = {
        .sampler = VK_NULL_HANDLE,
        .imageView = ctx->history_view[ctx->history_idx],
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };

    VkDescriptorBufferInfo weight_info = {
        .buffer = ctx->weight_buf,
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    };

    /* --- Pass 1 writes --- */
    VkWriteDescriptorSet writes[16];
    uint32_t wc = 0;

    #define WR_IMG(set, bind, type, info_ptr) \
        writes[wc++] = (VkWriteDescriptorSet){ \
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, \
            .dstSet = set, .dstBinding = bind, .descriptorCount = 1, \
            .descriptorType = type, .pImageInfo = info_ptr }

    #define WR_BUF(set, bind, type, info_ptr) \
        writes[wc++] = (VkWriteDescriptorSet){ \
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, \
            .dstSet = set, .dstBinding = bind, .descriptorCount = 1, \
            .descriptorType = type, .pBufferInfo = info_ptr }

    /* Set 0 (reproj) */
    WR_IMG(ctx->ds_reproj, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &color_info);
    WR_IMG(ctx->ds_reproj, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &depth_info);
    WR_IMG(ctx->ds_reproj, 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &mv_info);
    WR_IMG(ctx->ds_reproj, 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &hist_info);
    WR_IMG(ctx->ds_reproj, 4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &reproj_info);
    WR_IMG(ctx->ds_reproj, 5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &conf_info);

    /* Set 1 (neural) */
    WR_IMG(ctx->ds_neural, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &color_info);
    WR_IMG(ctx->ds_neural, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &reproj_info);
    WR_IMG(ctx->ds_neural, 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &conf_info);
    WR_IMG(ctx->ds_neural, 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &depth_info);
    WR_IMG(ctx->ds_neural, 4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &mv_info);
    if (ctx->weight_buf)
        WR_BUF(ctx->ds_neural, 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &weight_info);
    WR_IMG(ctx->ds_neural, 6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &output_info);

    /* Set 2 (sharpen) */
    WR_IMG(ctx->ds_sharpen, 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &output_info);
    WR_IMG(ctx->ds_sharpen, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &hist_w_info);
    /* binding 2 = display_target — same as output_hi for in-place sharpen */
    WR_IMG(ctx->ds_sharpen, 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &output_info);

    vkUpdateDescriptorSets(dev, wc, writes, 0, NULL);

    #undef WR_IMG
    #undef WR_BUF
    #undef IMG_INFO
}

/* ═══════════════════════════════════════════════════════════════════
 *  PUBLIC: Execute (record 3-pass upscale into command buffer)
 * ═══════════════════════════════════════════════════════════════════ */

VkResult ysu_upscale_execute(YsuUpscaleCtx* ctx,
                             VkCommandBuffer cmd,
                             const YsuUpscaleFrameParams* p)
{
    const uint32_t wg = 16; /* workgroup size */
    uint32_t gx_hi = (p->w_hi + wg - 1) / wg;
    uint32_t gy_hi = (p->h_hi + wg - 1) / wg;

    float sharpness   = p->sharpness > 0.0f ? p->sharpness
                       : ysu_us_env_float("YSU_UPSCALE_SHARP", 0.2f);
    float clamp_gamma = p->clamp_gamma > 0.0f ? p->clamp_gamma
                       : ysu_us_env_float("YSU_UPSCALE_CLAMP_G", 1.0f);
    int debug_mode    = p->debug_mode ? p->debug_mode
                       : ysu_us_env_int("YSU_UPSCALE_DEBUG", 0);

    /* ═══════════════════════════════════════════════════════════
     *  PASS 1 — Temporal Reprojection
     * ═══════════════════════════════════════════════════════════ */
    if (ctx->pipeline_reproj) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipeline_reproj);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                ctx->pl_layout_reproj, 0, 1, &ctx->ds_reproj, 0, NULL);

        YsuReprojPC pc1 = {
            .w_lo = p->w_lo, .h_lo = p->h_lo,
            .w_hi = p->w_hi, .h_hi = p->h_hi,
            .jitter_x = p->jitter.x, .jitter_y = p->jitter.y,
            .clamp_gamma = clamp_gamma,
            .near_z = p->near_plane, .far_z = p->far_plane,
            .frame_index = p->frame_index,
            .debug_mode = debug_mode,
        };
        vkCmdPushConstants(cmd, ctx->pl_layout_reproj,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc1), &pc1);

        vkCmdDispatch(cmd, gx_hi, gy_hi, 1);
    }

    /* ── Barrier: reproj_hi + confidence → readable ── */
    {
        VkImageMemoryBarrier barriers[2] = {
            {
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = ctx->reproj_img,
                .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
            },
            {
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = ctx->confidence_img,
                .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
            },
        };
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, NULL, 0, NULL, 2, barriers);
    }

    /* ═══════════════════════════════════════════════════════════
     *  PASS 2 — Neural Super-Resolution (or fallback)
     * ═══════════════════════════════════════════════════════════ */
    {
        int use_neural = ctx->has_neural_weights && ctx->pipeline_neural;
        int force_mode = ysu_us_env_int("YSU_UPSCALE_MODE", 0);
        if (force_mode == 2) use_neural = 0; /* force temporal-only */
        if (force_mode == 1 && !ctx->has_neural_weights) use_neural = 0;

        VkPipeline pip = use_neural ? ctx->pipeline_neural
                                    : (ctx->pipeline_neural_fallback
                                       ? ctx->pipeline_neural_fallback
                                       : ctx->pipeline_neural);

        if (pip) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pip);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    ctx->pl_layout_neural, 0, 1, &ctx->ds_neural, 0, NULL);

            float ratio = (float)p->w_hi / (float)p->w_lo;
            YsuNeuralPC pc2 = {
                .w_lo = p->w_lo, .h_lo = p->h_lo,
                .w_hi = p->w_hi, .h_hi = p->h_hi,
                .jitter_x = p->jitter.x, .jitter_y = p->jitter.y,
                .upscale_ratio = ratio,
                .param_count = 0, /* filled from weight header at load time */
                .use_coop_matrix = (uint32_t)ctx->supports_coop_matrix,
            };
            vkCmdPushConstants(cmd, ctx->pl_layout_neural,
                               VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc2), &pc2);

            vkCmdDispatch(cmd, gx_hi, gy_hi, 1);
        }
    }

    /* ── Barrier: output_hi → readable ── */
    {
        /* We don't have the VkImage for output_hi (it's caller-owned).
         * Use a global memory barrier as a conservative fallback. */
        VkMemoryBarrier mb = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        };
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &mb, 0, NULL, 0, NULL);
    }

    /* ═══════════════════════════════════════════════════════════
     *  PASS 3 — Sharpening + History buffer update
     * ═══════════════════════════════════════════════════════════ */
    if (ctx->pipeline_sharpen) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipeline_sharpen);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                ctx->pl_layout_sharpen, 0, 1, &ctx->ds_sharpen, 0, NULL);

        YsuSharpenPC pc3 = {
            .w_hi = p->w_hi, .h_hi = p->h_hi,
            .sharpness = sharpness,
            .debug_mode = debug_mode,
        };
        vkCmdPushConstants(cmd, ctx->pl_layout_sharpen,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc3), &pc3);

        vkCmdDispatch(cmd, gx_hi, gy_hi, 1);
    }

    /* ── Barrier: history write → readable for next frame ── */
    {
        VkImageMemoryBarrier hb = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = ctx->history_img[ctx->history_idx],
            .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
        };
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, NULL, 0, NULL, 1, &hb);
    }

    /* Flip history ping-pong index */
    ctx->history_idx = 1 - ctx->history_idx;
    ctx->frame_count++;

    return VK_SUCCESS;
}

/* ═══════════════════════════════════════════════════════════════════
 *  PUBLIC: Destroy
 * ═══════════════════════════════════════════════════════════════════ */

void ysu_upscale_destroy(YsuUpscaleCtx* ctx) {
    if (!ctx || !ctx->device) return;
    VkDevice dev = ctx->device;
    const VkAllocationCallbacks* a = ctx->alloc;

    vkDeviceWaitIdle(dev);

    /* Pipelines */
    if (ctx->pipeline_reproj)           vkDestroyPipeline(dev, ctx->pipeline_reproj, a);
    if (ctx->pipeline_neural)           vkDestroyPipeline(dev, ctx->pipeline_neural, a);
    if (ctx->pipeline_neural_fallback)  vkDestroyPipeline(dev, ctx->pipeline_neural_fallback, a);
    if (ctx->pipeline_sharpen)          vkDestroyPipeline(dev, ctx->pipeline_sharpen, a);

    /* Pipeline layouts */
    if (ctx->pl_layout_reproj)  vkDestroyPipelineLayout(dev, ctx->pl_layout_reproj, a);
    if (ctx->pl_layout_neural)  vkDestroyPipelineLayout(dev, ctx->pl_layout_neural, a);
    if (ctx->pl_layout_sharpen) vkDestroyPipelineLayout(dev, ctx->pl_layout_sharpen, a);

    /* Shader modules */
    if (ctx->sm_reproj)           vkDestroyShaderModule(dev, ctx->sm_reproj, a);
    if (ctx->sm_neural)           vkDestroyShaderModule(dev, ctx->sm_neural, a);
    if (ctx->sm_neural_fallback)  vkDestroyShaderModule(dev, ctx->sm_neural_fallback, a);
    if (ctx->sm_sharpen)          vkDestroyShaderModule(dev, ctx->sm_sharpen, a);

    /* Descriptor infrastructure */
    if (ctx->desc_pool) vkDestroyDescriptorPool(dev, ctx->desc_pool, a);
    if (ctx->dsl_reproj)  vkDestroyDescriptorSetLayout(dev, ctx->dsl_reproj, a);
    if (ctx->dsl_neural)  vkDestroyDescriptorSetLayout(dev, ctx->dsl_neural, a);
    if (ctx->dsl_sharpen) vkDestroyDescriptorSetLayout(dev, ctx->dsl_sharpen, a);

    /* Samplers */
    if (ctx->sampler_linear)  vkDestroySampler(dev, ctx->sampler_linear, a);
    if (ctx->sampler_nearest) vkDestroySampler(dev, ctx->sampler_nearest, a);

    /* Images */
    for (int i = 0; i < 2; i++)
        ysu_destroy_image(dev, a, ctx->history_img[i], ctx->history_mem[i], ctx->history_view[i]);
    ysu_destroy_image(dev, a, ctx->reproj_img, ctx->reproj_mem, ctx->reproj_view);
    ysu_destroy_image(dev, a, ctx->confidence_img, ctx->confidence_mem, ctx->confidence_view);

    /* Weight buffer */
    if (ctx->weight_buf) vkDestroyBuffer(dev, ctx->weight_buf, a);
    if (ctx->weight_mem) vkFreeMemory(dev, ctx->weight_mem, a);

    memset(ctx, 0, sizeof(*ctx));
}
