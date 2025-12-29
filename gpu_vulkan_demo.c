#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

static void die(const char* msg, VkResult r){
    fprintf(stderr, "FATAL: %s (VkResult=%d)\n", msg, (int)r);
    exit(1);
}

static uint8_t* read_file(const char* path, size_t* out_sz){
    FILE* f = fopen(path, "rb");
    if(!f){ fprintf(stderr,"can't open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t* buf = (uint8_t*)malloc((size_t)n);
    if(!buf){ fclose(f); exit(1); }
    if(fread(buf, 1, (size_t)n, f) != (size_t)n){ fclose(f); exit(1); }
    fclose(f);
    *out_sz = (size_t)n;
    return buf;
}

static uint32_t find_memtype(VkPhysicalDevice phy, uint32_t typeBits, VkMemoryPropertyFlags req){
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phy, &mp);
    for(uint32_t i=0;i<mp.memoryTypeCount;i++){
        if((typeBits & (1u<<i)) && ((mp.memoryTypes[i].propertyFlags & req) == req))
            return i;
    }
    return UINT32_MAX;
}

static void ppm_write_rgb8(const char* path, const float* rgba32f, int W, int H){
    FILE* f = fopen(path, "wb");
    if(!f){ fprintf(stderr,"can't write %s\n", path); return; }
    fprintf(f,"P6\n%d %d\n255\n", W, H);

    for(int y=0;y<H;y++){
        for(int x=0;x<W;x++){
            const float* p = rgba32f + 4*(y*W + x);
            float r=p[0], g=p[1], b=p[2];
            if(r<0)r=0; if(g<0)g=0; if(b<0)b=0;
            if(r>1)r=1; if(g>1)g=1; if(b>1)b=1;
            uint8_t rgb[3] = {
                (uint8_t)(r*255.0f + 0.5f),
                (uint8_t)(g*255.0f + 0.5f),
                (uint8_t)(b*255.0f + 0.5f),
            };
            fwrite(rgb,1,3,f);
        }
    }
    fclose(f);
}

static VkImage create_image_rgba32f(VkDevice dev, int W, int H, VkImageUsageFlags usage){
    VkImageCreateInfo ci = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    ci.extent.width  = (uint32_t)W;
    ci.extent.height = (uint32_t)H;
    ci.extent.depth  = 1;
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = usage;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImage img = 0;
    VkResult r = vkCreateImage(dev, &ci, NULL, &img);
    if(r!=VK_SUCCESS) die("vkCreateImage", r);
    return img;
}

static VkDeviceMemory alloc_bind_image_mem(VkPhysicalDevice phy, VkDevice dev, VkImage img){
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(dev, img, &req);

    uint32_t mt = find_memtype(phy, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if(mt == UINT32_MAX) die("no DEVICE_LOCAL memtype for image", VK_ERROR_OUT_OF_DEVICE_MEMORY);

    VkMemoryAllocateInfo ai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = mt;

    VkDeviceMemory mem = 0;
    VkResult r = vkAllocateMemory(dev, &ai, NULL, &mem);
    if(r!=VK_SUCCESS) die("vkAllocateMemory(image)", r);

    r = vkBindImageMemory(dev, img, mem, 0);
    if(r!=VK_SUCCESS) die("vkBindImageMemory", r);
    return mem;
}

static VkImageView create_image_view(VkDevice dev, VkImage img){
    VkImageViewCreateInfo iv = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    iv.image = img;
    iv.viewType = VK_IMAGE_VIEW_TYPE_2D;
    iv.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    iv.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    iv.subresourceRange.levelCount = 1;
    iv.subresourceRange.layerCount = 1;

    VkImageView view = 0;
    VkResult r = vkCreateImageView(dev, &iv, NULL, &view);
    if(r!=VK_SUCCESS) die("vkCreateImageView", r);
    return view;
}

static VkBuffer create_buffer(VkDevice dev, VkDeviceSize size, VkBufferUsageFlags usage){
    VkBufferCreateInfo bi = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buf = 0;
    VkResult r = vkCreateBuffer(dev, &bi, NULL, &buf);
    if(r!=VK_SUCCESS) die("vkCreateBuffer", r);
    return buf;
}

static VkDeviceMemory alloc_bind_buffer_mem(VkPhysicalDevice phy, VkDevice dev, VkBuffer buf, VkMemoryPropertyFlags flags){
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, buf, &req);

    uint32_t mt = find_memtype(phy, req.memoryTypeBits, flags);
    if(mt == UINT32_MAX) die("no memtype for buffer", VK_ERROR_OUT_OF_DEVICE_MEMORY);

    VkMemoryAllocateInfo ai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = mt;

    VkDeviceMemory mem = 0;
    VkResult r = vkAllocateMemory(dev, &ai, NULL, &mem);
    if(r!=VK_SUCCESS) die("vkAllocateMemory(buffer)", r);

    r = vkBindBufferMemory(dev, buf, mem, 0);
    if(r!=VK_SUCCESS) die("vkBindBufferMemory", r);

    return mem;
}

int main(){
    // 360 4K default; change if you want
    const int W = 4096;
    const int H = 2048;

    const char* spv_path = "shaders/fill.comp.spv";

    int frames = 128; // default SPP
    const char* e = getenv("YSU_GPU_SPP");
    if(e && *e) frames = atoi(e);
    if(frames < 1) frames = 1;

    int seed = 1337;
    const char* senv = getenv("YSU_GPU_SEED");
    if(senv && *senv) seed = atoi(senv);

    printf("[GPU] W=%d H=%d SPP=%d seed=%d\n", W, H, frames, seed);

    // ---------- Instance ----------
    VkApplicationInfo ai = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    ai.pApplicationName = "YSU Vulkan Accum Demo";
    ai.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ici.pApplicationInfo = &ai;

    VkInstance inst = 0;
    VkResult r = vkCreateInstance(&ici, NULL, &inst);
    if(r!=VK_SUCCESS) die("vkCreateInstance", r);

    // ---------- Physical device ----------
    uint32_t pcount = 0;
    vkEnumeratePhysicalDevices(inst, &pcount, NULL);
    if(!pcount) die("no Vulkan physical devices", VK_ERROR_INITIALIZATION_FAILED);

    VkPhysicalDevice* phys = (VkPhysicalDevice*)malloc(sizeof(VkPhysicalDevice)*pcount);
    vkEnumeratePhysicalDevices(inst, &pcount, phys);
    VkPhysicalDevice phy = phys[0];
    free(phys);

    // ---------- Queue family (compute) ----------
    uint32_t qfc = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phy, &qfc, NULL);
    VkQueueFamilyProperties* qfp = (VkQueueFamilyProperties*)malloc(sizeof(*qfp)*qfc);
    vkGetPhysicalDeviceQueueFamilyProperties(phy, &qfc, qfp);

    uint32_t q_compute = UINT32_MAX;
    for(uint32_t i=0;i<qfc;i++){
        if(qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT){
            q_compute = i;
            break;
        }
    }
    free(qfp);
    if(q_compute == UINT32_MAX) die("no compute queue family", VK_ERROR_INITIALIZATION_FAILED);

    float qprio = 1.0f;
    VkDeviceQueueCreateInfo dqci = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    dqci.queueFamilyIndex = q_compute;
    dqci.queueCount = 1;
    dqci.pQueuePriorities = &qprio;

    VkDeviceCreateInfo dci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &dqci;

    VkDevice dev = 0;
    r = vkCreateDevice(phy, &dci, NULL, &dev);
    if(r!=VK_SUCCESS) die("vkCreateDevice", r);

    VkQueue q = 0;
    vkGetDeviceQueue(dev, q_compute, 0, &q);

    // ---------- Command pool/buffer ----------
    VkCommandPoolCreateInfo cpci = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    cpci.queueFamilyIndex = q_compute;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    VkCommandPool pool = 0;
    r = vkCreateCommandPool(dev, &cpci, NULL, &pool);
    if(r!=VK_SUCCESS) die("vkCreateCommandPool", r);

    VkCommandBufferAllocateInfo cbai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cbai.commandPool = pool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;

    VkCommandBuffer cb = 0;
    r = vkAllocateCommandBuffers(dev, &cbai, &cb);
    if(r!=VK_SUCCESS) die("vkAllocateCommandBuffers", r);

    // ---------- Images: out + accum ----------
    VkImageUsageFlags img_usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    VkImage out_img   = create_image_rgba32f(dev, W, H, img_usage);
    VkImage accum_img = create_image_rgba32f(dev, W, H, img_usage);

    VkDeviceMemory out_mem   = alloc_bind_image_mem(phy, dev, out_img);
    VkDeviceMemory accum_mem = alloc_bind_image_mem(phy, dev, accum_img);

    VkImageView out_view   = create_image_view(dev, out_img);
    VkImageView accum_view = create_image_view(dev, accum_img);

    // ---------- Readback buffer ----------
    VkDeviceSize out_bytes = (VkDeviceSize)W * (VkDeviceSize)H * 16; // RGBA32F
    VkBuffer read_buf = create_buffer(dev, out_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    VkDeviceMemory read_mem = alloc_bind_buffer_mem(
        phy, dev, read_buf,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    // ---------- Descriptor set layout: binding0 outImg, binding1 accumImg ----------
    VkDescriptorSetLayoutBinding binds[2];
    memset(binds, 0, sizeof(binds));

    binds[0].binding = 0;
    binds[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    binds[0].descriptorCount = 1;
    binds[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    binds[1].binding = 1;
    binds[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    binds[1].descriptorCount = 1;
    binds[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo dslci = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    dslci.bindingCount = 2;
    dslci.pBindings = binds;

    VkDescriptorSetLayout dsl = 0;
    r = vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl);
    if(r!=VK_SUCCESS) die("vkCreateDescriptorSetLayout", r);

    VkDescriptorPoolSize dps = { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 };
    VkDescriptorPoolCreateInfo dpci = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    dpci.maxSets = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &dps;

    VkDescriptorPool dp = 0;
    r = vkCreateDescriptorPool(dev, &dpci, NULL, &dp);
    if(r!=VK_SUCCESS) die("vkCreateDescriptorPool", r);

    VkDescriptorSetAllocateInfo dsai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dsai.descriptorPool = dp;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &dsl;

    VkDescriptorSet ds = 0;
    r = vkAllocateDescriptorSets(dev, &dsai, &ds);
    if(r!=VK_SUCCESS) die("vkAllocateDescriptorSets", r);

    VkDescriptorImageInfo dii0 = {0};
    dii0.imageView = out_view;
    dii0.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorImageInfo dii1 = {0};
    dii1.imageView = accum_view;
    dii1.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet ws[2];
    memset(ws, 0, sizeof(ws));

    ws[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    ws[0].dstSet = ds;
    ws[0].dstBinding = 0;
    ws[0].descriptorCount = 1;
    ws[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    ws[0].pImageInfo = &dii0;

    ws[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    ws[1].dstSet = ds;
    ws[1].dstBinding = 1;
    ws[1].descriptorCount = 1;
    ws[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    ws[1].pImageInfo = &dii1;

    vkUpdateDescriptorSets(dev, 2, ws, 0, NULL);

    // ---------- Pipeline layout: push constants (W,H,frame,seed) = 16 bytes ----------
    VkPushConstantRange pcr = {0};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = 16;

    VkPipelineLayoutCreateInfo plci = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;

    VkPipelineLayout pl = 0;
    r = vkCreatePipelineLayout(dev, &plci, NULL, &pl);
    if(r!=VK_SUCCESS) die("vkCreatePipelineLayout", r);

    // ---------- Shader module ----------
    size_t spv_sz = 0;
    uint8_t* spv = read_file(spv_path, &spv_sz);

    VkShaderModuleCreateInfo smci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smci.codeSize = spv_sz;
    smci.pCode = (const uint32_t*)spv;

    VkShaderModule sm = 0;
    r = vkCreateShaderModule(dev, &smci, NULL, &sm);
    if(r!=VK_SUCCESS) die("vkCreateShaderModule", r);
    free(spv);

    VkComputePipelineCreateInfo cpi = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = sm;
    cpi.stage.pName = "main";
    cpi.layout = pl;

    VkPipeline pipe = 0;
    r = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, NULL, &pipe);
    if(r!=VK_SUCCESS) die("vkCreateComputePipelines", r);

    // ---------- Record command buffer ----------
    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    r = vkBeginCommandBuffer(cb, &bi);
    if(r!=VK_SUCCESS) die("vkBeginCommandBuffer", r);

    // Transition both images: UNDEFINED -> GENERAL
    VkImageMemoryBarrier ib[2];
    memset(ib, 0, sizeof(ib));
    for(int i=0;i<2;i++){
        ib[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        ib[i].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ib[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        ib[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        ib[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        ib[i].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        ib[i].subresourceRange.levelCount = 1;
        ib[i].subresourceRange.layerCount = 1;
        ib[i].srcAccessMask = 0;
        ib[i].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    }
    ib[0].image = out_img;
    ib[1].image = accum_img;

    vkCmdPipelineBarrier(
        cb,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        0,NULL, 0,NULL,
        2, ib
    );

    // Clear accumulation image to 0 (RGB=0, A=0 count)
    VkClearColorValue z = {{0.0f,0.0f,0.0f,0.0f}};
    VkImageSubresourceRange range = {0};
    range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    range.levelCount = 1;
    range.layerCount = 1;
    vkCmdClearColorImage(cb, accum_img, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &range);

    // Barrier: clear -> shader read/write (accum uses imageLoad)
    VkImageMemoryBarrier ib_clear = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    ib_clear.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    ib_clear.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    ib_clear.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ib_clear.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ib_clear.image = accum_img;
    ib_clear.subresourceRange = range;
    ib_clear.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    ib_clear.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

    vkCmdPipelineBarrier(
        cb,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0,NULL, 0,NULL, 1, &ib_clear
    );

    // Bind pipeline + descriptors once
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);

    uint32_t gx = (uint32_t)((W + 15) / 16);
    uint32_t gy = (uint32_t)((H + 15) / 16);

    // Progressive frames
    for(int f=0; f<frames; f++){
        int pc[4] = { W, H, f, seed };
        vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pc);
        vkCmdDispatch(cb, gx, gy, 1);
    }

    // Transition out_img for copy: GENERAL -> TRANSFER_SRC_OPTIMAL
    VkImageMemoryBarrier ib2 = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    ib2.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    ib2.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    ib2.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ib2.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ib2.image = out_img;
    ib2.subresourceRange = range;
    ib2.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    ib2.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    vkCmdPipelineBarrier(
        cb,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0,NULL, 0,NULL, 1, &ib2
    );

    // Copy out_img -> buffer
    VkBufferImageCopy bic = {0};
    bic.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    bic.imageSubresource.layerCount = 1;
    bic.imageExtent.width = (uint32_t)W;
    bic.imageExtent.height = (uint32_t)H;
    bic.imageExtent.depth = 1;

    vkCmdCopyImageToBuffer(cb, out_img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, read_buf, 1, &bic);

    // Buffer barrier for host read
    VkBufferMemoryBarrier bb = { VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER };
    bb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    bb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bb.buffer = read_buf;
    bb.size = VK_WHOLE_SIZE;

    vkCmdPipelineBarrier(
        cb,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT,
        0, 0,NULL, 1, &bb, 0, NULL
    );

    r = vkEndCommandBuffer(cb);
    if(r!=VK_SUCCESS) die("vkEndCommandBuffer", r);

    // ---------- Submit + wait ----------
    VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkFence fence = 0;
    r = vkCreateFence(dev, &fci, NULL, &fence);
    if(r!=VK_SUCCESS) die("vkCreateFence", r);

    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;

    r = vkQueueSubmit(q, 1, &si, fence);
    if(r!=VK_SUCCESS) die("vkQueueSubmit", r);

    r = vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX);
    if(r!=VK_SUCCESS) die("vkWaitForFences", r);

    // ---------- Readback ----------
    void* mapped = NULL;
    r = vkMapMemory(dev, read_mem, 0, out_bytes, 0, &mapped);
    if(r!=VK_SUCCESS) die("vkMapMemory", r);

    ppm_write_rgb8("output_gpu.ppm", (const float*)mapped, W, H);
    vkUnmapMemory(dev, read_mem);

    printf("[GPU] wrote output_gpu.ppm (%dx%d RGBA32F)  SPP=%d\n", W, H, frames);

    // ---------- Cleanup ----------
    vkDestroyFence(dev, fence, NULL);

    vkDestroyPipeline(dev, pipe, NULL);
    vkDestroyShaderModule(dev, sm, NULL);
    vkDestroyPipelineLayout(dev, pl, NULL);

    vkDestroyDescriptorPool(dev, dp, NULL);
    vkDestroyDescriptorSetLayout(dev, dsl, NULL);

    vkDestroyBuffer(dev, read_buf, NULL);
    vkFreeMemory(dev, read_mem, NULL);

    vkDestroyImageView(dev, out_view, NULL);
    vkDestroyImage(dev, out_img, NULL);
    vkFreeMemory(dev, out_mem, NULL);

    vkDestroyImageView(dev, accum_view, NULL);
    vkDestroyImage(dev, accum_img, NULL);
    vkFreeMemory(dev, accum_mem, NULL);

    vkDestroyCommandPool(dev, pool, NULL);
    vkDestroyDevice(dev, NULL);
    vkDestroyInstance(inst, NULL);
    return 0;
}
