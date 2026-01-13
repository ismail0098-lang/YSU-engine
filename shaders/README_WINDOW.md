# Window + Present ekleme (GLFW)

Bu pakette:
- gpu_vulkan_demo.c (senin dosyan)
- present.vert / present.frag (fullscreen triangle; sampler2D -> swapchain)

Kullanım özeti:
1) GLFW ile `glfwInit()` + `glfwCreateWindow(..., GLFW_NO_API)`.
2) Instance extension: `glfwGetRequiredInstanceExtensions`.
3) `glfwCreateWindowSurface` ile `VkSurfaceKHR`.
4) Device: `VK_KHR_swapchain` extension.
5) Swapchain + image views.
6) RenderPass: 1 color attachment (swapchain format).
7) Graphics pipeline:
   - Vertex: present.vert.spv
   - Fragment: present.frag.spv
   - Descriptor: combined image sampler (senin LDR output image view)
8) Frame loop:
   - `vkAcquireNextImageKHR`
   - Compute (tri.comp -> HDR)
   - Tonemap compute (HDR -> LDR)
   - RenderPass + `vkCmdDraw(3,1,0,0)`
   - `vkQueuePresentKHR`

İstersen bir sonraki mesajında `gpu_vulkan_demo.c` içinde swapchain/present eklemek istediğin noktayı söyle; ben sana **minimal patch (diff)** formatında direkt uygulanabilir şekilde çıkartayım.
