# build_shaders.ps1 - compile Vulkan compute shaders to SPIR-V
$ErrorActionPreference = "Stop"

$vk = $env:VULKAN_SDK
if (-not $vk) { $vk = "C:\VulkanSDK\1.4.335.0" }

$glslang = Join-Path $vk "Bin\glslangValidator.exe"
if (-not (Test-Path $glslang)) {
  throw "glslangValidator.exe not found at: $glslang (set VULKAN_SDK or install Vulkan SDK)"
}

New-Item -ItemType Directory -Force -Path "shaders" | Out-Null

& $glslang -V "shaders\tri.comp"     -o "shaders\tri.comp.spv"
& $glslang -V "shaders\tonemap.comp" -o "shaders\tonemap.comp.spv"

Write-Host "OK: shaders compiled."
