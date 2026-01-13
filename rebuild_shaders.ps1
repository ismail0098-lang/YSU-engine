param(
  [string]$VulkanSDKBin = ""
)

$ErrorActionPreference = "Stop"

# Try to auto-detect glslangValidator from VulkanSDK if not provided
if ($VulkanSDKBin -eq "") {
  if ($env:VULKAN_SDK) {
    $VulkanSDKBin = Join-Path $env:VULKAN_SDK "Bin"
  }
}

$glslang = Join-Path $VulkanSDKBin "glslangValidator.exe"

if (!(Test-Path $glslang)) {
  Write-Host "[ERROR] glslangValidator.exe not found."
  Write-Host "Set env VULKAN_SDK or call: .\rebuild_shaders.ps1 -VulkanSDKBin 'C:\VulkanSDK\X.Y.Z\Bin'"
  exit 1
}

Write-Host "[shader] compiling shaders/tri.comp -> shaders/tri.comp.spv"
& $glslang -V shaders\tri.comp -o shaders\tri.comp.spv
Write-Host "[shader] OK"
