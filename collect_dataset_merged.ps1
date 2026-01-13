param(
  [int]$Runs = 5000,
  [string]$Exe = ".\ysuengine.exe",
  [string]$OutDir = ".\DATASET",
  [string]$OutFile = ".\DATASET\baseline_merged.csv"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$hasHeader = Test-Path $OutFile

for ($i=0; $i -lt $Runs; $i++) {

  python .\gen_scene.py | Out-Host

  Remove-Item Env:\YSU_BVH_POLICY -ErrorAction SilentlyContinue
  & $Exe | Out-Host

  $src = ".\baseline_bvh.csv"
  if (!(Test-Path $src)) { throw "baseline_bvh.csv yok!" }

  if (-not $hasHeader) {
    $lines = Get-Content $src
    $lines[0] = $lines[0] + ",run_id"
    $lines | Set-Content -Encoding UTF8 $OutFile
    $hasHeader = $true
  } else {
    Get-Content $src | Select-Object -Skip 1 |
      ForEach-Object { $_ + ",run_" + ("{0:D5}" -f $i) } |
      Add-Content -Encoding UTF8 $OutFile
  }

  if ($i % 50 -eq 0) {
    Write-Host ("[merge] {0}/{1}" -f $i, $Runs)
  }
}

Write-Host "DONE -> $OutFile"
