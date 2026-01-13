param(
  [int]$Runs = 100,
  [string]$Exe = ".\ysuengine.exe",
  [string]$OutDir = ".\DATASET"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

for ($i=0; $i -lt $Runs; $i++) {
  python .\gen_scene.py | Out-Host

  Remove-Item Env:\YSU_BVH_POLICY -ErrorAction SilentlyContinue
  & $Exe | Out-Host

  if (!(Test-Path ".\baseline_bvh.csv")) { throw "baseline_bvh.csv yok. Engine yazıyor mu?" }

  $dst = Join-Path $OutDir ("baseline_{0:D4}.csv" -f $i)
  Copy-Item ".\baseline_bvh.csv" $dst -Force
  Write-Host "Saved $dst"
}

Write-Host "DONE. Collected $Runs baselines into $OutDir"
