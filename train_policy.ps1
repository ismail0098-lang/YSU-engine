# train_policy.ps1
# Usage: powershell -ExecutionPolicy Bypass -File .\train_policy.ps1
# Optional params:
#   -TrainRatio 0.35  (default)
#   -TrainSPP 1       (default)
#   -Exe .\ysuengine.exe
#   -DataDir .\DATA

param(
  [double]$TrainRatio = 0.35,
  [int]$TrainSPP = 1,
  [string]$Exe = ".\ysuengine.exe",
  [string]$DataDir = ".\DATA"
)

$ErrorActionPreference = "Stop"

function Ensure-Dir($p) {
  if (!(Test-Path $p)) { New-Item -ItemType Directory -Path $p | Out-Null }
}

function Run-Engine($envPolicy, $extraEnv) {
  if ($envPolicy -eq $null -or $envPolicy -eq "") {
    Remove-Item Env:\YSU_BVH_POLICY -ErrorAction SilentlyContinue
  } else {
    $env:YSU_BVH_POLICY = $envPolicy
  }

  # Optional env overrides (only if your engine reads them)
  foreach ($k in $extraEnv.Keys) { Set-Item -Path "Env:\$k" -Value $extraEnv[$k] }

  Write-Host "== Running: $Exe (YSU_BVH_POLICY=$($env:YSU_BVH_POLICY)) =="
  & $Exe
  if ($LASTEXITCODE -ne 0) { throw "Engine exited with code $LASTEXITCODE" }
}

Ensure-Dir $DataDir

# --- 1) TRAIN RUN (policy off) ---
Write-Host "`n=== TRAIN RUN (policy OFF) ==="
$trainEnv = @{
  "YSU_SPP" = "$TrainSPP"
}
Run-Engine "" $trainEnv

# baseline_bvh.csv is written to current directory by your engine
$baseline = ".\baseline_bvh.csv"
if (!(Test-Path $baseline)) {
  throw "baseline_bvh.csv not found in current directory. Make sure engine wrote it here."
}

# --- 2) GENERATE POLICY ---
Write-Host "`n=== GENERATE POLICY (ratio=$TrainRatio) ==="
$policyOut = Join-Path $DataDir "bvh_policy_RELEASE.csv"

python -c @"
import csv, math
base = r'$baseline'
out  = r'$policyOut'
ratio = float($TrainRatio)

rows=[]
with open(base,'r',newline='') as f:
    rd = csv.DictReader(f)
    # Expect: depth,start,count,visits,useful,node_id
    for r in rd:
        try:
            vid = int(float(r.get('visits',0) or 0))
            uid = int(float(r.get('useful',0) or 0))
            nid = int(float(r.get('node_id',0) or 0))
        except Exception:
            continue
        if vid<=0 or nid==0:
            continue
        rows.append((uid/vid, vid, uid, nid))

rows.sort(key=lambda x:(x[0], x[1]))
k = max(1, int(len(rows)*ratio))
pr = set(n for *_,n in rows[:k])

with open(out,'w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['node_id','prune'])
    w.writerow([0,0])  # never prune root
    for n in sorted(pr):
        w.writerow([n,1])

print(f'Wrote {out} | pruned {len(pr)} of {len(rows)} nodes | ratio={ratio}')
"@

if ($LASTEXITCODE -ne 0) { throw "Python policy generator failed." }

if (!(Test-Path $policyOut)) { throw "Policy output not created: $policyOut" }

Write-Host "Policy file lines:" (Get-Content $policyOut | Measure-Object).Count

# --- 3) FINAL RUN (policy on) ---
Write-Host "`n=== FINAL RUN (policy ON: RELEASE) ==="
$finalEnv = @{}  # keep your normal defaults (SPP=64 etc.) unless you want to force
Run-Engine (Resolve-Path $policyOut).Path $finalEnv

Write-Host "`nDONE. Release policy:" (Resolve-Path $policyOut).Path
