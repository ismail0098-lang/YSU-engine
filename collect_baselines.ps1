param(
  [int]$Runs = 50,
  [string]$OutDir = ".\DATASET",
  [string]$Exe = ".\ysuengine.exe"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# (Opsiyonel) hız için bazı şeyleri kapatabiliyorsan burada env ver:
# $env:YSU_DISABLE_360="1"
# $env:YSU_DISABLE_WRITE_PNG="1"
# $env:YSU_DISABLE_POSTFX="1"

# Burayı kendi sahne seçimine göre uyarlayacaksın:
# Örnek 1: scene file env ile seçiliyorsa
# $Scenes = @(".\DATA\scene1.json", ".\DATA\scene2.json", ".\DATA\scene3.json")
# Örnek 2: kod içinde sabitse, tek sahneyle de çalışır:
$Scenes = @("DEFAULT")

# SPP’yi sabitlemek istersen:
# $env:YSU_SPP="64"

for ($i=0; $i -lt $Runs; $i++) {
  $scene = $Scenes[$i % $Scenes.Count]

  # --- SAHNE SEÇİMİ (BURAYI 1 KEZ AYARLA) ---
  # Eğer scene loader env okuyorsa:
  # $env:YSU_SCENE = (Resolve-Path $scene).Path

  # Eğer scene loader arg alıyorsa (ör: .\ysuengine.exe -scene path):
  # & $Exe "-scene" $scene | Out-Host

  # Şimdilik direkt çalıştırıyoruz (DEFAULT sahne):
  Remove-Item Env:\YSU_BVH_POLICY -ErrorAction SilentlyContinue

  # Seed sabitlemek istersen (engine destekliyorsa):
  # $env:YSU_SEED = (1337 + $i).ToString()

  & $Exe | Out-Host

  # baseline dosyası köke yazılıyor (sende böyle)
  $src = ".\baseline_bvh.csv"
  if (!(Test-Path $src)) {
    throw "baseline_bvh.csv bulunamadı. Engine gerçekten yazıyor mu? Çalışma klasörü doğru mu?"
  }

  $tag = "{0:D4}" -f $i
  $dst = Join-Path $OutDir ("baseline_" + $tag + ".csv")
  Copy-Item $src $dst -Force

  Write-Host "Saved $dst"
}

Write-Host "DONE. Collected $Runs baselines into $OutDir"
