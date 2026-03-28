#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEX_DIR="$ROOT/tex"
OUT_DIR="$TEX_DIR/build"
LOG_FILE="${TMPDIR:-/tmp}/sm89_monograph_pdflatex.log"

GEN_ASSETS="$ROOT/scripts/generate_monograph_assets.py"
if [ -f "$GEN_ASSETS" ]; then
    python3 "$GEN_ASSETS"
else
    echo "note: $GEN_ASSETS not found; using existing processed data" >&2
fi

mkdir -p "$OUT_DIR"

cd "$TEX_DIR"
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$OUT_DIR" sm89_monograph.tex >"$LOG_FILE"
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$OUT_DIR" sm89_monograph.tex >"$LOG_FILE"

python3 "$ROOT/scripts/verify_monograph_pdf.py"
python3 "$ROOT/scripts/write_monograph_checksums.py"
python3 "$ROOT/scripts/verify_monograph_assets.py"

printf 'built_monograph_pdf\n'
printf '%s\n' "$OUT_DIR/sm89_monograph.pdf"
