#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
    echo "usage: $0 <uniform|cutlass> <outdir>" >&2
    exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FAMILY="$1"
OUTDIR="$2"
mkdir -p "$OUTDIR"

case "$FAMILY" in
    uniform)
        BASELINE="$ROOT/results/runs/uplop3_patch_sketch_20260323_011000/probe_uplop3_uniform_predicates_O3.cubin"
        RUNNER_CPP="$ROOT/runners/uplop3_uniform_predicates_cubin_driver_runner.cpp"
        KERNEL="probe_uplop3_uniform_predicates"
        SINGLE_DIR="$ROOT/results/runs/uplop3_uniform_predicates_occ_sweep_20260323_030000"
        COMBO_DIR="$ROOT/results/runs/uplop3_uniform_predicates_combo_sweep_20260323_031500"
        PAIR="$OUTDIR/occ2_occ5.cubin"
        if [[ ! -f "$PAIR" ]]; then
            python3 "$ROOT/scripts/patch_uplop3_cubin.py" \
                --input "$BASELINE" --output "$PAIR" \
                --occurrence 2 --occurrence 5 > "$OUTDIR/occ2_occ5.patch.txt"
        fi
        if [[ ! -f "$OUTDIR/occ1_occ2_occ5.cubin" ]]; then
            cp "$COMBO_DIR/occ1_occ2_occ5.cubin" "$OUTDIR/occ1_occ2_occ5.cubin"
        fi
        if [[ ! -f "$OUTDIR/occ1_occ5.cubin" ]]; then
            cp "$COMBO_DIR/occ1_occ5.cubin" "$OUTDIR/occ1_occ5.cubin"
        fi
        if [[ ! -f "$OUTDIR/occ1.cubin" ]]; then
            cp "$SINGLE_DIR/occ1.cubin" "$OUTDIR/occ1.cubin"
        fi
        bash "$ROOT/scripts/validate_uplop3_tandem_site.sh" \
            "$OUTDIR" "$RUNNER_CPP" "$PAIR" "$KERNEL" 2 \
            occ1 "$OUTDIR/occ1.cubin" \
            occ1_occ5 "$OUTDIR/occ1_occ5.cubin" \
            occ1_occ2_occ5 "$OUTDIR/occ1_occ2_occ5.cubin"
        python3 "$ROOT/scripts/summarize_uplop3_tandem.py" \
            "$OUTDIR" baseline occ1 occ1_occ5 occ1_occ2_occ5
        ;;
    cutlass)
        BASELINE="$ROOT/results/runs/uplop3_cutlass_predicate_matrix_20260323_085900/probe_cutlass_predicate_pipeline_O3.cubin"
        RUNNER_CPP="$ROOT/runners/uplop3_cutlass_predicate_cubin_driver_runner.cpp"
        KERNEL="probe_cutlass_predicate_pipeline"
        SINGLE_DIR="$ROOT/results/runs/uplop3_cutlass_predicate_matrix_20260323_085900"
        COMBO_DIR="$ROOT/results/runs/uplop3_cutlass_predicate_combo_matrix_20260323_090703"
        PAIR="$OUTDIR/occ2_occ5.cubin"
        if [[ ! -f "$PAIR" ]]; then
            cp "$SINGLE_DIR/occ2_occ5.cubin" "$PAIR"
        fi
        if [[ ! -f "$OUTDIR/occ4.cubin" ]]; then
            cp "$SINGLE_DIR/occ4.cubin" "$OUTDIR/occ4.cubin"
        fi
        if [[ ! -f "$OUTDIR/occ4_occ5.cubin" ]]; then
            cp "$COMBO_DIR/occ4_occ5.cubin" "$OUTDIR/occ4_occ5.cubin"
        fi
        if [[ ! -f "$OUTDIR/occ2_occ4_occ5.cubin" ]]; then
            cp "$COMBO_DIR/occ2_occ4_occ5.cubin" "$OUTDIR/occ2_occ4_occ5.cubin"
        fi
        bash "$ROOT/scripts/validate_uplop3_tandem_site.sh" \
            "$OUTDIR" "$RUNNER_CPP" "$PAIR" "$KERNEL" 2 \
            occ4 "$OUTDIR/occ4.cubin" \
            occ4_occ5 "$OUTDIR/occ4_occ5.cubin" \
            occ2_occ4_occ5 "$OUTDIR/occ2_occ4_occ5.cubin"
        python3 "$ROOT/scripts/summarize_uplop3_tandem.py" \
            "$OUTDIR" baseline occ4 occ4_occ5 occ2_occ4_occ5
        ;;
    *)
        echo "unknown family: $FAMILY" >&2
        exit 2
        ;;
esac

echo "$OUTDIR"
