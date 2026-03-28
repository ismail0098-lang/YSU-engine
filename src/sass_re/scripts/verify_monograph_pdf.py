#!/usr/bin/env python3
from __future__ import annotations

import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
PDF = ROOT / "tex/build/sm89_monograph.pdf"
TEX = ROOT / "tex/sm89_monograph.tex"
FIGURES = [
    ROOT / "tex/fig_inventory_closure.tex",
    ROOT / "tex/fig_p2r_frontier.tex",
    ROOT / "tex/fig_uplop3_runtime.tex",
    ROOT / "tex/fig_pair_baseline.tex",
]
PLOTS = [
    ROOT / "processed/monograph_20260323/inventory_plot.csv",
    ROOT / "processed/monograph_20260323/p2r_frontier_plot.csv",
    ROOT / "processed/monograph_20260323/uplop3_runtime_class_plot.csv",
    ROOT / "processed/monograph_20260323/uplop3_live_site_plot.csv",
    ROOT / "processed/monograph_20260323/uplop3_pair_baseline_plot.csv",
]


def main() -> int:
    errors: list[str] = []
    deps = [TEX, *FIGURES, *PLOTS]

    if not PDF.exists():
        errors.append(f"missing pdf: {PDF}")
    elif PDF.stat().st_size <= 0:
        errors.append(f"empty pdf: {PDF}")
    else:
        pdf_mtime = PDF.stat().st_mtime
        for dep in deps:
            if not dep.exists():
                errors.append(f"missing dependency: {dep}")
                continue
            if dep.stat().st_mtime > pdf_mtime:
                errors.append(f"stale pdf: {PDF} older than {dep}")

    if errors:
        for err in errors:
            print(err)
        return 1

    print("monograph_pdf_ok")
    print(f"pdf={PDF}")
    print(f"deps={len(deps)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
