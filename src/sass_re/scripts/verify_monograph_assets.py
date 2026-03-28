#!/usr/bin/env python3
from __future__ import annotations

import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
MONOGRAPH_MD = ROOT / "MONOGRAPH_SM89_SYNTHESIS.md"
MONOGRAPH_TEX = ROOT / "tex/sm89_monograph.tex"
PROCESSED = ROOT / "processed/monograph_20260323"

REQUIRED_FILES = [
    MONOGRAPH_MD,
    MONOGRAPH_TEX,
    ROOT / "MONOGRAPH_GLOSSARY.md",
    ROOT / "MONOGRAPH_BUILD.md",
    ROOT / "MONOGRAPH_THEOREM_APPENDIX.md",
    ROOT / "Thought_Processes.md",
    ROOT / "tex/appendix_selection_laws.tex",
    ROOT / "tex/fig_inventory_closure.tex",
    ROOT / "tex/fig_p2r_frontier.tex",
    ROOT / "tex/fig_uplop3_runtime.tex",
    ROOT / "tex/fig_pair_baseline.tex",
    PROCESSED / "inventory_numeric.csv",
    PROCESSED / "inventory_plot.csv",
    PROCESSED / "p2r_frontier_numeric.csv",
    PROCESSED / "p2r_frontier_plot.csv",
    PROCESSED / "uplop3_runtime_class_counts.csv",
    PROCESSED / "uplop3_runtime_class_plot.csv",
    PROCESSED / "uplop3_runtime_sites.csv",
    PROCESSED / "uplop3_live_site_numeric.csv",
    PROCESSED / "uplop3_live_site_plot.csv",
    PROCESSED / "uplop3_pair_baseline_numeric.csv",
    PROCESSED / "uplop3_pair_baseline_plot.csv",
    PROCESSED / "tool_effectiveness_numeric.csv",
    PROCESSED / "SHA256SUMS",
]

REQUIRED_HEADINGS = [
    "# SM89 Frontier Monograph Synthesis",
    "## 1. Problem Statement",
    "## 4. The P2R Frontier",
    "## 5. The UPLOP3 Frontier",
    "## 8. Open Gaps And Next Falsifiable Experiments",
]


def main() -> int:
    errors: list[str] = []
    for path in REQUIRED_FILES:
        if not path.exists():
            errors.append(f"missing file: {path}")
        elif path.is_file() and path.stat().st_size <= 0:
            errors.append(f"empty file: {path}")

    if MONOGRAPH_MD.exists():
        text = MONOGRAPH_MD.read_text(encoding="utf-8")
        for heading in REQUIRED_HEADINGS:
            if heading not in text:
                errors.append(f"missing heading: {heading}")

    if MONOGRAPH_TEX.exists():
        tex = MONOGRAPH_TEX.read_text(encoding="utf-8")
        for needle in [
            "\\begin{document}",
            "\\input{fig_inventory_closure.tex}",
            "\\input{fig_p2r_frontier.tex}",
            "\\input{fig_uplop3_runtime.tex}",
            "\\input{fig_pair_baseline.tex}",
            "\\input{appendix_selection_laws.tex}",
        ]:
            if needle not in tex:
                errors.append(f"tex missing token: {needle}")

    figure_expectations = {
        ROOT / "tex/fig_inventory_closure.tex": ["inventory_plot.csv", "\\begin{axis}"],
        ROOT / "tex/fig_p2r_frontier.tex": ["p2r_frontier_plot.csv", "\\begin{axis}"],
        ROOT / "tex/fig_uplop3_runtime.tex": [
            "uplop3_runtime_class_plot.csv",
            "uplop3_live_site_plot.csv",
            "\\begin{groupplot}",
        ],
        ROOT / "tex/fig_pair_baseline.tex": ["uplop3_pair_baseline_plot.csv", "\\begin{axis}"],
    }
    for path, needles in figure_expectations.items():
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for needle in needles:
            if needle not in text:
                errors.append(f"figure missing token: {path.name}: {needle}")

    if errors:
        for err in errors:
            print(err)
        return 1

    print("monograph_assets_ok")
    print(f"processed_dir={PROCESSED}")
    print(f"required_files={len(REQUIRED_FILES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
