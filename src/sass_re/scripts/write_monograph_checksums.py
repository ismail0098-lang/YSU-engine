#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "processed/monograph_20260323"
PDF = ROOT / "tex/build/sm89_monograph.pdf"
MANIFEST = PROCESSED / "SHA256SUMS"

FILES = [
    ROOT / "MONOGRAPH_BUILD.md",
    ROOT / "MONOGRAPH_THEOREM_APPENDIX.md",
    PROCESSED / "README.md",
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
    ROOT / "tex/sm89_monograph.tex",
    ROOT / "tex/appendix_selection_laws.tex",
    ROOT / "tex/fig_inventory_closure.tex",
    ROOT / "tex/fig_p2r_frontier.tex",
    ROOT / "tex/fig_uplop3_runtime.tex",
    ROOT / "tex/fig_pair_baseline.tex",
    PDF,
]


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    lines: list[str] = []
    for path in FILES:
        if not path.exists():
            raise FileNotFoundError(path)
        rel = path.relative_to(ROOT)
        lines.append(f"{sha256(path)}  {rel.as_posix()}")
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("monograph_checksums_written")
    print(MANIFEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
