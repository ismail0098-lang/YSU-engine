#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import re
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
DRAFT = ROOT / "PAPER_DRAFT_SM89.md"
CLAIMS = ROOT / "PAPER_CLAIMS_MATRIX.md"

REQUIRED_HEADINGS = [
    "## Draft Abstract",
    "## Draft Section 7.4: Ada-Only Frontier Status",
    "## Draft Section 8.5: Ada-Only Frontier Synthesis",
    "## Draft Conclusion",
]

REQUIRED_ASSET_TARGETS = [
    "tables/table_a1_sm89_inventory_summary.csv",
    "tables/table_a2_p2r_frontier_status.csv",
    "tables/table_a3_uplop3_structural_boundary.csv",
    "figures/uplop3_runtime_class_map.generated.svg",
    "tables/table_a4_live_uplop3_site_ranking.csv",
    "tables/table_a5_tool_effectiveness_matrix.csv",
]

FORBIDDEN_PHRASES = [
    "source-level `P2R.B*` has been reproduced",
    "source-level `UPLOP3.LUT` has been reproduced",
]

REF_CLAIM_RE = re.compile(r"`(C\d{2})`")


def main() -> int:
    text = DRAFT.read_text(encoding="utf-8")
    text_for_forbidden_check = text.split("## Bounded-Language Reminder", 1)[0]
    claim_text = CLAIMS.read_text(encoding="utf-8")
    claim_ids = set(re.findall(r"^\|+\s*(C\d{2})\s*\|", claim_text, flags=re.MULTILINE))
    errors: list[str] = []

    for heading in REQUIRED_HEADINGS:
        if heading not in text:
            errors.append(f"missing heading: {heading}")

    for target in REQUIRED_ASSET_TARGETS:
        if target not in text:
            errors.append(f"missing asset reference: {target}")
        elif not (ROOT / target).exists():
            errors.append(f"asset does not exist: {target}")

    for cid in REF_CLAIM_RE.findall(text):
        if cid not in claim_ids:
            errors.append(f"unknown claim id: {cid}")

    for phrase in FORBIDDEN_PHRASES:
        if phrase in text_for_forbidden_check:
            errors.append(f"forbidden phrase present: {phrase}")

    if errors:
        for err in errors:
            print(err)
        return 1

    print("paper_draft_ok")
    print(f"claims_referenced={len(REF_CLAIM_RE.findall(text))}")
    print(f"required_assets={len(REQUIRED_ASSET_TARGETS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
