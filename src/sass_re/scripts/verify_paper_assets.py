#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import re
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
CLAIMS_PATH = ROOT / "PAPER_CLAIMS_MATRIX.md"
OUTLINE_PATH = ROOT / "PAPER_OUTLINE.md"
FIG_PLAN_PATH = ROOT / "PAPER_FIGURE_TABLE_PLAN.md"
ASSETS_PATH = ROOT / "PAPER_ASSETS_SM89.md"
SECTION_COVERAGE_PATH = ROOT / "PAPER_SECTION_COVERAGE.md"

CLAIM_RE = re.compile(r"^\|\s*(C\d{2})\s*\|", re.MULTILINE)
REF_CLAIM_RE = re.compile(r"`(C\d{2})`")
MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def claim_ids(path: pathlib.Path) -> set[str]:
    return set(CLAIM_RE.findall(path.read_text(encoding="utf-8")))


def referenced_claim_ids(path: pathlib.Path) -> set[str]:
    return set(REF_CLAIM_RE.findall(path.read_text(encoding="utf-8")))


def markdown_targets(path: pathlib.Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [target for _, target in MD_LINK_RE.findall(text)]


def exists_local_markdown_target(base: pathlib.Path, target: str) -> bool:
    if target.startswith("http://") or target.startswith("https://"):
        return True
    if target.startswith("/home/"):
        return pathlib.Path(target).exists()
    return (base.parent / target).exists()


def local_markdown_target_path(base: pathlib.Path, target: str) -> pathlib.Path | None:
    if target.startswith("http://") or target.startswith("https://"):
        return None
    if target.startswith("/home/"):
        return pathlib.Path(target)
    return base.parent / target


def main() -> int:
    claim_doc_ids = claim_ids(CLAIMS_PATH)
    checked_docs = [OUTLINE_PATH, FIG_PLAN_PATH, ASSETS_PATH, SECTION_COVERAGE_PATH]
    errors: list[str] = []

    for doc in checked_docs:
        for cid in referenced_claim_ids(doc):
            if cid not in claim_doc_ids:
                errors.append(f"{doc}: unknown claim id {cid}")

    for doc in [OUTLINE_PATH, FIG_PLAN_PATH, ASSETS_PATH]:
        for target in markdown_targets(doc):
            if not exists_local_markdown_target(doc, target):
                errors.append(f"{doc}: missing link target {target}")
                continue
            local_target = local_markdown_target_path(doc, target)
            if local_target is None or not local_target.is_file():
                continue
            if local_target.suffix not in {".svg", ".csv"}:
                continue
            if local_target.stat().st_size <= 0:
                errors.append(f"{doc}: empty generated artifact {target}")

    if errors:
        for err in errors:
            print(err)
        return 1

    print("paper_assets_ok")
    print(f"claims={len(claim_doc_ids)}")
    print("checked_docs=" + ",".join(str(p.name) for p in checked_docs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
