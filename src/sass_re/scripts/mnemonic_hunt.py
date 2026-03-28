#!/usr/bin/env python3
"""
Compare emitted mnemonics against the checked-in mnemonic census.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys


def load_known(path: pathlib.Path) -> set[str]:
    known: set[str] = set()
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            mnemonic = (row.get("mnemonic") or "").strip()
            if mnemonic:
                known.add(mnemonic)
    return known


def load_emitted(paths: list[pathlib.Path]) -> set[str]:
    emitted: set[str] = set()
    for path in paths:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                item = line.strip()
                if item:
                    emitted.add(item)
    return emitted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Find emitted mnemonics absent from the checked-in census")
    parser.add_argument("mnemonic_files", nargs="+", help="one or more all_mnemonics/baseline files")
    parser.add_argument(
        "--known",
        default="src/sass_re/results/mnemonic_census.csv",
        help="checked-in CSV baseline",
    )
    args = parser.parse_args(argv)

    known = load_known(pathlib.Path(args.known))
    emitted = load_emitted([pathlib.Path(item) for item in args.mnemonic_files])
    novel = sorted(emitted - known)
    for item in novel:
        print(item)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
