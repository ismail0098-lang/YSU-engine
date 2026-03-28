#!/usr/bin/env python3
"""
Turn an auto-explorer proposal table into a compact execution queue.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys


def load_rows(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build an execution queue from auto-explorer proposals")
    parser.add_argument("proposal_csv", help="proposals.csv from auto_explorer.py")
    parser.add_argument("--outdir", required=True, help="output directory")
    parser.add_argument("--top-runtime", type=int, default=5, help="number of runtime items to keep")
    parser.add_argument("--top-symbolic", type=int, default=3, help="number of symbolic items to keep")
    args = parser.parse_args(argv)

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(pathlib.Path(args.proposal_csv))
    runtime = [row for row in rows if row.get("kind") == "runtime"]
    symbolic = [row for row in rows if row.get("kind") == "symbolic"]
    runtime = sorted(runtime, key=lambda row: (-float(row["score"]), row["name"]))[: args.top_runtime]
    symbolic = sorted(symbolic, key=lambda row: (-float(row["score"]), row["name"]))[: args.top_symbolic]

    with (outdir / "queue.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["phase", "rank", "name", "kind", "score", "derive_from", "features", "reasons", "description"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        rank = 1
        for row in runtime:
            writer.writerow(
                {
                    "phase": "runtime",
                    "rank": rank,
                    "name": row["name"],
                    "kind": row["kind"],
                    "score": row["score"],
                    "derive_from": row["derive_from"],
                    "features": row["features"],
                    "reasons": row["reasons"],
                    "description": row["description"],
                }
            )
            rank += 1
        rank = 1
        for row in symbolic:
            writer.writerow(
                {
                    "phase": "symbolic",
                    "rank": rank,
                    "name": row["name"],
                    "kind": row["kind"],
                    "score": row["score"],
                    "derive_from": row["derive_from"],
                    "features": row["features"],
                    "reasons": row["reasons"],
                    "description": row["description"],
                }
            )
            rank += 1

    with (outdir / "summary.txt").open("w", encoding="utf-8") as handle:
        handle.write("auto_explorer_queue\n")
        handle.write("===================\n\n")
        handle.write(f"source={args.proposal_csv}\n\n")
        handle.write("[runtime]\n")
        for idx, row in enumerate(runtime, start=1):
            handle.write(
                f"{idx}. {row['name']} score={row['score']} derive_from={row['derive_from']} "
                f"features={row['features']} reasons={row['reasons']}\n"
            )
            handle.write(f"   {row['description']}\n")
        handle.write("\n[symbolic]\n")
        for idx, row in enumerate(symbolic, start=1):
            handle.write(
                f"{idx}. {row['name']} score={row['score']} derive_from={row['derive_from']} "
                f"features={row['features']} reasons={row['reasons']}\n"
            )
            handle.write(f"   {row['description']}\n")

    print(outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
