#!/usr/bin/env python3
"""
Validate derived P2R patch sketches against observed B2/B3 control-half patterns.
"""

from __future__ import annotations

import argparse
import json
import pathlib


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patches", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    rows = json.loads(pathlib.Path(args.patches).read_text(encoding="utf-8"))
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    checked = []
    for row in rows:
        orig1 = int(row["orig_word1"], 16)
        patched1 = int(row["patched_word1"], 16)
        delta = patched1 ^ orig1
        expect = 0x2000 if row["target_lane"] == "B2" else 0x3000
        checked.append({
            **row,
            "ctrl_delta": f"0x{delta:016x}",
            "ctrl_lane_ok": delta == expect,
            "ctrl_lane_expected": f"0x{expect:016x}",
        })

    (outdir / "validation.json").write_text(json.dumps(checked, indent=2), encoding="utf-8")
    summary = ["p2r_patch_validation", "====================", ""]
    for row in checked[:24]:
        summary.append(
            f"- {pathlib.Path(row['sass']).name}:{row['function']} {row['pc']} lane={row['target_lane']} "
            f"delta={row['ctrl_delta']} expected={row['ctrl_lane_expected']} ok={row['ctrl_lane_ok']}"
        )
    (outdir / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
