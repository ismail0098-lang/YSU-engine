#!/usr/bin/env python3
"""
Auto-explorer for the SASS RE combo frontier.

This script ingests existing run artifacts, extracts mnemonic-chain and ncu
signals, and proposes the next experiments to run. It is stdlib-first, with
optional sklearn-based surrogate scoring when available.
"""

from __future__ import annotations

import argparse
import collections
import csv
import importlib.util
import itertools
import json
import math
import pathlib
import re
import statistics
import sys
import tomllib


SASS_LINE_RE = re.compile(r"/\*[^*]+\*/\s+([A-Z][A-Z0-9_.]*)")
PRIMARY_COLS = [
    "smsp__cycles_elapsed.avg",
    "smsp__inst_executed.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__t_sector_hit_rate.pct",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_static",
]
STALL_COLS = [
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_membar_per_warp_active.pct",
    "smsp__warp_issue_stalled_wait_per_warp_active.pct",
]
FEATURE_KEYS = [
    "async_cache",
    "uniform",
    "blockred",
    "warp",
    "divergent",
    "sys64",
    "store_sys",
    "dlcm_cg",
    "depth",
]
DEFAULT_INPUTS = [
    "src/sass_re/results/runs/combo_family_ncu_batch_20260322_022222",
    "src/sass_re/results/runs/combo_uniform_redsys_async_profile_safe_tranche_20260322_133653",
    "src/sass_re/results/runs/combo_atomg64sys_ops_profile_safe_tranche_20260322_135833",
    "src/sass_re/results/runs/combo_divergent_blockred_warp_atomg64sys_ops_profile_safe_tranche_20260322_140111",
]


def package_report() -> dict[str, bool]:
    names = [
        "numpy",
        "pandas",
        "networkx",
        "sklearn",
        "optuna",
        "onnx",
        "onnxruntime",
        "torch",
        "typer",
        "pydantic",
    ]
    return {name: importlib.util.find_spec(name) is not None for name in names}


def load_rows(path: pathlib.Path) -> list[str]:
    rows: list[str] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith('"ID"') or (line and line[0].isdigit()) or line.startswith('"'):
                rows.append(line)
    return rows


def kernel_names(csv_path: pathlib.Path) -> list[str]:
    reader = csv.DictReader(load_rows(csv_path))
    names: list[str] = []
    seen: set[str] = set()
    for row in reader:
        name = (row.get("Kernel Name") or "").strip()
        if name and name not in seen:
            names.append(name)
            seen.add(name)
    return names


def mean_metrics(csv_path: pathlib.Path, kernel_name: str, cols: list[str]) -> dict[str, float]:
    reader = csv.DictReader(load_rows(csv_path))
    values = {col: [] for col in cols}
    fieldnames = reader.fieldnames or []
    long_form = "Metric Name" in fieldnames and "Metric Value" in fieldnames
    for row in reader:
        if row.get("Kernel Name") != kernel_name:
            continue
        if long_form:
            metric_name = (row.get("Metric Name") or "").strip()
            if metric_name not in values:
                continue
            raw = (row.get("Metric Value") or "").strip()
            if raw:
                values[metric_name].append(float(raw))
            continue
        for col in cols:
            raw = (row.get(col) or "").strip()
            if raw:
                values[col].append(float(raw))
    return {col: statistics.fmean(items) for col, items in values.items() if items}


def parse_mnemonics(path: pathlib.Path) -> list[str]:
    items: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = SASS_LINE_RE.search(line)
            if match:
                items.append(match.group(1))
    return items


def ngrams(items: list[str], n: int) -> collections.Counter[tuple[str, ...]]:
    counts: collections.Counter[tuple[str, ...]] = collections.Counter()
    if len(items) < n:
        return counts
    for idx in range(len(items) - n + 1):
        counts[tuple(items[idx : idx + n])] += 1
    return counts


def classify_regime(metrics: dict[str, float]) -> str:
    long_score = metrics.get(STALL_COLS[2], 0.0)
    membar = metrics.get(STALL_COLS[3], 0.0)
    short_score = metrics.get(STALL_COLS[1], 0.0)
    if membar >= 20.0:
        return "membar_plus_latency"
    if long_score >= 35.0:
        return "long_scoreboard_dominated"
    if short_score >= 5.0:
        return "short_scoreboard_pressure"
    return "mixed_dependency"


def infer_features(run_name: str, mnemonics: list[str]) -> dict[str, int]:
    mnemonic_set = set(mnemonics)
    features = {key: 0 for key in FEATURE_KEYS}
    features["async_cache"] = int(
        "LDGSTS.E.BYPASS.LTC128B.128" in mnemonic_set
        and "LDGDEPBAR" in mnemonic_set
        and "DEPBAR.LE" in mnemonic_set
    )
    features["uniform"] = int(
        any(item.startswith("ULDC") for item in mnemonic_set)
        or "UIADD3" in mnemonic_set
        or "ULOP3.LUT" in mnemonic_set
        or "USHF.L.U64.HI" in mnemonic_set
        or "USHF.L.U32" in mnemonic_set
    )
    features["blockred"] = int(
        any(item.startswith("BAR.RED") for item in mnemonic_set) or "B2R.RESULT" in mnemonic_set
    )
    features["warp"] = int(
        "MATCH.ANY" in mnemonic_set
        or any(item.startswith("VOTE") for item in mnemonic_set)
        or any(item.startswith("REDUX") for item in mnemonic_set)
    )
    features["divergent"] = int(
        "BSSY" in mnemonic_set
        or "BSYNC" in mnemonic_set
        or "WARPSYNC" in mnemonic_set
        or "WARPSYNC.EXCLUSIVE" in mnemonic_set
        or "divergent" in run_name
    )
    features["sys64"] = int(
        any(".64.STRONG.SYS" in item for item in mnemonic_set)
        or "MEMBAR.SC.SYS" in mnemonic_set
    )
    features["store_sys"] = int("STG.E.64.STRONG.SYS" in mnemonic_set)
    features["dlcm_cg"] = int("dlcm_cg" in run_name)
    features["depth"] = int("depth" in run_name or "deep" in run_name)
    return features


def discover_run_dirs(inputs: list[pathlib.Path]) -> list[pathlib.Path]:
    run_dirs: list[pathlib.Path] = []
    for item in inputs:
        if not item.exists():
            continue
        if any(item.glob("*.sass")):
            run_dirs.append(item)
            continue
        for path in sorted(item.rglob("*.sass")):
            run_dirs.append(path.parent)
    unique: list[pathlib.Path] = []
    seen: set[pathlib.Path] = set()
    for path in run_dirs:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def load_record(run_dir: pathlib.Path) -> dict[str, object]:
    sass_files = sorted(run_dir.glob("*.sass"))
    mnemonics: list[str] = []
    for sass in sass_files:
        mnemonics.extend(parse_mnemonics(sass))
    bigrams = ngrams(mnemonics, 2)
    trigrams = ngrams(mnemonics, 3)
    kernel = ""
    metrics: dict[str, float] = {}
    ncu = run_dir / "ncu.csv"
    stalls = run_dir / "ncu_stalls.csv"
    names = kernel_names(ncu)
    if names:
        kernel = names[0]
        metrics.update(mean_metrics(ncu, kernel, PRIMARY_COLS))
        metrics.update(mean_metrics(stalls, kernel, STALL_COLS))
    features = infer_features(run_dir.name, mnemonics)
    return {
        "label": run_dir.name,
        "path": str(run_dir),
        "kernel": kernel,
        "mnemonics": mnemonics,
        "mnemonic_set": set(mnemonics),
        "bigrams": set(bigrams),
        "trigrams": set(trigrams),
        "metrics": metrics,
        "features": features,
        "regime": classify_regime(metrics) if metrics else "symbolic_only",
    }


def enrich_novelty(records: list[dict[str, object]]) -> None:
    bigram_freq: collections.Counter[tuple[str, ...]] = collections.Counter()
    trigram_freq: collections.Counter[tuple[str, ...]] = collections.Counter()
    mnemonic_freq: collections.Counter[str] = collections.Counter()
    for record in records:
        bigram_freq.update(record["bigrams"])  # type: ignore[arg-type]
        trigram_freq.update(record["trigrams"])  # type: ignore[arg-type]
        mnemonic_freq.update(record["mnemonic_set"])  # type: ignore[arg-type]
    for record in records:
        mnems: set[str] = record["mnemonic_set"]  # type: ignore[assignment]
        bigrams: set[tuple[str, ...]] = record["bigrams"]  # type: ignore[assignment]
        trigrams: set[tuple[str, ...]] = record["trigrams"]  # type: ignore[assignment]
        rare_mnemonic_score = sum(1.0 / mnemonic_freq[item] for item in mnems)
        chain_score = sum(1.0 / bigram_freq[item] for item in bigrams)
        chain_score += 1.5 * sum(1.0 / trigram_freq[item] for item in trigrams)
        record["rare_mnemonic_score"] = rare_mnemonic_score
        record["chain_novelty_score"] = chain_score


def fit_surrogates(records: list[dict[str, object]]) -> dict[str, object]:
    if importlib.util.find_spec("sklearn") is None:
        return {"enabled": False}
    from sklearn.ensemble import RandomForestRegressor

    usable = [record for record in records if record["metrics"]]
    if len(usable) < 6:
        return {"enabled": False}
    x_rows = [
        [record["features"][key] for key in FEATURE_KEYS]  # type: ignore[index]
        for record in usable
    ]
    targets = {
        "cycles": "smsp__cycles_elapsed.avg",
        "long_scoreboard": STALL_COLS[2],
        "membar": STALL_COLS[3],
    }
    models: dict[str, object] = {}
    for name, col in targets.items():
        y_rows = [record["metrics"].get(col, 0.0) for record in usable]  # type: ignore[index]
        model = RandomForestRegressor(n_estimators=256, random_state=7)
        model.fit(x_rows, y_rows)
        models[name] = model
    return {"enabled": True, "models": models}


def predict_surrogate(
    surrogate: dict[str, object], features: dict[str, int]
) -> dict[str, float]:
    if not surrogate.get("enabled"):
        return {}
    x_row = [[features[key] for key in FEATURE_KEYS]]
    result: dict[str, float] = {}
    for name, model in surrogate["models"].items():  # type: ignore[index]
        estimators = model.estimators_  # type: ignore[attr-defined]
        preds = [est.predict(x_row)[0] for est in estimators]
        result[f"{name}_pred"] = statistics.fmean(preds)
        result[f"{name}_std"] = statistics.pstdev(preds)
    return result


def load_search_space(path: pathlib.Path) -> dict[str, object]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def proposal_score(
    candidate: dict[str, object],
    predicted: dict[str, float],
    observed_vectors: set[tuple[int, ...]],
) -> tuple[float, list[str]]:
    features = {key: 0 for key in FEATURE_KEYS}
    for key in candidate["features"]:  # type: ignore[index]
        if key in features:
            features[key] = 1
    vector = tuple(features[key] for key in FEATURE_KEYS)
    score = 0.0
    reasons: list[str] = []
    priority = float(candidate.get("priority", 0.0))
    score += 3.0 * priority
    if vector not in observed_vectors:
        score += 3.0
        reasons.append("unseen_feature_vector")
    if features["uniform"] and features["sys64"]:
        score += 1.2
        reasons.append("bridges_uniform_and_sys64")
    if features["divergent"] and features["store_sys"]:
        score += 1.1
        reasons.append("adds_store_to_reconvergence_branch")
    if features["blockred"] and features["warp"] and features["sys64"]:
        score += 0.8
        reasons.append("extends_strongest_fused_family")
    if predicted:
        regime_gap = predicted.get("membar_pred", 0.0) / 20.0 + predicted.get("long_scoreboard_pred", 0.0) / 40.0
        uncertainty = (
            predicted.get("cycles_std", 0.0) / 500.0
            + predicted.get("long_scoreboard_std", 0.0) / 5.0
            + predicted.get("membar_std", 0.0) / 5.0
        )
        score += regime_gap + uncertainty
        if predicted.get("membar_pred", 0.0) >= 20.0:
            reasons.append("predicted_membar_regime")
        if predicted.get("long_scoreboard_pred", 0.0) >= 35.0:
            reasons.append("predicted_long_scoreboard_regime")
        if uncertainty >= 0.8:
            reasons.append("high_model_uncertainty")
    return score, reasons


def candidate_realized(
    candidate: dict[str, object], observed_labels: set[str]
) -> bool:
    markers = candidate.get("realized_by", [])
    if not isinstance(markers, list):
        return False
    for marker in markers:
        if not isinstance(marker, str):
            continue
        for label in observed_labels:
            if marker in label:
                return True
    return False


def write_csv(path: pathlib.Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a scored auto-explorer report for the SASS RE corpus")
    parser.add_argument(
        "inputs",
        nargs="*",
        default=DEFAULT_INPUTS,
        help="run directories or parent run roots",
    )
    parser.add_argument(
        "--search-space",
        default="src/sass_re/auto_explorer_search_space.toml",
        help="TOML candidate registry",
    )
    parser.add_argument(
        "--outdir",
        default=f"src/sass_re/results/runs/auto_explorer_{pathlib.Path.cwd().stat().st_mtime_ns}",
        help="output directory",
    )
    parser.add_argument("--top", type=int, default=8, help="top proposals to emit")
    args = parser.parse_args(argv)

    input_paths = [pathlib.Path(item) for item in args.inputs]
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    records = [load_record(path) for path in discover_run_dirs(input_paths)]
    records = [record for record in records if record["mnemonics"]]
    enrich_novelty(records)

    stack = package_report()
    search_space = load_search_space(pathlib.Path(args.search_space))
    surrogate = fit_surrogates(records)

    observed_vectors = {
        tuple(record["features"][key] for key in FEATURE_KEYS)  # type: ignore[index]
        for record in records
    }
    observed_labels = {str(record["label"]) for record in records}
    runtime_rows: list[dict[str, object]] = []
    for candidate in search_space.get("runtime_candidates", []):
        if candidate_realized(candidate, observed_labels):
            continue
        features = {key: 0 for key in FEATURE_KEYS}
        for key in candidate["features"]:
            if key in features:
                features[key] = 1
        predicted = predict_surrogate(surrogate, features)
        score, reasons = proposal_score(candidate, predicted, observed_vectors)
        runtime_rows.append(
            {
                "name": candidate["name"],
                "kind": "runtime",
                "priority": candidate.get("priority", 0.0),
                "score": round(score, 6),
                "derive_from": candidate.get("derive_from", ""),
                "features": ",".join(candidate["features"]),
                "pred_cycles": round(predicted.get("cycles_pred", 0.0), 4),
                "pred_long_scoreboard": round(predicted.get("long_scoreboard_pred", 0.0), 4),
                "pred_membar": round(predicted.get("membar_pred", 0.0), 4),
                "uncertainty": round(
                    predicted.get("cycles_std", 0.0)
                    + predicted.get("long_scoreboard_std", 0.0)
                    + predicted.get("membar_std", 0.0),
                    4,
                ),
                "reasons": ",".join(reasons),
                "description": candidate.get("description", ""),
            }
        )
    runtime_rows.sort(key=lambda item: (-float(item["score"]), item["name"]))

    symbolic_rows: list[dict[str, object]] = []
    for candidate in search_space.get("symbolic_candidates", []):
        symbolic_rows.append(
            {
                "name": candidate["name"],
                "kind": "symbolic",
                "priority": candidate.get("priority", 0.0),
                "score": round(2.0 * float(candidate.get("priority", 0.0)) + 1.0, 6),
                "derive_from": candidate.get("derive_from", ""),
                "features": ",".join(candidate["features"]),
                "pred_cycles": "",
                "pred_long_scoreboard": "",
                "pred_membar": "",
                "uncertainty": "",
                "reasons": "raw_sass_gap",
                "description": candidate.get("description", ""),
            }
        )
    symbolic_rows.sort(key=lambda item: (-float(item["score"]), item["name"]))

    record_rows: list[dict[str, object]] = []
    for record in records:
        metrics: dict[str, float] = record["metrics"]  # type: ignore[assignment]
        features: dict[str, int] = record["features"]  # type: ignore[assignment]
        row: dict[str, object] = {
            "label": record["label"],
            "regime": record["regime"],
            "mnemonic_count": len(record["mnemonics"]),  # type: ignore[arg-type]
            "unique_mnemonics": len(record["mnemonic_set"]),  # type: ignore[arg-type]
            "chain_novelty_score": round(float(record["chain_novelty_score"]), 6),
            "rare_mnemonic_score": round(float(record["rare_mnemonic_score"]), 6),
        }
        for key in FEATURE_KEYS:
            row[key] = features[key]
        row["cycles"] = round(metrics.get(PRIMARY_COLS[0], 0.0), 4)
        row["inst"] = round(metrics.get(PRIMARY_COLS[1], 0.0), 4)
        row["l2_hit"] = round(metrics.get(PRIMARY_COLS[3], 0.0), 4)
        row["long_scoreboard"] = round(metrics.get(STALL_COLS[2], 0.0), 4)
        row["membar"] = round(metrics.get(STALL_COLS[3], 0.0), 4)
        record_rows.append(row)

    write_csv(
        outdir / "observed_runs.csv",
        record_rows,
        [
            "label",
            "regime",
            "mnemonic_count",
            "unique_mnemonics",
            "chain_novelty_score",
            "rare_mnemonic_score",
            *FEATURE_KEYS,
            "cycles",
            "inst",
            "l2_hit",
            "long_scoreboard",
            "membar",
        ],
    )
    write_csv(
        outdir / "proposals.csv",
        runtime_rows + symbolic_rows,
        [
            "name",
            "kind",
            "priority",
            "score",
            "derive_from",
            "features",
            "pred_cycles",
            "pred_long_scoreboard",
            "pred_membar",
            "uncertainty",
            "reasons",
            "description",
        ],
    )
    with (outdir / "stack.json").open("w", encoding="utf-8") as handle:
        json.dump({"packages": stack, "surrogate_enabled": surrogate.get("enabled", False)}, handle, indent=2)

    top_runtime = runtime_rows[: args.top]
    top_symbolic = symbolic_rows[: min(3, len(symbolic_rows))]
    with (outdir / "summary.txt").open("w", encoding="utf-8") as handle:
        handle.write("auto_explorer\n")
        handle.write("=============\n\n")
        handle.write("inputs:\n")
        for item in input_paths:
            handle.write(f"- {item}\n")
        handle.write("\nstack:\n")
        for name, enabled in stack.items():
            handle.write(f"- {name}: {'yes' if enabled else 'no'}\n")
        handle.write(
            "\nobserved_regimes:\n"
        )
        for row in sorted(record_rows, key=lambda item: item["label"]):
            handle.write(
                f"- {row['label']}: regime={row['regime']}, cycles={row['cycles']}, "
                f"long_scoreboard={row['long_scoreboard']}%, membar={row['membar']}%\n"
            )
        handle.write("\nnext_runtime_candidates:\n")
        for row in top_runtime:
            handle.write(
                f"- {row['name']}: score={row['score']}, features={row['features']}, "
                f"pred_cycles={row['pred_cycles']}, pred_long={row['pred_long_scoreboard']}, "
                f"pred_membar={row['pred_membar']}, reasons={row['reasons']}\n"
            )
            handle.write(f"  derive_from={row['derive_from']}\n")
            handle.write(f"  {row['description']}\n")
        handle.write("\nremaining_symbolic_boundary:\n")
        for row in top_symbolic:
            handle.write(
                f"- {row['name']}: score={row['score']}, derive_from={row['derive_from']}, "
                f"features={row['features']}\n"
            )
            handle.write(f"  {row['description']}\n")
        handle.write("\nnotes:\n")
        handle.write(
            "- This explorer is stdlib-first. sklearn, when present, is used for a "
            "small RandomForest surrogate over the observed runtime-safe frontier.\n"
        )
        handle.write(
            "- The search-space registry is explicit on purpose: it borrows the "
            "registry/lineage idea from open_gororoba, but it does not depend on "
            "Cayley-Dickson or hypercomplex embeddings for first-pass search.\n"
        )
        handle.write(
            "- ONNX or tiny MLP surrogates are optional follow-ons, not a prerequisite, "
            "because the current frontier is still sparse and interpretable.\n"
        )

    print(outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
