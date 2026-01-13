import csv, os
from collections import defaultdict

INP  = r"DATASET\baseline_merged.csv"
OUT  = r"DATASET\labels_from_merged.csv"

MIN_VISITS = 2000
BOTTOM_PCT = 0.15
KEEP_DEPTH_MAX = None  # istersen 4 yap

def pick_col(cols, name):
    if name in cols:
        return name
    # BOM'lu olasılıkları yakala
    for c in cols:
        if c.replace("\ufeff", "").strip().lower() == name.lower():
            return c
    return None

rows_by_run = defaultdict(list)

with open(INP, "r", newline="", encoding="utf-8-sig") as f:
    rd = csv.DictReader(f)
    cols = rd.fieldnames or []
    # kolonu BOM temizleyerek eşle
    c_run   = pick_col(cols, "run_id")
    c_nid   = pick_col(cols, "node_id")
    c_vis   = pick_col(cols, "visits")
    c_use   = pick_col(cols, "useful")
    c_depth = pick_col(cols, "depth")

    miss = [k for k,v in [("run_id",c_run),("node_id",c_nid),("visits",c_vis),("useful",c_use),("depth",c_depth)] if v is None]
    if miss:
        raise SystemExit(f"CSV kolonları eksik: {miss} got={cols}")

    for r in rd:
        try:
            run_id = r[c_run]
            nid = int(float(r[c_nid]))
            v   = int(float(r[c_vis]))
            u   = int(float(r[c_use]))
            d   = int(float(r[c_depth]))
        except Exception:
            continue

        if nid == 0:
            continue
        if KEEP_DEPTH_MAX is not None and d > KEEP_DEPTH_MAX:
            continue
        if v < MIN_VISITS:
            continue

        ratio = u / max(1, v)
        rows_by_run[run_id].append((ratio, v, u, d, nid))

out_rows = []
for run_id, lst in rows_by_run.items():
    if not lst:
        continue
    lst.sort(key=lambda x:(x[0], x[1]))
    k = max(1, int(len(lst) * BOTTOM_PCT))
    bad = set(nid for *_, nid in lst[:k])

    for ratio, v, u, d, nid in lst:
        out_rows.append({
            "run_id": run_id,
            "node_id": str(nid),
            "label_prune": "1" if nid in bad else "0",
            "visits": str(v),
            "useful": str(u),
            "depth": str(d),
            "ratio": f"{ratio:.10f}",
        })

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["run_id","node_id","label_prune","visits","useful","depth","ratio"])
    w.writeheader()
    w.writerows(out_rows)

print("Wrote", OUT, "rows=", len(out_rows), "runs=", len(rows_by_run))
