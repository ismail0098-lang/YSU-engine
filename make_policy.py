import csv

INP  = "baseline_bvh.csv"
OUT  = r"DATA\bvh_policy.csv"

MIN_VISITS = 10000
THRESH = 0.0002   # 0.05% below 0.05 is useless
KEEP_DEPTH = 4     # depth<=3 means it never prunes

rows = []
with open(INP, "r", newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        if "node_id" not in row:
            raise SystemExit("CSV'de node_id yok. bvh_dump_stats'i node_id ile güncelle.")

        nid = int(row["node_id"])
        visits = int(row["visits"])
        useful = int(row["useful"])
        depth = int(row.get("depth", "999999"))  # depth yoksa "çok derin" say

        if nid == 0:
            prune = 0
        elif depth <= KEEP_DEPTH:
            prune = 0
        elif visits < MIN_VISITS:
            prune = 0
        else:
            ratio = useful / max(1, visits)
            prune = 1 if ratio < THRESH else 0

        rows.append((nid, prune))

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["node_id", "prune"])
    w.writerows(rows)

print("Wrote:", OUT, "entries:", len(rows))
