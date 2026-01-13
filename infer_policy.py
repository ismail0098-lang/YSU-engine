import csv, json, math

MODEL = r"DATA\bvh_ml_model.json"
INP   = r"baseline_bvh.csv"
OUT   = r"DATA\bvh_policy_ML.csv"

TARGET_PCT = 0.50   # sahnede prune edilecek oran
MIN_VISITS = 200    # düşük visits = gürültü, dokunma

def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)

def feats(depth, visits, useful, ratio):
    lv = math.log1p(max(0, visits))
    ur = useful / (1.0 + visits)
    return [1.0, float(depth), lv, float(ratio), ur]

with open(MODEL, "r", encoding="utf-8") as f:
    m = json.load(f)
w = m["weights"]

# p(prune) skorlarını topla
scored = []
all_nodes = []

with open(INP, "r", newline="", encoding="utf-8-sig") as f:
    rd = csv.DictReader(f)
    for r in rd:
        try:
            nid = int(r["node_id"])
            if nid == 0:
                continue
            depth = int(r["depth"])
            visits = int(r["visits"])
            useful = int(r["useful"])
            if visits < MIN_VISITS:
                all_nodes.append((nid, 0.0, visits))
                continue
            ratio = useful / max(1.0, visits)
        except Exception:
            continue

        x = feats(depth, visits, useful, ratio)
        p = sigmoid(sum(a*b for a,b in zip(w,x)))
        scored.append((p, nid, visits))
        all_nodes.append((nid, p, visits))

# hedef oran kadar en yüksek p'leri prune
scored.sort(reverse=True, key=lambda t: t[0])
k = max(1, int(len(scored) * TARGET_PCT))
prune_set = set(nid for _, nid, _ in scored[:k])

with open(OUT, "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f)
    wr.writerow(["node_id","prune"])
    wr.writerow([0,0])
    for nid, p, visits in all_nodes:
        if nid in prune_set:
            wr.writerow([nid, 1])

print("Nodes:", len(all_nodes), "eligible:", len(scored), "pruned:", len(prune_set), "target_pct:", TARGET_PCT)
if scored:
    print("Chosen threshold ~", scored[k-1][0])
print("Wrote", OUT)
