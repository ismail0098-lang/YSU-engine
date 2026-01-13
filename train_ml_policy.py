import csv, json, math, os, time
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np

# -----------------------------
# Config
# -----------------------------
BASELINE_CSV = r"DATASET\baseline_merged.csv"     # engine yazıyor (kök dizinde)
LABEL_POLICY = r"DATA\bvh_policy_FINAL.csv"       # senin "doğru" policy (label)
OUT_POLICY   = r"DATA\bvh_policy_ML.csv"          # ML'nin ürettiği policy
OUT_MODEL    = r"DATA\bvh_ml_model.json"          # model ağırlıkları

KEEP_DEPTH   = 4          # depth<=4 ASLA prune yok (hard safety)
MIN_VISITS   = 5000       # çok düşük visits: label'a göre bile olsa prune etme (isteğe bağlı güvenlik)

# Training
EPOCHS       = 2000       # 2k iyi başlangıç; 50k-200k de yapabilirsin
LR           = 0.05       # learning rate
L2           = 1e-4       # regularization
BATCH        = 256        # mini-batch
SEED         = 1337

# Inference
THRESH_PROB  = 0.70       # p(prune)=0.70 üstü prune (0.5-0.95 dene)
WRITE_ZEROES = False      # False: sadece prune=1 yaz (daha güvenli ve küçük dosya)
ALWAYS_WRITE_ROOT = True  # (0,0) yaz

# -----------------------------
# Utils
# -----------------------------
def sigmoid(x):
    # stable sigmoid
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))

def zscore_fit(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    return mu, sd

def zscore_apply(X, mu, sd):
    return (X - mu) / sd

def read_policy_labels(path: str) -> Dict[int, int]:
    lab = {}
    with open(path, "r", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            nid = int(r.get("node_id", "0"))
            pr  = int(float(r.get("prune", "0") or 0))
            lab[nid] = 1 if pr != 0 else 0
    return lab

@dataclass
class Row:
    node_id: int
    depth: int
    visits: int
    useful: int
    ratio: float

def read_baseline_rows(path: str) -> List[Row]:
    rows = []
    with open(path, "r", newline="") as f:
        rd = csv.DictReader(f)
        # expected columns: depth, visits, useful, node_id (others ignored)
        for r in rd:
            if "node_id" not in r:
                raise SystemExit("baseline_bvh.csv içinde node_id yok. dump'ı node_id ile yazdırmalısın.")
            nid = int(r.get("node_id", "0") or 0)
            depth = int(r.get("depth", "0") or 0)
            visits = int(float(r.get("visits", "0") or 0))
            useful = int(float(r.get("useful", "0") or 0))
            if visits <= 0 or nid == 0:
                continue
            ratio = useful / max(1, visits)
            rows.append(Row(nid, depth, visits, useful, ratio))
    return rows

def build_dataset(rows: List[Row], labels: Dict[int,int]) -> Tuple[np.ndarray, np.ndarray, List[int], List[Row]]:
    X_list = []
    y_list = []
    ids = []
    kept_rows = []

    for rw in rows:
        # hard safety constraints
        if rw.depth <= KEEP_DEPTH:
            continue
        if rw.visits < MIN_VISITS:
            continue

        y = labels.get(rw.node_id, 0)  # label yoksa prune=0 say

        # features (hepsi sayısal)
        # 1) depth
        # 2) log(visits)
        # 3) log(useful+1)
        # 4) ratio useful/visits
        # 5) logit(ratio+eps) benzeri (ayrıştırıcı)
        eps = 1e-9
        log_vis = math.log(rw.visits + 1.0)
        log_use = math.log(rw.useful + 1.0)
        ratio = rw.ratio
        # ratio çok küçükse logit negatif büyük olur; stabil tut
        logit_ratio = math.log((ratio + eps) / (1.0 - min(ratio, 1.0 - 1e-6)))

        X_list.append([rw.depth, log_vis, log_use, ratio, logit_ratio])
        y_list.append(y)
        ids.append(rw.node_id)
        kept_rows.append(rw)

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    return X, y, ids, kept_rows

def train_logreg(X, y, epochs=2000, lr=0.05, l2=1e-4, batch=256, seed=1337):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    w = np.zeros((d,), dtype=np.float32)
    b = np.float32(0.0)

    # class imbalance -> pos_weight
    pos = float((y > 0.5).sum())
    neg = float((y <= 0.5).sum())
    pos_weight = (neg / max(1.0, pos)) if pos > 0 else 1.0

    idx = np.arange(n)

    t0 = time.time()
    for ep in range(1, epochs + 1):
        rng.shuffle(idx)
        for s in range(0, n, batch):
            j = idx[s:s+batch]
            Xb = X[j]
            yb = y[j]

            logits = Xb @ w + b
            p = sigmoid(logits)

            # weighted BCE gradient
            # loss = -[ pos_weight*y*log(p) + (1-y)*log(1-p) ] + 0.5*l2*||w||^2
            # grad wrt logits = (p - y) * weight
            weight = np.where(yb > 0.5, pos_weight, 1.0).astype(np.float32)
            g = (p - yb) * weight

            gw = (Xb.T @ g) / max(1, len(j)) + l2 * w
            gb = g.mean()

            w -= lr * gw
            b -= lr * gb

        if ep % 100 == 0 or ep == 1:
            # quick metrics
            logits = X @ w + b
            p = sigmoid(logits)
            pred = (p >= 0.5).astype(np.float32)

            tp = float(((pred == 1) & (y == 1)).sum())
            tn = float(((pred == 0) & (y == 0)).sum())
            fp = float(((pred == 1) & (y == 0)).sum())
            fn = float(((pred == 0) & (y == 1)).sum())

            prec = tp / max(1.0, (tp + fp))
            rec  = tp / max(1.0, (tp + fn))
            acc  = (tp + tn) / max(1.0, (tp + tn + fp + fn))

            print(f"[ep {ep:5d}] acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} pos_w={pos_weight:.2f}")

    print("Training time:", f"{time.time()-t0:.2f}s")
    return w, float(b), float(pos_weight)

def save_model(path: str, w, b, mu, sd, meta: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    obj = {
        "w": w.tolist(),
        "b": b,
        "mu": mu.tolist(),
        "sd": sd.tolist(),
        "meta": meta,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def load_model(path: str):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    w = np.asarray(obj["w"], dtype=np.float32)
    b = float(obj["b"])
    mu = np.asarray(obj["mu"], dtype=np.float32)
    sd = np.asarray(obj["sd"], dtype=np.float32)
    meta = obj.get("meta", {})
    return w, b, mu, sd, meta

def write_policy_csv(path: str, pruned_ids: List[int], write_zeroes: bool):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node_id", "prune"])
        if ALWAYS_WRITE_ROOT:
            w.writerow([0, 0])
        if write_zeroes:
            # not recommended: requires knowing all nodes
            raise SystemExit("WRITE_ZEROES=True is not supported in this minimal writer. Keep it False.")
        else:
            for nid in sorted(set(pruned_ids)):
                if nid != 0:
                    w.writerow([nid, 1])

def main():
    print("Loading baseline:", BASELINE_CSV)
    rows = read_baseline_rows(BASELINE_CSV)

    print("Loading labels:", LABEL_POLICY)
    labels = read_policy_labels(LABEL_POLICY)

    X, y, ids, kept_rows = build_dataset(rows, labels)
    if len(y) < 50:
        raise SystemExit(f"Dataset too small ({len(y)}). Check KEEP_DEPTH/MIN_VISITS or input CSV.")

    pos = int((y > 0.5).sum())
    neg = int((y <= 0.5).sum())
    print(f"Dataset: n={len(y)}  pos(prune)= {pos}  neg= {neg}")

    # normalize
    mu, sd = zscore_fit(X)
    Xn = zscore_apply(X, mu, sd)

    print("Training logistic regression...")
    w, b, pos_w = train_logreg(Xn, y, epochs=EPOCHS, lr=LR, l2=L2, batch=BATCH, seed=SEED)

    save_model(
        OUT_MODEL, w, b, mu, sd,
        meta={
            "features": ["depth", "log_visits", "log_useful", "ratio", "logit_ratio"],
            "keep_depth": KEEP_DEPTH,
            "min_visits": MIN_VISITS,
            "train_epochs": EPOCHS,
            "lr": LR,
            "l2": L2,
            "batch": BATCH,
            "seed": SEED,
            "pos_weight": pos_w,
            "label_policy": LABEL_POLICY,
        }
    )
    print("Saved model:", OUT_MODEL)

    # Inference on all eligible nodes (same filters) -> prune if p>=THRESH_PROB
    logits = Xn @ w + b
    p = sigmoid(logits)

    pruned = [ids[i] for i in range(len(ids)) if p[i] >= THRESH_PROB]

    print(f"ML prune @ thresh={THRESH_PROB:.2f}: {len(pruned)} nodes")
    write_policy_csv(OUT_POLICY, pruned, WRITE_ZEROES)
    print("Wrote policy:", OUT_POLICY)

    # Quick sanity: how many 1s?
    # (We don't print file parse; just count list size)
    print("Done.")

if __name__ == "__main__":
    main()
