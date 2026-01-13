import csv, json, math, time, random

LABELS = r"DATASET\labels_from_merged.csv"
OUT_MODEL = r"DATA\bvh_ml_model.json"

# Eğitim ayarları
EPOCHS = 3              # 437k satır için 3-5 arası yeter
LR = 0.15
L2 = 1e-4
THRESH = 0.70           # policy üretirken prune eşiği

def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)

def feats(depth, visits, useful, ratio):
    # Stabil feature set:
    # - bias
    # - depth
    # - log(visits)
    # - ratio
    # - useful/log(visits+1) benzeri sinyal
    lv = math.log1p(max(0, visits))
    ur = useful / (1.0 + visits)
    return [1.0, float(depth), lv, float(ratio), ur]

# SGD Logistic Regression
w = [0.0, 0.0, 0.0, 0.0, 0.0]

def dot(a,b):
    return sum(x*y for x,y in zip(a,b))

t0 = time.time()

# Dosyayı her epoch yeniden oku (RAM’e yüklemiyoruz)
for ep in range(1, EPOCHS+1):
    n = 0
    correct = 0
    loss_sum = 0.0

    with open(LABELS, "r", newline="", encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)

        for r in rd:
            run_id = (r.get("run_id") or "").strip()
            if run_id == "":    # ilk run'ın boş kalan satırlarını atla
                continue

            try:
                y = int(r["label_prune"])
                depth = int(r["depth"])
                visits = int(r["visits"])
                useful = int(r["useful"])
                ratio = float(r["ratio"])
            except Exception:
                continue

            x = feats(depth, visits, useful, ratio)
            p = sigmoid(dot(w, x))

            # loss (logloss)
            eps = 1e-9
            loss = -(y*math.log(p+eps) + (1-y)*math.log(1-p+eps))
            loss_sum += loss

            pred = 1 if p >= 0.5 else 0
            if pred == y:
                correct += 1

            # gradient: (p - y)*x + L2*w
            g = (p - y)
            for i in range(len(w)):
                w[i] -= LR * (g * x[i] + L2 * w[i])

            n += 1

    acc = correct / max(1, n)
    print(f"[ep {ep}] n={n} acc={acc:.4f} loss={loss_sum/max(1,n):.6f}")

dt = time.time() - t0
print("Training time:", f"{dt:.2f}s")

model = {
    "type": "logreg_sgd",
    "weights": w,
    "features": ["bias","depth","log1p(visits)","ratio","useful/(1+visits)"],
    "threshold": THRESH
}
with open(OUT_MODEL, "w", encoding="utf-8") as f:
    json.dump(model, f, indent=2)

print("Saved model:", OUT_MODEL)
