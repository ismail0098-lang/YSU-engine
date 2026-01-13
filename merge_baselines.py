import csv, glob, os
IN_DIR="DATASET"
OUT=os.path.join(IN_DIR,"baseline_merged.csv")
files=sorted(glob.glob(os.path.join(IN_DIR,"baseline_*.csv")))
if not files: raise SystemExit("No baseline_*.csv found")
rows=[]
header=None
for f in files:
    run_id=os.path.splitext(os.path.basename(f))[0]
    with open(f,"r",newline="") as fp:
        rd=csv.DictReader(fp)
        if header is None: header=rd.fieldnames
        for r in rd:
            r["run_id"]=run_id
            rows.append(r)
with open(OUT,"w",newline="") as fp:
    w=csv.DictWriter(fp,fieldnames=header+["run_id"])
    w.writeheader()
    w.writerows(rows)
print("Wrote",OUT,"rows:",len(rows),"files:",len(files))
