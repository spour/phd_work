#!/usr/bin/env python3
# the training schema of the exponential cnn, as a tsv
import sys
import numpy as np

def log(*a): print(*a, file=sys.stderr, flush=True)
D = np.load("cnn_data.npz", allow_pickle=True); ok = D["ok"]
y = D["y"][ok]; gene = D["gene"][ok]; split = D["split"][ok]
UP, DN = int(D["up"]), int(D["dn"])
rows = []
for s in ("train", "val", "test"):
    m = split == s
    rows.append((s, int(m.sum()), int(y[m].sum()), int((y[m] == 0).sum()), len(set(gene[m]))))
    log(f"{s:5s} {rows[-1][1]:6d} peaks  {rows[-1][2]:5d} real  {rows[-1][3]:6d} competitor  "
        f"{rows[-1][4]:5d} genes")
for a, b in (("train", "test"), ("val", "test"), ("train", "val")):
    assert not (set(gene[split == a]) & set(gene[split == b])), f"gene spans {a}/{b}"
log("[check] no gene spans splits: OK")

tr = split == "train"
npos = float(y[tr].sum()); nneg = float((y[tr] == 0).sum())
posw = nneg / npos
te = split == "test"
base = float(y[te].mean())
T = None
for ln in open("calibration_diagnosis.tsv"):
    f = ln.rstrip("\n").split("\t")
    if len(f) > 2 and f[0] == "calibration" and f[1] == "platt_equiv_T": T = float(f[2])
assert T is not None, "platt_equiv_T missing from calibration_diagnosis.tsv"
log(f"pos_weight {posw:.2f}   base rate (test) {base:.5f}   window [{-UP},+{DN}] = {UP+DN} nt   "
    f"identifiable temperature {T:.3f}")

with open("cnn_training_schema.tsv", "w") as fh:
    fh.write("key\tvalue\n")
    for s, n, p, c, g in rows:
        fh.write(f"n_{s}\t{n}\nn_{s}_real\t{p}\nn_{s}_comp\t{c}\nn_{s}_genes\t{g}\n")
    fh.write(f"window_up\t{UP}\nwindow_dn\t{DN}\nwindow_len\t{UP+DN}\n")
    fh.write(f"pos_weight\t{posw:.4f}\nbase_rate_test\t{base:.5f}\ntemperature\t{T:.4f}\n")
    fh.write("real_label\tis_canonical\ncompetitor_label\tin_analysis_set\n")
    fh.write("split_rule\tchromosome holdout plus in_gene_holdout genes in test\n")
    fh.write("early_stop\tval within-gene win-rate\n")
log("[out] cnn_training_schema.tsv")
