#!/usr/bin/env python3
# how often does maxentscan call a strong splice site where there is none?
import csv, sys, os, numpy as np
from maxentpy import maxent
from maxentpy.maxent import load_matrix3, load_matrix5
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); PUB = os.path.join(ROOT, "..")
M3 = load_matrix3(); M5 = load_matrix5(); NT = np.array(list("ACGTN"))
R = list(csv.DictReader(open(os.path.join(PUB, "exon_ref_scores.tsv")), delimiter="\t"))
sa = np.array([float(r["three"]) for r in R if r["three"] not in ("", "nan")])
sd = np.array([float(r["five"]) for r in R if r["five"] not in ("", "nan")])
D = np.load(os.path.join(PUB, "cnn_data.npz"), allow_pickle=True)
comp = np.flatnonzero(D["y"] == 0)
pick = np.random.default_rng(0).choice(comp, 2000, replace=False)
XA = D["X"]
A3, D5, up3, up5 = [], [], [], []
for n, i in enumerate(pick):
    w = XA[i]; s = "".join(NT[np.minimum(w, 4)])
    a = [(j + 18, maxent.score3(s[j:j+23], matrix=M3)) for j in range(len(s) - 22)
         if s[j+18:j+20] == "AG" and "N" not in s[j:j+23]]
    d = [(j + 3, maxent.score5(s[j:j+9], matrix=M5)) for j in range(len(s) - 8)
         if s[j+3:j+5] == "GT" and "N" not in s[j:j+9]]
    A3.append(np.array([v for _, v in a])); D5.append(np.array([v for _, v in d]))
    up3.append(np.array([v for p, v in a if 0 <= p <= 2000 - 126]))
    up5.append(np.array([v for p, v in d if 0 <= p <= 2000 - 126]))
    if n % 500 == 0: print(f"  {n} windows", file=sys.stderr, flush=True)
a_all = np.concatenate(A3); d_all = np.concatenate(D5); L = 2500 * len(pick) / 1000.0
os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
with open(os.path.join(ROOT, "data/maxent_calibration.tsv"), "w") as fh:
    fh.write("site\tthreshold\tsensitivity_annotated\tfrac_anchored_background\tper_kb_background\tfrac_windows_upstream_hit\n")
    for t in np.arange(2, 14.01, 0.5):
        fh.write(f"acceptor\t{t:.1f}\t{np.mean(sa >= t):.4f}\t{np.mean(a_all >= t):.5f}\t{(a_all >= t).sum()/L:.4f}\t{np.mean([np.any(u >= t) for u in up3]):.4f}\n")
        fh.write(f"donor\t{t:.1f}\t{np.mean(sd >= t):.4f}\t{np.mean(d_all >= t):.5f}\t{(d_all >= t).sum()/L:.4f}\t{np.mean([np.any(u >= t) for u in up5]):.4f}\n")
print(f"[bg] {len(pick)} competitor windows, {len(a_all):,} AG-anchored and {len(d_all):,} GT-anchored positions", file=sys.stderr)
for site, arr, sens, up, t in (("acceptor", a_all, sa, up3, 6.5), ("donor", d_all, sd, up5, 8.0)):
    print(f"[{site} >= {t}] sensitivity {np.mean(sens >= t):.3f}; background {np.mean(arr >= t):.4f} of anchored positions, "
          f"{(arr >= t).sum()/L:.2f} per kb; windows with >= 1 upstream {np.mean([np.any(u >= t) for u in up]):.3f}", file=sys.stderr)
