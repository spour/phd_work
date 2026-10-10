#!/usr/bin/env python3
# test a and test b
import sys, csv, math
import numpy as np

def log(*a): print(*a, file=sys.stderr, flush=True)
def f(x):
    try:
        v = float(x); return v if np.isfinite(v) else np.nan
    except Exception: return np.nan
R = list(csv.DictReader(open("fig3_ucube_meta.tsv"), delimiter="\t"))
M = {(r["gene"], r["chrom"], int(r["center"])): r for r in R}
Z = np.load("cnn_wg_scores.npz", allow_pickle=True)
key = list(zip(Z["gene"], Z["chrom"], Z["center"].astype(int)))
hit = np.array([k in M for k in key])
L = np.asarray(Z["logit"])[hit]
rows = [M[k] for k, h in zip(key, hit) if h]
G = np.array([r["gene"] for r in rows]); C = np.array([r["is_canonical"] == "1" for r in rows])
FEAT = {}
for c in ("acc", "edef_best5", "nearest_dn", "u1_dw_dn", "u1_dw_up", "interv_donor", "pas"):
    FEAT[c] = np.array([f(r[c]) for r in rows])
log(f"[join] {hit.sum():,} of {len(key):,} peaks matched to the splice feature table "
    f"({C.sum():,} real 3' ends, {(~C).sum():,} competitors, {len(set(G)):,} genes)")
HASDN = np.isfinite(FEAT["nearest_dn"]) & (FEAT["nearest_dn"] <= 300)
NODN = ~np.isfinite(FEAT["nearest_dn"])
log(f"[donor] no strong downstream donor at all: {100*NODN.mean():.1f}% of peaks "
    f"({100*NODN[C].mean():.1f}% of real 3' ends, {100*NODN[~C].mean():.1f}% of competitors)")
log(f"[donor] strong donor within +300 nt:      {100*HASDN.mean():.1f}% of peaks "
    f"({100*HASDN[C].mean():.1f}% of real 3' ends, {100*HASDN[~C].mean():.1f}% of competitors)")

def pear(x, y):
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    n = len(x)
    if n < 10: return np.nan, np.nan, np.nan, n
    r = float(np.corrcoef(x, y)[0, 1])
    z = 0.5*math.log((1+r)/(1-r)); se = 1/math.sqrt(n-3)
    return r, math.tanh(z-1.96*se), math.tanh(z+1.96*se), n
uq, inv = np.unique(G, return_inverse=True); cnt = np.bincount(inv)
def within_gene(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    g = inv[m]; u, i2 = np.unique(g, return_inverse=True); c2 = np.bincount(i2)
    k = c2[i2] >= 2
    xa = x[m][k] - (np.bincount(i2, x[m])/c2)[i2[k]]
    ya = y[m][k] - (np.bincount(i2, y[m])/c2)[i2[k]]
    return pear(xa, ya)

SIGN = {"acc": "+ if the model reads acceptor strength",
        "edef_best5": "- if the model wants NO downstream donor",
        "u1_dw_dn": "- if the model wants NO downstream donor",
        "nearest_dn": "+ if the model wants the donor FAR away",
        "interv_donor": "- if the model penalises internal-exon architecture",
        "u1_dw_up": "(upstream U1, for contrast)"}
out = []
log(f"\nTEST A and B  CNN logit against real splice architecture, {len(L):,} peaks")
log(f"  {'feature':<13} {'scope':<20} {'n':>7} {'r':>9} {'95% CI':>20}   expected sign")
for c in ("acc", "edef_best5", "u1_dw_dn", "nearest_dn", "interv_donor", "u1_dw_up"):
    v = FEAT[c]
    for nm, m in (("all peaks", np.ones(len(L), bool)),
                  ("real 3' ends", C), ("competitors", ~C)):
        r, lo, hi, n = pear(v[m], L[m])
        log(f"  {c:<13} {nm:<20} {n:>7,} {r:>+9.4f} {f'[{lo:+.4f}, {hi:+.4f}]':>20}"
            f"{'   ' + SIGN[c] if nm == 'all peaks' else ''}")
        out.append((c, nm, n, r, lo, hi))
    r, lo, hi, n = within_gene(v, L)
    log(f"  {c:<13} {'within gene':<20} {n:>7,} {r:>+9.4f} {f'[{lo:+.4f}, {hi:+.4f}]':>20}")
    out.append((c, "within gene", n, r, lo, hi))
with open("terminal_exon_test.tsv", "w") as fh:
    fh.write("feature\tscope\tn\tpearson_r\tci_lo\tci_hi\n")
    for o in out: fh.write("\t".join(str(x) for x in o) + "\n")

log(f"\nTEST A  mean CNN logit by presence of a strong downstream donor")
log(f"  {'group':<34} {'n':>7} {'all':>9} {'real ends':>11} {'competitors':>12}")
for nm, m in (("no strong downstream donor at all", NODN),
              ("strong donor beyond +300 nt", np.isfinite(FEAT["nearest_dn"]) & ~HASDN),
              ("strong donor within +300 nt", HASDN)):
    log(f"  {nm:<34} {m.sum():>7,} {L[m].mean():>+9.3f} {L[m & C].mean():>+11.3f} "
        f"{L[m & ~C].mean():>+12.3f}")
d = L[NODN & C].mean() - L[HASDN & C].mean()
se = math.sqrt(L[NODN & C].var(ddof=1)/max((NODN & C).sum(),1) +
               L[HASDN & C].var(ddof=1)/max((HASDN & C).sum(),1))
log(f"  within real 3' ends, no-donor minus donor-within-300: {d:+.3f} +/- {se:.3f} z = {d/se:+.1f}")
d2 = L[NODN & ~C].mean() - L[HASDN & ~C].mean()
se2 = math.sqrt(L[NODN & ~C].var(ddof=1)/max((NODN & ~C).sum(),1) +
                L[HASDN & ~C].var(ddof=1)/max((HASDN & ~C).sum(),1))
log(f"  within competitors,  no-donor minus donor-within-300: {d2:+.3f} +/- {se2:.3f} z = {d2/se2:+.1f}")

log(f"\nTEST A  mean CNN logit by decile of downstream donor strength (edef_best5)")
v = FEAT["edef_best5"]; ok = np.isfinite(v)
q = np.quantile(v[ok], np.linspace(0, 1, 11))
with open("terminal_exon_binned.tsv", "w") as fh:
    fh.write("decile\tlo\thi\tn\tmean_logit\tmean_logit_real\tmean_logit_comp\tfrac_real\n")
    log(f"  {'decile':>6} {'edef_best5':>14} {'n':>7} {'all':>9} {'real ends':>11} {'competitors':>12}")
    for i in range(10):
        m = ok & (v >= q[i]) & ((v <= q[i+1]) if i == 9 else (v < q[i+1]))
        if m.sum() < 5: continue
        log(f"  {i+1:>6} {f'{q[i]:.2f}-{q[i+1]:.2f}':>14} {m.sum():>7,} {L[m].mean():>+9.3f} "
            f"{L[m & C].mean():>+11.3f} {L[m & ~C].mean():>+12.3f}")
        fh.write(f"{i+1}\t{q[i]:.4f}\t{q[i+1]:.4f}\t{m.sum()}\t{L[m].mean():.5f}\t"
                 f"{L[m&C].mean():.5f}\t{L[m&~C].mean():.5f}\t{C[m].mean():.5f}\n")
log("[out] terminal_exon_test.tsv terminal_exon_binned.tsv")
