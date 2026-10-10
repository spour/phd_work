#!/usr/bin/env python3
# characterise what the 4-feature reranker gets wrong
import csv, numpy as np, pandas as pd
from collections import defaultdict
SEED, RIDGE = 42, 1.0
FINAL = ["scan_n3ss_dn5k", "spl_upstream_max_me3", "spl_pas_prox_u1", "spl_n_5ss_1k"]
rng = np.random.default_rng(SEED)

D = pd.read_parquet("shortlist_feat_all.parquet", engine="fastparquet")
M = list(csv.DictReader(open("fig3_new_positional_features.tsv"), delimiter="\t"))
D = D[D.gene.isin(set(r["gene"] for r in M))].reset_index(drop=True)
gene = D.gene.to_numpy(); chrom = D.chrom.to_numpy(); y = D.is_canonical.to_numpy().astype(float)
S = np.load("splice_feats_cache.npz")
km = {(r["chrom"], int(r["center"]), r["strand"], r["gene"]): i for i, r in enumerate(M)}
ridx = np.array([km[(c, int(ce), st, g)] for c, ce, st, g in zip(D.chrom, D.center, D.strand, D.gene)])
col = {"scan_n3ss_dn5k": S["C_n3ss_dn5k"][ridx]}
for n in FINAL:
    if n not in col: col[n] = D[n].to_numpy(float)
X = np.column_stack([col[n] for n in FINAL])

CLS = ["first exon", "internal exon", "last exon-3'UTR", "first intron", "internal intron",
       "last intron", "intergenic-other"]
fc_all = np.load("fine_compartment.npy")
Lb = list(csv.DictReader(open("cpa_peaks_labeled.tsv"), delimiter="\t"))
fmap = {(r["chrom"], int(r["center"]), r["strand"], r["gene"]): int(fc_all[i]) for i, r in enumerate(Lb)}
fine = np.array([fmap[(c, int(ce), st, g)] for c, ce, st, g in zip(D.chrom, D.center, D.strand, D.gene)])
At = np.load("atlas_peak_cache.npz")
a_dist = At["dist"][ridx]; a_tpm = At["tpm"][ridx]
matched = a_dist <= 25
gi = defaultdict(list)
for i in range(len(gene)): gi[gene[i]].append(i)
gi = {g: np.array(v) for g, v in gi.items()}
ufrac = np.zeros(len(gene))
for g, ix in gi.items():
    s = a_tpm[ix].sum()
    if s > 0: ufrac[ix] = a_tpm[ix] / s

def fit_ridge(Xt, yt, lam=RIDGE):
    P = np.eye(Xt.shape[1]) * lam; P[0, 0] = 0.0; b = np.zeros(Xt.shape[1])
    for _ in range(80):
        eta = np.clip(Xt @ b, -30, 30); mu = 1 / (1 + np.exp(-eta)); w = np.maximum(mu * (1 - mu), 1e-9)
        z = eta + (yt - mu) / w; XtW = Xt.T * w
        nb = np.linalg.solve(XtW @ Xt + P, XtW @ z + P @ b)
        if np.max(np.abs(nb - b)) < 1e-9: b = nb; break
        b = nb
    return b
pred = np.full(len(y), np.nan)
for c in sorted(set(chrom)):
    te = chrom == c; tr = ~te
    mu = np.nanmean(X[tr], 0); sd = np.nanstd(X[tr], 0); sd[sd == 0] = 1.0
    A = (np.where(np.isfinite(X[tr]), X[tr], mu) - mu) / sd
    B = (np.where(np.isfinite(X[te]), X[te], mu) - mu) / sd
    pred[te] = np.column_stack([np.ones(len(B)), B]) @ fit_ridge(np.column_stack([np.ones(len(A)), A]), y[tr])

GRP = {"correct pick": [], "marginal miss": [], "confident miss": []}
GAP = {k: [] for k in GRP}
for g, ix in gi.items():
    s = pred[ix]
    if len(ix) < 2 or not np.isfinite(s).all(): continue
    c = int(np.argmax(y[ix])); top = int(np.argmax(s + rng.random(len(ix)) * 1e-12))
    span = s.max() - s.min(); gap = 0.0 if span <= 0 else (s[top] - s[c]) / span
    key = "correct pick" if top == c else ("confident miss" if gap > 0.5 else "marginal miss")
    GRP[key].append(ix[top]); GAP[key].append(float(gap))
POOL = np.where(y == 0)[0]
SETS = [(k, np.array(v)) for k, v in GRP.items()] + [("competitor pool", POOL)]

OUT = open("rerank_min_misses.tsv", "w"); W = lambda *a: OUT.write("\t".join(str(x) for x in a) + "\n")
W("section", "group", "n", "key", "value", "value2")
print("\n=====  WHAT THE PICK IS, BY OUTCOME  (LOCO, 4-feature model)  =====")
print(f"{'group':<18} {'n':>6} {'atlas <=25nt':>13} {'usage>=10%':>11} {'median usage':>13} "
      + " ".join(f"{c[:14]:>15}" for c in CLS))
for nm, idx in SETS:
    n = len(idx)
    am = float(matched[idx].mean()); u10 = float((matched[idx] & (ufrac[idx] >= 0.10)).mean())
    um = float(np.median(ufrac[idx][matched[idx]])) if matched[idx].any() else 0.0
    comp = np.array([float((fine[idx] == k).mean()) for k in range(7)])
    print(f"{nm:<18} {n:>6} {am:>13.3f} {u10:>11.3f} {um:>13.3f} " + " ".join(f"{v:>15.3f}" for v in comp))
    W("atlas", nm, n, "matched_25nt", f"{am:.5f}", "")
    W("atlas", nm, n, "matched_usage_ge10", f"{u10:.5f}", "")
    W("atlas", nm, n, "median_usage_if_matched", f"{um:.5f}", "")
    for k in range(7): W("composition", nm, n, CLS[k], f"{comp[k]:.5f}", "")
    for q in (0.25, 0.5, 0.75, 0.9):
        W("usage", nm, n, f"q{int(q*100)}", f"{float(np.quantile(ufrac[idx],q)):.5f}", "")

cm = np.array(GRP["confident miss"]); cp = np.array(GRP["correct pick"])
print("\nconfident misses vs correct picks (difference in the picked peak's description):")
print(f"  atlas match           {matched[cm].mean():.3f}  vs  {matched[cp].mean():.3f}   "
      f"({matched[cm].mean()-matched[cp].mean():+.3f})")
print(f"  annotated, usage>=10% {(matched[cm]&(ufrac[cm]>=.10)).mean():.3f}  vs  "
      f"{(matched[cp]&(ufrac[cp]>=.10)).mean():.3f}   "
      f"({(matched[cm]&(ufrac[cm]>=.10)).mean()-(matched[cp]&(ufrac[cp]>=.10)).mean():+.3f})")
for k in (2, 4, 5):
    print(f"  {CLS[k]:<20} {(fine[cm]==k).mean():.3f}  vs  {(fine[cp]==k).mean():.3f}   "
          f"({(fine[cm]==k).mean()-(fine[cp]==k).mean():+.3f})")
print(f"\nof {len(cm)} confident misses, {int(matched[cm].sum())} ({matched[cm].mean()*100:.1f}%) land on a PolyASite "
      f"cluster within 25 nt; of those the median within-gene usage share is "
      f"{np.median(ufrac[cm][matched[cm]]):.3f} and {(ufrac[cm][matched[cm]]>=.10).mean()*100:.1f}% are at 10% usage or more")
gcm = np.array(GAP["confident miss"])
sub = [("atlas match, usage >= 10%", matched[cm] & (ufrac[cm] >= 0.10)),
       ("atlas match, usage < 10%", matched[cm] & (ufrac[cm] < 0.10)),
       ("no atlas match", ~matched[cm])]
print("\nthe confident misses are two populations:")
W("#")
W("split", "confident_miss", len(cm), "total", "", "")
for lab, m in sub:
    n = int(m.sum()); med = float(np.median(gcm[m])) if n else 0.0
    intr = float(np.isin(fine[cm][m], [3, 4, 5]).mean()) if n else 0.0
    print(f"  {lab:<26} {n:>5} ({n/len(cm)*100:>4.1f}%)  median gap {med:.3f} of range  intronic {intr:.3f}")
    W("split", lab, n, "median_gap", f"{med:.5f}", f"{intr:.5f}")
W("summary", "confident_miss", len(cm), "atlas_frac", f"{matched[cm].mean():.5f}", "")
W("summary", "correct_pick", len(cp), "atlas_frac", f"{matched[cp].mean():.5f}", "")
OUT.close()
print("\nwrote rerank_min_misses.tsv")
