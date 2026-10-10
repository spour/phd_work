#!/usr/bin/env python3
# contrast 1: competitor vs competitor within gene, location held constant
import argparse, sys
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--cube", required=True)
ap.add_argument("--cube_meta_atlas", required=True,
                help="output of atlas_join.py: cube meta plus atlas_score / annotation")
ap.add_argument("--out", default="fig3_contrast1.tsv")
ap.add_argument("--usage_col", default="atlas_score")
ap.add_argument("--nboot", type=int, default=2000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--min_genes", type=int, default=20)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)

z = np.load(a.cube)
U = z["cube"].astype(np.float64)
Mc = z["me5cube"].astype(np.float64) if "me5cube" in z else None
EDGES = z["edges"]; UTHR = z["thrs"]
MTHR = z["me5_thrs"] if "me5_thrs" in z else None
M = pd.read_csv(a.cube_meta_atlas, sep="\t")
assert len(M) == U.shape[0], "cube and meta row counts differ"
if Mc is None:
    print("[warn] no me5cube in the npz; 5'ss density rows will be skipped", file=sys.stderr)

b = lambda x: int(np.searchsorted(EDGES, x))
def dens(cube, thrs, lo, hi, t):
    ti = 3 + int(np.argmin(np.abs(np.asarray(thrs) - t)))
    return cube[:, b(lo):b(hi), ti].sum(1) / ((hi - lo) / 1000.0)
def wmax(cube, lo, hi):
    with np.errstate(invalid="ignore"):
        return np.nanmax(cube[:, b(lo):b(hi), 2], axis=1)

FEATS = [("U1 density/kb thr 4.0 (winner)", "U1", "[-500,+250] spans site", dens(U, UTHR, -500, 250, 4.0)),
         ("U1 max (discrete site)",         "U1", "[-450,+50] spans site",  wmax(U, -450, 50)),
         ("U1 density/kb thr 8.0 (discrete site)", "U1", "[-500,+400] spans site", dens(U, UTHR, -500, 400, 8.0)),
         ("U1 density/kb thr 4.0",          "U1", "[0,+250] 3' of site",    dens(U, UTHR, 0, 250, 4.0)),
         ("5'ss edef_best5",                "5ss", "50-300nt past anchor",  M.edef_best5.values.astype(float))]
if Mc is not None:
    FEATS += [("5'ss density/kb thr 8.0", "5ss", "[-500,+250] spans site", dens(Mc, MTHR, -500, 250, 8.0)),
              ("5'ss density/kb thr 8.0", "5ss", "[-500,0] exon body",     dens(Mc, MTHR, -500, 0, 8.0)),
              ("5'ss density/kb thr 8.0", "5ss", "[0,+250] 3' of site",    dens(Mc, MTHR, 0, 250, 8.0))]

K = (M.is_canonical.values == 0) & np.isfinite(M[a.usage_col].values)
idx = np.where(K)[0]
use = M[a.usage_col].values[idx].astype(float)
gene = M.gene.values[idx]
o = np.argsort(gene, kind="stable")
idx, use, gene = idx[o], use[o], gene[o]
cuts = np.r_[0, np.flatnonzero(gene[1:] != gene[:-1]) + 1, len(gene)]
groups = [(cuts[i], cuts[i+1]) for i in range(len(cuts)-1) if cuts[i+1]-cuts[i] >= 2]
print(f"[data] {len(idx):,} scored competitors ; {len(groups):,} genes with >=2", file=sys.stderr)

rows = []
for name, layer, region, vec in FEATS:
    f = vec[idx]
    per_gene = []
    for s, e in groups:
        fu, uu = f[s:e], use[s:e]
        ok = np.isfinite(fu) & np.isfinite(uu)
        if ok.sum() < 2: continue
        fu, uu = fu[ok], uu[ok]
        df_ = fu[:, None] - fu[None, :]
        du_ = uu[:, None] - uu[None, :]
        iu = np.triu(np.ones(df_.shape, bool), 1) & (df_ != 0) & (du_ != 0)
        if not iu.any(): continue
        conc = ((df_ > 0) & (du_ < 0))[iu].sum() + ((df_ < 0) & (du_ > 0))[iu].sum()
        per_gene.append(conc / iu.sum())
    if len(per_gene) < a.min_genes: continue
    pg = np.array(per_gene, float)
    bs = pg[rng.integers(0, len(pg), size=(a.nboot, len(pg)))].mean(1)
    lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    rows.append(dict(feature=name, layer=layer, region=region, genes=len(pg),
                     concordance=float(pg.mean()), ci_lo=lo, ci_hi=hi,
                     clears_null=bool(lo > 0.5)))

R = pd.DataFrame(rows).sort_values("concordance", ascending=False).reset_index(drop=True)
R.to_csv(a.out, sep="\t", index=False)
pd.set_option("display.width", 200)
print(R.to_string(index=False))
print(f"\n[out] {a.out}", file=sys.stderr)
