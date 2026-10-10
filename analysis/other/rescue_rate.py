#!/usr/bin/env python3
# on the genes the baseline cannot resolve
import sys, collections
import numpy as np
def log(*a): print(*a, file=sys.stderr, flush=True)
import pyarrow.parquet as pq

D = pq.read_table("shortlist_feat_all.parquet", columns=["gene", "is_canonical"]).to_pandas()
ncomp = collections.Counter(D.loc[~D.is_canonical, "gene"])
allg = sorted(set(D.loc[D.is_canonical, "gene"]))
contested = sorted([g for g in allg if ncomp.get(g, 0) >= 1])
UNC = len(allg)-len(contested); FRAC_UNC = UNC/len(allg); FRAC_CON = len(contested)/len(allg)
log(f"[genes] {len(allg):,} with a canonical site")
log(f"        uncontested, baseline already right : {UNC:,}  ({100*FRAC_UNC:.1f}%)")
log(f"        contested, baseline wrong by construction : {len(contested):,}  ({100*FRAC_CON:.1f}%)")
log(f"        identical under RF, LR and CNN: the shortlist rule is a consensus")
log(f"        baseline genome-wide P@1 = {FRAC_UNC:.4f}")

Z = np.load("cnn_wg_scores.npz", allow_pickle=True)
zg = np.array([str(x) for x in Z["gene"]]); zc = Z["center"].astype(np.int64)
zy = Z["y"].astype(int); zl = Z["logit"].astype(float)
S = np.load("splice26_loco_scores.npz", allow_pickle=True)
sg = np.array([str(x) for x in S["gene"]]); sc = S["center"].astype(np.int64)
sv = S["score"].astype(float); sy = S["y"].astype(int)
smap = {(g, int(c)): i for g, c, i in zip(sg, sc, range(len(sg)))}
zmap = {(g, int(c)): i for g, c, i in zip(zg, zc, range(len(zg)))}
log(f"\n[scores] extended CNN LOCO over {len(zg):,} peaks; splice 26 LOCO over {len(sg):,} peaks")

def zs(v):
    s = v.std()
    return (v-v.mean())/(s if s > 1e-9 else 1.0)
def p1(v, y): return int(y[np.argmax(v)] == 1)

# candidate set a: the analysis set, both models complete
ga = collections.defaultdict(list)
for i in range(len(sg)): ga[sg[i]].append(i)
rowsA = []
for g, idx in ga.items():
    idx = np.array(idx); y = sy[idx]
    if y.sum() != 1 or len(idx) < 2: continue
    keys = [(g, int(sc[i])) for i in idx]
    if not all(k in zmap for k in keys): continue
    a = zs(sv[idx]); b = zs(zl[np.array([zmap[k] for k in keys])])
    rowsA.append(dict(gene=g, n=len(idx), spl=p1(a, y), cnn=p1(b, y), sum=p1(a+b, y)))
log(f"[set A] analysis-set candidates: {len(rowsA):,} contested genes, "
    f"median {np.median([r['n'] for r in rowsA]):.0f} candidates, both models complete")

# candidate set b: the full shortlist, cnn only
gb = collections.defaultdict(list)
for i in range(len(zg)): gb[zg[i]].append(i)
rowsB = []
for g, idx in gb.items():
    idx = np.array(idx); y = zy[idx]
    if y.sum() != 1 or len(idx) < 2: continue
    rowsB.append(dict(gene=g, n=len(idx), cnn=p1(zs(zl[idx]), y)))
log(f"[set B] full shortlist candidates:  {len(rowsB):,} contested genes, "
    f"median {np.median([r['n'] for r in rowsB]):.0f} candidates, CNN only")

def rep(rows, key, lab, n_note=""):
    v = np.mean([r[key] for r in rows])
    lift = FRAC_UNC+v*FRAC_CON
    log(f"    {lab:<26} rescue {v:.4f}   genome-wide {FRAC_UNC:.4f} -> {lift:.4f}   "
        f"(+{100*(lift-FRAC_UNC):.1f} points, n={len(rows):,}){n_note}")
    return v, lift
log("\n[RESCUE RATE, both models out of fold on every gene]")
log(f"\n  candidate set A, the analysis set, {len(rowsA):,} genes")
outA = {}
for key, lab in (("cnn", "extended CNN"), ("spl", "splice 26"), ("sum", "z-sum of the two")):
    outA[key] = rep(rowsA, key, lab)
log(f"\n  candidate set B, the full shortlist, {len(rowsB):,} genes")
outB = {"cnn": rep(rowsB, "cnn", "extended CNN")}
log(f"\n  the two sets differ because A drops the APA-annotated and terminal-exon competitors,")
log(f"  which makes each gene's contest smaller and easier")
with open("rescue_rate.tsv", "w") as f:
    f.write("candidate_set\tmodel\trescue_rate\tn_genes\tbaseline_genomewide\tlifted_genomewide\n")
    for key, lab in (("cnn", "extended_CNN"), ("spl", "splice26"), ("sum", "z_sum")):
        v, l = outA[key]; f.write(f"analysis_set\t{lab}\t{v:.6f}\t{len(rowsA)}\t{FRAC_UNC:.6f}\t{l:.6f}\n")
    v, l = outB["cnn"]; f.write(f"full_shortlist\textended_CNN\t{v:.6f}\t{len(rowsB)}\t{FRAC_UNC:.6f}\t{l:.6f}\n")
with open("rescue_rate_genes.tsv", "w") as f:
    f.write("candidate_set\tgene\tn_candidates\tsplice\tcnn\tzsum\n")
    for r in rowsA: f.write(f"analysis_set\t{r['gene']}\t{r['n']}\t{r['spl']}\t{r['cnn']}\t{r['sum']}\n")
    for r in rowsB: f.write(f"full_shortlist\t{r['gene']}\t{r['n']}\t\t{r['cnn']}\t\n")
log("\n[out] rescue_rate.tsv / rescue_rate_genes.tsv")
