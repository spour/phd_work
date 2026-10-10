#!/usr/bin/env python3
# why does the splice model re-rank the rf's shortlist badly?
import sys, os, gzip, csv
import numpy as np
def log(*a): print(*a, file=sys.stderr, flush=True)
UP, DN = 2000, 500; NT = "ACGT"; THR = 4.0; MINSEP = 500; TOL = 250
FA = "/scratch/spour98/jobs/hg38.fa"
SG = "/scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/score_genome"
PGD = os.path.join(SG, "out_scores", "per_gene_dir")
sys.path.insert(0, "scripts")
import pysam, u1_5ss_landscape_data as L
from maxentpy import maxent
from maxentpy.maxent import load_matrix3
M3 = load_matrix3()
U1 = np.asarray(L.U1, float); POW = np.array([int(L.POW[p]) for p in range(9)], np.int64)
SM = np.load("splice_model_26.npz", allow_pickle=True)
beta = SM["beta"]; mu = SM["mu"]; sd = SM["sd"]; BINLO = SM["bin_lo"].astype(int)
AC = np.load("analysis_columns.npz", allow_pickle=True)
gene = np.array([str(x) for x in AC["gene"]]); center = AC["center"].astype(np.int64)
strand = np.array([str(x) for x in AC["strand"]]); iscan = AC["is_canonical"].astype(bool)
BODY = {}
for line in open(os.path.join(SG, "pc_genes.bed")):
    p = line.rstrip("\n").split("\t"); BODY[p[3]] = (p[0], int(p[1]), int(p[2]), p[5])
PICK = [r["gene"] for r in csv.DictReader(open("pilot_allwindows_genes.tsv"), delimiter="\t")]
fa = pysam.FastaFile(FA)
def tok_of(s):
    a = np.frombuffer(s.upper().encode(), np.uint8); o = np.full(a.shape, 4, np.uint8)
    for i, b in enumerate(b"ACGT"): o[a == b] = i
    return o
def rc(t): return np.where(t < 4, 3-t, 4)[::-1]
def score3_arr(t):
    n = max(len(t)-22, 0); s = "".join(NT[x] if x < 4 else "N" for x in t)
    out = np.full(n, np.nan)
    for i in range(n):
        w = s[i:i+23]
        if "N" not in w: out[i] = maxent.score3(w, matrix=M3)
    return out
def nms(sc, cs, need):
    keep = []
    for i in np.argsort(-sc):
        if not np.isfinite(sc[i]): continue
        if all(abs(cs[i]-cs[j]) >= MINSEP for j in keep):
            keep.append(int(i))
            if len(keep) >= need: break
    return np.array(keep, int)
rows = []; dist_rf = []; dist_pk = []
for gi, g in enumerate(PICK, 1):
    ch, gs, ge, st = BODY[g]
    ci = np.flatnonzero((gene == g) & iscan)
    if len(ci) != 1: continue
    ctrue = int(center[ci[0]])
    lo, hi = gs-UP-32, ge+DN+32
    t = tok_of(fa.fetch(ch, max(0, lo), hi))
    if lo < 0: t = np.concatenate([np.full(-lo, 4, np.uint8), t])
    if st == "-": t = rc(t)
    n = len(t)
    to_t = (lambda p: p-lo) if st == "+" else (lambda p: hi-1-p)
    c0, c1 = sorted([to_t(gs), to_t(ge-1)])
    cs = np.arange(max(c0, UP), min(c1, n-DN)+1)
    if len(cs) < 100: continue
    ct = to_t(ctrue)
    rf = np.full(len(cs), np.nan)
    fp = os.path.join(PGD, ch, f"{g}.bed6.gz")
    if not os.path.exists(fp): continue
    pos, val = [], []
    with gzip.open(fp, "rt") as fh:
        for line in fh:
            q = line.split("\t"); pos.append((int(q[1])+int(q[2]))//2); val.append(float(q[4]))
    pos = np.array(pos); val = np.array(val)
    tp = np.array([to_t(p) for p in pos]); o = np.argsort(tp)
    rf = np.interp(cs, tp[o], val[o], left=np.nan, right=np.nan)
    if not np.isfinite(rf).any(): continue
    K = n-8
    Xi = t.astype(np.int64); bad = Xi > 3; Xi[bad] = 0
    code = np.zeros(K, np.int64)
    for p in range(9): code += Xi[p:p+K]*POW[p]
    h = U1[code] >= THR
    inv = np.zeros(K, bool)
    for p in range(9): inv |= bad[p:p+K]
    h &= ~inv
    cum = np.concatenate([[0], np.cumsum(h)])
    s3 = score3_arr(t); bb = np.arange(len(s3))+20
    ordv = np.full(n+1, -1e9)
    np.maximum.at(ordv, np.clip(bb, 0, n), np.where(np.isfinite(s3), s3, -1e9))
    def wmax(a):
        pad = np.concatenate([ordv, np.full(UP, -1e9)]); m = np.full(len(a), -1e9)
        for s0 in range(0, UP, 512):
            m = np.maximum(m, np.stack([pad[a+q] for q in range(s0, min(s0+512, UP))]).max(0))
        return m
    def splice_at(ix):
        F = np.empty((len(ix), 26))
        for j in range(25):
            F[:, j] = (cum[np.clip(cs[ix]+BINLO[j]+100, 0, K)]-cum[np.clip(cs[ix]+BINLO[j], 0, K)])*10.0
        F[:, 25] = wmax(np.clip(cs[ix]-UP, 0, n))
        return beta[0]+(((F-F.mean(0, keepdims=True))-mu)/sd)@beta[1:]
    sel = nms(np.where(np.isfinite(rf), rf, -np.inf), cs, 20)
    d = cs[sel]-ct
    dist_rf += list(d)
    pk = np.flatnonzero(gene == g)
    dpk = np.array([to_t(int(center[i]))-ct for i in pk if not iscan[i]])
    dist_pk += list(dpk)
    for k in (10, 20):
        s = sel[:k]
        near = np.abs(cs[s]-ct) <= TOL
        if not near.any(): continue
        sv = splice_at(s)
        hit_all = int(near[int(np.argmax(sv))])
        up = s[(cs[s]-ct) <= TOL]
        hit_up = None
        if len(up) >= 2 and (np.abs(cs[up]-ct) <= TOL).any():
            su = splice_at(up)
            hit_up = int((np.abs(cs[up]-ct) <= TOL)[int(np.argmax(su))])
        rows.append(dict(gene=g, k=k, n_shortlist=len(s), n_upstream=int(len(up)),
                         hit_all=hit_all, hit_up=hit_up))
    if gi % 6 == 0: log(f"   {gi}/{len(PICK)}")
dist_rf = np.array(dist_rf, float); dist_pk = np.array(dist_pk, float)
log(f"\n[where the candidates sit relative to the true 3' end, transcript coords, + is DOWNSTREAM]")
log(f"  RF NMS top-20      n={len(dist_rf):>5}   downstream {100*np.mean(dist_rf>TOL):5.1f}%   "
    f"upstream {100*np.mean(dist_rf<-TOL):5.1f}%   median {np.median(dist_rf):+,.0f}")
log(f"  analysis-set peaks n={len(dist_pk):>5}   downstream {100*np.mean(dist_pk>TOL):5.1f}%   "
    f"upstream {100*np.mean(dist_pk<-TOL):5.1f}%   median {np.median(dist_pk):+,.0f}")
log(f"\n[splice as re-ranker, all candidates vs upstream-only]")
for k in (10, 20):
    r = [x for x in rows if x["k"] == k]
    ra = [x for x in r if x["hit_up"] is not None]
    log(f"  k={k:<3} genes where the true end is in the shortlist: {len(r)}")
    log(f"        P@1 over ALL shortlist candidates      {np.mean([x['hit_all'] for x in r]):.3f}")
    if ra: log(f"        P@1 with DOWNSTREAM candidates removed {np.mean([x['hit_up'] for x in ra]):.3f}"
               f"   ({len(ra)} genes, median {np.median([x['n_upstream'] for x in ra]):.0f} left of "
               f"{np.median([x['n_shortlist'] for x in ra]):.0f})")
with open("shortlist_shift.tsv", "w") as f:
    f.write("gene\tk\tn_shortlist\tn_upstream\thit_all\thit_upstream_only\n")
    for x in rows: f.write(f"{x['gene']}\t{x['k']}\t{x['n_shortlist']}\t{x['n_upstream']}\t"
                           f"{x['hit_all']}\t{'' if x['hit_up'] is None else x['hit_up']}\n")
log("[out] shortlist_shift.tsv")
