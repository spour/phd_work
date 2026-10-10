#!/usr/bin/env python3
# why does the 26-feature splice model collapse in the all-windows test?
import sys, os
import numpy as np
def log(*a): print(*a, file=sys.stderr, flush=True)
UP, DN = 2000, 500; NT = "ACGT"; THR = 4.0
FA = "/scratch/spour98/jobs/hg38.fa"
SG = "/scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/score_genome"
sys.path.insert(0, "scripts")
import pysam, u1_5ss_landscape_data as L
from maxentpy import maxent
from maxentpy.maxent import load_matrix3
M3 = load_matrix3()
U1 = np.asarray(L.U1, float); POW = np.array([int(L.POW[p]) for p in range(9)], np.int64)
SM = np.load("splice_model_26.npz", allow_pickle=True)
beta = SM["beta"]; mu = SM["mu"]; sd = SM["sd"]; BINLO = SM["bin_lo"].astype(int)
AC = np.load("analysis_columns.npz", allow_pickle=True)
gene = np.array([str(x) for x in AC["gene"]]); chrom = np.array([str(x) for x in AC["chrom"]])
center = AC["center"].astype(np.int64); strand = np.array([str(x) for x in AC["strand"]])
iscan = AC["is_canonical"].astype(bool); me3c = AC["spl_upstream_max_me3"].astype(float)
D = np.load("cnn_data.npz", allow_pickle=True); ok = D["ok"]
split = np.array([str(x) for x in D["split"][ok]])
BODY = {}
for line in open(os.path.join(SG, "pc_genes.bed")):
    p = line.rstrip("\n").split("\t"); BODY[p[3]] = (p[0], int(p[1]), int(p[2]), p[5])
import csv
PICK = [r["gene"] for r in csv.DictReader(open("pilot_allwindows_genes.tsv"), delimiter="\t")]
log(f"[diag] {len(PICK)} genes from the pilot sample")
fa = pysam.FastaFile(FA)
def tok_of(s):
    a = np.frombuffer(s.upper().encode(), np.uint8); o = np.full(a.shape, 4, np.uint8)
    for i, b in enumerate(b"ACGT"): o[a == b] = i
    return o
def rc(t): return np.where(t < 4, 3-t, 4)[::-1]
def score3_arr(t):
    n = len(t)-22
    s = "".join(NT[x] if x < 4 else "N" for x in t)
    out = np.full(max(n, 0), np.nan)
    for i in range(n):
        w = s[i:i+23]
        if "N" not in w: out[i] = maxent.score3(w, matrix=M3)
    return out
def winmax_prefixfree(ordv, a, b):
    wdt = int(b[0]-a[0]); assert np.all(b-a == wdt)
    pad = np.concatenate([ordv, np.full(wdt, -1e9)])
    m = np.full(len(a), -1e9)
    step = 512
    for s0 in range(0, wdt, step):
        e0 = min(s0+step, wdt)
        blk = np.stack([pad[a+o] for o in range(s0, e0)])
        m = np.maximum(m, blk.max(0))
    return m
def feats(g, cs_abs):
    ch, gs, ge, st = BODY[g]
    lo, hi = gs-UP-32, ge+DN+32
    t = tok_of(fa.fetch(ch, max(0, lo), hi))
    if lo < 0: t = np.concatenate([np.full(-lo, 4, np.uint8), t])
    if st == "-": t = rc(t)
    n = len(t)
    to_t = (lambda p: p-lo) if st == "+" else (lambda p: hi-1-p)
    cs = np.array([to_t(p) for p in cs_abs])
    K = n-8
    Xi = t.astype(np.int64); bad = Xi > 3; Xi[bad] = 0
    code = np.zeros(K, np.int64)
    for p in range(9): code += Xi[p:p+K]*POW[p]
    h = U1[code] >= THR
    inv = np.zeros(K, bool)
    for p in range(9): inv |= bad[p:p+K]
    h &= ~inv
    cum = np.concatenate([[0], np.cumsum(h)])
    F = np.empty((len(cs), 26))
    for j in range(25):
        F[:, j] = (cum[np.clip(cs+BINLO[j]+100, 0, K)]-cum[np.clip(cs+BINLO[j], 0, K)])*10.0
    s3 = score3_arr(t); b = np.arange(len(s3))+20
    ordv = np.full(n+1, -1e9)
    np.maximum.at(ordv, np.clip(b, 0, n), np.where(np.isfinite(s3), s3, -1e9))
    F[:, 25] = winmax_prefixfree(ordv, np.clip(cs-UP, 0, n), np.clip(cs-UP, 0, n)+UP)
    allc = np.arange(max(UP, 0), n-DN)
    Fa = np.empty((len(allc), 26))
    for j in range(25):
        Fa[:, j] = (cum[np.clip(allc+BINLO[j]+100, 0, K)]-cum[np.clip(allc+BINLO[j], 0, K)])*10.0
    Fa[:, 25] = winmax_prefixfree(ordv, np.clip(allc-UP, 0, n), np.clip(allc-UP, 0, n)+UP)
    return F, Fa.mean(0)
def score(Fc): return beta[0]+((Fc-mu)/sd)@beta[1:]
def wr_p1(s, y):
    if y.sum() != 1 or len(y) < 2: return None, None
    t = s[y == 1][0]; o = s[y == 0]
    return float((t > o).mean()+0.5*(t == o).mean()), int((o < t).all())
res = {k: [] for k in "ABCD"}
for gi, g in enumerate(PICK, 1):
    idx = np.flatnonzero(gene == g)
    if len(idx) < 2 or iscan[idx].sum() != 1: continue
    F, allmean = feats(g, center[idx])
    y = iscan[idx].astype(int)
    Fcache = F.copy(); Fcache[:, 25] = me3c[idx]
    for key, Fx, cen in (("A", Fcache, "peak"), ("B", F, "peak"),
                         ("C", F, "all"), ("D", Fcache, "all")):
        Fc = Fx-(Fx.mean(0, keepdims=True) if cen == "peak" else allmean[None, :])
        w, p = wr_p1(score(Fc), y)
        if w is not None: res[key].append((w, p))
    if gi % 6 == 0: log(f"   {gi}/{len(PICK)}")
log(f"\n[result] over {len(res['A'])} genes of the pilot sample")
LAB = {"A": "cached me3, peak-centred   (the validated path)",
       "B": "recomputed me3, peak-centred",
       "C": "recomputed me3, ALL-WINDOW-centred",
       "D": "cached me3, ALL-WINDOW-centred"}
for k in "ABCD":
    a = np.array(res[k])
    log(f"  {k}  {LAB[k]:<46} win rate {a[:,0].mean():.4f}   P@1 {a[:,1].mean():.4f}")
log("\n  reference: acceptor_options_metrics.tsv reports 0.6993 / 0.4720 on the 1,623 test genes")
with open("splice_diagnose.tsv", "w") as f:
    f.write("variant\tdescription\twinrate\tp_at_1\tn_genes\n")
    for k in "ABCD":
        a = np.array(res[k])
        f.write(f"{k}\t{LAB[k]}\t{a[:,0].mean():.6f}\t{a[:,1].mean():.6f}\t{len(a)}\n")
log("[out] splice_diagnose.tsv")
