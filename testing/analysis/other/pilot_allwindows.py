#!/usr/bin/env python3
# rank the true 3' end against every 1-nt-stride window in the gene
import sys, os, gzip, argparse, math
import numpy as np
def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser()
P.add_argument("--ngene", type=int, default=40)
P.add_argument("--tol", type=int, default=250)
P.add_argument("--maxlen", type=int, default=400000)
P.add_argument("--seed", type=int, default=42)
P.add_argument("--model", default="cnn_model.pt")
A = P.parse_args()
UP, DN, W = 2000, 500, 2500
NT = "ACGT"; THR = 4.0
FA = "/scratch/spour98/jobs/hg38.fa"
SG = "/scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/score_genome"
PGD = os.path.join(SG, "out_scores", "per_gene_dir")

import torch, pysam
sys.path.insert(0, "scripts")
import u1_5ss_landscape_data as L
from maxentpy import maxent
from maxentpy.maxent import load_matrix3
M3 = load_matrix3()
U1 = np.asarray(L.U1, float); POW = np.array([int(L.POW[p]) for p in range(9)], np.int64)

# sample
AC = np.load("analysis_columns.npz", allow_pickle=True)
gene = np.array([str(x) for x in AC["gene"]]); chrom = np.array([str(x) for x in AC["chrom"]])
center = AC["center"].astype(np.int64); strand = np.array([str(x) for x in AC["strand"]])
iscan = AC["is_canonical"].astype(bool); me3_cached = AC["spl_upstream_max_me3"].astype(float)
D = np.load("cnn_data.npz", allow_pickle=True); ok = D["ok"]
split = np.array([str(x) for x in D["split"][ok]])
assert np.array_equal(np.array([str(x) for x in D["gene"][ok]]), gene)
BODY = {}
for line in open(os.path.join(SG, "pc_genes.bed")):
    p = line.rstrip("\n").split("\t")
    BODY[p[3]] = (p[0], int(p[1]), int(p[2]), p[5])
import collections
ncand = collections.Counter(gene)
tg = sorted({g for g, s in zip(gene, split) if s == "test"} & set(BODY))
allg = sorted(set(gene) & set(BODY))
def glen(g): _, s, e, _ = BODY[g]; return e-s
LEN_ALL = np.array([glen(g) for g in allg])
cand = [g for g in tg if glen(g) <= A.maxlen and ncand[g] >= 2]
log(f"[sample] {len(tg):,} held-out test genes with a body; {len(cand):,} under {A.maxlen:,} nt "
    f"with at least two called peaks")
rng = np.random.default_rng(A.seed)
Lc = np.array([glen(g) for g in cand]); Cc = np.array([ncand[g] for g in cand])
lq = np.quantile(Lc, [1/3, 2/3]); cq = np.quantile(Cc, [1/3, 2/3])
lb = np.digitize(Lc, lq); cb = np.digitize(Cc, cq)
per = max(1, A.ngene//9); pick = []
for i in range(3):
    for j in range(3):
        m = np.flatnonzero((lb == i) & (cb == j))
        if len(m) == 0: continue
        pick += [cand[k] for k in rng.choice(m, min(per, len(m)), replace=False)]
pick = sorted(set(pick))[:A.ngene]
LS = np.array([glen(g) for g in pick])
log(f"[sample] {len(pick)} genes, stratified 3 length x 3 candidate-count strata")
qs = [0, 10, 25, 50, 75, 90, 100]
log( "         gene length percentiles (nt)")
log(f"           sample     " + "  ".join(f"{np.percentile(LS,q):>9,.0f}" for q in qs))
log(f"           genomewide " + "  ".join(f"{np.percentile(LEN_ALL,q):>9,.0f}" for q in qs))
log(f"         (percentiles {qs})")

# me3 definition, validated
def tok_of(s):
    a = np.frombuffer(s.upper().encode(), np.uint8)
    o = np.full(a.shape, 4, np.uint8)
    for i, b in enumerate(b"ACGT"): o[a == b] = i
    return o
def rc_tok(t): return np.where(t < 4, 3-t, 4)[::-1]
fa = pysam.FastaFile(FA)
def fetch_tok(ch, s, e, st):
    s0, e0 = max(0, s), e
    t = tok_of(fa.fetch(ch, s0, e0))
    if s < 0: t = np.concatenate([np.full(-s, 4, np.uint8), t])
    return rc_tok(t) if st == "-" else t
def score3_arr(t):
    n = len(t)-22
    if n <= 0: return np.zeros(0)
    s = "".join(NT[x] if x < 4 else "N" for x in t)
    out = np.full(n, np.nan)
    for i in range(n):
        w = s[i:i+23]
        if "N" in w: continue
        out[i] = maxent.score3(w, matrix=M3)
    return out
log("\n[me3] validating the acceptor scalar definition against analysis_columns.npz")
vi = rng.choice(np.flatnonzero(iscan), 40, replace=False)
got, want = [], []
for i in vi:
    c = int(center[i]); st = strand[i]
    lo = c-UP-40; hi = c+40
    t = fetch_tok(chrom[i], lo if st == "+" else c-40, hi if st == "+" else c+UP+40, st)
    s3 = score3_arr(t)
    cpos = UP+40
    b = np.arange(len(s3))+20
    m = (b >= cpos-UP) & (b < cpos)
    if m.sum() and np.isfinite(s3[m]).any():
        got.append(np.nanmax(s3[m])); want.append(me3_cached[i])
got = np.array(got); want = np.array(want)
r = np.corrcoef(got, want)[0, 1]
log(f"      recomputed vs cached on {len(got)} canonical peaks: r = {r:+.4f}, "
    f"median |diff| {np.median(np.abs(got-want)):.3f}")
if r < 0.9: log("      WARNING: definition does not reproduce the cache; treat the splice arm as approximate")

# models
from cnn_model_def import Net
dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(A.model, map_location=dev, weights_only=False)
net = Net(ck["args"]["dropout"]).to(dev); net.load_state_dict(ck["state"]); net.eval()
for p in net.parameters(): p.requires_grad_(False)
TBL = np.zeros((256, 4), np.float32); TBL[4] = .25
for i in range(4): TBL[i, i] = 1.
TBLt = torch.from_numpy(TBL).to(dev)
SM = np.load("splice_model_26.npz", allow_pickle=True)
beta = SM["beta"]; mu = SM["mu"]; sd = SM["sd"]; BINLO = SM["bin_lo"].astype(int)

SC = np.load("splice26_scores.npz", allow_pickle=True)
CN = np.load("cnn_model_scores.npz", allow_pickle=True)
sp_pk = SC["score"].astype(float); cn_pk = CN["logit"].astype(float)
yl = (iscan).astype(float)
fitm = np.isin(split, ["train", "val"])
def gcen(v, g):
    o = v.copy()
    for gg, idx in collections.defaultdict(list, {k: [] for k in set(g)}).items(): pass
    dd = collections.defaultdict(list)
    for i, k in enumerate(g): dd[k].append(i)
    for k, idx in dd.items():
        idx = np.array(idx); o[idx] -= v[idx].mean()
    return o
Xs = np.column_stack([gcen(sp_pk, gene), gcen(cn_pk, gene)])
mS, sS = Xs[fitm].mean(0), Xs[fitm].std(0)
Z = (Xs-mS)/sS
def irls(X, y, lam=1.0, it=60):
    Xd = np.column_stack([np.ones(len(X)), X]); b = np.zeros(Xd.shape[1])
    for _ in range(it):
        p = 1/(1+np.exp(-Xd@b)); w = np.clip(p*(1-p), 1e-6, None)
        z = Xd@b+(y-p)/w
        R = np.eye(Xd.shape[1])*lam; R[0, 0] = 0
        b = np.linalg.solve(Xd.T@(Xd*w[:, None])+R, Xd.T@(w*z))
    return b
bc = irls(Z[fitm], yl[fitm])
log(f"\n[combo] logistic fit on {int(fitm.sum()):,} train and val peaks: "
    f"intercept {bc[0]:+.3f}, splice {bc[1]:+.3f}, CNN {bc[2]:+.3f}")

# score every window
def cnn_scores(t, cs, bs=1024):
    out = np.empty(len(cs))
    for i in range(0, len(cs), bs):
        cc = cs[i:i+bs]
        blk = np.stack([t[c-UP:c+DN] for c in cc])
        x = TBLt[torch.from_numpy(blk.astype(np.int64)).to(dev)].permute(0, 2, 1).contiguous()
        with torch.inference_mode(): out[i:i+bs] = net(x).float().cpu().numpy()
    return out
rows, ranks = [], []
for gi, g in enumerate(pick, 1):
    ch, gs, ge, st = BODY[g]
    ci = np.flatnonzero((gene == g) & iscan)
    if len(ci) != 1: continue
    ctrue = int(center[ci[0]])
    lo, hi = gs-UP-32, ge+DN+32
    t = fetch_tok(ch, lo, hi, st)
    n = len(t)
    def to_t(p): return (p-lo) if st == "+" else (hi-1-p)
    c0, c1 = to_t(gs), to_t(ge-1)
    if st == "-": c0, c1 = c1, c0
    cs = np.arange(max(c0, UP), min(c1, n-DN)+1)
    if len(cs) < 100: continue
    ct = to_t(ctrue)
    K = n-8
    Xi = t.astype(np.int64); bad = Xi > 3; Xi[bad] = 0
    code = np.zeros(K, np.int64)
    for p in range(9): code += Xi[p:p+K]*POW[p]
    hit = (U1[code] >= THR)
    inv = np.zeros(K, bool)
    for p in range(9): inv |= bad[p:p+K]
    hit &= ~inv
    cum = np.concatenate([[0], np.cumsum(hit)])
    def binsum(a, b): return cum[np.clip(b, 0, K)]-cum[np.clip(a, 0, K)]
    F = np.empty((len(cs), 26))
    for j in range(25):
        F[:, j] = binsum(cs+BINLO[j], cs+BINLO[j]+100)*10.0
    s3 = score3_arr(t)
    b = np.arange(len(s3))+20
    big = np.where(np.isfinite(s3), s3, -1e9)
    import numpy.lib.stride_tricks as stt
    ordv = np.full(n+1, -1e9)
    ordv[np.clip(b, 0, n)] = np.maximum(ordv[np.clip(b, 0, n)], big)
    pref = np.maximum.accumulate(np.concatenate([[-1e9], ordv]))
    def winmax(a, bnd):
        out = np.empty(len(a))
        from collections import deque
        dq = deque(); j = 0
        order = np.argsort(a)
        for k in order:
            s_, e_ = a[k], bnd[k]
            while dq and dq[0] < s_: dq.popleft()
            while j < e_:
                v = ordv[j] if j < len(ordv) else -1e9
                while dq and ordv[dq[-1]] <= v: dq.pop()
                dq.append(j); j += 1
            out[k] = ordv[dq[0]] if dq else -1e9
        return out
    F[:, 25] = winmax(np.clip(cs-UP, 0, n), np.clip(cs, 0, n))
    Fc = F-F.mean(0, keepdims=True)
    spl = beta[0]+((Fc-mu)/sd)@beta[1:]
    cnn = cnn_scores(t, cs)
    zs = ((spl-spl.mean())/(spl.std()+1e-9)-mS[0]/1)*0+ (spl-spl.mean())/(spl.std()+1e-9)
    zc = (cnn-cnn.mean())/(cnn.std()+1e-9)
    combo = bc[0]+bc[1]*zs+bc[2]*zc
    rf = np.full(len(cs), np.nan)
    fp = os.path.join(PGD, ch, f"{g}.bed6.gz")
    if os.path.exists(fp):
        pos, val = [], []
        with gzip.open(fp, "rt") as fh:
            for line in fh:
                q = line.split("\t"); pos.append((int(q[1])+int(q[2]))//2); val.append(float(q[4]))
        if pos:
            pos = np.array(pos); val = np.array(val)
            tp = np.array([to_t(p) for p in pos]); o = np.argsort(tp)
            rf = np.interp(cs, tp[o], val[o], left=np.nan, right=np.nan)
    def rank_of(v):
        good = np.isfinite(v)
        if not good.any(): return None, None
        near = np.abs(cs-ct) <= A.tol
        if not (near & good).any(): return None, None
        best = np.nanmax(v[near & good])
        return int((v[good] > best).sum())+1, int(np.nanargmin(np.abs(cs-ct)))
    r_spl, _ = rank_of(spl); r_cnn, _ = rank_of(cnn)
    r_com, _ = rank_of(combo); r_rf, _ = rank_of(rf)
    rows.append(dict(gene=g, chrom=ch, strand=st, length=ge-gs, n_windows=len(cs),
                     n_peaks=int(ncand[g]), rank_splice=r_spl, rank_cnn=r_cnn,
                     rank_combo=r_com, rank_rf=r_rf))
    log(f"  [{gi:>2}/{len(pick)}] {g:<12} {ge-gs:>8,} nt  {len(cs):>8,} windows   "
        f"ranks  splice {str(r_spl):>7}  CNN {str(r_cnn):>7}  combo {str(r_com):>7}  RF {str(r_rf):>7}")
R = rows
log(f"\n[pilot] {len(R)} genes scored; windows per gene "
    f"{min(r['n_windows'] for r in R):,} to {max(r['n_windows'] for r in R):,}, "
    f"total {sum(r['n_windows'] for r in R):,}")
def rep(k, lab):
    v = np.array([r[k] for r in R if r[k] is not None], float)
    if not len(v): log(f"  {lab:<10} no ranks"); return
    log(f"  {lab:<10} P@1 {np.mean(v==1):.3f}   top5 {np.mean(v<=5):.3f}   top10 {np.mean(v<=10):.3f}"
        f"   top50 {np.mean(v<=50):.3f}   median rank {np.median(v):,.0f}   "
        f"best {v.min():,.0f}  worst {v.max():,.0f}   n={len(v)}")
log("\n[ranks of the true 3' end among ALL windows in the gene]")
for k, lab in (("rank_rf","baseline RF"),("rank_splice","splice 26"),("rank_cnn","CNN"),("rank_combo","combo")):
    rep(k, lab)
with open("pilot_allwindows_genes.tsv", "w") as f:
    ks = ["gene","chrom","strand","length","n_windows","n_peaks","rank_rf","rank_splice","rank_cnn","rank_combo"]
    f.write("\t".join(ks)+"\n")
    for r in R: f.write("\t".join("" if r[k] is None else str(r[k]) for k in ks)+"\n")
with open("pilot_allwindows_summary.tsv", "w") as f:
    f.write("model\tP_at_1\ttop5\ttop10\ttop50\tmedian_rank\tn_genes\n")
    for k, lab in (("rank_rf","baseline_RF"),("rank_splice","splice26"),("rank_cnn","CNN"),("rank_combo","combo")):
        v = np.array([r[k] for r in R if r[k] is not None], float)
        if len(v): f.write(f"{lab}\t{np.mean(v==1):.4f}\t{np.mean(v<=5):.4f}\t{np.mean(v<=10):.4f}\t"
                           f"{np.mean(v<=50):.4f}\t{np.median(v):.0f}\t{len(v)}\n")
log("[out] pilot_allwindows_genes.tsv / pilot_allwindows_summary.tsv")
