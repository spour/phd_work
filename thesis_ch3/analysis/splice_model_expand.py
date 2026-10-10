#!/usr/bin/env python3
# how far does an interpretable splicing model get if it is given terminal- exon-definition
import os, sys, collections, numpy as np, pysam
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); PUB = os.path.join(ROOT, "..")
def log(*a): print(*a, file=sys.stderr, flush=True)
D = np.load(os.path.join(PUB, "cnn_data.npz"), allow_pickle=True); y = D["y"].astype(int); g = D["gene"]
ch, cen, st = D["chrom"], D["center"].astype(np.int64), D["strand"]; N = len(y)
C = np.load(os.path.join(PUB, "cnn_loco_scores.npz"), allow_pickle=True); assert (C["gene"] == g).all(); logit = C["logit"].astype(float)
key = lambda a, b, c, d: (str(a), str(b), int(float(c)), str(d)); dk = [key(*t) for t in zip(g, ch, cen, st)]
def table(path, cols):
    with open(path) as fh:
        hdr = fh.readline().rstrip("\n").split("\t"); ix = {h: j for j, h in enumerate(hdr)}; out = {}
        for line in fh:
            f = line.rstrip("\n").split("\t")
            out[key(f[ix["gene"]], f[ix["chrom"]], f[ix["center"]], f[ix["strand"]])] = [float(f[ix[c]]) if f[ix[c]] not in ("", "nan", "NA") else np.nan for c in cols]
    return out
UC = ["acc", "interv_donor", "edef_best5", "nearest_dn", "u1_dw_up", "u1_dw_dn"]
U = table(os.path.join(PUB, "fig3_ucube_meta.tsv"), UC); SA = table(os.path.join(PUB, "sa_peak_scalars_final.tsv"), ["sa_acc_max", "sa_don_sum"])
X = {c: np.array([U[k][j] for k in dk]) for j, c in enumerate(UC)}
X["nearest_dn"] = np.log1p(X["nearest_dn"])
X["sa_acc_max"] = np.array([SA[k][0] for k in dk]); X["sa_don_sum"] = np.array([SA[k][1] for k in dk])
W = np.load(os.path.join(PUB, "cnn_wg_scores.npz"), allow_pickle=True); S = np.load(os.path.join(ROOT, "data/stress_context.npz"), allow_pickle=True)
wrow = {key(*t): i for i, t in enumerate(zip(W["gene"], W["chrom"], W["center"], W["strand"]))}
X["exon_upstream_500"] = S["exo"][[wrow[k] for k in dk]].astype(float)
fa = pysam.FastaFile(os.path.join(PUB, "../train/data/hg38.fa")); tr = str.maketrans("ACGTN", "TGCAN")
for c in ("ugua", "u_dn", "gu_dn", "gc_up", "pas"): X[c] = np.zeros(N)
for i in range(N):
    p = int(cen[i]); c_ = str(ch[i])
    s = fa.fetch(c_, p - 500, p + 41).upper() if st[i] == "+" else fa.fetch(c_, p - 40, p + 501).upper().translate(tr)[::-1]
    s = s.replace("T", "U"); z = 500; dn = s[z + 1:z + 41]
    X["ugua"][i] = sum(s[k:k + 4] == "UGUA" for k in range(z - 100, z - 23)); X["u_dn"][i] = dn.count("U") / 40
    X["gu_dn"][i] = sum(dn[k:k + 2] == "GU" for k in range(39)); X["gc_up"][i] = (s[:z].count("G") + s[:z].count("C")) / z
    X["pas"][i] = any(h in s[z - 35:z - 4] for h in ("AAUAAA", "AUUAAA"))
for c in X:
    v = X[c]; X[c] = np.where(np.isnan(v), np.nanmedian(v), v)
X["acc_x_interv_donor"] = X["acc"] * X["interv_donor"]; X["sa_acc_x_don"] = X["sa_acc_max"] * X["sa_don_sum"]
SETS = [
    ("splice model as in the chapter: MaxEnt acceptor + U1 upstream", ["acc", "u1_dw_up"]),
    ("+ terminal-exon definition (intervening donor, distance to next donor, U1 downstream, interaction)",
     ["acc", "u1_dw_up", "interv_donor", "edef_best5", "nearest_dn", "u1_dw_dn", "acc_x_interv_donor"]),
    ("SpliceAI acceptor + donor + interaction", ["sa_acc_max", "sa_don_sum", "sa_acc_x_don"]),
    ("MaxEnt terminal-exon set + SpliceAI", ["acc", "u1_dw_up", "interv_donor", "edef_best5", "nearest_dn", "u1_dw_dn", "acc_x_interv_donor", "sa_acc_max", "sa_don_sum", "sa_acc_x_don"]),
    ("local cleavage features only (PAS, UGUA, U and GU downstream, GC upstream)", ["pas", "ugua", "u_dn", "gu_dn", "gc_up"]),
    ("MaxEnt terminal-exon set + SpliceAI + cleavage", ["acc", "u1_dw_up", "interv_donor", "edef_best5", "nearest_dn", "u1_dw_dn", "acc_x_interv_donor", "sa_acc_max", "sa_don_sum", "sa_acc_x_don", "pas", "ugua", "u_dn", "gu_dn", "gc_up"]),
    ("reference: annotated exon upstream 500 nt (annotation-derived)", ["exon_upstream_500"]),
]
gid = collections.defaultdict(list)
for i, gg in enumerate(g): gid[gg].append(i)
genes = [np.array(v) for v in gid.values() if (y[v] == 1).sum() == 1 and len(v) > 1]
fold = np.random.default_rng(7).integers(0, 5, len(genes))
def fit(M, gl, lam=1e-2, it=30):
    b = np.zeros(M.shape[1])
    for _ in range(it):
        gr = -lam * b; H = -lam * np.eye(len(b))
        for ii in gl:
            Z = M[ii]; e = Z @ b; e -= e.max(); p = np.exp(e); p /= p.sum(); r = ii[y[ii] == 1][0]
            mu = p @ Z; gr += M[r] - mu; H -= (Z * p[:, None]).T @ Z - np.outer(mu, mu)
        step = np.linalg.solve(H, gr); b -= step
        if np.abs(step).max() < 1e-6: break
    return b
def p1(score, gl): return np.array([float(score[ii][y[ii] == 1][0] > score[ii][y[ii] == 0].max()) for ii in gl])
rng = np.random.default_rng(1)
def ci(v): return np.percentile([v[rng.integers(0, len(v), len(v))].mean() for _ in range(1000)], [2.5, 97.5])
OUT = []; net = p1(logit, genes); chance = np.mean([1 / len(ii) for ii in genes])
lo, hi = ci(net); OUT.append(("network (LOCO)", "", f"{net.mean():.3f}", f"[{lo:.3f}, {hi:.3f}]", ""))
log(f"genes {len(genes)}  chance {chance:.3f}  network {net.mean():.3f} [{lo:.3f},{hi:.3f}]")
for name, cols in SETS:
    M = np.column_stack([X[c] for c in cols]).astype(float); score = np.zeros(N); coefs = []
    for f in range(5):
        trg = [ii for ii, k in zip(genes, fold) if k != f]; teg = [ii for ii, k in zip(genes, fold) if k == f]
        tri = np.concatenate(trg); mu, sd = M[tri].mean(0), M[tri].std(0) + 1e-9; Mz = (M - mu) / sd
        b = fit(Mz, trg); coefs.append(b)
        for ii in teg: score[ii] = Mz[ii] @ b
    v = p1(score, genes); lo, hi = ci(v); b = np.mean(coefs, 0)
    cs = ", ".join(f"{c} {x:+.2f}" for c, x in zip(cols, b))
    OUT.append((name, ";".join(cols), f"{v.mean():.3f}", f"[{lo:.3f}, {hi:.3f}]", cs))
    log(f"  P@1 {v.mean():.3f} [{lo:.3f},{hi:.3f}]  {name}\n      {cs}")
with open(os.path.join(ROOT, "data/splice_model_expand.tsv"), "w") as fh:
    fh.write(f"# genes {len(genes)}; chance {chance:.3f}\nmodel\tfeatures\tp_at_1\tci95\tcoefficients_per_sd\n")
    for r in OUT: fh.write("\t".join(r) + "\n")
log("[out] data/splice_model_expand.tsv")
