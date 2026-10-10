#!/usr/bin/env python3
# protein occupancy and u1-dependent usage at real 3' ends, competitors and intronic decoys
import os, sys, math, collections, numpy as np, pysam
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); PUB = os.path.join(ROOT, "..")
def log(*a): print(*a, file=sys.stderr, flush=True)
Z = np.load(os.path.join(ROOT, "data/occupancy_profiles.npz"), allow_pickle=True); T = list(Z["tracks"]); bs = Z["bin_start"]
D = np.load(os.path.join(PUB, "cnn_data.npz"), allow_pickle=True); y = D["y"].astype(int); g = D["gene"]; st = D["strand"]; ch = D["chrom"]; cen = D["center"].astype(np.int64)
S = np.where((st == "+")[None, :, None], Z["prof"][:, 0], Z["prof"][:, 1]).astype(float)
Zd = np.load(os.path.join(ROOT, "data/occupancy_profiles_decoy.npz"), allow_pickle=True); gd = Zd["gene"]
Sd = np.where((Zd["strand"] == "+")[None, :, None], Zd["prof"][:, 0], Zd["prof"][:, 1]).astype(float)
ti = {n: i for i, n in enumerate(T)}
genes = sorted(set(g)); gix = {x: k for k, x in enumerate(genes)}; NG = len(genes)
CL = ("real 3' end", "competitor", "decoy")
def gene_mean(X, gg, mask=None):
    out = np.zeros((NG, X.shape[1])); n = np.zeros(NG)
    k = np.array([gix[x] for x in gg])
    if mask is not None: X, k = X[mask], k[mask]
    np.add.at(out, k, X); np.add.at(n, k, 1)
    out[n == 0] = np.nan; return out / np.maximum(n, 1)[:, None]
GM = {}
for t in T:
    GM[t] = {"real 3' end": gene_mean(S[ti[t]], g, y == 1), "competitor": gene_mean(S[ti[t]], g, y == 0), "decoy": gene_mean(Sd[ti[t]], gd)}
W = lambda a, b: (bs >= a) & (bs < b)
expr = {c: np.nan_to_num(GM["input"][c]).mean(1) > 0 for c in CL}
RNG = np.random.default_rng(0); NBOOT = 2000
def boot(fn, keep):
    k = np.flatnonzero(keep); est = fn(k); b = np.array([fn(k[RNG.integers(0, len(k), len(k))]) for _ in range(NBOOT)])
    b = b[np.isfinite(b)]; lo, hi = np.percentile(b, [2.5, 97.5]); p = 2 * min((b <= 0).mean(), (b >= 0).mean()); return est, lo, hi, max(p, 1 / NBOOT), len(k)
OUT = []
def out(*a): OUT.append(a); log("  ".join(str(x) for x in a))
fmt = lambda e, lo, hi, p, n: (f"{e:+.3f}", f"[{lo:+.3f}, {hi:+.3f}]", f"{p:.3g}", n)
# a metagene
with open(os.path.join(ROOT, "data/occupancy_metagene.tsv"), "w") as fh:
    fh.write("track\tclass\tbin_start\tmean_signal\tgenes\n")
    keep = expr["real 3' end"] & expr["competitor"] & expr["decoy"]
    for t in T:
        for c in CL:
            m = GM[t][c][keep].mean(0)
            for b_, v in zip(bs, m): fh.write(f"{t}\t{c}\t{b_}\t{v:.6g}\t{int(keep.sum())}\n")
log(f"[A] metagene over {int(keep.sum())} genes with input signal in all three classes")
# b site over flank
log("\n[B] log2 (site -100..+100 / flank -1000..-500), ratio of sums over genes")
site, flank = W(-100, 100), W(-1000, -500)
for t in T:
    if t.startswith("polyA"): continue
    for c in CL:
        A, B = GM[t][c][:, site].mean(1), GM[t][c][:, flank].mean(1)
        e = boot(lambda k: np.log2(A[k].sum() / B[k].sum()), expr[c]); out("site_over_flank", t, c, *fmt(*e))
    A1, B1 = GM[t]["competitor"][:, site].mean(1), GM[t]["competitor"][:, flank].mean(1)
    A2, B2 = GM[t]["decoy"][:, site].mean(1), GM[t]["decoy"][:, flank].mean(1)
    e = boot(lambda k: np.log2(A1[k].sum() / B1[k].sum()) - np.log2(A2[k].sum() / B2[k].sum()), expr["competitor"] & expr["decoy"])
    out("site_over_flank", t, "competitor minus decoy", *fmt(*e))
# c u1 amo
log("\n[C] poly(A)+ RNA, log2 (upstream -200..0 / downstream 0..+200)")
up, dn = W(-200, 0), W(0, 200)
def step(c, cond, k): return np.log2(GM[cond][c][k][:, up].sum() / GM[cond][c][k][:, dn].sum())
for c in CL:
    ok = expr[c] & (np.nan_to_num(GM["polyA_cAMO"][c]).sum(1) > 0) & (np.nan_to_num(GM["polyA_U1AMO"][c]).sum(1) > 0)
    for cond in ("polyA_cAMO", "polyA_U1AMO"):
        out("u1amo_step", cond, c, *fmt(*boot(lambda k: step(c, cond, k), ok)))
    out("u1amo_step", "change", c, *fmt(*boot(lambda k: step(c, "polyA_U1AMO", k) - step(c, "polyA_cAMO", k), ok)))
ok = np.ones(NG, bool)
for c in ("competitor", "decoy"): ok &= expr[c] & (np.nan_to_num(GM["polyA_cAMO"][c]).sum(1) > 0) & (np.nan_to_num(GM["polyA_U1AMO"][c]).sum(1) > 0)
dch = lambda c, k: step(c, "polyA_U1AMO", k) - step(c, "polyA_cAMO", k)
out("u1amo_step", "change", "competitor minus decoy", *fmt(*boot(lambda k: dch("competitor", k) - dch("decoy", k), ok)))
for c in ("competitor", "decoy"):
    ok = expr["real 3' end"] & expr[c]
    for cond in ("polyA_cAMO", "polyA_U1AMO"):
        out("u1amo_relative_upstream", cond, c + " over real 3' end", *fmt(*boot(lambda k: np.log2(GM[cond][c][k][:, up].sum() / GM[cond]["real 3' end"][k][:, up].sum()), ok)))
    out("u1amo_relative_upstream", "change", c + " over real 3' end",
        *fmt(*boot(lambda k: np.log2(GM["polyA_U1AMO"][c][k][:, up].sum() / GM["polyA_U1AMO"]["real 3' end"][k][:, up].sum())
                   - np.log2(GM["polyA_cAMO"][c][k][:, up].sum() / GM["polyA_cAMO"]["real 3' end"][k][:, up].sum()), ok)))
key = lambda a, b, c, d: (str(a), str(b), int(float(c)), str(d)); dk = [key(*t) for t in zip(g, ch, cen, st)]
U = {}
with open(os.path.join(PUB, "fig3_ucube_meta.tsv")) as fh:
    h = fh.readline().rstrip("\n").split("\t"); ix = {k: j for j, k in enumerate(h)}
    for l in fh:
        f = l.rstrip("\n").split("\t"); U[key(f[0], f[ix["chrom"]], f[ix["center"]], f[ix["strand"]])] = (float(f[ix["u1_dw_up"]]), float(f[ix["acc"]]))
u1 = np.array([U[k][0] for k in dk]); acc = np.array([U[k][1] for k in dk])
logit = np.load(os.path.join(PUB, "cnn_loco_scores.npz"), allow_pickle=True)["logit"].astype(float)
gl = collections.defaultdict(list)
for i, x in enumerate(g): gl[x].append(i)
def centre(v):
    o = np.full(len(v), np.nan)
    for ii in gl.values():
        ii = np.array(ii); c = ii[y[ii] == 0]; o[c] = v[c] - v[c].mean()
    return o
comp = y == 0
PRED = {"U1 sites upstream (sequence)": u1, "network logit": logit, "U1C occupancy, -500 to 0": S[ti["U1C"]][:, W(-500, 0)].mean(1)}
for name, v in PRED.items():
    vc = centre(v); multi = np.array([len(gl[x]) > 3 for x in g]); m0 = comp & multi
    q = np.quantile(vc[m0], [1 / 3, 2 / 3]); lab = np.where(vc <= q[0], 0, np.where(vc <= q[1], 1, 2))
    ch_ = []
    for k_, tn in enumerate(("low", "middle", "high")):
        gm = {cond: gene_mean(S[ti[cond]], g, m0 & (lab == k_)) for cond in ("polyA_cAMO", "polyA_U1AMO", "input")}
        ok = (np.nan_to_num(gm["input"]).mean(1) > 0) & (np.nan_to_num(gm["polyA_cAMO"]).sum(1) > 0) & (np.nan_to_num(gm["polyA_U1AMO"]).sum(1) > 0)
        f = lambda k: (np.log2(gm["polyA_U1AMO"][k][:, up].sum() / gm["polyA_U1AMO"][k][:, dn].sum()) - np.log2(gm["polyA_cAMO"][k][:, up].sum() / gm["polyA_cAMO"][k][:, dn].sum()))
        out("u1amo_tertile", name, tn, *fmt(*boot(f, ok)))
# d sequence features against occupancy
log("\n[D] within-gene Spearman among competitors with input signal")
fa = pysam.FastaFile(os.path.join(PUB, "../train/data/hg38.fa")); tr = str.maketrans("ACGTN", "TGCAN")
ugua = np.zeros(len(y)); udn = np.zeros(len(y)); gudn = np.zeros(len(y))
for i in np.flatnonzero(comp):
    p = int(cen[i]); c_ = str(ch[i])
    s = fa.fetch(c_, p - 100, p + 101).upper() if st[i] == "+" else fa.fetch(c_, p - 100, p + 101).upper().translate(tr)[::-1]
    ugua[i] = sum(s[k:k + 4] == "TGTA" for k in range(0, 97)); d = s[101:201]
    udn[i] = d.count("T") / max(1, len(d)); gudn[i] = sum(d[k:k + 2] == "GT" for k in range(len(d) - 1))
def rank(v):
    o = np.argsort(v, kind="mergesort"); r = np.empty(len(v)); r[o] = np.arange(len(v)); u, inv = np.unique(v, return_inverse=True)
    s = np.zeros(len(u)); n = np.zeros(len(u)); np.add.at(s, inv, r); np.add.at(n, inv, 1); return (s / n)[inv]
inp_site = S[ti["input"]].mean(1) > 0
PAIRS = [("U1 sites upstream (sequence)", u1, "U1A", (-500, 0)), ("U1 sites upstream (sequence)", u1, "U1C", (-500, 0)),
         ("UGUA, -100 to 0", ugua, "CFIm25", (-100, 0)), ("U fraction, +1 to +100", udn, "CstF64", (0, 100)), ("GU count, +1 to +100", gudn, "CstF64", (0, 100)),
         ("U fraction, +1 to +100", udn, "hnRNPC", (0, 100)), ("upstream acceptor score3", acc, "SF3B1", (-1000, 0))]
for name, v, t, (a, b) in PAIRS:
    m = comp & inp_site; x = S[ti[t]][:, W(a, b)].mean(1); X = centre(np.where(m, v, np.nan)); Y = np.full(len(y), np.nan)
    for ii in gl.values():
        ii = np.array(ii); c = ii[m[ii]]
        if len(c) > 1: Y[c] = x[c] - x[c].mean(); X[c] = v[c] - v[c].mean()
        else: X[c] = np.nan
    k = np.isfinite(X) & np.isfinite(Y); rho = np.corrcoef(rank(X[k]), rank(Y[k]))[0, 1]; n = int(k.sum())
    z = rho * math.sqrt(n - 1); p = math.erfc(abs(z) / math.sqrt(2))
    out("concordance", name, f"{t} occupancy, {a} to {b}", f"{rho:+.3f}", "", f"{p:.3g}", n)
with open(os.path.join(ROOT, "data/occupancy_stats.tsv"), "w") as fh:
    fh.write("analysis\ttrack_or_feature\tclass_or_target\testimate\tci95\tP\tn\n")
    for r in OUT: fh.write("\t".join(str(x) for x in r) + "\n")
log("[out] data/occupancy_metagene.tsv data/occupancy_stats.tsv")
