#!/usr/bin/env python3
# are competitor sites used when u1 snrnp is inhibited
import os, sys, math, collections, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); PUB = os.path.join(ROOT, "..")
def log(*a): print(*a, file=sys.stderr, flush=True)
src = open(os.path.join(HERE, "stats_ch3.py")).read(); cut = src.index("# ---------------------------------------------------------------- data")
ns = {"__file__": os.path.join(HERE, "stats_ch3.py")}; exec(src[:cut], ns); wilcoxon = ns["wilcoxon"]
Z = np.load(os.path.join(ROOT, "data/occupancy_profiles.npz"), allow_pickle=True); T = list(Z["tracks"]); bs = Z["bin_start"]; ti = {n: i for i, n in enumerate(T)}
D = np.load(os.path.join(PUB, "cnn_data.npz"), allow_pickle=True); y = D["y"].astype(int); g = D["gene"]; st = D["strand"]; ch = D["chrom"]; cen = D["center"].astype(np.int64)
Zd = np.load(os.path.join(ROOT, "data/occupancy_profiles_decoy.npz"), allow_pickle=True); gd = Zd["gene"]
sense = lambda P, s, t: np.where((s == "+")[:, None], P[ti[t], 0], P[ti[t], 1]).astype(float)
A = {"control": sense(Z["prof"], st, "polyA_cAMO"), "U1 AMO": sense(Z["prof"], st, "polyA_U1AMO")}
Ad = {"control": sense(Zd["prof"], Zd["strand"], "polyA_cAMO"), "U1 AMO": sense(Zd["prof"], Zd["strand"], "polyA_U1AMO")}
genes = sorted(set(g)); gix = {x: k for k, x in enumerate(genes)}; NG = len(genes)
kg = np.array([gix[x] for x in g]); kd = np.array([gix[x] for x in gd])
def gene_sum(X, k, mask=None):
    out = np.zeros((NG, X.shape[1])); n = np.zeros(NG)
    if mask is not None: X, k = X[mask], k[mask]
    np.add.at(out, k, X); np.add.at(n, k, 1); return out / np.maximum(n, 1)[:, None], n > 0
W = lambda a, b: (bs >= a) & (bs < b)
RNG = np.random.default_rng(0); NB = 2000
def boot(fn, keep):
    k = np.flatnonzero(keep); est = fn(k); b = np.array([fn(k[RNG.integers(0, len(k), len(k))]) for _ in range(NB)]); b = b[np.isfinite(b)]
    lo, hi = np.percentile(b, [2.5, 97.5]); return est, lo, hi, max(2 * min((b <= 0).mean(), (b >= 0).mean()), 1 / NB), len(k)
OUT = []
def out(*a): OUT.append(a); log("  ".join(str(x) for x in a))
CLS = {"real 3' end": (A, kg, y == 1), "competitor": (A, kg, y == 0), "decoy": (Ad, kd, None)}
GS = {c: {cond: gene_sum(src_[cond], k, m) for cond in ("control", "U1 AMO")} for c, (src_, k, m) in CLS.items()}
def step(M, k, w): return np.log2(M[k][:, W(-w, 0)].sum() / M[k][:, W(0, w)].sum())
for w in (100, 200, 500):
    log(f"\n== window {w} nt")
    ok = {}
    for c in CLS:
        M0, has = GS[c]["control"]; M1, _ = GS[c]["U1 AMO"]
        ok[c] = has & (M0[:, W(-w, w)].sum(1) > 0) & (M1[:, W(-w, w)].sum(1) > 0)
        for cond, M in (("control", M0), ("U1 AMO", M1)):
            e = boot(lambda k: step(M, k, w), ok[c]); out(w, "pooled step", c, cond, f"{e[0]:+.3f}", f"[{e[1]:+.3f}, {e[2]:+.3f}]", f"{e[3]:.3g}", e[4])
        e = boot(lambda k: step(M1, k, w) - step(M0, k, w), ok[c]); out(w, "pooled derepression", c, "U1 AMO minus control", f"{e[0]:+.3f}", f"[{e[1]:+.3f}, {e[2]:+.3f}]", f"{e[3]:.3g}", e[4])
        idx = lambda M: (M[:, W(-w, 0)].sum(1) - M[:, W(0, w)].sum(1)) / np.maximum(M[:, W(-w, w)].sum(1), 1e-12)
        d = (idx(M1) - idx(M0))[ok[c]]; z, p, n = wilcoxon(d)
        out(w, "per-gene index change", c, "U1 AMO minus control", f"{np.mean(d):+.3f}", f"median {np.median(d):+.3f}; increased in {np.mean(d > 0):.3f}", f"z {z:.2f}, r {abs(z)/math.sqrt(max(n,1)):.3f}, P {p:.2g}", n)
    both = ok["competitor"] & ok["decoy"]
    f = lambda c, k: step(GS[c]["U1 AMO"][0], k, w) - step(GS[c]["control"][0], k, w)
    e = boot(lambda k: f("competitor", k) - f("decoy", k), both); out(w, "pooled derepression", "competitor minus decoy", "", f"{e[0]:+.3f}", f"[{e[1]:+.3f}, {e[2]:+.3f}]", f"{e[3]:.3g}", e[4])
    idx = lambda M: (M[:, W(-w, 0)].sum(1) - M[:, W(0, w)].sum(1)) / np.maximum(M[:, W(-w, w)].sum(1), 1e-12)
    dc = idx(GS["competitor"]["U1 AMO"][0]) - idx(GS["competitor"]["control"][0]); dd = idx(GS["decoy"]["U1 AMO"][0]) - idx(GS["decoy"]["control"][0])
    d = (dc - dd)[both]; z, p, n = wilcoxon(d)
    out(w, "per-gene index change", "competitor minus decoy", "", f"{np.mean(d):+.3f}", f"median {np.median(d):+.3f}; positive in {np.mean(d > 0):.3f}", f"z {z:.2f}, r {abs(z)/math.sqrt(max(n,1)):.3f}, P {p:.2g}", n)
# by score
key = lambda a, b, c, d: (str(a), str(b), int(float(c)), str(d)); dk = [key(*t) for t in zip(g, ch, cen, st)]
U = {}
with open(os.path.join(PUB, "fig3_ucube_meta.tsv")) as fh:
    h = fh.readline().rstrip("\n").split("\t"); ix = {k: j for j, k in enumerate(h)}
    for l in fh:
        f_ = l.rstrip("\n").split("\t"); U[key(f_[0], f_[ix["chrom"]], f_[ix["center"]], f_[ix["strand"]])] = (float(f_[ix["u1_dw_up"]]), float(f_[ix["acc"]]))
SP = np.load(os.path.join(PUB, "splice26_loco_scores.npz"), allow_pickle=True)
SCORE = {"sequence network": np.load(os.path.join(PUB, "cnn_loco_scores.npz"), allow_pickle=True)["logit"].astype(float),
         "splice-feature model": None, "U1 sites upstream": np.array([U[k][0] for k in dk]), "upstream acceptor score3": np.array([U[k][1] for k in dk]),
         "U1C occupancy, -500 to 0": sense(Z["prof"], st, "U1C")[:, W(-500, 0)].mean(1)}
skey = [k for k in SP.files if SP[k].shape == (len(y),) and SP[k].dtype.kind == "f"]
log("splice score arrays:", skey, "gene order matches" if "gene" in SP.files and (SP["gene"] == g).all() else "CHECK ORDER")
SCORE["splice-feature model"] = SP[skey[0]].astype(float)
gl = collections.defaultdict(list)
for i, x in enumerate(g): gl[x].append(i)
w = 200
for name, v in SCORE.items():
    lab = np.full(len(y), -1)
    for ii in gl.values():
        ii = np.array(ii); c = ii[y[ii] == 0]
        if len(c) < 3: continue
        r = np.argsort(np.argsort(v[c], kind="mergesort"), kind="mergesort") / len(c); lab[c] = np.minimum((r * 3).astype(int), 2)
    G = {t: {cond: gene_sum(A[cond], kg, lab == t)[0] for cond in ("control", "U1 AMO")} for t in range(3)}
    okt = {t: (G[t]["control"][:, W(-w, w)].sum(1) > 0) & (G[t]["U1 AMO"][:, W(-w, w)].sum(1) > 0) for t in range(3)}
    f = lambda t, k: step(G[t]["U1 AMO"], k, w) - step(G[t]["control"], k, w)
    for t, tn in enumerate(("low", "middle", "high")):
        e = boot(lambda k: f(t, k), okt[t]); out(w, "derepression by score", name, tn, f"{e[0]:+.3f}", f"[{e[1]:+.3f}, {e[2]:+.3f}]", f"{e[3]:.3g}", e[4])
    e = boot(lambda k: f(2, k) - f(0, k), okt[0] & okt[2]); out(w, "derepression by score", name, "high minus low", f"{e[0]:+.3f}", f"[{e[1]:+.3f}, {e[2]:+.3f}]", f"{e[3]:.3g}", e[4])
with open(os.path.join(ROOT, "data/u1amo_derepression.tsv"), "w") as fh:
    fh.write("window\tanalysis\tclass_or_score\tcondition_or_tertile\testimate\tinterval\tP\tgenes\n")
    for r in OUT: fh.write("\t".join(str(x) for x in r) + "\n")
with open(os.path.join(ROOT, "data/u1amo_metagene.tsv"), "w") as fh:
    fh.write("class\tcondition\tbin_start\tmean_fraction_of_gene_signal\tgenes\n")
    for c in CLS:
        M0, has = GS[c]["control"]; M1, _ = GS[c]["U1 AMO"]; ok = has & (M0.sum(1) > 0) & (M1.sum(1) > 0)
        for cond, M in (("control", M0), ("U1 AMO", M1)):
            p = (M[ok] / M[ok].sum(1, keepdims=True)).mean(0)
            for b_, v_ in zip(bs, p): fh.write(f"{c}\t{cond}\t{b_}\t{v_:.5g}\t{int(ok.sum())}\n")
log("[out] data/u1amo_derepression.tsv data/u1amo_metagene.tsv")
