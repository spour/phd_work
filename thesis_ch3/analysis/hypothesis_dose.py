#!/usr/bin/env python3
# test of the hypothesis that a stronger upstream 3' splice site makes a site more like
import os, sys, math, collections, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); PUB = os.path.join(ROOT, "..")
sys.path.insert(0, "/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/Compiler/gcccore/arrow/17.0.0/lib/python3.11/site-packages")
import pyarrow.parquet as pq
def log(*a): print(*a, file=sys.stderr, flush=True)
src = open(os.path.join(HERE, "stats_ch3.py")).read(); cut = src.index("# ---------------------------------------------------------------- data")
ns = {"__file__": os.path.join(HERE, "stats_ch3.py")}; exec(src[:cut], ns); wilcoxon = ns["wilcoxon"]
key = lambda a, b, c, d: (str(a), str(b), int(float(c)), str(d))
M = {}
with open(os.path.join(PUB, "fig3_ucube_meta.tsv")) as fh:
    h = fh.readline().rstrip("\n").split("\t"); ix = {k: j for j, k in enumerate(h)}
    for l in fh:
        f = l.rstrip("\n").split("\t"); g = lambda c: float(f[ix[c]]) if f[ix[c]] not in ("", "nan", "NA") else np.nan
        M[key(f[0], f[ix["chrom"]], f[ix["center"]], f[ix["strand"]])] = (g("acc"), g("interv_donor"))
T = pq.read_table(os.path.join(PUB, "cpa_peaks_final.parquet")).to_pydict()
K = [key(*t) for t in zip(T["gene"], T["chrom"], T["center"], T["strand"])]; N = len(K)
gene = np.array(T["gene"]); can = np.array(T["is_canonical"]); apa = np.array(T["flag_apa_annotated"]); te = np.array(T["flag_reference_terminal_exon"])
dn = np.array(T["flag_downstream_of_canonical"]); ana = np.array(T["in_analysis_set"])
acc = np.array([M[k][0] for k in K]); don = np.array([M[k][1] for k in K])
log(f"missing: 3'ss {np.isnan(acc).mean():.3f}, 5'ss {np.isnan(don).mean():.3f}")
don = np.where(np.isnan(don), np.nanmin(don), don); acc = np.where(np.isnan(acc), np.nanmin(acc), acc)
gi = collections.defaultdict(list)
for i, g_ in enumerate(gene): gi[g_].append(i)
OUT = []; DOSE = []
def out(*a): OUT.append(a); log("  ".join(str(x) for x in a))
RNG = np.random.default_rng(0)
# 1 dose
GA = [np.array([i for i in v if ana[i]]) for v in gi.values()]; GA = [ii for ii in GA if len(ii) >= 2 and can[ii].sum() == 1]
def within_rank(v):
    r = np.full(N, np.nan)
    for ii in GA:
        o = np.argsort(np.argsort(v[ii] + RNG.normal(0, 1e-9, len(ii)))); r[ii] = (o + 0.5) / len(ii)
    return r
ra, rd = within_rank(acc), within_rank(don); gof = np.full(N, -1)
for k, ii in enumerate(GA): gof[ii] = k
expct = np.zeros(N)
for ii in GA: expct[ii] = 1 / len(ii)
def dose(r, mask, name, sub):
    for q in range(5):
        m = mask & (r >= q / 5) & (r < (q + 1) / 5) if q < 4 else mask & (r >= 0.8)
        m &= gof >= 0
        def f(gs):
            sel = np.isin(gof, gs) & m; return can[sel].sum() / expct[sel].sum()
        e = can[m].sum() / expct[m].sum(); gs0 = np.arange(len(GA)); b = []
        for _ in range(500):
            gs = RNG.integers(0, len(GA), len(GA)); w = np.bincount(gs, minlength=len(GA))[gof[m]]; b.append((can[m] * w).sum() / (expct[m] * w).sum())
        lo, hi = np.percentile(b, [2.5, 97.5]); DOSE.append((name, sub, q + 1, int(m.sum()), int(can[m].sum()), f"{e:.3f}", f"{lo:.3f}", f"{hi:.3f}"))
        log(f"  {name:20s} {sub:22s} fifth {q+1}  n {int(m.sum()):6d}  real/expected {e:.2f} [{lo:.2f},{hi:.2f}]")
allm = np.ones(N, bool)
dose(ra, allm, "3' splice site score", "all"); dose(rd, allm, "5' splice site score", "all")
dose(ra, rd < 1 / 3, "3' splice site score", "weak 5' splice site"); dose(ra, rd >= 2 / 3, "3' splice site score", "strong 5' splice site")
# 2 conditional logit
idx = np.concatenate(GA); mu_a, sd_a = acc[idx].mean(), acc[idx].std(); mu_d, sd_d = don[idx].mean(), don[idx].std()
za = (acc - mu_a) / sd_a; zd = (don - mu_d) / sd_d
def clogit(X):
    b = np.zeros(X.shape[1])
    for _ in range(30):
        gr = np.zeros_like(b); H = np.zeros((len(b), len(b)))
        for ii in GA:
            Z = X[ii]; e = Z @ b; e -= e.max(); p = np.exp(e); p /= p.sum(); mu = p @ Z
            gr += X[ii[can[ii]][0]] - mu; H += (Z * p[:, None]).T @ Z - np.outer(mu, mu)
        step = np.linalg.solve(H, gr); b += step
        if np.abs(step).max() < 1e-8: break
    se = np.sqrt(np.diag(np.linalg.inv(H))); return b, se
for name, X, labs in (("3' splice site alone", np.column_stack([za]), ["3' splice site"]), ("5' splice site alone", np.column_stack([zd]), ["5' splice site"]),
                      ("joint", np.column_stack([za, zd]), ["3' splice site", "5' splice site"]), ("joint with product", np.column_stack([za, zd, za * zd]), ["3' splice site", "5' splice site", "product"])):
    b, se = clogit(X)
    for l_, bb, ss in zip(labs, b, se):
        z = bb / ss; out("model", name, l_, f"OR per SD {math.exp(bb):.3f}", f"[{math.exp(bb-1.96*ss):.3f}, {math.exp(bb+1.96*ss):.3f}]", f"z {z:.1f}", f"P {math.erfc(abs(z)/math.sqrt(2)):.2g}", len(GA))
# 3 usage gradient
cls = np.full(N, "", dtype=object); cls[can] = "real 3' end"; cls[~can & ana] = "unannotated competitor"; cls[~can & apa & ~dn & ~te] = "annotated poly(A) site"
for fname, v in (("3' splice site score", acc), ("5' splice site score", don)):
    for c in ("unannotated competitor", "annotated poly(A) site", "real 3' end"): out("usage", fname, c, "median", f"{np.median(v[cls == c]):.2f}", "n", int((cls == c).sum()), "")
    for a_, b_ in (("annotated poly(A) site", "unannotated competitor"), ("real 3' end", "annotated poly(A) site"), ("real 3' end", "unannotated competitor")):
        d = []
        for ii in gi.values():
            ii = np.array(ii); x = v[ii][cls[ii] == a_]; y_ = v[ii][cls[ii] == b_]
            if len(x) and len(y_): d.append(x.mean() - y_.mean())
        d = np.array(d); z, p, n = wilcoxon(d)
        out("usage", fname, f"{a_} minus {b_}", f"median {np.median(d):+.3f}", f"higher in {np.mean(d > 0):.3f}", f"z {z:.1f}, r {abs(z)/math.sqrt(max(n,1)):.3f}", f"P {p:.2g}", len(d))
D = np.load(os.path.join(PUB, "cnn_data.npz"), allow_pickle=True); R = np.load(os.path.join(ROOT, "data/u1_replication_per_site.npz"), allow_pickle=True); O = np.load(os.path.join(ROOT, "data/oh2017_per_site.npz"), allow_pickle=True)
nD = len(D["gene"]); ctl = [k for k, f in enumerate(R["files"]) if any(t in f for t in ("Control-", "293-ctr", "SW480-NC", "HeLa-ctr", "_c25_", "_c50_", "_dmso_"))]
used = (np.nan_to_num(R["signal"][ctl][:, :nD]) > 0).any(0) | (np.nan_to_num(O["control"][:nD]) > 0)
row = {k: i for i, k in enumerate(K)}; j = np.array([row[key(*t)] for t in zip(D["gene"], D["chrom"], D["center"], D["strand"])]); comp = D["y"].astype(int) == 0
log(f"analysis-set competitors with 3' end reads in control cells: {int((used & comp).sum())} of {int(comp.sum())} ({(used & comp).sum()/comp.sum():.4f}); control libraries {len(ctl) + 1}")
gD = collections.defaultdict(list)
for i, g_ in enumerate(D["gene"]): gD[g_].append(i)
for fname, v in (("3' splice site score", acc[j]), ("5' splice site score", don[j])):
    d = []
    for ii in gD.values():
        ii = np.array(ii); u = ii[comp[ii] & used[ii]]; n_ = ii[comp[ii] & ~used[ii]]
        if len(u) and len(n_): d.append(v[u].mean() - v[n_].mean())
    d = np.array(d); z, p, n = wilcoxon(d)
    out("usage", fname, "competitors with reads minus competitors without", f"median {np.median(d):+.3f}", f"higher in {np.mean(d > 0):.3f}", f"z {z:.1f}, r {abs(z)/math.sqrt(max(n,1)):.3f}", f"P {p:.2g}", len(d))
with open(os.path.join(ROOT, "data/hypothesis_dose.tsv"), "w") as fh:
    fh.write("score\tsubset\tfifth\tcandidates\treal\treal_over_expected\tlo\thi\n")
    for r in DOSE: fh.write("\t".join(str(x) for x in r) + "\n")
with open(os.path.join(ROOT, "data/hypothesis_stats.tsv"), "w") as fh:
    for r in OUT: fh.write("\t".join(str(x) for x in r) + "\n")
log("[out] data/hypothesis_dose.tsv data/hypothesis_stats.tsv")
