#!/usr/bin/env python3
# significance tests and effect sizes for every comparison in chapter 3, in one table
import os, sys, csv, math, collections, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); PUB = os.path.join(ROOT, "..")
def log(*a): print(*a, file=sys.stderr, flush=True)
RNG = np.random.default_rng(1); NB = 2000
ROWS = []
def out(i, what, n, eff, ci, stat, p):
    ROWS.append((i, what, n, eff, ci, stat, p)); log(f"{i:4s} {what:70s} n={n}  {eff}  {ci}  {stat}  P={p}")
def fp(p):
    if p == 0 or p < 1e-300: return "< 1e-300"
    return f"{p:.2g}"
# distributions
def norm_sf(z): return 0.5 * math.erfc(z / math.sqrt(2))
def p2(z): return min(1.0, 2 * norm_sf(abs(z)))
def betacf(a, b, x):
    qab, qap, qam = a + b, a + 1, a - 1; c, d = 1.0, 1 - qab * x / qap; d = 1 / (d if abs(d) > 1e-30 else 1e-30); h = d
    for m in range(1, 300):
        m2 = 2 * m; aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d; d = 1 / (d if abs(d) > 1e-30 else 1e-30); c = 1 + aa / c; c = c if abs(c) > 1e-30 else 1e-30; h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d; d = 1 / (d if abs(d) > 1e-30 else 1e-30); c = 1 + aa / c; c = c if abs(c) > 1e-30 else 1e-30; de = d * c; h *= de
        if abs(de - 1) < 3e-14: break
    return h
def betai(a, b, x):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    lb = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    return math.exp(lb) * betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1 - math.exp(lb) * betacf(b, a, 1 - x) / b
def t_p2(t, df): return betai(df / 2, 0.5, df / (df + t * t))
def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    t = (a.mean() - b.mean()) / math.sqrt(va + vb); df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return a.mean() - b.mean(), t, df, t_p2(t, df)
def tq975(df):
    lo, hi = 0.0, 50.0
    for _ in range(80):
        m = (lo + hi) / 2
        if t_p2(m, df) > 0.05: lo = m
        else: hi = m
    return (lo + hi) / 2
def wilcoxon(d):
    d = np.asarray(d, float); d = d[d != 0]; n = len(d)
    a = np.abs(d); o = np.argsort(a, kind="mergesort"); r = np.empty(n); r[o] = np.arange(1, n + 1)
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); avg = (csum - cnt + 1 + csum) / 2; r = avg[inv]
    W = r[d > 0].sum(); mu = n * (n + 1) / 4
    var = n * (n + 1) * (2 * n + 1) / 24 - (cnt ** 3 - cnt).sum() / 48
    z = (W - mu) / math.sqrt(var); return z, p2(z), n
def mcnemar(b, c):
    chi = (abs(b - c) - 1) ** 2 / (b + c); z = math.sqrt(chi); return chi, p2(z)
def boot_genes(fn, groups, B=NB):
    k = len(groups); v = [fn([groups[j] for j in RNG.integers(0, k, k)]) for _ in range(B)]
    return np.percentile(v, [2.5, 97.5])
# data
D = np.load(os.path.join(PUB, "cnn_data.npz"), allow_pickle=True)
X = D["X"]; y = D["y"].astype(int); g = D["gene"]; split = D["split"]; cen = D["center"]
gidx = collections.defaultdict(list)
for i, gg in enumerate(g): gidx[gg].append(i)
gidx = {k: np.array(v) for k, v in gidx.items()}
GENES = list(gidx)
# a  splice environment
from maxentpy import maxent
from maxentpy.maxent import load_matrix5
M5 = load_matrix5(); NT = np.array(list("ACGTN"))
def n_donors(w):
    s = "".join(NT[np.minimum(w[1500:2000 + 8], 4)]); n = 0
    for j in range(0, 500):
        if s[j+3:j+5] == "GT" and "N" not in s[j:j+9] and maxent.score5(s[j:j+9], matrix=M5) >= 8: n += 1
    return n
dn = np.array([n_donors(X[i]) for i in range(len(y))], float) * 2.0
d1 = np.array([dn[ii][y[ii] == 1][0] - dn[ii][y[ii] == 0].mean() for ii in gidx.values()])
z, p, n = wilcoxon(d1)
out("A1", "strong donors per kb, upstream 500 nt: real minus mean competitor (within gene)", len(d1),
    f"mean {d1.mean():+.3f}; real lower in {np.mean(d1 < 0):.3f}, higher in {np.mean(d1 > 0):.3f}",
    f"mean CI [{np.percentile([RNG.choice(d1, len(d1)).mean() for _ in range(NB)], 2.5):+.3f}, {np.percentile([RNG.choice(d1, len(d1)).mean() for _ in range(NB)], 97.5):+.3f}]",
    f"Wilcoxon z = {z:.2f}, r = {abs(z)/math.sqrt(n):.3f} (n non-zero {n})", fp(p))
AP = list(csv.DictReader(open(os.path.join(PUB, "acceptor_position_scan.tsv")), delimiter="\t"))
acc = collections.defaultdict(lambda: ([], []))
for r in AP:
    if r["class"] == "real": acc[r["gene"]][0].append(float(r["max_me3"]))
    elif r["class"] == "competitor": acc[r["gene"]][1].append(float(r["max_me3"]))
d2 = np.array([np.mean(a) - np.mean(b) for a, b in acc.values() if a and b])
z, p, n = wilcoxon(d2)
out("A2", "strongest upstream acceptor score3: real minus mean competitor (within gene)", len(d2),
    f"median {np.median(d2):+.2f}; positive in {np.mean(d2 > 0):.3f}", "", f"Wilcoxon z = {z:.2f}, r = {abs(z)/math.sqrt(n):.3f}", fp(p))
pos = collections.defaultdict(list)
for r in AP:
    if float(r["max_me3"]) >= 6.5: pos[r["class"]].append(float(r["position"]) >= -1000)
pr, pc = np.mean(pos["real"]), np.mean(pos["competitor"]); nr, nc = len(pos["real"]), len(pos["competitor"])
pp = (pr * nr + pc * nc) / (nr + nc); z = (pr - pc) / math.sqrt(pp * (1 - pp) * (1 / nr + 1 / nc))
se = math.sqrt(pr * (1 - pr) / nr + pc * (1 - pc) / nc)
out("A3", "strongest acceptor within 1,000 nt of the site: real vs competitor", f"{nr}/{nc}",
    f"{pr:.3f} vs {pc:.3f}, difference {pr-pc:+.3f}", f"[{pr-pc-1.96*se:+.3f}, {pr-pc+1.96*se:+.3f}]", f"two-proportion z = {z:.2f}", fp(p2(z)))
W = {r[0]: r[1] for r in csv.reader(open(os.path.join(PUB, "splice_env_wilcoxon.tsv")), delimiter="\t")}
z = float(W["z"]); n = int(W["n_genes"])
out("A4", "U1 site density per kb: real minus mean competitor (within gene)", n, f"mean {float(W['mean_delta']):+.2f}, median {float(W['median_delta']):+.2f}; below zero in {float(W['frac_below_zero']):.3f}",
    "", f"Wilcoxon z = {z:.2f}, r = {abs(z)/math.sqrt(n):.3f}", W["p"])
# b, c  models
RR = list(csv.DictReader(open(os.path.join(PUB, "rescue_ranks.tsv")), delimiter="\t"))
hit = {k: np.array([int(r[f"rank_{k}"]) == 1 for r in RR]) for k in ("splice", "cnn", "combo")}
ncand = np.array([int(r["n_candidates"]) for r in RR])
def ci_mean(v):
    v = np.asarray(v, float); b = [v[RNG.integers(0, len(v), len(v))].mean() for _ in range(NB)]; return np.percentile(b, [2.5, 97.5])
for k, lab in (("splice", "splice model"), ("cnn", "sequence network"), ("combo", "combined ranking")):
    lo, hi = ci_mean(hit[k]); ch = (1 / ncand).mean()
    out(f"B{'123'[['splice','cnn','combo'].index(k)]}", f"P@1 on the analysis set, {lab} (rescue fit) vs chance", len(RR),
        f"{hit[k].mean():.3f} vs chance {ch:.3f}, lift {hit[k].mean()-ch:+.3f}", f"[{lo:.3f}, {hi:.3f}]", "", "")
b = int((hit["cnn"] & ~hit["splice"]).sum()); c = int((~hit["cnn"] & hit["splice"]).sum()); chi, p = mcnemar(b, c)
dd = hit["cnn"].astype(float) - hit["splice"]; lo, hi = ci_mean(dd)
out("C1", "P@1, network minus splice model (paired by gene)", len(RR), f"{dd.mean():+.3f} (network only {b}, splice only {c})", f"[{lo:+.3f}, {hi:+.3f}]", f"McNemar chi2 = {chi:.1f}", fp(p))
b = int((hit["combo"] & ~hit["cnn"]).sum()); c = int((~hit["combo"] & hit["cnn"]).sum()); chi, p = mcnemar(b, c)
dd = hit["combo"].astype(float) - hit["cnn"]; lo, hi = ci_mean(dd)
out("C2", "P@1, combined ranking minus network (paired by gene)", len(RR), f"{dd.mean():+.3f} (combined only {b}, network only {c})", f"[{lo:+.3f}, {hi:+.3f}]", f"McNemar chi2 = {chi:.1f}", fp(p))
C = np.load(os.path.join(PUB, "cnn_loco_scores.npz"), allow_pickle=True); S = np.load(os.path.join(PUB, "splice26_loco_scores.npz"), allow_pickle=True)
assert (C["gene"] == g).all() and (S["gene"] == g).all() and (S["center"] == cen).all()
def wr(v, ii):
    r = v[ii][y[ii] == 1]; c = v[ii][y[ii] == 0]
    return ((r[:, None] > c[None, :]).sum() + 0.5 * (r[:, None] == c[None, :]).sum()) / (len(r) * len(c))
wc = np.array([wr(C["logit"], ii) for ii in gidx.values()]); ws = np.array([wr(S["score"], ii) for ii in gidx.values()])
z, p, n = wilcoxon(wc - ws); lo, hi = ci_mean(wc - ws)
out("C3", "within-gene win-rate, network minus splice model (paired by gene)", len(wc), f"{wc.mean():.3f} vs {ws.mean():.3f}, difference {np.mean(wc-ws):+.3f}", f"[{lo:+.3f}, {hi:+.3f}]", f"Wilcoxon z = {z:.2f}, r = {abs(z)/math.sqrt(n):.3f}", fp(p))
WG = [r for r in csv.DictReader(open(os.path.join(PUB, "wg_per_gene.tsv")), delimiter="\t") if r["uncontested"] == "0"]
hc = np.array([r["cnn"] == "1" for r in WG]); hs = np.array([r["splice"] == "1" for r in WG])
b = int((hc & ~hs).sum()); c = int((~hc & hs).sum()); chi, p = mcnemar(b, c); lo, hi = ci_mean(hc.astype(float) - hs)
out("C4", "P@1 on 6,583 contested genes, network minus splice model", len(WG), f"{hc.mean():.3f} vs {hs.mean():.3f}, difference {hc.mean()-hs.mean():+.3f}", f"[{lo:+.3f}, {hi:+.3f}]", f"McNemar chi2 = {chi:.1f}", fp(p))
CS = list(csv.DictReader(open(os.path.join(ROOT, "data/context_sweep.tsv")), delimiter="\t"))
cs = collections.defaultdict(list)
for r in CS: cs[r["context"]].append(float(r["test_p_at_1"]))
for a, bb, i in (("up2000", "span500", "D1"), ("up2000", "up1500", "D2"), ("up1500", "up1000", "D3")):
    d, t, df, p = welch(cs[a], cs[bb]); se = math.sqrt(np.var(cs[a], ddof=1) / 5 + np.var(cs[bb], ddof=1) / 5); q = tq975(df)
    out(i, f"P@1, window {a} minus {bb} (5 seeds each, fixed split)", "5 vs 5", f"{np.mean(cs[a]):.3f} vs {np.mean(cs[bb]):.3f}, difference {d:+.3f}", f"[{d-q*se:+.3f}, {d+q*se:+.3f}]", f"Welch t = {t:.2f}, df = {df:.1f}", fp(p))
L = np.load(os.path.join(ROOT, "data/scramble_logits.npz")); ti = L["test_idx"]; yt = y[ti]; gt = g[ti]
tg = collections.defaultdict(list)
for k, gg in enumerate(gt): tg[gg].append(k)
tg = [np.array(v) for v in tg.values()]
def p1_per_gene(Lg):
    h = np.zeros(len(tg))
    for j, ii in enumerate(tg):
        r = Lg[:, ii][:, yt[ii] == 1][:, 0]; c = Lg[:, ii][:, yt[ii] == 0].max(1); h[j] = np.mean(r > c)
    return h
base = p1_per_gene(L["intact"])
for mode in ("scramble", "resample"):
    for t, lab in (("acc", "acceptors"), ("don", "donors"), ("both", "acceptors and donors")):
        a = p1_per_gene(L[f"{mode}_{t}"]); cc = p1_per_gene(L[f"{mode}_ctrl_{t}"])
        d = a - cc; lo, hi = ci_mean(d); d0 = a - base; lo0, hi0 = ci_mean(d0); z, p, n = wilcoxon(d)
        out(f"D{mode[0]}{t[0]}", f"P@1, strong {lab} {'permuted' if mode=='scramble' else 'resampled'}: minus matched control / minus intact", len(tg),
            f"{d.mean():+.4f} vs control; {d0.mean():+.4f} vs intact", f"[{lo:+.4f}, {hi:+.4f}]; [{lo0:+.4f}, {hi0:+.4f}]", f"Wilcoxon z = {z:.2f}", fp(p))
for r in csv.DictReader(open(os.path.join(PUB, "loco_shuffle.tsv")), delimiter="\t"):
    out("Dsh", f"dinucleotide shuffle, {r['model']}: win-rate real minus shuffled", r["n_genes"], f"{float(r['real']):.3f} to {float(r['shuffled']):.3f}, drop {float(r['drop']):.3f} ({float(r['pct_of_lift_lost']):.0f}% of lift)",
        f"[{float(r['drop_lo']):.3f}, {float(r['drop_hi']):.3f}]", "", "")
AT = list(csv.DictReader(open(os.path.join(PUB, "acceptor_arch_train.tsv")), delimiter="\t"))
a = [float(r["test_within_gene"]) for r in AT if r["variant"] == "aux"]; bb = [float(r["test_within_gene"]) for r in AT if r["variant"] == "base"]
d, t, df, p = welch(a, bb); se = math.sqrt(np.var(a, ddof=1) / 8 + np.var(bb, ddof=1) / 8); q = tq975(df)
out("Dax", "win-rate, network with minus without acceptor score (8 seeds each)", "8 vs 8", f"{d:+.4f}", f"[{d-q*se:+.4f}, {d+q*se:+.4f}]", f"Welch t = {t:.2f}, df = {df:.1f}", fp(p))
E = np.loadtxt(os.path.join(ROOT, "data/exon_context.tsv"), skiprows=1, usecols=(3, 4, 5)); ex5 = E[:, 2]; ex2 = E[:, 0]
eh = np.array([float(ex5[ii][y[ii] == 1][0] > ex5[ii][y[ii] == 0].max()) for ii in gidx.values()])
lo, hi = ci_mean(eh)
out("G1", "P@1 of the exonic fraction of the upstream 500 nt (annotation only)", len(eh), f"{eh.mean():.3f}", f"[{lo:.3f}, {hi:.3f}]", "", "")
rows = np.array([(ex2[ii][y[ii] == 1][0], float(C["logit"][ii][y[ii] == 1][0] > C["logit"][ii][y[ii] == 0].max())) for ii in gidx.values()])
edges = [0, 0.25, 0.5, 0.75, 1.0001]; parts = []
for a0, a1 in zip(edges[:-1], edges[1:]):
    m = (rows[:, 0] >= a0) & (rows[:, 0] < a1); lo, hi = ci_mean(rows[m, 1]); parts.append(f"{a0:.2f}-{min(a1,1):.2f}: {rows[m,1].mean():.3f} [{lo:.3f}, {hi:.3f}] (n={m.sum()})")
xb = np.digitize(rows[:, 0], edges[1:-1]); N = len(rows); R_ = rows[:, 1].sum(); pbar = R_ / N
T = sum(k * (rows[xb == k, 1].sum() - pbar * (xb == k).sum()) for k in range(4))
V = pbar * (1 - pbar) * (sum(k * k * (xb == k).sum() for k in range(4)) - sum(k * (xb == k).sum() for k in range(4)) ** 2 / N)
z = T / math.sqrt(V)
out("G2", "network P@1 by the real end's exonic fraction of the upstream 2 kb", N, "; ".join(parts), "", f"Cochran-Armitage z = {z:.2f}", fp(p2(z)))
cz = np.zeros(len(y))
for ii in gidx.values():
    cc = ii[y[ii] == 0]; cz[cc] = C["logit"][cc] - C["logit"][cc].mean()
hi_ = (y == 0) & (ex5 > 0.5); lo_ = (y == 0) & (ex5 == 0)
d, t, df, p = welch(cz[hi_], cz[lo_])
out("G3", "gene-centred network logit of competitors, >50% vs 0% exonic upstream 500 nt", f"{hi_.sum()} vs {lo_.sum()}", f"{cz[hi_].mean():+.3f} vs {cz[lo_].mean():+.3f}, difference {d:+.3f}", "", f"Welch t = {t:.2f}, df = {df:.0f}", fp(p))
P = np.load(os.path.join(ROOT, "data/patching.npz"), allow_pickle=True); pg = P["gene"]
pgi = [np.flatnonzero(pg == k) for k in np.unique(pg)]
gap = P["gap"]
for k in ("upstream", "core", "downstream"):
    v = P[f"gain_{k}"]; f = lambda gs, v=v: np.mean(np.concatenate([v[s] for s in gs])) / np.mean(np.concatenate([gap[s] for s in gs]))
    lo, hi = boot_genes(f, pgi, B=500)
    out(f"H{k[0]}", f"patching: logit gain of competitor from the real end's {k} region, fraction of gap", len(gap), f"{v.mean():+.3f} of {gap.mean():+.3f} ({v.mean()/gap.mean():.2f})", f"[{lo:.2f}, {hi:.2f}]", "", "")
with open(os.path.join(ROOT, "data/stats_ch3.tsv"), "w") as fh:
    fh.write("id\tcomparison\tn\teffect\teffect_ci\tstatistic\tp\n")
    for r in ROWS: fh.write("\t".join(str(x) for x in r) + "\n")
log("[out] data/stats_ch3.tsv")
