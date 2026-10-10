#!/usr/bin/env python3
# three further tests of what the network uses, on the held-out genes of the fixed split
import sys, os, time, collections, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, pysam
def log(*a): print(*a, file=sys.stderr, flush=True)
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); PUB = os.path.join(ROOT, "..")
dev = "cuda" if torch.cuda.is_available() else "cpu"; log("device", dev)
D = np.load(os.path.join(PUB, "cnn_data.npz"), allow_pickle=True); X = D["X"]; y = D["y"].astype(int); s = D["split"]
rng = np.random.default_rng(0)
ci_ = np.flatnonzero((s == "test") & (y == 0)); ci_ = np.sort(rng.choice(ci_, 3000, replace=False)); ri_ = np.flatnonzero((s == "test") & (y == 1))
def dishuffle(v, r):
    v = v.astype(int); last = v[-1]; edges = collections.defaultdict(list)
    for a, b in zip(v[:-1], v[1:]): edges[a].append(b)
    while True:
        le = {a: e[r.integers(len(e))] for a, e in edges.items() if a != last}; okk = True
        for a in le:
            seen = set(); c = a
            while c != last and c in le and c not in seen: seen.add(c); c = le[c]
            if c != last: okk = False; break
        if okk: break
    out = [v[0]]; pools = {}
    for a, e in edges.items():
        e = list(e)
        if a in le: e.remove(le[a])
        e = list(r.permutation(e)); 
        if a in le: e.append(le[a])
        pools[a] = e[::-1]
    c = v[0]
    while pools.get(c): c = pools[c].pop(); out.append(c)
    assert len(out) == len(v); return np.array(out, np.uint8)
BG = None
TBL = torch.zeros(256, 4); TBL[255] = 0.25
for k in range(4): TBL[k, k] = 1.0
TBL = TBL.to(dev)
class V(nn.Module):
    def __init__(s):
        super().__init__()
        s.c1 = nn.Conv1d(4, 32, 19, padding=9, bias=False); s.b1 = nn.BatchNorm1d(32)
        s.c2 = nn.Conv1d(32, 64, 7, padding=3, bias=False); s.b2 = nn.BatchNorm1d(64)
        s.c3 = nn.Conv1d(64, 96, 5, padding=2, bias=False); s.b3 = nn.BatchNorm1d(96)
        s.do = nn.Dropout(0.4); s.fc1 = nn.Linear(96 * 25, 64); s.fc2 = nn.Linear(64, 1)
    def forward(s, x):
        h = F.max_pool1d(torch.exp(torch.clamp(s.b1(s.c1(x)), max=8.0)), 10)
        h = F.max_pool1d(F.relu(s.b2(s.c2(h))), 5)
        h = F.max_pool1d(F.relu(s.b3(s.c3(h))), 2)
        return s.fc2(s.do(F.relu(s.fc1(s.do(h.flatten(1)))))).squeeze(1)
nets = []
for seed in (42, 43, 44, 45, 46):
    n = V().to(dev); n.load_state_dict(torch.load(os.path.join(PUB, f"acc_arch_base_s{seed}.pt"), map_location=dev)["state"]); n.eval(); nets.append(n)
@torch.no_grad()
def score(Xa, bs=2048):
    out = np.zeros(len(Xa), np.float32)
    for k in range(0, len(Xa), bs):
        xb = TBL[torch.from_numpy(Xa[k:k+bs].astype(np.int64)).to(dev)].permute(0, 2, 1).contiguous()
        out[k:k+bs] = np.mean([n(xb).float().cpu().numpy() for n in nets], 0)
    return out
CODE = {c: i for i, c in enumerate("ACGT")}; enc = lambda q: np.array([CODE[c] for c in q], np.uint8)
CODE = {c: i for i, c in enumerate("ACGT")}; enc = lambda q: np.array([CODE[c] for c in q], np.uint8)
GTF = os.path.join(PUB, "../train/data/annotation/gencode.v49.annotation.gtf"); tx = collections.defaultdict(list)
with open(GTF) as fh:
    for line in fh:
        if line.startswith("#"): continue
        f = line.split("\t", 8)
        if f[2] == "exon" and 'transcript_type "protein_coding"' in f[8] and f[0] in {f"chr{i}" for i in list(range(1, 23)) + ["X"]}:
            tid = f[8].split('transcript_id "')[1].split('"')[0]; tx[tid].append((f[0], int(f[3]) - 1, int(f[4]), f[6]))
acc, don = set(), set()
for ex in tx.values():
    if len(ex) < 2: continue
    st = ex[0][3]; ex = sorted(ex, key=lambda e: e[1], reverse=(st == "-"))
    for k, (c, a, b, _) in enumerate(ex):
        if k > 0: acc.add((c, a if st == "+" else b, st))
        if k < len(ex) - 1: don.add((c, b if st == "+" else a, st))
fa = pysam.FastaFile(os.path.join(PUB, "../train/data/hg38.fa")); tr = str.maketrans("ACGT", "TGCA")
def fetch(c, p, st, up, dn):
    q = fa.fetch(c, p - up, p + dn).upper() if st == "+" else fa.fetch(c, p - dn, p + up).upper().translate(tr)[::-1]
    return q
def sample(sites, up, dn, test, n=300):
    sites = sorted(sites); out = []
    for k in rng.permutation(len(sites)):
        q = fetch(*sites[k], up, dn)
        if len(q) == up + dn and set(q) <= set("ACGT") and test(q): out.append(enc(q))
        if len(out) == n: break
    return out

from maxentpy import maxent
from maxentpy.maxent import load_matrix3, load_matrix5
M3, M5 = load_matrix3(), load_matrix5()
Xc = X[ci_].copy(); Xr = X[ri_].copy(); bc = score(Xc); br = score(Xr); log("base", bc.mean(), br.mean())
def stat(d): b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(500)]; return d.mean(), *np.percentile(b, [2.5, 97.5])
dec = lambda v: "".join("ACGT"[c] for c in v)
# 1 strength
def pool(sites, up, dn, test, scorer, groups, n=200):
    sites = sorted(sites); G = {g: [] for g in groups}
    for k in rng.permutation(len(sites))[:60000]:
        q = fetch(*sites[k], up, dn)
        if len(q) != up + dn or not set(q) <= set("ACGT") or not test(q): continue
        s_ = scorer(q)
        for g, (lo, hi) in groups.items():
            if lo <= s_ < hi and len(G[g]) < n: G[g].append(enc(q))
        if all(len(v) >= n for v in G.values()): break
    return G
A3 = pool(acc, 47, 3, lambda q: q[45:47] == "AG", lambda q: maxent.score3(q[27:50], matrix=M3), {"<6": (-99, 6), "6-8": (6, 8), "8-10": (8, 10), "10-12": (10, 12), ">=12": (12, 99)})
D5 = pool(don, 3, 6, lambda q: q[3:5] == "GT", lambda q: maxent.score5(q, matrix=M5), {"<6": (-99, 6), "6-8": (6, 8), "8-10": (8, 10), ">=10": (10, 99)})
with open(os.path.join(ROOT, "data/deep_strength.tsv"), "w") as fh:
    fh.write("site\tscore_group\tsites\toffset\tdiff\tdiff_lo\tdiff_hi\n")
    for name, G in (("3' splice site", A3), ("5' splice site", D5)):
        for g, P in G.items():
            if len(P) < 30: continue
            L = len(P[0]); pick = rng.integers(0, len(P), len(Xc)); ins = np.stack([P[k] for k in pick]); ctl = [np.stack([rng.permutation(q) for q in ins]) for _ in range(3)]
            for off in (-300, -100):
                a = off + 2000; A = Xc.copy(); A[:, a:a+L] = ins; d = score(A) - bc; dk = np.zeros(len(Xc), np.float32)
                for c_ in ctl:
                    B = Xc.copy(); B[:, a:a+L] = c_; dk += (score(B) - bc) / 3
                m, lo, hi = stat(d - dk); fh.write(f"{name}\t{g}\t{len(P)}\t{off}\t{m:.4f}\t{lo:.4f}\t{hi:.4f}\n"); log(name, g, len(P), off, round(m, 3))
# 2 necessity
import re
ELEM = [("UGUA", "TGTA", -500, 0), ("U run (6 or more)", "T{6,}", -500, 0), ("GAAGAA", "GAAGAA", -500, 0), ("CUCUCU", "CTCTCT", -500, 0), ("UAGGG", "TAGGG", -500, 0),
        ("poly(A) signal", "AATAAA|ATTAAA", -40, -5 + 6), ("strong 5' splice site (score5 >= 8)", None, -500, 0)]
with open(os.path.join(ROOT, "data/deep_necessity.tsv"), "w") as fh:
    fh.write("element\tclass\twindows_with_element\tmean_occurrences\tdiff\tdiff_lo\tdiff_hi\n")
    for cname, Xw, b0 in (("real 3' end", Xr, br), ("competitor", Xc, bc)):
        S = [dec(x) for x in Xw]
        for name, pat, lo, hi in ELEM:
            K = Xw.copy(); C = [Xw.copy() for _ in range(3)]; has = np.zeros(len(Xw), bool); nocc = []
            for i, s_ in enumerate(S):
                reg = s_[2000 + lo:2000 + hi]
                if pat is None:
                    iv = [(m.start() - 3, m.start() + 6) for m in re.finditer("(?=GT)", reg) if m.start() >= 3 and m.start() + 6 <= len(reg) and maxent.score5(reg[m.start() - 3:m.start() + 6], matrix=M5) >= 8]
                else: iv = [(m.start(), m.end()) for m in re.finditer(pat, reg)]
                if not iv: continue
                has[i] = True; nocc.append(len(iv)); comp_ = np.bincount(Xw[i][Xw[i] < 4], minlength=4) / (Xw[i] < 4).sum()
                for a, b in iv: K[i, 2000 + lo + a:2000 + lo + b] = rng.choice(4, b - a, p=comp_)
                for Cm in C:
                    for a, b in iv:
                        p0 = rng.integers(2000 - 500, 2000 - (b - a)); Cm[i, p0:p0 + b - a] = rng.choice(4, b - a, p=comp_)
            if has.sum() < 30: continue
            d = score(K[has]) - b0[has]; dk = np.mean([score(Cm[has]) - b0[has] for Cm in C], 0); m, l_, h_ = stat(d - dk)
            fh.write(f"{name}\t{cname}\t{int(has.sum())}\t{np.mean(nocc):.2f}\t{m:.4f}\t{l_:.4f}\t{h_:.4f}\n"); log(name, cname, int(has.sum()), round(m, 3))
# 3 mutagenesis
LO, HI = -400, 100; prof = {}; km = {}
for cname, Xw, b0 in (("real 3' end", Xr, br), ("competitor", Xc[:1500], bc[:1500])):
    imp = np.zeros((len(Xw), HI - LO), np.float32)
    for j, p in enumerate(range(2000 + LO, 2000 + HI)):
        tot = np.zeros(len(Xw), np.float32)
        for k in (1, 2, 3):
            A = Xw.copy(); A[:, p] = (Xw[:, p] + k) % 4; tot += score(A) - b0
        imp[:, j] = -tot / 3
        if j % 100 == 0: log(cname, "position", LO + j)
    prof[cname] = (imp.mean(0), np.abs(imp).mean(0)); s6 = collections.defaultdict(list)
    for i in range(len(Xw)):
        q = dec(Xw[i][2000 + LO:2000 + HI])
        for j in range(0, HI - LO - 5): s6[q[j:j + 6]].append(imp[i, j:j + 6].sum())
    km[cname] = {k: (np.mean(v), len(v)) for k, v in s6.items() if len(v) >= 200}
    np.save(os.path.join(ROOT, f"data/deep_ism_{'real' if cname.startswith('real') else 'competitor'}.npy"), imp)
with open(os.path.join(ROOT, "data/deep_ism_profile.tsv"), "w") as fh:
    fh.write("class\tposition\tmean_importance\tmean_absolute_importance\n")
    for c_, (m, a) in prof.items():
        for j in range(HI - LO): fh.write(f"{c_}\t{LO + j}\t{m[j]:.5f}\t{a[j]:.5f}\n")
with open(os.path.join(ROOT, "data/deep_ism_kmers.tsv"), "w") as fh:
    fh.write("class\tkmer\toccurrences\tmean_importance\n")
    for c_, d_ in km.items():
        for k, (m, n_) in sorted(d_.items(), key=lambda kv: -kv[1][0]): fh.write(f"{c_}\t{k.replace('T','U')}\t{n_}\t{m:.4f}\n")
log("[out] data/deep_*.tsv")
