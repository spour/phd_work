#!/usr/bin/env python3
# three sequence-design experiments on the five 2,500 nt networks (fixed split
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
import itertools
M3, M5 = load_matrix3(), load_matrix5()
_cwd = os.getcwd(); os.chdir(PUB); sys.path.insert(0, PUB); sys.path.insert(0, "scripts"); import u1_5ss_landscape_data as L; os.chdir(_cwd)
U1T = np.asarray(L.U1, float); POW = np.array([int(L.POW[p]) for p in range(9)], np.int64)
gene = D["gene"]; Xc = X[ci_].copy(); Xr = X[ri_].copy(); bc = score(Xc); br = score(Xr); log("base", bc.mean(), br.mean())
def stat(d): b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(500)]; return d.mean(), *np.percentile(b, [2.5, 97.5])
dec = lambda v: "".join("ACGT"[c] for c in v)
def put(Xw, off, ins):
    A = Xw.copy(); a = off + 2000
    if ins.ndim == 1: A[:, a:a + len(ins)] = ins
    else: A[:, a:a + ins.shape[1]] = ins
    return A
# 1 u1 series
cons = "CAGGTAAGT"; V = {cons: ()}
for k in (1, 2):
    for pos in itertools.combinations(range(9), k):
        for sub in itertools.product("ACGT", repeat=k):
            if any(sub[j] == cons[p] for j, p in enumerate(pos)): continue
            q = list(cons)
            for j, p in enumerate(pos): q[p] = sub[j]
            V["".join(q)] = pos
log(len(V), "U1 variants")
with open(os.path.join(ROOT, "data/design_u1_series.tsv"), "w") as fh:
    fh.write("sequence\tmismatches\tpositions_changed\tu1_dG\tscore5\tdiff\tdiff_lo\tdiff_hi\n")
    for q, pos in V.items():
        e = enc(q); dG = U1T[int((e.astype(np.int64) * POW).sum())]; s5 = maxent.score5(q, matrix=M5) if q[3:5] == "GT" else float("nan")
        d = score(put(Xc, -300, e)) - bc; dk = np.mean([score(put(Xc, -300, rng.permutation(e))) - bc for _ in range(3)], 0); m, lo, hi = stat(d - dk)
        fh.write(f"{q.replace('T','U')}\t{len(pos)}\t{','.join(str(p + 1) for p in pos)}\t{dG:.2f}\t{s5:.2f}\t{m:.4f}\t{lo:.4f}\t{hi:.4f}\n")
log("U1 series done")
# strong annotated sites
def pool(sites, up, dn, test, scorer, lo, n=200):
    sites = sorted(sites); out = []
    for k in rng.permutation(len(sites))[:80000]:
        q = fetch(*sites[k], up, dn)
        if len(q) == up + dn and set(q) <= set("ACGT") and test(q) and scorer(q) >= lo: out.append(enc(q))
        if len(out) == n: break
    return out
A3 = pool(acc, 47, 3, lambda q: q[45:47] == "AG", lambda q: maxent.score3(q[27:50], matrix=M3), 10)
D5 = pool(don, 3, 6, lambda q: q[3:5] == "GT", lambda q: maxent.score5(q, matrix=M5), 9); log(len(A3), len(D5), "strong sites")
def draw(P, n): k = rng.integers(0, len(P), n); return np.stack([P[j] for j in k])
def perm(I): return np.stack([rng.permutation(q) for q in I])
# 2 relay grid
APOS = [-1500, -1200, -900, -700, -500, -400, -300, -200, -100]; DPOS = [-1500, -1200, -1000, -800, -650, -450, -350, -250, -150, -100, -50, 100]
n = len(Xc); ia = draw(A3, n); idn = draw(D5, n); pa = perm(ia); pd = perm(idn)
ea = {a: score(put(Xc, a - 47, ia)) - score(put(Xc, a - 47, pa)) for a in APOS}
ed = {d: score(put(Xc, d - 3, idn)) - score(put(Xc, d - 3, pd)) for d in DPOS}
with open(os.path.join(ROOT, "data/design_relay_grid.tsv"), "w") as fh:
    fh.write("acceptor_pos\tdonor_pos\tdonor_minus_acceptor\teffect_acceptor\teffect_donor\teffect_both\tinteraction\tint_lo\tint_hi\n")
    for a in APOS:
        for d in DPOS:
            if not (d - 3 >= a + 3 or d + 6 <= a - 47): continue
            both = score(put(put(Xc, a - 47, ia), d - 3, idn)) - score(put(put(Xc, a - 47, pa), d - 3, pd))
            m, lo, hi = stat(both - ea[a] - ed[d]); fh.write(f"{a}\t{d}\t{d - a}\t{ea[a].mean():.4f}\t{ed[d].mean():.4f}\t{both.mean():.4f}\t{m:.4f}\t{lo:.4f}\t{hi:.4f}\n")
    log("relay grid done")
# 3 ladders
MOT = {}
with open(os.path.join(PUB, "../cisbp_rna.meme")) as fh_:
    name = None; rows = []
    for line in fh_:
        if line.startswith("MOTIF"):
            if name and rows: MOT[name] = np.array(rows)
            name = line.split()[1]; rows = []
        elif name and line.strip() and line.strip()[0].isdigit():
            v = [float(x) for x in line.split()]
            if len(v) == 4: rows.append(v)
    if name and rows: MOT[name] = np.array(rows)
def trim(P):
    P = P / P.sum(1, keepdims=True); ic = 2 + (P * np.log2(np.clip(P, 1e-9, 1))).sum(1); k = np.flatnonzero(ic >= 0.25); return P[k[0]:k[-1] + 1]
def samp(mid, n):
    P = trim(MOT[mid]); cdf = np.cumsum(P, 1); u = rng.random((n, len(P))); return (u[:, :, None] > cdf[None, :, :]).sum(2).clip(0, 3).astype(np.uint8)
import csv
best = {r["protein"]: r["motif"] for r in csv.DictReader(open(os.path.join(ROOT, "data/implant_cisbp_best.tsv")), delimiter="\t")}
gidx = {}
for i in ri_: gidx[gene[i]] = i
real_of = np.array([br[list(ri_).index(gidx[g_])] if g_ in gidx else np.nan for g_ in gene[ci_]])
allc = np.flatnonzero((s == "test") & (y == 0)); sc_all = score(X[allc]); bestc = collections.defaultdict(lambda: -1e9)
for g_, v in zip(gene[allc], sc_all): bestc[g_] = max(bestc[g_], v)
best_comp = np.array([bestc[g_] for g_ in gene[ri_]])
Xs = np.stack([dishuffle(x, rng) for x in Xc])
BUILD = [("poly(A) signal AAUAAA at -21", lambda n: (-21, np.tile(enc("AATAAA"), (n, 1)))),
         ("CstF64 motif at +10", lambda n: (10, samp(best["CSTF2"], n))),
         ("CFIm25 motif at -50", lambda n: (-50, samp(best["NUDT21"], n))), ("CFIm25 motif at -70", lambda n: (-70, samp(best["NUDT21"], n))),
         ("U run at -38", lambda n: (-38, np.tile(enc("TTTTTT"), (n, 1)))),
         ("SRSF7 motif at -200", lambda n: (-200, samp(best["SRSF7"], n))), ("SRSF7 motif at -260", lambda n: (-260, samp(best["SRSF7"], n))), ("SRSF7 motif at -320", lambda n: (-320, samp(best["SRSF7"], n))),
         ("strong 3' splice site at -600", lambda n: (-647, draw(A3, n)))]
BREAK = [("strong 5' splice site at -300", lambda n: (-303, draw(D5, n))), ("strong 5' splice site at -150", lambda n: (-153, draw(D5, n))),
         ("hnRNP A0 motif at -200", lambda n: (-200, samp(best["HNRNPA0"], n))), ("hnRNP A0 motif at -250", lambda n: (-250, samp(best["HNRNPA0"], n))), ("hnRNP A0 motif at -100", lambda n: (-100, samp(best["HNRNPA0"], n))),
         ("CU tract at -60", lambda n: (-60, np.tile(enc("CTCTCTCTCTCT"), (n, 1)))), ("intron end UCUCAG at -120", lambda n: (-120, np.tile(enc("TCTCAG"), (n, 1))))]
with open(os.path.join(ROOT, "data/design_ladders.tsv"), "w") as fh:
    fh.write("ladder\tbackground\tstep\tadded\tmode\tmean_logit\tlo\thi\tfraction_flipped\tn\n")
    def run(lname, bname, X0, steps, flip):
        for mode in ("cumulative", "alone"):
            cur = X0.copy(); s0 = score(X0); m, lo, hi = stat(s0)
            if mode == "cumulative": fh.write(f"{lname}\t{bname}\t0\tnone\t{mode}\t{m:.4f}\t{lo:.4f}\t{hi:.4f}\t{flip(s0):.4f}\t{len(X0)}\n")
            for k, (nm, f_) in enumerate(steps, 1):
                off, ins = f_(len(X0)); cur = put(cur if mode == "cumulative" else X0, off, ins); sc = score(cur); m, lo, hi = stat(sc)
                fh.write(f"{lname}\t{bname}\t{k}\t{nm}\t{mode}\t{m:.4f}\t{lo:.4f}\t{hi:.4f}\t{flip(sc):.4f}\t{len(X0)}\n"); log(lname, bname, mode, k, nm, round(m, 3), round(flip(sc), 3))
    okc = np.isfinite(real_of)
    run("build", "dinucleotide-shuffled competitor windows", Xs, BUILD, lambda sc: float(np.mean(sc[okc] > real_of[okc])))
    run("build", "competitor windows", Xc, BUILD, lambda sc: float(np.mean(sc[okc] > real_of[okc])))
    run("break", "real 3' end windows", Xr, BREAK, lambda sc: float(np.mean(sc <= best_comp)))
log("[out] data/design_*.tsv")
