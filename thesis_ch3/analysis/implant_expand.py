#!/usr/bin/env python3
# checks and extensions of the implantation experiments
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
BG = {"competitor windows": X[ci_].copy(), "competitor windows, dinucleotide-shuffled": np.stack([dishuffle(x, rng) for x in X[ci_]]), "real 3' end windows": X[ri_].copy()}
for k, v in BG.items(): log(k, v.shape)
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
# native annotated splice sites
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
NAT = {"native 3' splice sites (50 nt)": sample(acc, 47, 3, lambda q: q[45:47] == "AG"),
       "native 5' splice sites (9 nt)": sample(don, 3, 6, lambda q: q[3:5] == "GT"),
       "native 5' splice sites (20 nt)": sample(don, 3, 17, lambda q: q[3:5] == "GT")}
FIX = [("optimised 3' splice site", "TTTAACTTTCTTTTTTACAGACG"), ("strongest MaxEnt 3' splice site", "TTTCCTCTCTCTCCCCGCAGGTG"), ("optimised 5' splice site", "CAGGTGAGT"),
       ("UGUA", "TGTA"), ("UGUA x2", "TGTATATGTA"), ("AAUAAA", "AATAAA"), ("AUUAAA", "ATTAAA"), ("GU-rich", "TGTGTGTG"), ("U-rich", "TTTTTTTT"),
       ("C-rich", "CCCCCCCC"), ("G-rich", "GGGAGGGG"), ("A-rich", "AAAAAAAA")]
OFFS = [-1500, -1000, -500, -300, -200, -100, -50, 50, 200]
out = open(os.path.join(ROOT, "data/implant_expand.tsv"), "w")
out.write("background\tinsert\tkind\tlength\toffset\tdelta_insert\tdelta_control\tdiff\tdiff_lo\tdiff_hi\tn_windows\n"); t0 = time.time()
def stat(d): b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(500)]; return d.mean(), *np.percentile(b, [2.5, 97.5])
for bname, Xb in BG.items():
    base = score(Xb); n = len(Xb)
    for name, pool in NAT.items():
        L = len(pool[0]); pick = rng.integers(0, len(pool), n); ins = np.stack([pool[k] for k in pick])
        ctl = [np.stack([rng.permutation(q) for q in ins]) for _ in range(3)]
        for off in OFFS:
            a = off + 2000; A = Xb.copy(); A[:, a:a+L] = ins; dc = score(A) - base; dk = np.zeros(n, np.float32)
            for cset in ctl:
                B = Xb.copy(); B[:, a:a+L] = cset; dk += (score(B) - base) / 3
            m, lo, hi = stat(dc - dk); out.write(f"{bname}\t{name}\tnative\t{L}\t{off}\t{dc.mean():.4f}\t{dk.mean():.4f}\t{m:.4f}\t{lo:.4f}\t{hi:.4f}\t{n}\n"); out.flush()
    for name, q in FIX:
        ins = enc(q); L = len(ins); ctl = [rng.permutation(ins) for _ in range(3)]
        if len(set(q)) == 1: ctl = [enc("".join(rng.choice(list("ACGT"), L))) for _ in range(3)]
        for off in OFFS:
            a = off + 2000; A = Xb.copy(); A[:, a:a+L] = ins; dc = score(A) - base; dk = np.zeros(n, np.float32)
            for cq in ctl:
                B = Xb.copy(); B[:, a:a+L] = cq; dk += (score(B) - base) / 3
            m, lo, hi = stat(dc - dk); out.write(f"{bname}\t{name}\tfixed\t{L}\t{off}\t{dc.mean():.4f}\t{dk.mean():.4f}\t{m:.4f}\t{lo:.4f}\t{hi:.4f}\t{n}\n"); out.flush()
    log(f"{bname} done {time.time()-t0:.0f}s; base mean logit {base.mean():.3f}")
out.close(); log("[out] data/implant_expand.tsv")
