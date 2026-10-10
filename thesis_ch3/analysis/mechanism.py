#!/usr/bin/env python3
# has the network learned the mechanisms known to couple splicing and 3' end formation?
import os, re, sys, time, collections, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
def log(*a): print(*a, file=sys.stderr, flush=True)
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); PUB = os.path.join(ROOT, "..")
GTF = os.path.join(PUB, "../train/data/annotation/gencode.v49.annotation.gtf")
dev = "cuda" if torch.cuda.is_available() else "cpu"; log("device", dev)
RNG = np.random.default_rng(0)
D = np.load(os.path.join(PUB, "cnn_data.npz"), allow_pickle=True)
X = D["X"]; y = D["y"].astype(int); g = D["gene"]; split = D["split"]; ch = D["chrom"]; cen = D["center"].astype(np.int64); st = D["strand"]
te = np.flatnonzero(split == "test"); realte = te[y[te] == 1]; compte = te[y[te] == 0]
# annotation
TID = re.compile(r'transcript_id "([^"]+)"')
tx = collections.defaultdict(list); info = {}
with open(GTF) as fh:
    for line in fh:
        if line.startswith("#"): continue
        f = line.split("\t", 8)
        if f[2] != "exon" or 'transcript_type "protein_coding"' not in f[8]: continue
        t = TID.search(f[8]).group(1); tx[t].append((int(f[3]) - 1, int(f[4]))); info[t] = (f[0], f[6])
last = collections.defaultdict(list); donors = collections.defaultdict(set)
for t, ex in tx.items():
    if len(ex) < 2: continue
    c_, s_ = info[t]; ex.sort()
    if s_ == "+":
        last[(c_, s_)].append(ex[-1])
        for e in ex[:-1]: donors[(c_, s_)].add(e[1])
    else:
        last[(c_, s_)].append(ex[0])
        for e in ex[1:]: donors[(c_, s_)].add(e[0] - 1)
DON = {k: np.array(sorted(v)) for k, v in donors.items()}
log(f"[gtf] {len(tx):,} protein-coding transcripts")
def acc_offset(i):
    c = int(cen[i]); k = (ch[i], st[i]); best = None
    for s_, e_ in last.get(k, []):
        if not (s_ <= c < e_): continue
        if st[i] == "+": a = s_ - c; endd = e_ - 1 - c
        else: a = c - (e_ - 1); endd = c - s_
        if best is None or abs(endd) < best[1]: best = (a, abs(endd))
    return None if best is None else best[0]
def don_offset(i):
    c = int(cen[i]); k = (ch[i], st[i]); Dk = DON.get(k)
    if Dk is None: return None
    if st[i] == "+":
        j = np.searchsorted(Dk, c) - 1
        if j < 0: return None
        b = Dk[j] - c
    else:
        j = np.searchsorted(Dk, c, side="right")
        if j >= len(Dk): return None
        b = c - Dk[j]
    return b if -1990 <= b <= -60 else None
A_OFF = {i: acc_offset(i) for i in realte}; A_OFF = {i: a for i, a in A_OFF.items() if a is not None and -1980 <= a <= -60}
D_OFF = {i: don_offset(i) for i in compte}; D_OFF = {i: b for i, b in D_OFF.items() if b is not None}
okA = np.mean([X[i][2000 + a - 2] == 0 and X[i][2000 + a - 1] == 2 for i, a in A_OFF.items()])
okD = np.mean([X[i][2000 + b] == 2 and X[i][2000 + b + 1] == 3 for i, b in D_OFF.items()])
log(f"[sites] real 3' ends with an annotated terminal-exon acceptor in the window: {len(A_OFF)} of {len(realte)}, AG present {okA:.3f}")
log(f"[sites] competitors with an annotated donor upstream in the window: {len(D_OFF)} of {len(compte)}, GT present {okD:.3f}")
# networks
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
def score(Xa, bs=4096):
    out = np.zeros(len(Xa), np.float32)
    for k in range(0, len(Xa), bs):
        xb = TBL[torch.from_numpy(Xa[k:k+bs].astype(np.int64)).to(dev)].permute(0, 2, 1).contiguous()
        out[k:k+bs] = np.mean([n(xb).float().cpu().numpy() for n in nets], 0)
    return out
ROWS = []
def out(*a): ROWS.append(a); log("  ".join(str(x) for x in a))
def boot(v, B=1000):
    v = np.asarray(v, float); return np.percentile([v[RNG.integers(0, len(v), len(v))].mean() for _ in range(B)], [2.5, 97.5])
def resample(w, lo, hi):
    acgt = w[w < 4]; p = np.bincount(acgt, minlength=4) / max(1, len(acgt)); w[lo:hi] = RNG.choice(4, size=hi - lo, p=p).astype(w.dtype)
def random_control(i, L, avoid):
    while True:
        s0 = int(RNG.integers(0, 2500 - L))
        if s0 + L <= 1960 or s0 >= 2040:
            if s0 + L <= avoid[0] or s0 >= avoid[1]: return s0
# 1, 2 knockouts
DB = [(-2000, -1000), (-1000, -500), (-500, -250), (-250, -60)]
def knockout(sites, left, right, label, want):
    sites = {i: o for i, o in sites.items() if 2000 + o - left >= 0 and 2000 + o + right <= 1960}
    idx = np.array(list(sites)); off = np.array([sites[i] for i in idx]); base = score(X[idx])
    Kx = X[idx].copy(); Cx = [X[idx].copy() for _ in range(3)]
    for r, (i, o) in enumerate(zip(idx, off)):
        lo, hi = 2000 + o - left, 2000 + o + right; resample(Kx[r], lo, hi)
        for Cc in Cx:
            s0 = random_control(i, hi - lo, (lo, hi)); resample(Cc[r], s0, s0 + hi - lo)
    dk = score(Kx) - base; dc = np.mean([score(Cc) - base for Cc in Cx], 0); d = dk - dc; lo_, hi_ = boot(d)
    out(label, "all", f"n {len(idx)}", f"knockout {dk.mean():+.3f}", f"control {dc.mean():+.3f}", f"difference {d.mean():+.3f} [{lo_:+.3f}, {hi_:+.3f}]", f"expected sign {want}")
    for a, b in DB:
        m = (off >= a) & (off < b)
        if m.sum() < 30: continue
        l2, h2 = boot(d[m]); out(label, f"site at {a} to {b} nt", f"n {m.sum()}", f"difference {d[m].mean():+.3f} [{l2:+.3f}, {h2:+.3f}]")
    return idx, dk, dc
t0 = time.time()
ia, dka, dca = knockout(A_OFF, 20, 3, "terminal-exon 3' splice site 23-mer knockout in real 3' end windows", "negative")
knockout(A_OFF, 47, 3, "terminal-exon 3' splice site 50-mer knockout in real 3' end windows", "negative")
knockout(D_OFF, 3, 6, "upstream 5' splice site 9-mer knockout in competitor windows", "positive")
knockout(D_OFF, 3, 17, "upstream 5' splice site 20-mer knockout in competitor windows", "positive")
Lall = score(X[te]); pos = {i: k for k, i in enumerate(te)}
Kx = X[ia].copy()
for r, i in enumerate(ia): o = A_OFF[i]; resample(Kx[r], max(0, 2000 + o - 47), 2000 + o + 3)
Lk = score(Kx); gset = collections.defaultdict(list)
for i in te: gset[g[i]].append(i)
h0 = []; h1 = []
for r, i in enumerate(ia):
    cc = [j for j in gset[g[i]] if y[j] == 0]
    if not cc: continue
    cm = max(Lall[pos[j]] for j in cc); h0.append(Lall[pos[i]] > cm); h1.append(Lk[r] > cm)
h0 = np.array(h0, float); h1 = np.array(h1, float); lo_, hi_ = boot(h1 - h0)
out("terminal-exon 3' splice site 50-mer knockout: P@1 of the knocked-out real 3' end", f"genes {len(h0)}", f"intact {h0.mean():.3f}", f"knockout {h1.mean():.3f}", f"difference {np.mean(h1-h0):+.3f} [{lo_:+.3f}, {hi_:+.3f}]")
log(f"  knockouts done {time.time()-t0:.0f}s")
# 3 architecture
CODE = {c: i for i, c in enumerate("ACGT")}; enc = lambda q: np.array([CODE[c] for c in q], np.uint8)
ACC = enc("TTTAACTTTCTTTTTTACAGACG"); DNR = enc("CAGGTGAGT")
Xc = X[compte]; base = score(Xc)
def implant(parts, shuffle=False):
    A = Xc.copy()
    for seq, start in parts:
        q = RNG.permutation(seq) if shuffle else seq; A[:, 2000 + start:2000 + start + len(q)] = q
    return score(A) - base
def net(parts):
    d = implant(parts) - np.mean([implant(parts, True) for _ in range(3)], 0); return d
AOFF = -300; acc_part = (ACC, AOFF - 20)
def don_part(d): return (DNR, d - 3)
dA = net([acc_part]); out("architecture", "acceptor alone at -300", f"{dA.mean():+.3f}")
for d in (-600, -450, -350, -250, -200, -150, -100):
    dD = net([don_part(d)]); dAD = net([acc_part, don_part(d)]); inter = dAD - dA - dD; lo_, hi_ = boot(inter)
    kind = "donor downstream of acceptor (internal exon)" if d > AOFF else "donor upstream of acceptor (intron, then terminal exon)"
    out("architecture", f"donor at {d}", kind, f"donor alone {dD.mean():+.3f}", f"both {dAD.mean():+.3f}", f"interaction {inter.mean():+.3f} [{lo_:+.3f}, {hi_:+.3f}]")
log(f"  architecture done {time.time()-t0:.0f}s")
for name, unit in (("TTGTAC (UGUA)", "TTGTAC"), ("AATAAA (PAS)", "AATAAA")):
    u = enc(unit); vals = []
    for off in range(-200, 101, 10):
        d = net([(u, off)]); lo_, hi_ = boot(d, 300); vals.append((off, d.mean(), lo_, hi_))
        out("scan", name, f"offset {off}", f"{d.mean():+.4f} [{lo_:+.4f}, {hi_:+.4f}]")
    b = max(vals, key=lambda v: v[1]); out("scan", name, "peak", f"offset {b[0]}", f"{b[1]:+.4f}")
with open(os.path.join(ROOT, "data/mechanism.tsv"), "w") as fh:
    for r in ROWS: fh.write("\t".join(str(x) for x in r) + "\n")
log(f"[out] data/mechanism.tsv {time.time()-t0:.0f}s")
