#!/usr/bin/env python3
# implant real acceptor regions cut out of real introns, instead of synthetic optima
import sys, time, argparse, gzip, random
import numpy as np
import torch
from cnn_interp_lib import NT

def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser()
P.add_argument("--model", default="cnn_model.pt")
P.add_argument("--gtf", default="/scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/"
                               "score_genome/GRCh38.gencode.v43.annotation.gtf")
P.add_argument("--fa", default="/scratch/spour98/jobs/hg38.fa")
P.add_argument("--nreg", type=int, default=150)
P.add_argument("--lengths", default="60,120")
P.add_argument("--gia_bg", type=int, default=256)
P.add_argument("--seed", type=int, default=42)
A = P.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
UP, WIN = 2000, 2500
LENS = [int(x) for x in A.lengths.split(",")]
rng = np.random.default_rng(A.seed)

# genome
FAI = {}
for line in open(A.fa + ".fai"):
    n, ln, off, lb, wb = line.split()[:5]
    FAI[n] = (int(ln), int(off), int(lb), int(wb))
fh = open(A.fa, "rb")
def fetch(c, s, e):
    if c not in FAI: return None
    ln, off, lb, wb = FAI[c]
    if s < 0 or e > ln or e <= s: return None
    fh.seek(off + s // lb * wb + s % lb)
    raw = fh.read((e - s) + (e - s) // lb + 2).decode()
    return raw.replace("\n", "")[:e - s].upper()
def rc(s): return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]

# acceptors from the annotation
TX = {}
op = gzip.open if A.gtf.endswith(".gz") else open
with op(A.gtf, "rt") as g:
    for line in g:
        if line[0] == "#": continue
        f = line.split("\t")
        if len(f) < 9 or f[2] != "exon" or 'transcript_type "protein_coding"' not in f[8]: continue
        i = f[8].find('transcript_id "'); tid = f[8][i+15:f[8].find('"', i+15)]
        TX.setdefault(tid, (f[0], f[6], []))[2].append((int(f[3]), int(f[4])))
log(f"[gtf] {len(TX):,} protein-coding transcripts")

from maxentpy import maxent
from maxentpy.maxent import load_matrix3
M3 = load_matrix3()
def s3(s):
    try: return float(maxent.score3(s, matrix=M3))
    except Exception: return -99.0

ACC = []
seen = set()
for tid, (c, st, ex) in TX.items():
    if c not in FAI or len(ex) < 2: continue
    ex.sort()
    for k in range(len(ex) - 1):
        istart, iend = ex[k][1] + 1, ex[k+1][0] - 1
        if iend - istart + 1 < 300: continue
        if st == "+":
            key = (c, "+", ex[k+1][0])
            if key in seen: continue
            s = fetch(c, ex[k+1][0] - 21, ex[k+1][0] + 2)
        else:
            key = (c, "-", ex[k][1])
            if key in seen: continue
            s = fetch(c, ex[k][1] - 3, ex[k][1] + 20)
            if s: s = rc(s)
        if not s or len(s) != 23 or set(s) - set("ACGT"): continue
        seen.add(key)
        if s[18:20] != "AG": continue
        ACC.append((c, st, key[2], istart, iend, s3(s)))
SC = np.array([a[5] for a in ACC])
log(f"[acc] {len(ACC):,} distinct AG acceptors with an intron of at least 300 nt; "
    f"score3 median {np.median(SC):.2f}, p90 {np.percentile(SC,90):.2f}, max {SC.max():.2f}")

o = np.argsort(-SC)
top = [ACC[i] for i in o[:A.nreg]]
mid_lo, mid_hi = np.percentile(SC, 45), np.percentile(SC, 55)
midpool = [a for a in ACC if mid_lo <= a[5] <= mid_hi]
mid = [midpool[i] for i in rng.choice(len(midpool), A.nreg, replace=False)]
log(f"[acc] top {A.nreg}: score3 {np.mean([a[5] for a in top]):.2f} mean, "
    f"{min(a[5] for a in top):.2f} lowest; mid {A.nreg}: {np.mean([a[5] for a in mid]):.2f} mean")

def frag(a, L, kind):
    c, st, pos, istart, iend, _ = a
    if kind == "acc":
        return (fetch(c, pos - 1 - (L - 3), pos + 2) if st == "+"
                else (lambda s: rc(s) if s else None)(fetch(c, pos - 3, pos - 3 + L)))
    m = (istart + iend) // 2
    if m - istart < 200 or iend - m < 200: return None
    s = fetch(c, m - L // 2, m - L // 2 + L)
    return s if st == "+" else (rc(s) if s else None)

def mono(s, key):
    return "".join(np.random.default_rng(key).permutation(list(s)))

SEQ = []
for L in LENS:
    for nm, pool in (("acc_top", top), ("acc_mid", mid), ("intron_ctl", top)):
        got = 0
        for i, a in enumerate(pool):
            s = frag(a, L, "acc" if nm != "intron_ctl" else "ctl")
            if not s or len(s) != L or set(s) - set("ACGT"): continue
            SEQ.append((f"{nm}_L{L}_{got}", s, nm, L, "native", got))
            SEQ.append((f"{nm}_L{L}_{got}_mono", mono(s, [A.seed, L, i]), nm, L, "mono", got))
            got += 1
            if got >= A.nreg: break
        log(f"[frag] {nm:<11} L={L:>3}  {got} fragments")
log(f"[seq] {len(SEQ):,} sequences to score")

# score
from cnn_model_def import Net
ck = torch.load(A.model, map_location=dev, weights_only=False)
net = Net(ck["args"]["dropout"]).to(dev); net.load_state_dict(ck["state"]); net.eval()
for p in net.parameters(): p.requires_grad_(False)
D = np.load("cnn_data.npz", allow_pickle=True); ok = D["ok"]
X = D["X"][ok]; y = D["y"][ok]; te = np.flatnonzero(D["split"][ok] == "test")
TBL = np.zeros((256, 4), np.float32); TBL[255] = 0.25
for i in range(4): TBL[i, i] = 1.0
TBLt = torch.from_numpy(TBL).to(dev)
def oh(a): return TBLt[torch.from_numpy(a.astype(np.int64)).to(dev)].permute(0, 2, 1).contiguous()
def fwd(a, bs=1024):
    out = []
    with torch.no_grad():
        for k in range(0, len(a), bs): out.append(net(oh(a[k:k+bs])).float().cpu().numpy())
    return np.concatenate(out)

BGC = X[rng.choice(te[y[te] == 1], A.gia_bg, replace=False)]
BGX = X[rng.choice(te[y[te] == 0], A.gia_bg, replace=False)]
POS = sorted(set(list(range(-300, 101, 10)) + list(range(-UP + 100, 500, 100))))
log(f"\n[gia] {len(SEQ):,} x {len(POS)} positions x 2 backgrounds x {A.gia_bg}")
rows = []; t0 = time.time()
for bn, B in (("real_canonical", BGC), ("real_competitor", BGX)):
    base = fwd(B)
    for k_, s_, cls, L, arm, idx in SEQ:
        m = np.array([NT.index(ch) for ch in s_], np.uint8)
        best = (None, -1e9)
        for p_ in POS:
            s0 = p_ + UP - L // 2
            if s0 < 0 or s0 + L > WIN: continue
            q = B.copy(); q[:, s0:s0+L] = m
            d = float((fwd(q) - base).mean())
            if d > best[1]: best = (p_, d)
        rows.append((bn, k_, cls, L, arm, idx, best[0], best[1]))
    log(f"[gia] {bn} done  {time.time()-t0:.0f}s")
with open("gia_native_acceptor.tsv", "w") as f:
    f.write("background\tseq_id\tclass\tlength\tarm\tidx\tbest_position\tbest_delta\n")
    for r in rows: f.write("\t".join(str(x) for x in r) + "\n")
log("[out] gia_native_acceptor.tsv")

V = {(r[0], r[2], r[3], r[4], r[5]): r[7] for r in rows}
for bn in ("real_canonical", "real_competitor"):
    log(f"\n[{bn}]  native fragment, its own shuffle, and the paired difference")
    log(f"  {'class':<11} {'L':>4} {'n':>4} {'native':>9} {'shuffle':>9} {'difference':>12} {'z':>6}")
    for L in LENS:
        for cls in ("acc_top", "acc_mid", "intron_ctl"):
            n = [V[(bn, cls, L, "native", i)] for i in range(A.nreg) if (bn, cls, L, "native", i) in V]
            s = [V[(bn, cls, L, "mono", i)] for i in range(A.nreg) if (bn, cls, L, "mono", i) in V]
            if not n: continue
            n = np.array(n); s = np.array(s); d = n - s
            se = d.std(ddof=1) / np.sqrt(len(d))
            log(f"  {cls:<11} {L:>4} {len(n):>4} {n.mean():>+9.4f} {s.mean():>+9.4f} "
                f"{d.mean():>+12.4f} {d.mean()/se:>+6.1f}")
