#!/usr/bin/env python3
import sys, time, argparse
import numpy as np
import torch
from cnn_interp_lib import NT

def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser()
P.add_argument("--model", default="cnn_model.pt")
P.add_argument("--nreal", type=int, default=50)
P.add_argument("--ncomp", type=int, default=4096)
P.add_argument("--seed", type=int, default=42)
A = P.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
UP, WIN = 2000, 2500
POS = [-1500, -1200, -1000, -800, -600, -400, -300, -200, -100, -50, 0, 100, 300]
rng = np.random.default_rng(A.seed)

# the native acceptors, same extraction as gia_native_acceptor.py
FA = "/scratch/spour98/jobs/hg38.fa"
GTF = ("/scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/score_genome/"
       "GRCh38.gencode.v43.annotation.gtf")
FAI = {}
for line in open(FA + ".fai"):
    n, ln, off, lb, wb = line.split()[:5]; FAI[n] = (int(ln), int(off), int(lb), int(wb))
fh = open(FA, "rb")
def fetch(c, s, e):
    if c not in FAI: return None
    ln, off, lb, wb = FAI[c]
    if s < 0 or e > ln or e <= s: return None
    fh.seek(off + s // lb * wb + s % lb)
    return fh.read((e-s) + (e-s)//lb + 2).decode().replace("\n", "")[:e-s].upper()
def rc(s): return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]
from maxentpy import maxent
from maxentpy.maxent import load_matrix3
M3 = load_matrix3()
def s3(s):
    try: return float(maxent.score3(s, matrix=M3))
    except Exception: return -99.0
TX = {}
with open(GTF) as g:
    for line in g:
        if line[0] == "#": continue
        f = line.split("\t")
        if len(f) < 9 or f[2] != "exon" or 'transcript_type "protein_coding"' not in f[8]: continue
        i = f[8].find('transcript_id "'); tid = f[8][i+15:f[8].find('"', i+15)]
        TX.setdefault(tid, (f[0], f[6], []))[2].append((int(f[3]), int(f[4])))
ACC = []; seen = set()
for tid, (c, st, ex) in TX.items():
    if c not in FAI or len(ex) < 2: continue
    ex.sort()
    for k in range(len(ex)-1):
        if ex[k+1][0]-1 - (ex[k][1]+1) + 1 < 300: continue
        key = (c, st, ex[k+1][0] if st == "+" else ex[k][1])
        if key in seen: continue
        s = (fetch(c, ex[k+1][0]-21, ex[k+1][0]+2) if st == "+"
             else (lambda z: rc(z) if z else None)(fetch(c, ex[k][1]-3, ex[k][1]+20)))
        if not s or len(s) != 23 or set(s)-set("ACGT"): continue
        seen.add(key)
        if s[18:20] != "AG": continue
        ACC.append((c, st, key[2], s3(s)))
ACC.sort(key=lambda a: -a[3])
NAT = []
for c, st, pos, sc in ACC:
    z = (fetch(c, pos-1-117, pos+2) if st == "+"
         else (lambda q: rc(q) if q else None)(fetch(c, pos-3, pos-3+120)))
    if z and len(z) == 120 and not (set(z)-set("ACGT")): NAT.append(z)
    if len(NAT) >= A.nreal: break
log(f"[native] {len(NAT)} real acceptor fragments of 120 nt, mean terminal score3 "
    f"{np.mean([s3(z[-23:]) for z in NAT]):.2f}")
SYN = "TTTCCTCTCTCTCCCCGCAGGTG"
assert abs(s3(SYN) - 17.172) < 0.01
def mono(s, k): return "".join(np.random.default_rng(k).permutation(list(s)))
UNITS = ([("synthetic", "motif", SYN)] +
         [("synthetic", "mono", mono(SYN, [A.seed, 1, j])) for j in range(10)] +
         [("native", "motif", z) for z in NAT] +
         [("native", "mono", mono(z, [A.seed, 2, i])) for i, z in enumerate(NAT)])
log(f"[units] {len(UNITS)} sequences x {len(POS)} fixed positions, NO maximum taken")

from cnn_model_def import Net
ck = torch.load(A.model, map_location=dev, weights_only=False)
net = Net(ck["args"]["dropout"]).to(dev); net.load_state_dict(ck["state"]); net.eval()
for p in net.parameters(): p.requires_grad_(False)
D = np.load("cnn_data.npz", allow_pickle=True); ok = D["ok"]
X = D["X"][ok]; y = D["y"][ok]
te = np.flatnonzero((D["split"][ok] == "test") & (X.max(1) <= 3))
TBL = np.zeros((256, 4), np.float32); TBL[255] = 0.25
for i in range(4): TBL[i, i] = 1.0
TBLt = torch.from_numpy(TBL).to(dev)
def oh(a): return TBLt[torch.from_numpy(a.astype(np.int64)).to(dev)].permute(0, 2, 1).contiguous()
def fwd(a, bs=1024):
    o = []
    with torch.no_grad():
        for k in range(0, len(a), bs): o.append(net(oh(a[k:k+bs])).float().cpu().numpy())
    return np.concatenate(o)
BGC = X[te[y[te] == 1]]
cx = te[y[te] == 0]; BGX = X[rng.choice(cx, min(A.ncomp, len(cx)), replace=False)]
log(f"[bg] {len(BGC):,} real 3' ends (all of them) and {len(BGX):,} competitors")

rows = []; t0 = time.time()
for bn, B in (("real_canonical", BGC), ("real_competitor", BGX)):
    base = fwd(B)
    for cls, arm, s_ in UNITS:
        m = np.array([NT.index(c) for c in s_], np.uint8); Lm = len(m)
        for p_ in POS:
            s0 = p_ + UP - Lm // 2
            if s0 < 0 or s0 + Lm > WIN: continue
            q = B.copy(); q[:, s0:s0+Lm] = m
            d = fwd(q) - base
            rows.append((bn, cls, arm, p_, Lm, float(d.mean()),
                         float(d.std(ddof=1)/np.sqrt(len(d)))))
    log(f"[run] {bn} done  {time.time()-t0:.0f}s")
with open("acceptor_positional.tsv", "w") as f:
    f.write("background\tunit\tarm\tposition\tlength\tdelta_logit\tse\n")
    for r in rows: f.write("\t".join(str(x) for x in r) + "\n")
log("[out] acceptor_positional.tsv")

V = {}
for bn, cls, arm, p_, Lm, d, se in rows: V.setdefault((bn, cls, arm, p_), []).append(d)
for bn in ("real_canonical", "real_competitor"):
    log(f"\n[{bn}] effect AT each fixed position, no maximum taken")
    log(f"  {'position':>9} | {'synthetic motif':>15} {'its shuffle':>12} {'diff':>8} "
        f"| {'native motif':>13} {'its shuffle':>12} {'diff':>8}")
    for p_ in POS:
        out = []
        for cls in ("synthetic", "native"):
            a = V.get((bn, cls, "motif", p_)); b = V.get((bn, cls, "mono", p_))
            if not a: out += ["", "", ""]; continue
            out += [f"{np.mean(a):>+15.4f}", f"{np.mean(b):>+12.4f}", f"{np.mean(a)-np.mean(b):>+8.4f}"]
        log(f"  {p_:>9} | {out[0]} {out[1]} {out[2]} | {out[3]} {out[4]} {out[5]}")
