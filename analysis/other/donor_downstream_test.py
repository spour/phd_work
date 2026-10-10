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
UP, WIN, LFR = 2000, 2500, 120
POS = [-800, -400, -200, -100, 0, 50, 100, 150, 200, 250, 300, 400]
DNPOS = [p for p in POS if 0 <= p <= 300]
rng = np.random.default_rng(A.seed)

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
    fh.seek(off + s//lb*wb + s%lb)
    return fh.read((e-s) + (e-s)//lb + 2).decode().replace("\n", "")[:e-s].upper()
def rc(s): return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]
from maxentpy import maxent
from maxentpy.maxent import load_matrix5
M5 = load_matrix5()
def s5(s):
    try: return float(maxent.score5(s, matrix=M5))
    except Exception: return -99.0
sys.path.insert(0, "scripts")
import u1_5ss_landscape_data as Lu
U1X = "".join("ACGT"[(int(np.nanargmax(Lu.U1)) // int(Lu.POW[p])) % 4] for p in range(9))
assert U1X == "CAGGTGAGT"

TX = {}
with open(GTF) as g:
    for line in g:
        if line[0] == "#": continue
        f = line.split("\t")
        if len(f) < 9 or f[2] != "exon" or 'transcript_type "protein_coding"' not in f[8]: continue
        i = f[8].find('transcript_id "'); tid = f[8][i+15:f[8].find('"', i+15)]
        TX.setdefault(tid, (f[0], f[6], []))[2].append((int(f[3]), int(f[4])))
DON = []; seen = set()
for tid, (c, st, ex) in TX.items():
    if c not in FAI or len(ex) < 2: continue
    ex.sort()
    for k in range(len(ex)-1):
        if ex[k+1][0]-1 - (ex[k][1]+1) + 1 < 300: continue
        pos = ex[k][1] if st == "+" else ex[k+1][0]
        key = (c, st, pos)
        if key in seen: continue
        s = (fetch(c, pos-3, pos+6) if st == "+"
             else (lambda z: rc(z) if z else None)(fetch(c, pos-7, pos+2)))
        if not s or len(s) != 9 or set(s)-set("ACGT"): continue
        seen.add(key)
        if s[3:5] != "GT": continue
        DON.append((c, st, pos, s5(s)))
DON.sort(key=lambda a: -a[3])
NAT = []
for c, st, pos, sc in DON:
    z = (fetch(c, pos-3, pos-3+LFR) if st == "+"
         else (lambda q: rc(q) if q else None)(fetch(c, pos-LFR+2, pos+2)))
    if z and len(z) == LFR and not (set(z)-set("ACGT")): NAT.append(z)
    if len(NAT) >= A.nreal: break
nsc = np.array([s5(z[:9]) for z in NAT])
log(f"[donors] {len(DON):,} distinct GT donors; top {len(NAT)} taken, mean score5 of the fragment's "
    f"own first 9 nt {nsc.mean():.2f} (synthetic U1 optimum {U1X}, score5 {s5(U1X):.2f})")
assert all(z[3:5] == "GT" for z in NAT), "a fragment does not carry GT at its donor register"
assert nsc.min() > 8.0, f"a fragment's own donor scores {nsc.min():.2f}; register is wrong"
def mono(s, k): return "".join(np.random.default_rng(k).permutation(list(s)))
SEQ = ([(0, "synthetic", "motif", U1X)] +
       [(j, "synthetic", "mono", mono(U1X, [A.seed, 1, j])) for j in range(10)] +
       [(i, "native", "motif", z) for i, z in enumerate(NAT)] +
       [(i, "native", "mono", mono(z, [A.seed, 2, i])) for i, z in enumerate(NAT)])
log(f"[units] {len(SEQ)} sequences x {len(POS)} fixed positions, no maximum taken")

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
def oh(a): return TBLt[torch.from_numpy(np.ascontiguousarray(a).astype(np.int64)).to(dev)].permute(0,2,1).contiguous()
def fwd(a, bs=1024):
    o = np.empty(len(a), np.float64)
    with torch.inference_mode():
        for k in range(0, len(a), bs): o[k:k+bs] = net(oh(a[k:k+bs])).float().cpu().numpy()
    return o
BGC = X[te[y[te] == 1]]
cx = te[y[te] == 0]; BGX = X[rng.choice(cx, min(A.ncomp, len(cx)), replace=False)]
log(f"[bg] {len(BGC):,} real 3' ends and {len(BGX):,} competitors")

V = {}; rows = []; t0 = time.time()
for bn, B in (("real_canonical", BGC), ("real_competitor", BGX)):
    base = fwd(B)
    for i, cls, arm, s_ in SEQ:
        m = np.array([NT.index(c) for c in s_], np.uint8); Lm = len(m)
        for p_ in POS:
            s0 = p_ + UP - Lm//2
            if s0 < 0 or s0 + Lm > WIN: continue
            q = B.copy(); q[:, s0:s0+Lm] = m
            d = float((fwd(q) - base).mean())
            V[(bn, cls, arm, p_, i)] = d
            rows.append((bn, cls, arm, p_, i, Lm, d))
    log(f"[run] {bn} done  {time.time()-t0:.0f}s")
with open("donor_downstream.tsv", "w") as f:
    f.write("background\tclass\tarm\tposition\tfragment\tlength\tdelta_logit\n")
    for r in rows: f.write("\t".join(str(x) for x in r) + "\n")
log("[out] donor_downstream.tsv")

n = len(NAT)
def vec(bn, arm, p): return np.array([V[(bn, "native", arm, p, i)] for i in range(n)])
for bn in ("real_canonical", "real_competitor"):
    log(f"\n[{bn}] donor implanted AT each position, no maximum taken")
    log(f"  {'pos':>6} | {'synthetic':>10} {'its shuffle':>12} {'diff':>8} | "
        f"{'native':>9} {'its shuffle':>12} {'diff':>8} {'SE':>7} {'z':>6}")
    for p in POS:
        sy = V.get((bn, "synthetic", "motif", p, 0))
        sh = np.mean([V[(bn, "synthetic", "mono", p, j)] for j in range(10)
                      if (bn, "synthetic", "mono", p, j) in V]) if sy is not None else np.nan
        a = vec(bn, "motif", p); b = vec(bn, "mono", p); d = a - b
        se = d.std(ddof=1)/np.sqrt(n)
        log(f"  {p:>6} | {sy:>+10.4f} {sh:>+12.4f} {sy-sh:>+8.4f} | {a.mean():>+9.4f} "
            f"{b.mean():>+12.4f} {d.mean():>+8.4f} {se:>7.4f} {d.mean()/se:>+6.1f}")
    Dn = np.mean([vec(bn, "motif", p) - vec(bn, "mono", p) for p in DNPOS], axis=0)
    se = Dn.std(ddof=1)/np.sqrt(n)
    log(f"  pooled downstream {DNPOS[0]} to {DNPOS[-1]}, native donor vs own shuffle: "
        f"{Dn.mean():+.4f} +/- {se:.4f} [{Dn.mean()-1.96*se:+.4f}, {Dn.mean()+1.96*se:+.4f}] "
        f"z = {Dn.mean()/se:+.1f}")
