#!/usr/bin/env python3
# the deciding control
import sys, time, argparse
import numpy as np
import torch
from cnn_interp_lib import NT

def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser()
P.add_argument("--model", default="cnn_model.pt")
P.add_argument("--npair", type=int, default=50)
P.add_argument("--ncomp", type=int, default=4096)
P.add_argument("--seed", type=int, default=42)
A = P.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
UP, WIN, LFR = 2000, 2500, 120
POS = [-1500, -1000, -800, -600, -400, -200, -50, 0]
UPPOS = [p for p in POS if p <= -400]
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
        istart, iend = ex[k][1]+1, ex[k+1][0]-1
        if iend - istart + 1 < 300: continue
        key = (c, st, ex[k+1][0] if st == "+" else ex[k][1])
        if key in seen: continue
        s = (fetch(c, ex[k+1][0]-21, ex[k+1][0]+2) if st == "+"
             else (lambda z: rc(z) if z else None)(fetch(c, ex[k][1]-3, ex[k][1]+20)))
        if not s or len(s) != 23 or set(s)-set("ACGT"): continue
        seen.add(key)
        if s[18:20] != "AG": continue
        ACC.append((c, st, key[2], istart, iend, s3(s)))
ACC.sort(key=lambda a: -a[5])
def acc_frag(a):
    c, st, pos, _, _, _ = a
    return (fetch(c, pos-1-(LFR-3), pos+2) if st == "+"
            else (lambda z: rc(z) if z else None)(fetch(c, pos-3, pos-3+LFR)))
def ctl_frag(a):
    c, st, _, istart, iend, _ = a
    m = (istart+iend)//2
    if m-istart < 200 or iend-m < 200: return None
    z = fetch(c, m-LFR//2, m-LFR//2+LFR)
    return z if st == "+" else (rc(z) if z else None)
def okseq(z): return z and len(z) == LFR and not (set(z)-set("ACGT"))

PAIRS = []
for a in ACC:
    x, y = acc_frag(a), ctl_frag(a)
    if okseq(x) and okseq(y): PAIRS.append((x, y, a[5]))
    if len(PAIRS) >= A.npair: break
log(f"[pairs] {len(PAIRS)} introns contributing BOTH an acceptor fragment and an acceptor-free "
    f"control, paired within intron")
ia = np.array([s3(x[-23:]) for x, _, _ in PAIRS])
ic = np.array([max(s3(y[i:i+23]) for i in range(LFR-22)) for _, y, _ in PAIRS])
log(f"[check] terminal score3 of the acceptor fragments: median {np.median(ia):.2f}; "
    f"best score3 ANYWHERE in the control fragments: median {np.median(ic):.2f}, "
    f"max {ic.max():.2f}, {100*np.mean(ic>=14):.0f}% reach 14")

def mono(s, k): return "".join(np.random.default_rng(k).permutation(list(s)))
SEQ = []
for i, (x, y, _) in enumerate(PAIRS):
    SEQ.append((i, "acceptor", "motif", x)); SEQ.append((i, "acceptor", "mono", mono(x, [A.seed, 1, i])))
    SEQ.append((i, "intron",   "motif", y)); SEQ.append((i, "intron",   "mono", mono(y, [A.seed, 2, i])))
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
def oh(a): return TBLt[torch.from_numpy(a.astype(np.int64)).to(dev)].permute(0, 2, 1).contiguous()
def fwd(a, bs=1024):
    o = []
    with torch.no_grad():
        for k in range(0, len(a), bs): o.append(net(oh(a[k:k+bs])).float().cpu().numpy())
    return np.concatenate(o)
BGC = X[te[y[te] == 1]]
cx = te[y[te] == 0]; BGX = X[rng.choice(cx, min(A.ncomp, len(cx)), replace=False)]
log(f"[bg] {len(BGC):,} real 3' ends (all) and {len(BGX):,} competitors")

V = {}; rows = []; t0 = time.time()
for bn, B in (("real_canonical", BGC), ("real_competitor", BGX)):
    base = fwd(B)
    for i, cls, arm, s_ in SEQ:
        m = np.array([NT.index(c) for c in s_], np.uint8)
        for p_ in POS:
            s0 = p_ + UP - LFR // 2
            q = B.copy(); q[:, s0:s0+LFR] = m
            d = float((fwd(q) - base).mean())
            V[(bn, cls, arm, p_, i)] = d
            rows.append((bn, cls, arm, p_, i, d))
    log(f"[run] {bn} done  {time.time()-t0:.0f}s")
with open("acceptor_vs_intron_positional.tsv", "w") as f:
    f.write("background\tclass\tarm\tposition\tfragment\tdelta_logit\n")
    for r in rows: f.write("\t".join(str(x) for x in r) + "\n")
log("[out] acceptor_vs_intron_positional.tsv")

n = len(PAIRS)
def vec(bn, cls, arm, p): return np.array([V[(bn, cls, arm, p, i)] for i in range(n)])
def stat(d): 
    se = d.std(ddof=1)/np.sqrt(len(d)); return d.mean(), se, d.mean()/se
for bn in ("real_canonical", "real_competitor"):
    log(f"\n[{bn}] each class against a shuffle of its own bases, at each fixed position")
    log(f"  {'pos':>6} | {'acceptor':>9} {'SE':>7} {'z':>6} | {'intron':>9} {'SE':>7} {'z':>6} "
        f"| {'acc - intron':>13} {'SE':>7} {'z':>6}")
    for p in POS:
        da = vec(bn,"acceptor","motif",p) - vec(bn,"acceptor","mono",p)
        di = vec(bn,"intron","motif",p)   - vec(bn,"intron","mono",p)
        dd = da - di
        (ma,sa,za),(mi,si,zi),(md,sd,zd) = stat(da), stat(di), stat(dd)
        log(f"  {p:>6} | {ma:>+9.4f} {sa:>7.4f} {za:>+6.1f} | {mi:>+9.4f} {si:>7.4f} {zi:>+6.1f} "
            f"| {md:>+13.4f} {sd:>7.4f} {zd:>+6.1f}")
    A_ = np.mean([vec(bn,"acceptor","motif",p)-vec(bn,"acceptor","mono",p) for p in UPPOS], axis=0)
    I_ = np.mean([vec(bn,"intron","motif",p)-vec(bn,"intron","mono",p) for p in UPPOS], axis=0)
    Dd = A_ - I_
    for nm, d in (("acceptor vs own shuffle", A_), ("intron   vs own shuffle", I_),
                  ("acceptor MINUS intron  ", Dd)):
        m_, s_, z_ = stat(d)
        log(f"  pooled {UPPOS[0]} to {UPPOS[-1]}, {nm}: {m_:+.4f} +/- {s_:.4f} "
            f"[{m_-1.96*s_:+.4f}, {m_+1.96*s_:+.4f}]  z = {z_:+.1f}"
            f"{'' if abs(z_) >= 2 else '   NOT significant'}")
    log(f"  acceptor beats intron in {int((Dd > 0).sum())} of {len(Dd)} fragments")
