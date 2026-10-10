#!/usr/bin/env python3
# disrupt the acceptor by point mutation instead of by shuffling
import sys, time, argparse
import numpy as np
import torch
from cnn_interp_lib import NT

def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser()
P.add_argument("--model", default="cnn_model.pt")
P.add_argument("--nfrag", type=int, default=150)
P.add_argument("--ncomp", type=int, default=4096)
P.add_argument("--seed", type=int, default=42)
A = P.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
UP, WIN, LFR = 2000, 2500, 120
POS = [-800, -400, -200, 0]
rng = np.random.default_rng(A.seed)
FA = "/scratch/spour98/jobs/hg38.fa"
GTF = ("/scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/score_genome/"
       "GRCh38.gencode.v43.annotation.gtf")
FAI = {}
for line in open(FA+".fai"):
    n, ln, off, lb, wb = line.split()[:5]; FAI[n] = (int(ln), int(off), int(lb), int(wb))
fh = open(FA, "rb")
def fetch(c, s, e):
    if c not in FAI: return None
    ln, off, lb, wb = FAI[c]
    if s < 0 or e > ln or e <= s: return None
    fh.seek(off + s//lb*wb + s%lb)
    return fh.read((e-s)+(e-s)//lb+2).decode().replace("\n","")[:e-s].upper()
def rc(s): return s.translate(str.maketrans("ACGT","TGCA"))[::-1]
from maxentpy import maxent
from maxentpy.maxent import load_matrix3
M3 = load_matrix3()
def s3(s):
    try: return float(maxent.score3(s, matrix=M3))
    except Exception: return -99.0
TX = {}
with open(GTF) as g:
    for line in g:
        if line[0]=="#": continue
        f = line.split("\t")
        if len(f)<9 or f[2]!="exon" or 'transcript_type "protein_coding"' not in f[8]: continue
        i = f[8].find('transcript_id "'); tid = f[8][i+15:f[8].find('"', i+15)]
        TX.setdefault(tid,(f[0],f[6],[]))[2].append((int(f[3]),int(f[4])))
ACC = []; seen = set()
for tid,(c,st,ex) in TX.items():
    if c not in FAI or len(ex)<2: continue
    ex.sort()
    for k in range(len(ex)-1):
        if ex[k+1][0]-1-(ex[k][1]+1)+1 < 300: continue
        key = (c,st,ex[k+1][0] if st=="+" else ex[k][1])
        if key in seen: continue
        s = (fetch(c,ex[k+1][0]-21,ex[k+1][0]+2) if st=="+"
             else (lambda z: rc(z) if z else None)(fetch(c,ex[k][1]-3,ex[k][1]+20)))
        if not s or len(s)!=23 or set(s)-set("ACGT"): continue
        seen.add(key)
        if s[18:20]!="AG": continue
        ACC.append((c,st,key[2],s3(s)))
ACC.sort(key=lambda a:-a[3])
FR = []
for c,st,pos,sc in ACC:
    z = (fetch(c,pos-1-(LFR-3),pos+2) if st=="+"
         else (lambda q: rc(q) if q else None)(fetch(c,pos-3,pos-3+LFR)))
    if z and len(z)==LFR and not (set(z)-set("ACGT")): FR.append(z)
    if len(FR)>=A.nfrag: break
AGPOS = LFR-5
for z in FR: assert z[AGPOS:AGPOS+2]=="AG", z[-8:]
log(f"[frag] {len(FR)} real 120 nt acceptors; the invariant AG is at fragment positions "
    f"{AGPOS},{AGPOS+1} with three exonic bases after it, mean terminal score3 "
    f"{np.mean([s3(z[-23:]) for z in FR]):.2f}")
def mut(z, kind, i):
    j = AGPOS
    if kind=="wt": return z
    if kind=="ag_to_ac": return z[:j]+"AC"+z[j+2:]
    if kind=="ag_to_tg": return z[:j]+"TG"+z[j+2:]
    if kind=="t_to_c":
        t = z[j-20:j].replace("T","C"); return z[:j-20]+t+z[j:]
    if kind=="reversed": return z[::-1]
    if kind=="mono": return "".join(np.random.default_rng([A.seed,7,i]).permutation(list(z)))
KINDS = ["wt","ag_to_ac","ag_to_tg","t_to_c","reversed","mono"]
SEQ = [(i,k,mut(z,k,i)) for i,z in enumerate(FR) for k in KINDS]
w0 = np.mean([s3(mut(z,"wt",i)[-23:]) for i,z in enumerate(FR)])
for k in KINDS[1:]:
    v = np.mean([s3(mut(z,k,i)[-23:]) for i,z in enumerate(FR)])
    d = np.mean([sum(a!=b for a,b in zip(mut(z,"wt",i), mut(z,k,i))) for i,z in enumerate(FR)])
    log(f"[check] {k:<9} score3 of the fragment's own acceptor {w0:>6.2f} -> {v:>7.2f}   "
        f"bases changed {d:>5.1f} of {LFR}")
from cnn_model_def import Net
ck = torch.load(A.model, map_location=dev, weights_only=False)
net = Net(ck["args"]["dropout"]).to(dev); net.load_state_dict(ck["state"]); net.eval()
for p in net.parameters(): p.requires_grad_(False)
D = np.load("cnn_data.npz", allow_pickle=True); ok = D["ok"]
X = D["X"][ok]; y = D["y"][ok]
te = np.flatnonzero((D["split"][ok]=="test") & (X.max(1)<=3))
TBL = np.zeros((256,4), np.float32); TBL[255]=.25
for i in range(4): TBL[i,i]=1.
TBLt = torch.from_numpy(TBL).to(dev)
def oh(a): return TBLt[torch.from_numpy(np.ascontiguousarray(a).astype(np.int64)).to(dev)].permute(0,2,1).contiguous()
def fwd(a, bs=1024):
    o = np.empty(len(a), np.float64)
    with torch.inference_mode():
        for k in range(0,len(a),bs): o[k:k+bs] = net(oh(a[k:k+bs])).float().cpu().numpy()
    return o
BGC = X[te[y[te]==1]]
cx = te[y[te]==0]; BGX = X[rng.choice(cx, min(A.ncomp,len(cx)), replace=False)]
log(f"[bg] {len(BGC):,} real 3' ends and {len(BGX):,} competitors; "
    f"{len(SEQ):,} sequences x {len(POS)} fixed positions")
V = {}; rows = []; t0 = time.time()
for bn,B in (("real_canonical",BGC),("real_competitor",BGX)):
    base = fwd(B)
    for i,k,s_ in SEQ:
        m = np.array([NT.index(ch) for ch in s_], np.uint8)
        for p_ in POS:
            s0 = p_+UP-LFR//2
            q = B.copy(); q[:, s0:s0+LFR] = m
            d = float((fwd(q)-base).mean())
            V[(bn,k,p_,i)] = d; rows.append((bn,k,p_,i,d))
    log(f"[run] {bn} done  {time.time()-t0:.0f}s")
with open("acceptor_pointmut.tsv","w") as f:
    f.write("background\tvariant\tposition\tfragment\tdelta_logit\n")
    for r in rows: f.write("\t".join(str(x) for x in r)+"\n")
log("[out] acceptor_pointmut.tsv")
n = len(FR)
def vec(bn,k,p): return np.array([V[(bn,k,p,i)] for i in range(n)])
for bn in ("real_canonical","real_competitor"):
    log(f"\n[{bn}] each variant against the WILD TYPE fragment, paired within fragment (n={n})")
    log(f"  {'variant':<10}" + "".join(f"{p:>+12}" for p in POS) + f"{'pooled':>12}{'SE':>8}{'z':>7}")
    for k in KINDS[1:]:
        per = [vec(bn,k,p)-vec(bn,"wt",p) for p in POS]
        pooled = np.mean(per, axis=0); se = pooled.std(ddof=1)/np.sqrt(n)
        log(f"  {k:<10}" + "".join(f"{d.mean():>+12.4f}" for d in per) +
            f"{pooled.mean():>+12.4f}{se:>8.4f}{pooled.mean()/se:>+7.1f}")
    log(f"  {'(wt raw)':<10}" + "".join(f"{vec(bn,'wt',p).mean():>+12.4f}" for p in POS))
