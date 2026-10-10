#!/usr/bin/env python3
# the dose figure's control, redone as a point mutation
import sys, time, argparse
import numpy as np
import torch
from cnn_interp_lib import NT
def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser()
P.add_argument("--model", default="cnn_model.pt")
P.add_argument("--max_copies", type=int, default=7)
P.add_argument("--nscram", type=int, default=5)
P.add_argument("--ncomp", type=int, default=4096)
P.add_argument("--seed", type=int, default=42)
A = P.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
UP, WIN = 2000, 2500
rng = np.random.default_rng(A.seed)
SPACER = "CTCTCT"
UNITS = {"acc_ppt":  ("TTTTTTTTTTTTTCAGG", 14),
         "acc_noC":  ("TTTTTTTTTGTGTGTTTTAGGTT", 18),
         "acc_best": ("TTTCCTCTCTCTCCCCGCAGGTG", 18)}
for k,(s,j) in UNITS.items(): assert s[j:j+2]=="AG", (k, s[j:j+2])
def ag_ko(s,j): return s[:j]+"AC"+s[j+2:]
def t2c(s,j):   return s[:j].replace("T","C")+s[j:]
log("[units] the unit, its AG knocked out by one base, and its tract T converted to C")
for k,(s,j) in UNITS.items():
    log(f"  {k:<9} wt {s}")
    log(f"  {'':9} ag {ag_ko(s,j)}   ({sum(a!=b for a,b in zip(s,ag_ko(s,j)))} base changed)")
    log(f"  {'':9} t2c {t2c(s,j)}   ({sum(a!=b for a,b in zip(s,t2c(s,j)))} bases; pyrimidine "
        f"{sum(c in 'CT' for c in s)/len(s):.2f} -> {sum(c in 'CT' for c in t2c(s,j))/len(t2c(s,j)):.2f})")
def mono(s,k): return "".join(np.random.default_rng(k).permutation(list(s)))
SEQ=[]
for k,(s,j) in UNITS.items():
    for n in range(1, A.max_copies+1):
        for arm, u in (("wt", s), ("ag_ko", ag_ko(s,j)), ("t2c", t2c(s,j))):
            SEQ.append((k, arm, n, (u+SPACER)*(n-1)+u))
        base=(s+SPACER)*(n-1)+s
        for q in range(A.nscram):
            SEQ.append((k,"mono",n, mono(base,[A.seed,sum(map(ord,k)),n,q])))
log(f"[seq] {len(SEQ)} constructs")
from cnn_model_def import Net
ck=torch.load(A.model,map_location=dev,weights_only=False)
net=Net(ck["args"]["dropout"]).to(dev); net.load_state_dict(ck["state"]); net.eval()
for p in net.parameters(): p.requires_grad_(False)
D=np.load("cnn_data.npz",allow_pickle=True); ok=D["ok"]
X=D["X"][ok]; y=D["y"][ok]
te=np.flatnonzero((D["split"][ok]=="test")&(X.max(1)<=3))
TBL=np.zeros((256,4),np.float32); TBL[255]=.25
for i in range(4): TBL[i,i]=1.
TBLt=torch.from_numpy(TBL).to(dev)
def oh(a): return TBLt[torch.from_numpy(np.ascontiguousarray(a).astype(np.int64)).to(dev)].permute(0,2,1).contiguous()
def fwd(a,bs=1024):
    o=np.empty(len(a),np.float64)
    with torch.inference_mode():
        for k in range(0,len(a),bs): o[k:k+bs]=net(oh(a[k:k+bs])).float().cpu().numpy()
    return o
BGC=X[te[y[te]==1]]; cx=te[y[te]==0]
BGX=X[rng.choice(cx,min(A.ncomp,len(cx)),replace=False)]
POS=sorted(set(list(range(-300,101,10))+list(range(-UP+100,500,100))))
log(f"[bg] {len(BGC):,} real 3' ends and {len(BGX):,} competitors; {len(POS)} positions")
V={}; rows=[]; t0=time.time()
for bn,B in (("real_canonical",BGC),("real_competitor",BGX)):
    base=fwd(B)
    for k,arm,n,s_ in SEQ:
        m=np.array([NT.index(c) for c in s_],np.uint8); best=-1e9
        for p_ in POS:
            s0=p_+UP-len(m)//2
            if s0<0 or s0+len(m)>WIN: continue
            q=B.copy(); q[:,s0:s0+len(m)]=m
            best=max(best,float((fwd(q)-base).mean()))
        V.setdefault((bn,k,arm,n),[]).append(best)
        rows.append((bn,k,arm,n,len(s_),best))
    log(f"[run] {bn} done  {time.time()-t0:.0f}s")
with open("acceptor_dose_pointmut.tsv","w") as f:
    f.write("background\tunit\tarm\tn_copies\tlength\tbest_delta\n")
    for r in rows: f.write("\t".join(str(x) for x in r)+"\n")
log("[out] acceptor_dose_pointmut.tsv")
for bn in ("real_canonical","real_competitor"):
    log(f"\n[{bn}]  n | raw | vs each control   (positive = the intact unit scores HIGHER)")
    log(f"  {'unit':<9}{'n':>3}{'RAW':>9}{'- shuffle':>11}{'- AG knockout':>15}{'- T to C':>10}")
    for k in UNITS:
        for n in range(1,A.max_copies+1):
            w=V[(bn,k,'wt',n)][0]
            log(f"  {k:<9}{n:>3}{w:>+9.3f}"
                f"{w-np.mean(V[(bn,k,'mono',n)]):>+11.3f}"
                f"{w-V[(bn,k,'ag_ko',n)][0]:>+15.3f}"
                f"{w-V[(bn,k,'t2c',n)][0]:>+10.3f}")
