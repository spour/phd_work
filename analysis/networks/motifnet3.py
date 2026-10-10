#!/usr/bin/env python3
# keep the performance and use the 3' splice site and u1
import sys, math, time, re, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser()
P.add_argument("--seeds", default="42,43,44")
P.add_argument("--epochs", type=int, default=60)
P.add_argument("--batch", type=int, default=128)
P.add_argument("--lr", type=float, default=1e-3)
P.add_argument("--wd", type=float, default=1e-6)
P.add_argument("--dropout", type=float, default=0.4)
P.add_argument("--patience", type=int, default=12)
P.add_argument("--nacc", type=int, default=16)
P.add_argument("--nu1", type=int, default=16)
P.add_argument("--wacc", type=int, default=45)
P.add_argument("--wu1", type=int, default=15)
P.add_argument("--lam", type=float, default=1.0, help="weight on the motif supervision")
A = P.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
UP, WIN = 2000, 2500
REG = [("far_up", 0, UP-300), ("near_up", UP-300, UP-50), ("site", UP-50, UP+50),
       ("down", UP+50, WIN)]

class Net3(nn.Module):
    def __init__(s, nacc=16, nu1=16, wacc=45, wu1=15, drop=0.4):
        super().__init__()
        # the reported trunk, unchanged
        s.c1 = nn.Conv1d(4, 32, 19, padding=9, bias=False); s.b1 = nn.BatchNorm1d(32)
        s.c2 = nn.Conv1d(32, 64, 7, padding=3, bias=False); s.b2 = nn.BatchNorm1d(64)
        s.c3 = nn.Conv1d(64, 96, 5, padding=2, bias=False); s.b3 = nn.BatchNorm1d(96)
        # the motif branch
        s.ma = nn.Conv1d(4, nacc, wacc, padding=wacc//2, bias=True)
        s.mu = nn.Conv1d(4, nu1, wu1, padding=wu1//2, bias=True)
        nn.init.normal_(s.ma.weight, 0, .05); nn.init.constant_(s.ma.bias, -2.0)
        nn.init.normal_(s.mu.weight, 0, .05); nn.init.constant_(s.mu.bias, -2.0)
        s.bna = nn.BatchNorm1d(nacc); s.bnu = nn.BatchNorm1d(nu1)
        s.nacc, s.nu1 = nacc, nu1
        s.nmot = (nacc+nu1)*len(REG)
        s.do = nn.Dropout(drop)
        s.fc1 = nn.Linear(96*25 + s.nmot, 64); s.fc2 = nn.Linear(64, 1)
        s.h_acc = nn.Linear(nacc, 4)
        s.h_u1 = nn.Linear(nu1*len(REG), 1)
    def parts(s, x):
        h = F.max_pool1d(torch.exp(torch.clamp(s.b1(s.c1(x)), max=8.0)), 10)
        h = F.max_pool1d(F.relu(s.b2(s.c2(h))), 5)
        h = F.max_pool1d(F.relu(s.b3(s.c3(h))), 2)
        trunk = h.flatten(1)
        a = torch.exp(torch.clamp(s.bna(s.ma(x)), max=8.0))
        u = torch.exp(torch.clamp(s.bnu(s.mu(x)), max=8.0))
        Ra = [a[:, :, lo:hi].amax(2) for _, lo, hi in REG]
        Ru = [u[:, :, lo:hi].amax(2) for _, lo, hi in REG]
        return trunk, torch.cat(Ra+Ru, 1), Ra[0], torch.cat(Ru, 1)
    def forward(s, x):
        trunk, mot, af, ua = s.parts(x)
        f = torch.cat([trunk, mot], 1)
        y = s.fc2(s.do(F.relu(s.fc1(s.do(f))))).squeeze(1)
        return y, s.h_acc(af), s.h_u1(ua).squeeze(1)

D = np.load("cnn_data.npz", allow_pickle=True); ok = D["ok"]
X = D["X"][ok]; y = D["y"][ok].astype(np.float32); gene = D["gene"][ok]; split = D["split"][ok]
idx = {s: np.flatnonzero(split == s) for s in ("train","val","test")}
S3 = np.load("acceptor3_scalar.npz")
TGT = np.column_stack([S3["bp"], S3["ppt"], S3["run"], S3["ag"]]).astype(np.float32)
RAW = {"bp": S3["raw_bp"], "ppt": S3["raw_ppt"], "run": S3["raw_run"], "ag": S3["raw_ag"]}
U1z = np.load("u1_scalar.npz")["u1"].astype(np.float32)
U1z = ((U1z-U1z[idx["train"]].mean())/U1z[idx["train"]].std()).astype(np.float32)
RAWU1 = np.load("u1_scalar.npz")["u1"].astype(float)
TBL = np.zeros((256,4), np.float32); TBL[255] = .25
for i in range(4): TBL[i,i] = 1.
TBLt = torch.from_numpy(TBL).to(dev)
Xt = torch.from_numpy(X.astype(np.int64)); yt = torch.from_numpy(y)
Tt = torch.from_numpy(TGT); Ut = torch.from_numpy(U1z)
def bo(ii): return TBLt[Xt[ii].to(dev)].permute(0,2,1).contiguous()
log("  " + "  ".join(f"{s} {len(idx[s]):,}" for s in idx) + f"   lam {A.lam}, acceptor filters "
    f"{A.nacc} of width {A.wacc} (spans -{A.wacc//2} to +{A.wacc//2})")

def wingene(sc, ii):
    g = gene[ii]; yy = y[ii]; by = {}
    for k in range(len(ii)):
        r,c = by.setdefault(g[k],([],[])); (r if yy[k]==1 else c).append(sc[k])
    v=[]
    for r,c in by.values():
        if not r or not c: continue
        r=np.array(r); c=np.array(c)
        v.append(((r[:,None]>c[None,:]).sum()+.5*(r[:,None]==c[None,:]).sum())/(len(r)*len(c)))
    return float(np.mean(v)), len(v)
def auroc(s, yy):
    o=np.argsort(s,kind="mergesort"); r=np.empty(len(s),float); r[o]=np.arange(1,len(s)+1)
    ss=s[o]; i=0
    while i<len(ss):
        j=i
        while j+1<len(ss) and ss[j+1]==ss[i]: j+=1
        if j>i: r[o[i:j+1]]=(i+1+j+1)/2
        i=j+1
    n1=int(yy.sum()); n0=len(yy)-n1
    return (r[yy==1].sum()-n1*(n1+1)/2)/(n1*n0)
def pear(a,b):
    m=np.isfinite(a)&np.isfinite(b); a,b=a[m],b[m]; n=len(a)
    r=float(np.corrcoef(a,b)[0,1]); z=.5*math.log((1+r)/(1-r)); se=1/math.sqrt(n-3)
    return r, math.tanh(z-1.96*se), math.tanh(z+1.96*se)

RES=[]; FILT=[]
for SEED in [int(s) for s in A.seeds.split(",")]:
    torch.manual_seed(SEED); np.random.seed(SEED)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    net = Net3(A.nacc, A.nu1, A.wacc, A.wu1, A.dropout).to(dev)
    npos=float(y[idx["train"]].sum()); nneg=len(idx["train"])-npos
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(nneg/npos, device=dev))
    opt = torch.optim.Adam(net.parameters(), lr=A.lr, weight_decay=A.wd)
    log(f"\n[seed {SEED}] {sum(p.numel() for p in net.parameters()):,} parameters "
        f"(trunk 96x25 = 2400 features + {net.nmot} motif features)")
    def allout(ii, bs=384):
        net.eval(); o=np.empty((len(ii),6), np.float32)
        with torch.no_grad():
            for k in range(0,len(ii),bs):
                j=ii[k:k+bs]; a,b,c = net(bo(j))
                o[k:k+bs,0]=a.float().cpu().numpy()
                o[k:k+bs,1:5]=b.float().cpu().numpy(); o[k:k+bs,5]=c.float().cpu().numpy()
        return o
    best=(-1,-1,None); t0=time.time()
    for ep in range(1, A.epochs+1):
        net.train(); perm=np.random.permutation(idx["train"]); L=[0.,0.,0.]
        for k in range(0,len(perm),A.batch):
            j=perm[k:k+A.batch]
            opt.zero_grad(set_to_none=True)
            out,pa,pu = net(bo(j))
            l1=crit(out, yt[j].to(dev))
            l2=F.mse_loss(pa, Tt[j].to(dev)); l3=F.mse_loss(pu, Ut[j].to(dev))
            (l1 + A.lam*(l2+l3)).backward(); opt.step()
            L[0]+=float(l1)*len(j); L[1]+=float(l2)*len(j); L[2]+=float(l3)*len(j)
        o=allout(idx["val"]); wg,_=wingene(o[:,0], idx["val"])
        if ep%5==0 or ep<=3:
            log(f"  ep {ep:3d} cls {L[0]/len(perm):.4f} acc-mse {L[1]/len(perm):.4f} "
                f"u1-mse {L[2]/len(perm):.4f} val within-gene {wg:.4f}  {time.time()-t0:.0f}s")
        if wg>best[0]: best=(wg,ep,{k:v.detach().cpu().clone() for k,v in net.state_dict().items()})
        elif ep-best[1]>=A.patience:
            log(f"  early stop at {ep}, best epoch {best[1]}"); break
    net.load_state_dict(best[2]); torch.save({"state":best[2],"args":vars(A)}, f"motifnet3_s{SEED}.pt")
    te=idx["test"]; o=allout(te); st=o[:,0]; c=y[te]==1
    wgt,_=wingene(st,te); aut=auroc(st,y[te])
    log(f"[seed {SEED}] TEST within-gene {wgt:.4f}  pooled AUROC {aut:.4f}   "
        f"(reported CNN 0.8211 / 0.8185)")
    hp={}
    for i,nm in enumerate(("bp","ppt","run","ag")):
        r=pear(o[:,1+i], RAW[nm][te]); hp[nm]=r[0]
        log(f"[seed {SEED}]  head learns {nm:<4} from sequence: r = {r[0]:+.4f} [{r[1]:+.4f},{r[2]:+.4f}]")
    ru=pear(o[:,5], RAWU1[te]); hp["u1"]=ru[0]
    log(f"[seed {SEED}]  head learns u1   from sequence: r = {ru[0]:+.4f} [{ru[1]:+.4f},{ru[2]:+.4f}]")
    sc={}
    for nm in ("bp","ppt","run","ag"):
        r=pear(RAW[nm][te][c], st[c]); sc[nm]=r
        log(f"[seed {SEED}]  SCORE vs real {nm:<4} within real 3' ends: r = {r[0]:+.4f} "
            f"[{r[1]:+.4f},{r[2]:+.4f}]")
    ru2=pear(o[:,5], st); ru3=pear(o[:,5][~c], st[~c])
    log(f"[seed {SEED}]  SCORE vs predicted U1: all {ru2[0]:+.4f} [{ru2[1]:+.4f},{ru2[2]:+.4f}], "
        f"within competitors {ru3[0]:+.4f}   (negative = U1 is repressive)")
    RES.append((SEED,wgt,aut,hp["bp"],hp["ppt"],hp["run"],hp["ag"],hp["u1"],
                sc["bp"][0],sc["ppt"][0],sc["run"][0],sc["ag"][0],sc["ag"][1],sc["ag"][2],
                ru2[0],ru3[0],best[1]))
    for tag,conv,n in (("acceptor",net.ma,A.nacc),("u1",net.mu,A.nu1)):
        W=conv.weight.detach().cpu().numpy()
        for f_ in range(n):
            p=np.exp(W[f_]); p/=p.sum(0,keepdims=True)
            ic=float(np.mean(2.+np.sum(np.where(p>0,p*np.log2(np.clip(p,1e-12,None)),0),0)))
            con="".join("ACGT"[i] for i in p.argmax(0))
            FILT.append((SEED,tag,f_,con,round(ic,4),
                         max(len(m.group(0)) for m in re.finditer(r"(.)\1*",con))))
with open("motifnet3_results.tsv","w") as fh:
    fh.write("seed\twithin_gene\tauroc\th_bp\th_ppt\th_run\th_ag\th_u1\ts_bp\ts_ppt\ts_run\t"
             "s_ag\ts_ag_lo\ts_ag_hi\ts_u1_all\ts_u1_comp\tbest_epoch\n")
    for r in RES: fh.write("\t".join(str(x) for x in r)+"\n")
with open("motifnet3_filters.tsv","w") as fh:
    fh.write("seed\tbank\tfilter\tconsensus\tic\tlongest_run\n")
    for r in FILT: fh.write("\t".join(str(x) for x in r)+"\n")
R=np.array([[r[1],r[2],r[3],r[6],r[7],r[11],r[14]] for r in RES])
log(f"\n{'':16}{'within-gene':>12}{'pooled':>9}{'BP head':>9}{'AG head':>9}{'U1 head':>9}"
    f"{'score vs AG':>13}{'score vs U1':>13}")
log(f"{'MotifNet3':<16}{R[:,0].mean():>12.4f}{R[:,1].mean():>9.4f}{R[:,2].mean():>9.4f}"
    f"{R[:,3].mean():>9.4f}{R[:,4].mean():>9.4f}{R[:,5].mean():>13.4f}{R[:,6].mean():>13.4f}")
log(f"{'reported CNN':<16}{0.8211:>12.4f}{0.8185:>9.4f}{'-':>9}{'-':>9}{'-':>9}{0.0259:>13.4f}{'-':>13}")
for tag in ("acceptor","u1"):
    v=[f for f in FILT if f[1]==tag]
    log(f"[filters] {tag:<9} mean IC {np.mean([f[4] for f in v]):.2f} bits, "
        f"{100*np.mean([f[5]>=8 for f in v]):.0f}% homopolymer-like")
log("[out] motifnet3_results.tsv motifnet3_filters.tsv")
