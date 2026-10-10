#!/usr/bin/env python3
# v1 removed composition and left nothing in its place
import sys, csv, math, time, re, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser()
P.add_argument("--seeds", default="42,43,44")
P.add_argument("--epochs", type=int, default=80)
P.add_argument("--batch", type=int, default=128)
P.add_argument("--lr", type=float, default=1e-3)
P.add_argument("--wd", type=float, default=1e-6)
P.add_argument("--dropout", type=float, default=0.3)
P.add_argument("--patience", type=int, default=15)
P.add_argument("--nbank", type=int, default=16)
P.add_argument("--width", type=int, default=23)
P.add_argument("--lam_acc", type=float, default=1.0)
P.add_argument("--lam_u1", type=float, default=1.0)
P.add_argument("--adv", type=float, default=0.2)
A = P.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
UP, WIN = 2000, 2500
REGIONS = [("far_up", 0, UP-300), ("near_up", UP-300, UP-50),
           ("site", UP-50, UP+50), ("down", UP+50, WIN)]
NB = A.nbank
def load_pwm(p):
    R = list(csv.DictReader(open(p), delimiter="\t"))
    return np.array([[float(r[c]) for c in "ACGT"] for r in R]).T
def lo(p, eps=1e-3):
    p = np.clip(p, eps, None); p = p/p.sum(0, keepdims=True); return np.log2(p/0.25)
ACC_PWM, U1_PWM = lo(load_pwm("me3_pwm.tsv")), lo(load_pwm("u1_pwm.tsv"))

class GradRev(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, l): ctx.l = l; return x.view_as(x)
    @staticmethod
    def backward(ctx, g): return -ctx.l*g, None

class MotifNet2(nn.Module):
    def __init__(s, nb=16, w=23, drop=0.3):
        super().__init__()
        s.nb = nb; s.w = w; s.nf = 3*nb
        s.conv = nn.Conv1d(4, s.nf, w, padding=w//2, bias=True)
        nn.init.normal_(s.conv.weight, 0, 0.05); nn.init.constant_(s.conv.bias, -2.0)
        with torch.no_grad():
            for i in range(nb):
                m = ACC_PWM[:, :w] if ACC_PWM.shape[1] >= w else np.pad(ACC_PWM,((0,0),(0,w-ACC_PWM.shape[1])))
                s.conv.weight[i] = torch.tensor(m*0.5, dtype=torch.float32)+0.05*torch.randn(4,w)
            for i in range(nb, 2*nb):
                m = np.zeros((4,w)); o = (w-U1_PWM.shape[1])//2
                m[:, o:o+U1_PWM.shape[1]] = U1_PWM
                s.conv.weight[i] = torch.tensor(m*0.5, dtype=torch.float32)+0.05*torch.randn(4,w)
        s.bn = nn.BatchNorm1d(s.nf)
        s.nfeat = s.nf*len(REGIONS)
        s.do = nn.Dropout(drop)
        s.fc1 = nn.Linear(s.nfeat, 64); s.fc2 = nn.Linear(64, 1)
        s.head_acc = nn.Linear(nb, 1)
        s.head_u1 = nn.Linear(nb*len(REGIONS), 1)
        s.adv = nn.Sequential(nn.Linear(s.nfeat,32), nn.ReLU(), nn.Linear(32,4))
    def parts(s, x):
        h = torch.exp(torch.clamp(s.bn(s.conv(x)), max=8.0))
        R = [h[:, :, a:b].amax(2) for _, a, b in REGIONS]
        f = torch.cat(R, 1)
        acc_far = R[0][:, :s.nb]
        u1_all = torch.cat([r[:, s.nb:2*s.nb] for r in R], 1)
        return f, acc_far, u1_all
    def forward(s, x, lam=0.0):
        f, af, uu = s.parts(x)
        y = s.fc2(s.do(F.relu(s.fc1(s.do(f))))).squeeze(1)
        return (y, s.head_acc(af).squeeze(1), s.head_u1(uu).squeeze(1),
                s.adv(GradRev.apply(f, lam)) if lam > 0 else None)

D = np.load("cnn_data.npz", allow_pickle=True); ok = D["ok"]
X = D["X"][ok]; y = D["y"][ok].astype(np.float32); gene = D["gene"][ok]; split = D["split"][ok]
idx = {s: np.flatnonzero(split == s) for s in ("train","val","test")}
ACCS = np.load("acceptor_scalar.npz")["acc"].astype(np.float32)
U1S = np.load("u1_scalar.npz")["u1"].astype(np.float32)
za = lambda v: ((v-v[idx["train"]].mean())/v[idx["train"]].std()).astype(np.float32)
ACCz, U1z = za(ACCS), za(U1S)
COMP = np.stack([(X==i).mean(1) for i in range(4)],1).astype(np.float32)
COMPz = ((COMP-COMP[idx["train"]].mean(0))/COMP[idx["train"]].std(0)).astype(np.float32)
TBL = np.zeros((256,4), np.float32); TBL[255] = 0.25
for i in range(4): TBL[i,i] = 1.0
TBLt = torch.from_numpy(TBL).to(dev)
Xt = torch.from_numpy(X.astype(np.int64)); yt = torch.from_numpy(y)
At, Ut, Ct = torch.from_numpy(ACCz), torch.from_numpy(U1z), torch.from_numpy(COMPz)
def bo(ii): return TBLt[Xt[ii].to(dev)].permute(0,2,1).contiguous()
log("  " + "  ".join(f"{s} {len(idx[s]):,}" for s in idx) +
    f"   lam_acc {A.lam_acc}  lam_u1 {A.lam_u1}  adv {A.adv}")

def wingene(sc, ii):
    g = gene[ii]; yy = y[ii]; by = {}
    for k in range(len(ii)):
        r,c = by.setdefault(g[k],([],[])); (r if yy[k]==1 else c).append(sc[k])
    v=[]
    for r,c in by.values():
        if not r or not c: continue
        r=np.array(r); c=np.array(c)
        v.append(((r[:,None]>c[None,:]).sum()+0.5*(r[:,None]==c[None,:]).sum())/(len(r)*len(c)))
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
    r=float(np.corrcoef(a,b)[0,1]); z=0.5*math.log((1+r)/(1-r)); se=1/math.sqrt(n-3)
    return r, math.tanh(z-1.96*se), math.tanh(z+1.96*se)

RES=[]; FILT=[]
for SEED in [int(s) for s in A.seeds.split(",")]:
    torch.manual_seed(SEED); np.random.seed(SEED)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    net = MotifNet2(NB, A.width, A.dropout).to(dev)
    npos=float(y[idx["train"]].sum()); nneg=len(idx["train"])-npos
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(nneg/npos, device=dev))
    opt = torch.optim.Adam(net.parameters(), lr=A.lr, weight_decay=A.wd)
    log(f"\n[seed {SEED}] {sum(p.numel() for p in net.parameters()):,} parameters, "
        f"{net.nf} filters in three banks of {NB}")
    def allout(ii, bs=512):
        net.eval(); o=np.empty((len(ii),3), np.float32)
        with torch.no_grad():
            for k in range(0,len(ii),bs):
                j=ii[k:k+bs]; a,b,c,_ = net(bo(j))
                o[k:k+bs,0]=a.float().cpu().numpy(); o[k:k+bs,1]=b.float().cpu().numpy()
                o[k:k+bs,2]=c.float().cpu().numpy()
        return o
    best=(-1,-1,None); t0=time.time()
    for ep in range(1, A.epochs+1):
        lam = A.adv*min(1.0, ep/10)
        net.train(); perm=np.random.permutation(idx["train"]); L=[0.0,0.0,0.0]
        for k in range(0,len(perm),A.batch):
            j=perm[k:k+A.batch]
            opt.zero_grad(set_to_none=True)
            out,pa,pu,pc = net(bo(j), lam)
            l1=crit(out, yt[j].to(dev))
            l2=F.mse_loss(pa, At[j].to(dev)); l3=F.mse_loss(pu, Ut[j].to(dev))
            l4=F.mse_loss(pc, Ct[j].to(dev)) if pc is not None else torch.zeros((),device=dev)
            (l1 + A.lam_acc*l2 + A.lam_u1*l3 + l4).backward(); opt.step()
            L[0]+=float(l1)*len(j); L[1]+=float(l2)*len(j); L[2]+=float(l3)*len(j)
        o=allout(idx["val"]); wg,ng = wingene(o[:,0], idx["val"])
        if ep%10==0 or ep<=3:
            log(f"  ep {ep:3d} cls {L[0]/len(perm):.4f} acc-mse {L[1]/len(perm):.4f} "
                f"u1-mse {L[2]/len(perm):.4f} val within-gene {wg:.4f}  {time.time()-t0:.0f}s")
        if wg>best[0]: best=(wg,ep,{k:v.detach().cpu().clone() for k,v in net.state_dict().items()})
        elif ep-best[1]>=A.patience:
            log(f"  early stop at {ep}, best epoch {best[1]}"); break
    net.load_state_dict(best[2]); torch.save({"state":best[2],"args":vars(A)}, f"motifnet2_s{SEED}.pt")
    te=idx["test"]; o=allout(te); st=o[:,0]
    wgt,_=wingene(st,te); aut=auroc(st,y[te]); c=y[te]==1
    r_head_a = pear(o[:,1], ACCz[te]); r_head_u = pear(o[:,2], U1z[te])
    r_sc_a = pear(ACCS[te][c], st[c]); r_sc_all = pear(ACCS[te], st)
    wu = net.fc1.weight.detach().cpu().numpy()
    u1cols = np.concatenate([np.arange(r*net.nf+NB, r*net.nf+2*NB) for r in range(len(REGIONS))])
    acccols = np.concatenate([np.arange(r*net.nf, r*net.nf+NB) for r in range(len(REGIONS))])
    log(f"[seed {SEED}] TEST within-gene {wgt:.4f}  pooled AUROC {aut:.4f}")
    log(f"[seed {SEED}] 1 the banks learned the motifs FROM SEQUENCE: acceptor head vs real "
        f"score3 r = {r_head_a[0]:+.4f} [{r_head_a[1]:+.4f},{r_head_a[2]:+.4f}];  "
        f"U1 head vs real u1_dw_up r = {r_head_u[0]:+.4f} [{r_head_u[1]:+.4f},{r_head_u[2]:+.4f}]")
    log(f"[seed {SEED}] 2 classifier score vs REAL acceptor strength: within real 3' ends "
        f"r = {r_sc_a[0]:+.4f} [{r_sc_a[1]:+.4f},{r_sc_a[2]:+.4f}]  (all peaks {r_sc_all[0]:+.4f})")
    log(f"[seed {SEED}] 3 mean first-layer weight on the U1 bank {wu[:,u1cols].mean():+.5f} "
        f"vs the acceptor bank {wu[:,acccols].mean():+.5f}")
    RES.append((SEED,wgt,aut,r_head_a[0],r_head_u[0],r_sc_a[0],r_sc_a[1],r_sc_a[2],
                float(wu[:,u1cols].mean()), float(wu[:,acccols].mean()), best[1]))
    W=net.conv.weight.detach().cpu().numpy()
    for f_ in range(W.shape[0]):
        p=np.exp(W[f_]); p/=p.sum(0,keepdims=True)
        ic=float(np.mean(2.0+np.sum(np.where(p>0,p*np.log2(np.clip(p,1e-12,None)),0),0)))
        con="".join("ACGT"[i] for i in p.argmax(0))
        run=max(len(m.group(0)) for m in re.finditer(r"(.)\1*", con))
        FILT.append((SEED,f_,con,round(ic,4),run,
                     "acceptor" if f_<NB else ("u1" if f_<2*NB else "free")))
with open("motifnet2_results.tsv","w") as fh:
    fh.write("seed\ttest_within_gene\ttest_auroc\tr_head_acc\tr_head_u1\tr_score_acc_real\tlo\thi\t"
             "w_u1_bank\tw_acc_bank\tbest_epoch\n")
    for r in RES: fh.write("\t".join(str(x) for x in r)+"\n")
with open("motifnet2_filters.tsv","w") as fh:
    fh.write("seed\tfilter\tconsensus\tic_per_col\tlongest_run\tbank\n")
    for r in FILT: fh.write("\t".join(str(x) for x in r)+"\n")
R=np.array([[r[1],r[2],r[3],r[4],r[5]] for r in RES])
log(f"\n{'':16}{'within-gene':>12}{'pooled':>9}{'acc head':>10}{'U1 head':>9}{'score vs acceptor':>19}")
log(f"{'MotifNet2':<16}{R[:,0].mean():>12.4f}{R[:,1].mean():>9.4f}{R[:,2].mean():>10.4f}"
    f"{R[:,3].mean():>9.4f}{R[:,4].mean():>19.4f}")
log(f"{'MotifNet v1':<16}{0.7051:>12.4f}{0.6730:>9.4f}{'-':>10}{'-':>9}{0.0199:>19.4f}")
log(f"{'reported CNN':<16}{0.8211:>12.4f}{0.8185:>9.4f}{'-':>10}{'-':>9}{0.0259:>19.4f}")
for bank in ("acceptor","u1","free"):
    v=[f for f in FILT if f[5]==bank]
    log(f"[filters] {bank:<9} mean IC {np.mean([f[3] for f in v]):.2f} bits, "
        f"median longest run {np.median([f[4] for f in v]):.0f}, "
        f"{100*np.mean([f[4]>=8 for f in v]):.0f}% homopolymer-like")
log("[out] motifnet2_results.tsv motifnet2_filters.tsv")
