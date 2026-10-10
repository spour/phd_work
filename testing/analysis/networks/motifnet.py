#!/usr/bin/env python3
# a cnn built to learn the acceptor and u1 as motifs from sequence
import sys, csv, math, time, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser()
P.add_argument("--seeds", default="42,43,44")
P.add_argument("--epochs", type=int, default=60)
P.add_argument("--batch", type=int, default=128)
P.add_argument("--lr", type=float, default=1e-3)
P.add_argument("--wd", type=float, default=1e-6)
P.add_argument("--dropout", type=float, default=0.3)
P.add_argument("--patience", type=int, default=12)
P.add_argument("--nf", type=int, default=48)
P.add_argument("--width", type=int, default=23)
P.add_argument("--adv", type=float, default=1.0, help="composition adversary weight")
P.add_argument("--tag", default="motifnet")
A = P.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
UP, WIN = 2000, 2500
REGIONS = [("far_up", 0, UP-300), ("near_up", UP-300, UP-50),
           ("site", UP-50, UP+50), ("down", UP+50, WIN)]
log(f"[env] device {dev}; regions " + ", ".join(f"{n}[{a-UP},{b-UP}]" for n, a, b in REGIONS))

def load_pwm(path, cols=("A","C","G","T")):
    R = list(csv.DictReader(open(path), delimiter="\t"))
    return np.array([[float(r[c]) for c in cols] for r in R]).T
def logodds(p, eps=1e-3):
    p = np.clip(p, eps, None); p = p/p.sum(0, keepdims=True)
    return np.log2(p/0.25)
ACC_PWM = logodds(load_pwm("me3_pwm.tsv"))
U1_PWM = logodds(load_pwm("u1_pwm.tsv"))
log(f"[init] acceptor matrix {ACC_PWM.shape[1]} columns, U1 matrix {U1_PWM.shape[1]} columns")

class GradRev(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, l): ctx.l = l; return x.view_as(x)
    @staticmethod
    def backward(ctx, g): return -ctx.l * g, None

class MotifNet(nn.Module):
    def __init__(s, nf=48, w=23, drop=0.3):
        super().__init__()
        s.w = w
        s.conv = nn.Conv1d(4, nf, w, padding=w//2, bias=True)
        nn.init.normal_(s.conv.weight, 0, 0.05); nn.init.constant_(s.conv.bias, -2.0)
        with torch.no_grad():
            for i in range(8):
                m = ACC_PWM[:, :w] if ACC_PWM.shape[1] >= w else np.pad(
                    ACC_PWM, ((0,0),(0, w-ACC_PWM.shape[1])))
                s.conv.weight[i] = torch.tensor(m*0.5, dtype=torch.float32) + \
                    0.05*torch.randn(4, w)
            for i in range(8, 16):
                m = np.zeros((4, w)); o = (w-U1_PWM.shape[1])//2
                m[:, o:o+U1_PWM.shape[1]] = U1_PWM
                s.conv.weight[i] = torch.tensor(m*0.5, dtype=torch.float32) + \
                    0.05*torch.randn(4, w)
        s.bn = nn.BatchNorm1d(nf)
        s.nf = nf; s.nfeat = nf*len(REGIONS)
        s.do = nn.Dropout(drop)
        s.fc1 = nn.Linear(s.nfeat, 64); s.fc2 = nn.Linear(64, 1)
        s.adv = nn.Sequential(nn.Linear(s.nfeat, 32), nn.ReLU(), nn.Linear(32, 4))
    def feats(s, x):
        h = torch.exp(torch.clamp(s.bn(s.conv(x)), max=8.0))
        return torch.cat([h[:, :, a:b].amax(2) for _, a, b in REGIONS], 1)
    def forward(s, x, lam=0.0):
        f = s.feats(x)
        y = s.fc2(s.do(F.relu(s.fc1(s.do(f))))).squeeze(1)
        c = s.adv(GradRev.apply(f, lam)) if lam > 0 else None
        return y, c

D = np.load("cnn_data.npz", allow_pickle=True); ok = D["ok"]
X = D["X"][ok]; y = D["y"][ok].astype(np.float32); gene = D["gene"][ok]; split = D["split"][ok]
idx = {s: np.flatnonzero(split == s) for s in ("train","val","test")}
COMP = np.stack([(X == i).mean(1) for i in range(4)], 1).astype(np.float32)
cm, cs = COMP[idx["train"]].mean(0), COMP[idx["train"]].std(0)
COMPz = ((COMP-cm)/cs).astype(np.float32)
TBL = np.zeros((256,4), np.float32); TBL[255] = 0.25
for i in range(4): TBL[i,i] = 1.0
TBLt = torch.from_numpy(TBL).to(dev)
Xt = torch.from_numpy(X.astype(np.int64)); yt = torch.from_numpy(y)
Ct = torch.from_numpy(COMPz)
def bo(ii): return TBLt[Xt[ii].to(dev)].permute(0,2,1).contiguous()
log("  " + "  ".join(f"{s} {len(idx[s]):,} ({int(y[idx[s]].sum())} real)" for s in idx))

ACCS = np.load("acceptor_scalar.npz")["acc"].astype(float)
def wingene(sc, ii):
    g = gene[ii]; yy = y[ii]; by = {}
    for k in range(len(ii)):
        r,c = by.setdefault(g[k], ([],[])); (r if yy[k]==1 else c).append(sc[k])
    v = []
    for r,c in by.values():
        if not r or not c: continue
        r = np.array(r); c = np.array(c)
        v.append(((r[:,None]>c[None,:]).sum()+0.5*(r[:,None]==c[None,:]).sum())/(len(r)*len(c)))
    return float(np.mean(v)), len(v)
def auroc(s, yy):
    o = np.argsort(s,kind="mergesort"); r = np.empty(len(s),float); r[o] = np.arange(1,len(s)+1)
    ss = s[o]; i = 0
    while i < len(ss):
        j = i
        while j+1 < len(ss) and ss[j+1]==ss[i]: j += 1
        if j > i: r[o[i:j+1]] = (i+1+j+1)/2
        i = j+1
    n1 = int(yy.sum()); n0 = len(yy)-n1
    return (r[yy==1].sum()-n1*(n1+1)/2)/(n1*n0)
def pear(a, b):
    m = np.isfinite(a)&np.isfinite(b); a,b = a[m],b[m]; n = len(a)
    r = float(np.corrcoef(a,b)[0,1]); z = 0.5*math.log((1+r)/(1-r)); se = 1/math.sqrt(n-3)
    return r, math.tanh(z-1.96*se), math.tanh(z+1.96*se)

RES = []; FILT = []
for SEED in [int(s) for s in A.seeds.split(",")]:
    torch.manual_seed(SEED); np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    net = MotifNet(A.nf, A.width, A.dropout).to(dev)
    npos = float(y[idx["train"]].sum()); nneg = len(idx["train"])-npos
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(nneg/npos, device=dev))
    opt = torch.optim.Adam(net.parameters(), lr=A.lr, weight_decay=A.wd)
    log(f"\n[seed {SEED}] {sum(p.numel() for p in net.parameters()):,} parameters, "
        f"{net.nfeat} region-max features, adversary weight {A.adv}")
    def sc(ii, bs=512):
        net.eval(); o = np.empty(len(ii), np.float32)
        with torch.no_grad():
            for k in range(0,len(ii),bs):
                j = ii[k:k+bs]; o[k:k+bs] = net(bo(j))[0].float().cpu().numpy()
        return o
    best = (-1,-1,None); t0 = time.time()
    for ep in range(1, A.epochs+1):
        lam = A.adv*min(1.0, ep/10)
        net.train(); perm = np.random.permutation(idx["train"]); tl = al = 0.0
        for k in range(0,len(perm),A.batch):
            j = perm[k:k+A.batch]
            opt.zero_grad(set_to_none=True)
            out, cpred = net(bo(j), lam)
            l1 = crit(out, yt[j].to(dev))
            l2 = F.mse_loss(cpred, Ct[j].to(dev)) if cpred is not None else torch.zeros((), device=dev)
            (l1+l2).backward(); opt.step()
            tl += float(l1)*len(j); al += float(l2)*len(j)
        sv = sc(idx["val"]); wg, ng = wingene(sv, idx["val"])
        if ep % 5 == 0 or ep <= 3:
            log(f"  ep {ep:3d} cls {tl/len(perm):.4f} adv {al/len(perm):.4f} "
                f"val within-gene {wg:.4f}  {time.time()-t0:.0f}s")
        if wg > best[0]: best = (wg, ep, {k:v.detach().cpu().clone() for k,v in net.state_dict().items()})
        elif ep-best[1] >= A.patience:
            log(f"  early stop at {ep}, best epoch {best[1]}"); break
    net.load_state_dict(best[2]); torch.save({"state":best[2],"args":vars(A)},
                                             f"{A.tag}_s{SEED}.pt")
    te = idx["test"]; st = sc(te)
    wgt, ngt = wingene(st, te); aut = auroc(st, y[te])
    a = ACCS[te]; c = y[te] == 1
    ra = pear(a[c], st[c]); rall = pear(a, st)
    log(f"[seed {SEED}] TEST within-gene {wgt:.4f}  pooled AUROC {aut:.4f}")
    log(f"[seed {SEED}] score vs REAL acceptor strength: all r = {rall[0]:+.4f} "
        f"[{rall[1]:+.4f},{rall[2]:+.4f}]   within real 3' ends r = {ra[0]:+.4f} "
        f"[{ra[1]:+.4f},{ra[2]:+.4f}]")
    RES.append((SEED, wgt, aut, rall[0], ra[0], ra[1], ra[2], best[1]))
    W = net.conv.weight.detach().cpu().numpy()
    for f_ in range(W.shape[0]):
        p = np.exp(W[f_]); p /= p.sum(0, keepdims=True)
        ic = float(np.mean(2.0 + np.sum(np.where(p>0, p*np.log2(np.clip(p,1e-12,None)), 0), 0)))
        con = "".join("ACGT"[i] for i in p.argmax(0))
        run = max(len(r) for r in __import__("re").findall(r"(.)\1*", con) for r in [r]) \
              if con else 0
        runs = [len(m.group(0)) for m in __import__("re").finditer(r"(.)\1*", con)]
        FILT.append((SEED, f_, con, ic, max(runs), "seeded_acc" if f_ < 8 else
                     ("seeded_u1" if f_ < 16 else "free")))
with open("motifnet_results.tsv","w") as fh:
    fh.write("seed\ttest_within_gene\ttest_auroc\tr_acceptor_all\tr_acceptor_real\tlo\thi\tbest_epoch\n")
    for r in RES: fh.write("\t".join(str(x) for x in r)+"\n")
with open("motifnet_filters.tsv","w") as fh:
    fh.write("seed\tfilter\tconsensus\tic_per_col\tlongest_run\tinit\n")
    for r in FILT: fh.write("\t".join(str(x) for x in r)+"\n")
R = np.array([[r[1],r[2],r[4]] for r in RES])
log(f"\n{'':18}{'within-gene':>12} {'pooled AUROC':>13} {'r vs acceptor (real ends)':>26}")
log(f"{'MotifNet':<18}{R[:,0].mean():>12.4f} {R[:,1].mean():>13.4f} {R[:,2].mean():>26.4f}")
log(f"{'reported CNN':<18}{0.8211:>12.4f} {0.8185:>13.4f} {0.0259:>26.4f}")
hp = np.array([f[4] for f in FILT])
log(f"\n[filters] longest single-base run in a filter consensus: median {np.median(hp):.0f}, "
    f"{100*np.mean(hp>=8):.0f}% of filters have a run of 8 or more "
    f"(a homopolymer detector); mean IC {np.mean([f[3] for f in FILT]):.2f} bits")
log("[out] motifnet_results.tsv motifnet_filters.tsv")
