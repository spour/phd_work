#!/usr/bin/env python3
import sys, time, json, math, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser()
P.add_argument("--data", default="cnn_data.npz")
P.add_argument("--variants", default="base,aux,fine50,fine125")
P.add_argument("--epochs", type=int, default=60)
P.add_argument("--batch", type=int, default=128)
P.add_argument("--lr", type=float, default=1e-3)
P.add_argument("--wd", type=float, default=1e-6)
P.add_argument("--dropout", type=float, default=0.4)
P.add_argument("--patience", type=int, default=10)
P.add_argument("--seeds", default="42,43,44")
A = P.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
log(f"device {dev}  torch {torch.__version__}")

D = np.load(A.data, allow_pickle=True); ok = D["ok"]
X = D["X"][ok]; y = D["y"][ok].astype(np.float32); gene = D["gene"][ok]; split = D["split"][ok]
S = np.load("acceptor_scalar.npz")
ACC = S["acc"].astype(np.float32)
idx = {s: np.flatnonzero(split == s) for s in ("train", "val", "test")}
mu, sd = ACC[idx["train"]].mean(), ACC[idx["train"]].std()
ACCz = ((ACC - mu) / sd).astype(np.float32)
log("  " + "  ".join(f"{s} {len(idx[s]):,} ({int(y[idx[s]].sum())} real)" for s in idx))
log(f"  acceptor scalar standardised on train: mean {mu:.3f} sd {sd:.3f}")
TBL = np.zeros((256, 4), np.float32); TBL[255] = 0.25
for i in range(4): TBL[i, i] = 1.0
TBLt = torch.from_numpy(TBL).to(dev)
Xt = torch.from_numpy(X.astype(np.int64)); yt = torch.from_numpy(y)
At = torch.from_numpy(ACCz)
def bo(ii): return TBLt[Xt[ii].to(dev)].permute(0, 2, 1).contiguous()

class V(nn.Module):
    def __init__(s, drop=0.4, pools=(10, 5, 2), aux=False):
        super().__init__()
        s.c1 = nn.Conv1d(4, 32, 19, padding=9, bias=False); s.b1 = nn.BatchNorm1d(32)
        s.c2 = nn.Conv1d(32, 64, 7, padding=3, bias=False); s.b2 = nn.BatchNorm1d(64)
        s.c3 = nn.Conv1d(64, 96, 5, padding=2, bias=False); s.b3 = nn.BatchNorm1d(96)
        s.p = pools; s.aux = aux
        nb = 2500 // (pools[0]*pools[1]*pools[2])
        s.do = nn.Dropout(drop)
        s.fc1 = nn.Linear(96*nb + (1 if aux else 0), 64); s.fc2 = nn.Linear(64, 1)
        s.nb = nb
    def forward(s, x, a=None):
        h = F.max_pool1d(torch.exp(torch.clamp(s.b1(s.c1(x)), max=8.0)), s.p[0])
        h = F.max_pool1d(F.relu(s.b2(s.c2(h))), s.p[1])
        h = F.max_pool1d(F.relu(s.b3(s.c3(h))), s.p[2])
        h = h.flatten(1)
        if s.aux: h = torch.cat([h, a[:, None]], 1)
        return s.fc2(s.do(F.relu(s.fc1(s.do(h))))).squeeze(1)

def within_gene(sc, ii):
    g = gene[ii]; yy = y[ii]; by = {}
    for k in range(len(ii)):
        r, c = by.setdefault(g[k], ([], []))
        (r if yy[k] == 1 else c).append(sc[k])
    v = []
    for r, c in by.values():
        if not r or not c: continue
        r = np.array(r); c = np.array(c)
        v.append(((r[:,None] > c[None,:]).sum() + 0.5*(r[:,None] == c[None,:]).sum())/(len(r)*len(c)))
    return float(np.mean(v)), len(v)
def auroc(pos, neg):
    a = np.concatenate([pos, neg])
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    cs = np.cumsum(cnt); rk = ((cs-cnt)+cs+1)/2.0
    return float((rk[inv][:len(pos)].sum() - len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg)))

wga, nga = within_gene(ACC[idx["test"]], idx["test"])
log(f"[floor] the acceptor scalar ALONE, on these test genes: within-gene {wga:.4f} ({nga} genes)")

CFG = {"base": ((10,5,2), False), "aux": ((10,5,2), True),
       "fine50": ((5,5,2), False), "fine125": ((5,2,2), False)}
OUT = []
SEEDS = [int(x) for x in A.seeds.split(",")]
for name in A.variants.split(","):
  for SEED in SEEDS:
    pools, aux = CFG[name]
    torch.manual_seed(SEED); np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    net = V(A.dropout, pools, aux).to(dev)
    npos = float(y[idx["train"]].sum()); nneg = len(idx["train"]) - npos
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(nneg/npos, device=dev))
    opt = torch.optim.Adam(net.parameters(), lr=A.lr, weight_decay=A.wd)
    log(f"\n[{name} seed {SEED}] pools {pools}  bins {net.nb}  aux {aux}  "
        f"parameters {sum(p.numel() for p in net.parameters()):,}")
    def sc(ii, bs=512):
        net.eval(); o = np.empty(len(ii), np.float32)
        with torch.no_grad():
            for k in range(0, len(ii), bs):
                j = ii[k:k+bs]; o[k:k+bs] = net(bo(j), At[j].to(dev)).float().cpu().numpy()
        return o
    best = (-1, -1, None); t0 = time.time()
    for ep in range(1, A.epochs+1):
        net.train(); perm = np.random.permutation(idx["train"]); tot = 0.0
        for k in range(0, len(perm), A.batch):
            j = perm[k:k+A.batch]
            opt.zero_grad(set_to_none=True)
            loss = crit(net(bo(j), At[j].to(dev)), yt[j].to(dev))
            loss.backward(); opt.step(); tot += float(loss)*len(j)
        sv = sc(idx["val"]); wg, ng = within_gene(sv, idx["val"])
        if ep % 5 == 0 or ep <= 3:
            log(f"  ep {ep:3d} loss {tot/len(perm):.4f} val within-gene {wg:.4f}  "
                f"{time.time()-t0:.0f}s")
        if wg > best[0]: best = (wg, ep, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
        elif ep - best[1] >= A.patience:
            log(f"  early stop at {ep}, best epoch {best[1]}"); break
    net.load_state_dict(best[2])
    st = sc(idx["test"]); wgt, ngt = within_gene(st, idx["test"])
    aut = auroc(st[y[idx["test"]] == 1], st[y[idx["test"]] == 0])
    log(f"[{name} seed {SEED}] val within-gene {best[0]:.4f} (epoch {best[1]})   "
        f"TEST within-gene {wgt:.4f} ({ngt} genes)   TEST pooled AUROC {aut:.4f}")
    torch.save({"state": best[2], "args": vars(A), "variant": name, "pools": pools, "aux": aux},
               f"acc_arch_{name}_s{SEED}.pt")
    OUT.append((name, SEED, str(pools), int(aux), net.nb, best[1], best[0], wgt, aut, ngt))
with open("acceptor_arch_train.tsv", "w") as fh:
    fh.write("variant\tseed\tpools\taux\tbins\tbest_epoch\tval_within_gene\t"
             "test_within_gene\ttest_auroc\tn_genes\n")
    for o in OUT: fh.write("\t".join(str(x) for x in o) + "\n")
import collections
BY = collections.defaultdict(list)
for o in OUT: BY[o[0]].append(o[7])
log(f"\n{'variant':<9} {'bins':>5} {'aux':>4} {'seeds':>6} {'TEST within-gene':>18} {'range':>8}")
for name in A.variants.split(","):
    v = np.array(BY[name]); c = CFG[name]
    nb = 2500 // (c[0][0]*c[0][1]*c[0][2])
    log(f"{name:<9} {nb:>5} {int(c[1]):>4} {len(v):>6} "
        f"{v.mean():>10.4f} +/- {v.std(ddof=1)/np.sqrt(len(v)):.4f} {v.max()-v.min():>8.4f}")
bs = np.array(BY["base"]) if "base" in BY else None
if bs is not None:
    for name in A.variants.split(","):
        if name == "base": continue
        v = np.array(BY[name]); d = v.mean() - bs.mean()
        se = math.sqrt(v.var(ddof=1)/len(v) + bs.var(ddof=1)/len(bs))
        log(f"  {name} minus base: {d:+.4f} +/- {se:.4f}  z = {d/se:+.1f}"
            f"{'' if abs(d/se) >= 2 else '   within seed noise'}")
log(f"  acceptor scalar alone on the same test genes: {wga:.4f}")
log("[out] acceptor_arch_train.tsv")
