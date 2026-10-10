#!/usr/bin/env python3
# does performance come from reading more upstream sequence?
import sys, time, argparse, os
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser()
P.add_argument("--data", default="../cnn_data.npz")
P.add_argument("--out", default="data/context_sweep.tsv")
P.add_argument("--contexts", default="up2000,up1500,up1000,up500,up250,up100,span500")
P.add_argument("--seeds", default="42,43,44,45,46")
P.add_argument("--epochs", type=int, default=60)
P.add_argument("--batch", type=int, default=128)
P.add_argument("--lr", type=float, default=1e-3)
P.add_argument("--wd", type=float, default=1e-6)
P.add_argument("--dropout", type=float, default=0.4)
P.add_argument("--patience", type=int, default=10)
P.add_argument("--smoke", action="store_true", help="shape check on CPU, no training")
A = P.parse_args()
dev = "cuda" if torch.cuda.is_available() and not A.smoke else "cpu"
log(f"device {dev}  torch {torch.__version__}")

UP, DN = 2000, 500
CTX = {"up2000": (-2000, 500), "up1500": (-1500, 500), "up1000": (-1000, 500),
       "up500": (-500, 500), "up250": (-250, 500), "up100": (-100, 500),
       "span500": (-250, 250)}

D = np.load(A.data, allow_pickle=True); ok = D["ok"]
X = D["X"][ok]; y = D["y"][ok].astype(np.float32); gene = D["gene"][ok]; split = D["split"][ok]
assert X.shape[1] == UP + DN, X.shape
idx = {s: np.flatnonzero(split == s) for s in ("train", "val", "test")}
log("  " + "  ".join(f"{s} {len(idx[s]):,} ({int(y[idx[s]].sum())} real)" for s in idx))
TBL = np.zeros((256, 4), np.float32); TBL[255] = 0.25
for i in range(4): TBL[i, i] = 1.0
TBLt = torch.from_numpy(TBL).to(dev)
Xall = torch.from_numpy(X.astype(np.int64)); yt = torch.from_numpy(y)

class V(nn.Module):
    def __init__(s, L, drop=0.4, pools=(10, 5, 2)):
        super().__init__()
        s.c1 = nn.Conv1d(4, 32, 19, padding=9, bias=False); s.b1 = nn.BatchNorm1d(32)
        s.c2 = nn.Conv1d(32, 64, 7, padding=3, bias=False); s.b2 = nn.BatchNorm1d(64)
        s.c3 = nn.Conv1d(64, 96, 5, padding=2, bias=False); s.b3 = nn.BatchNorm1d(96)
        s.p = pools
        nb = L // (pools[0]*pools[1]*pools[2])
        s.do = nn.Dropout(drop)
        s.fc1 = nn.Linear(96*nb, 64); s.fc2 = nn.Linear(64, 1)
        s.nb = nb
    def forward(s, x):
        h = F.max_pool1d(torch.exp(torch.clamp(s.b1(s.c1(x)), max=8.0)), s.p[0])
        h = F.max_pool1d(F.relu(s.b2(s.c2(h))), s.p[1])
        h = F.max_pool1d(F.relu(s.b3(s.c3(h))), s.p[2])
        return s.fc2(s.do(F.relu(s.fc1(s.do(h.flatten(1)))))).squeeze(1)

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
def p_at_1(sc, ii):
    g = gene[ii]; yy = y[ii]; by = {}
    for k in range(len(ii)):
        r, c = by.setdefault(g[k], ([], []))
        (r if yy[k] == 1 else c).append(sc[k])
    v = [float(max(r) > max(c)) for r, c in by.values() if r and c]
    return float(np.mean(v))
def auroc(pos, neg):
    a = np.concatenate([pos, neg])
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    cs = np.cumsum(cnt); rk = ((cs-cnt)+cs+1)/2.0
    return float((rk[inv][:len(pos)].sum() - len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg)))

if A.smoke:
    for name in A.contexts.split(","):
        lo, hi = CTX[name]; a, b = lo + UP, hi + UP; L = b - a
        net = V(L).eval(); xb = TBLt[Xall[:4, a:b]].permute(0, 2, 1)
        with torch.no_grad(): o = net(xb)
        log(f"[smoke] {name:8s} [{lo:+5d},{hi:+4d})  L {L}  bins {net.nb}  "
            f"params {sum(p.numel() for p in net.parameters()):,}  out {tuple(o.shape)}")
    sys.exit(0)

os.makedirs(os.path.dirname(A.out) or ".", exist_ok=True)
OUT = []
SEEDS = [int(s) for s in A.seeds.split(",")]
for name in A.contexts.split(","):
  lo, hi = CTX[name]; a, b = lo + UP, hi + UP; L = b - a
  Xc = Xall[:, a:b].contiguous()
  def bo(ii): return TBLt[Xc[ii].to(dev)].permute(0, 2, 1).contiguous()
  for SEED in SEEDS:
    torch.manual_seed(SEED); np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    net = V(L, A.dropout).to(dev)
    npos = float(y[idx["train"]].sum()); nneg = len(idx["train"]) - npos
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(nneg/npos, device=dev))
    opt = torch.optim.Adam(net.parameters(), lr=A.lr, weight_decay=A.wd)
    log(f"\n[{name} seed {SEED}] [{lo:+d},{hi:+d})  L {L}  bins {net.nb}")
    def sc(ii, bs=512):
        net.eval(); o = np.empty(len(ii), np.float32)
        with torch.no_grad():
            for k in range(0, len(ii), bs):
                j = ii[k:k+bs]; o[k:k+bs] = net(bo(j)).float().cpu().numpy()
        return o
    best = (-1, -1, None); t0 = time.time()
    for ep in range(1, A.epochs+1):
        net.train(); perm = np.random.permutation(idx["train"])
        for k in range(0, len(perm), A.batch):
            j = perm[k:k+A.batch]
            opt.zero_grad(set_to_none=True)
            loss = crit(net(bo(j)), yt[j].to(dev)); loss.backward(); opt.step()
        wg, _ = within_gene(sc(idx["val"]), idx["val"])
        if wg > best[0]: best = (wg, ep, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
        elif ep - best[1] >= A.patience: break
    net.load_state_dict(best[2])
    st = sc(idx["test"]); wgt, ngt = within_gene(st, idx["test"])
    p1 = p_at_1(st, idx["test"])
    aut = auroc(st[y[idx["test"]] == 1], st[y[idx["test"]] == 0])
    log(f"[{name} seed {SEED}] best epoch {best[1]}  TEST within-gene {wgt:.4f}  "
        f"P@1 {p1:.4f}  AUROC {aut:.4f}  ({ngt} genes, {time.time()-t0:.0f}s)")
    OUT.append((name, lo, hi, L, SEED, best[1], best[0], wgt, p1, aut, ngt))
    with open(A.out, "w") as fh:
        fh.write("context\tstart\tend\tlength\tseed\tbest_epoch\tval_within_gene\t"
                 "test_within_gene\ttest_p_at_1\ttest_auroc\tn_genes\n")
        for o in OUT: fh.write("\t".join(str(x) for x in o) + "\n")
log(f"[out] {A.out}")
