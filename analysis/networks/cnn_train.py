#!/usr/bin/env python3
# interpretable cnn, real 3' ends vs intragenic competitors from [-2000,+500] sequence
import sys, os, json, time, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

def log(*a): print(*a, file=sys.stderr, flush=True)

P = argparse.ArgumentParser()
P.add_argument("--data", default="cnn_data.npz")
P.add_argument("--out", default="cnn_model")
P.add_argument("--epochs", type=int, default=60)
P.add_argument("--batch", type=int, default=128)
P.add_argument("--lr", type=float, default=1e-3)
P.add_argument("--wd", type=float, default=1e-6)
P.add_argument("--dropout", type=float, default=0.4)
P.add_argument("--patience", type=int, default=10)
P.add_argument("--seed", type=int, default=42)
A = P.parse_args()

torch.manual_seed(A.seed); np.random.seed(A.seed)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
dev = "cuda" if torch.cuda.is_available() else "cpu"
log(f"device {dev}  torch {torch.__version__}  cuda {torch.version.cuda}")

# data
D = np.load(A.data, allow_pickle=True)
ok = D["ok"]; X = D["X"][ok]; y = D["y"][ok].astype(np.float32)
gene = D["gene"][ok]; split = D["split"][ok]
idx = {s: np.flatnonzero(split == s) for s in ("train", "val", "test")}
log("  " + "  ".join(f"{s} {len(idx[s])} ({int(y[idx[s]].sum())} real)" for s in idx))

TBL = np.zeros((256, 4), np.float32); TBL[255] = 0.25
for i in range(4): TBL[i, i] = 1.0
TBL = torch.from_numpy(TBL).to(dev)
Xt = torch.from_numpy(X.astype(np.int64))
yt = torch.from_numpy(y)

def batch_onehot(ii):
    return TBL[Xt[ii].to(dev)].permute(0, 2, 1).contiguous()

# model
from cnn_model_def import Net

net = Net(A.dropout).to(dev)
log(f"parameters {sum(p.numel() for p in net.parameters()):,}")

npos = float(y[idx["train"]].sum()); nneg = len(idx["train"]) - npos
crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(nneg / npos, device=dev))
log(f"pos_weight {nneg/npos:.2f}")
opt = torch.optim.Adam(net.parameters(), lr=A.lr, weight_decay=A.wd)

# metrics
def scores(ii, bs=512):
    net.eval(); out = np.empty(len(ii), np.float32)
    with torch.no_grad():
        for k in range(0, len(ii), bs):
            j = ii[k:k + bs]; out[k:k + bs] = net(batch_onehot(j)).float().cpu().numpy()
    return out

def within_gene(s, ii):
    g = gene[ii]; yy = y[ii]; by = {}
    for k in range(len(ii)):
        r, c = by.setdefault(g[k], ([], []))
        (r if yy[k] == 1 else c).append(s[k])
    v = []
    for r, c in by.values():
        if not r or not c: continue
        r = np.array(r); c = np.array(c)
        v.append(((r[:, None] > c[None, :]).sum() + 0.5 * (r[:, None] == c[None, :]).sum())
                 / (len(r) * len(c)))
    return float(np.mean(v)), len(v)

def auroc(pos, neg):
    a = np.concatenate([pos, neg])
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    cs = np.cumsum(cnt); rk = ((cs - cnt) + cs + 1) / 2.0
    return float((rk[inv][:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

# train
best = (-1, -1, None); hist = []
tr = idx["train"]
for ep in range(1, A.epochs + 1):
    net.train(); perm = np.random.permutation(tr); tot = 0.0; t0 = time.time()
    for k in range(0, len(perm), A.batch):
        j = perm[k:k + A.batch]
        opt.zero_grad(set_to_none=True)
        loss = crit(net(batch_onehot(j)), yt[j].to(dev))
        loss.backward(); opt.step(); tot += float(loss) * len(j)
    sv = scores(idx["val"]); wg, ng = within_gene(sv, idx["val"])
    au = auroc(sv[y[idx["val"]] == 1], sv[y[idx["val"]] == 0])
    hist.append(dict(epoch=ep, loss=tot / len(perm), val_wg=wg, val_auroc=au))
    log(f"  ep {ep:3d}  loss {tot/len(perm):.4f}  val within-gene {wg:.4f} ({ng} genes)  "
        f"val pooled AUROC {au:.4f}  {time.time()-t0:.0f}s")
    if wg > best[0]:
        best = (wg, ep, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
    elif ep - best[1] >= A.patience:
        log(f"  early stop at {ep}, best epoch {best[1]} (val within-gene {best[0]:.4f})"); break
net.load_state_dict(best[2])
log(f"restored epoch {best[1]}, val within-gene {best[0]:.4f}")

sv = torch.from_numpy(scores(idx["val"])).to(dev); yv = yt[idx["val"]].to(dev)
logT = torch.zeros(1, device=dev, requires_grad=True)
optT = torch.optim.LBFGS([logT], lr=0.1, max_iter=100)
bce = nn.BCEWithLogitsLoss()
def closure():
    optT.zero_grad(); l = bce(sv / torch.exp(logT), yv); l.backward(); return l
optT.step(closure)
Temp = float(torch.exp(logT))
log(f"temperature {Temp:.4f}")

allidx = np.arange(len(y)); s_all = scores(allidx)
np.savez_compressed(A.out + "_scores.npz", logit=s_all, logit_cal=s_all / Temp,
                    y=y, gene=gene, split=split, temperature=Temp)
torch.save(dict(state=net.state_dict(), args=vars(A), temperature=Temp, epoch=best[1]),
           A.out + ".pt")
with open(A.out + "_history.tsv", "w") as fh:
    fh.write("epoch\tloss\tval_within_gene\tval_pooled_auroc\n")
    for h in hist: fh.write(f"{h['epoch']}\t{h['loss']:.5f}\t{h['val_wg']:.5f}\t{h['val_auroc']:.5f}\n")
for s in ("val", "test"):
    ss = s_all[idx[s]]; wg, ng = within_gene(ss, idx[s])
    log(f"[{s}] within-gene {wg:.4f} ({ng} genes)  pooled AUROC "
        f"{auroc(ss[y[idx[s]]==1], ss[y[idx[s]]==0]):.4f}")
log(f"[out] {A.out}.pt {A.out}_scores.npz {A.out}_history.tsv")
