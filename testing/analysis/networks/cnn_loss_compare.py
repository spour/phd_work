#!/usr/bin/env python3
# does the cnn's within-gene ranking depend on the pooled-bce training formulation?
import sys, os, time, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

def log(*a): print(*a, file=sys.stderr, flush=True)

P = argparse.ArgumentParser()
P.add_argument("--loss", required=True, choices=["bce", "pairwise", "listwise"])
P.add_argument("--data", default="cnn_data.npz")
P.add_argument("--out", default=None)
P.add_argument("--epochs", type=int, default=60)
P.add_argument("--batch", type=int, default=128, help="sites per step, bce arm")
P.add_argument("--gene_batch", type=int, default=16, help="genes per step, ranking arms")
P.add_argument("--max_comp", type=int, default=32, help="competitors kept per gene per epoch")
P.add_argument("--lr", type=float, default=1e-3)
P.add_argument("--wd", type=float, default=1e-6)
P.add_argument("--dropout", type=float, default=0.4)
P.add_argument("--patience", type=int, default=10)
P.add_argument("--seed", type=int, default=42)
A = P.parse_args()
TAG = A.out or f"cnn_loss_{A.loss}_s{A.seed}"

torch.manual_seed(A.seed); np.random.seed(A.seed)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
dev = "cuda" if torch.cuda.is_available() else "cpu"
log(f"[env] device {dev}  torch {torch.__version__}  cuda {torch.version.cuda}"
    + (f"  {torch.cuda.get_device_name(0)}" if dev == "cuda" else ""))
log(f"[run] loss={A.loss} seed={A.seed} -> {TAG}")

# data, exactly cnn_train.py's
D = np.load(A.data, allow_pickle=True)
ok = D["ok"]; X = D["X"][ok]; y = D["y"][ok].astype(np.float32)
gene = D["gene"][ok]; split = D["split"][ok]
idx = {s: np.flatnonzero(split == s) for s in ("train", "val", "test")}
log("[data] " + "  ".join(f"{s} {len(idx[s])} ({int(y[idx[s]].sum())} real)" for s in idx))

TBL = np.zeros((256, 4), np.float32); TBL[255] = 0.25
for i in range(4): TBL[i, i] = 1.0
TBL = torch.from_numpy(TBL).to(dev)
Xt = torch.from_numpy(X.astype(np.int64))

def batch_onehot(ii):
    return TBL[Xt[ii].to(dev)].permute(0, 2, 1).contiguous()

from cnn_model_def import Net
net = Net(A.dropout).to(dev)
log(f"[model] cnn_model_def.Net, {sum(p.numel() for p in net.parameters()):,} parameters")
opt = torch.optim.Adam(net.parameters(), lr=A.lr, weight_decay=A.wd)

# gene structure
def groups(ii):
    by = {}
    for k in ii:
        r, c = by.setdefault(gene[k], ([], []))
        (r if y[k] == 1 else c).append(k)
    return by

GTR = groups(idx["train"])
RANKG = [(np.array(r), np.array(c)) for r, c in GTR.values() if len(r) == 1 and len(c) >= 1]
n_multi = sum(1 for r, c in GTR.values() if len(r) != 1)
n_nocomp = sum(1 for r, c in GTR.values() if len(c) == 0)
log(f"[genes] train {len(GTR)} genes; {len(RANKG)} usable for the ranking losses "
    f"({n_nocomp} dropped for having no competitor, {n_multi} for not having exactly one real)")
assert n_multi == 0, "a train gene does not have exactly one real site; listwise target is wrong"

npos = float(y[idx["train"]].sum()); nneg = len(idx["train"]) - npos
crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(nneg / npos, device=dev))
yt = torch.from_numpy(y)
if A.loss == "bce": log(f"[loss] BCEWithLogits, pos_weight {nneg/npos:.2f}")
else: log(f"[loss] {A.loss}, <= {A.max_comp} competitors/gene/epoch, {A.gene_batch} genes/step")

# metrics (project conventions)
def scores(ii, bs=1024):
    net.eval(); out = np.empty(len(ii), np.float32)
    with torch.no_grad():
        for k in range(0, len(ii), bs):
            out[k:k + bs] = net(batch_onehot(ii[k:k + bs])).float().cpu().numpy()
    return out

def within_gene(s, ii):
    g = gene[ii]; yy = y[ii]; by = {}
    for k in range(len(ii)):
        r, c = by.setdefault(g[k], ([], []))
        (r if yy[k] == 1 else c).append(s[k])
    v = [((np.array(r)[:, None] > np.array(c)[None, :]).sum()
          + 0.5 * (np.array(r)[:, None] == np.array(c)[None, :]).sum()) / (len(r) * len(c))
         for r, c in by.values() if r and c]
    return float(np.mean(v)), len(v)

def auroc(pos, neg):
    a = np.concatenate([pos, neg])
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    cs = np.cumsum(cnt); rk = ((cs - cnt) + cs + 1) / 2.0
    return float((rk[inv][:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

# ragged batch -> padded matrix
def rank_batch(chunk, rng):
    rows, lens = [], []
    for r, c in chunk:
        cs = c if len(c) <= A.max_comp else rng.choice(c, A.max_comp, replace=False)
        rows.append(r[0]); rows.extend(cs.tolist()); lens.append(1 + len(cs))
    return np.asarray(rows), np.asarray(lens)

def pad(s, lens):
    G, m = len(lens), int(lens.max())
    out = torch.full((G, m), float("-inf"), device=s.device, dtype=s.dtype)
    off = np.r_[0, np.cumsum(lens)]
    for i in range(G):
        out[i, :lens[i]] = s[off[i]:off[i + 1]]
    return out

best = (-1.0, -1, None); hist = []
tr = idx["train"]
for ep in range(1, A.epochs + 1):
    net.train(); rng = np.random.default_rng([A.seed, ep]); tot = 0.0; nstep = 0; t0 = time.time()
    if A.loss == "bce":
        perm = np.random.permutation(tr)
        for k in range(0, len(perm), A.batch):
            j = perm[k:k + A.batch]
            opt.zero_grad(set_to_none=True)
            loss = crit(net(batch_onehot(j)), yt[j].to(dev))
            loss.backward(); opt.step(); tot += float(loss.detach()) * len(j); nstep += 1
        tot /= len(perm)
    else:
        order = rng.permutation(len(RANKG))
        for k in range(0, len(order), A.gene_batch):
            chunk = [RANKG[i] for i in order[k:k + A.gene_batch]]
            rows, lens = rank_batch(chunk, rng)
            opt.zero_grad(set_to_none=True)
            M = pad(net(batch_onehot(rows)), lens)
            if A.loss == "listwise":
                loss = F.cross_entropy(M, torch.zeros(len(lens), dtype=torch.long, device=dev))
            else:
                nc = torch.from_numpy(lens - 1).to(dev)
                m = torch.arange(M.shape[1] - 1, device=dev)[None, :] < nc[:, None]
                d = M[:, :1] - M[:, 1:].masked_fill(~m, 0.0)
                per = (F.softplus(-d) * m).sum(1) / nc.clamp(min=1).float()
                loss = per.mean()
            loss.backward(); opt.step(); tot += float(loss.detach()); nstep += 1
        tot /= max(nstep, 1)
    sv = scores(idx["val"]); wg, ng = within_gene(sv, idx["val"])
    au = auroc(sv[y[idx["val"]] == 1], sv[y[idx["val"]] == 0])
    hist.append(dict(epoch=ep, loss=tot, val_wg=wg, val_auroc=au, steps=nstep))
    log(f"  ep {ep:3d}  loss {tot:.4f}  {nstep} steps  val within-gene {wg:.4f} ({ng} genes)  "
        f"val pooled AUROC {au:.4f}  {time.time()-t0:.0f}s")
    if wg > best[0]:
        best = (wg, ep, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
    elif ep - best[1] >= A.patience:
        log(f"  early stop at {ep}, best epoch {best[1]} (val within-gene {best[0]:.4f})"); break
net.load_state_dict(best[2])
log(f"[fit] restored epoch {best[1]}, val within-gene {best[0]:.4f}")

s_all = scores(np.arange(len(y)))
np.savez_compressed(f"{TAG}_scores.npz", logit=s_all, y=y, gene=gene, split=split,
                    loss=A.loss, seed=A.seed, best_epoch=best[1], val_wg=best[0])
with open(f"{TAG}_history.tsv", "w") as fh:
    fh.write("epoch\tloss\tsteps\tval_within_gene\tval_pooled_auroc\n")
    for h in hist:
        fh.write(f"{h['epoch']}\t{h['loss']:.5f}\t{h['steps']}\t{h['val_wg']:.5f}\t"
                 f"{h['val_auroc']:.5f}\n")
for s in ("val", "test"):
    ss = s_all[idx[s]]; wg, ng = within_gene(ss, idx[s])
    log(f"[{s}] within-gene {wg:.4f} ({ng} genes)  pooled AUROC "
        f"{auroc(ss[y[idx[s]]==1], ss[y[idx[s]]==0]):.4f}")
log(f"[out] {TAG}_scores.npz {TAG}_history.tsv")
