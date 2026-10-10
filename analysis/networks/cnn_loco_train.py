#!/usr/bin/env python3
# leave-one-chromosome-out scores for the sequence cnn
import sys, os, time, argparse
import numpy as np
import torch, torch.nn as nn
from cnn_interp_lib import dinuc_shuffle

def log(*a): print(*a, file=sys.stderr, flush=True)

P = argparse.ArgumentParser()
P.add_argument("--data", default="cnn_data.npz")
P.add_argument("--epochs", type=int, default=60)
P.add_argument("--batch", type=int, default=128)
P.add_argument("--lr", type=float, default=1e-3)
P.add_argument("--wd", type=float, default=1e-6)
P.add_argument("--dropout", type=float, default=0.4)
P.add_argument("--patience", type=int, default=10)
P.add_argument("--seed", type=int, default=42)
P.add_argument("--ksh", type=int, default=3, help="dinucleotide shuffles per site")
P.add_argument("--inner_frac", type=float, default=0.25)
P.add_argument("--folds", default="", help="comma-separated subset, for smoke tests")
A = P.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
log(f"[env] device {dev}  torch {torch.__version__}"
    + (f"  {torch.cuda.get_device_name(0)}" if dev == "cuda" else ""))

D = np.load(A.data, allow_pickle=True)
ok = D["ok"]; X = D["X"][ok]; y = D["y"][ok].astype(np.float32)
gene = D["gene"][ok]; chrom = D["chrom"][ok].astype(str); split = D["split"][ok]
N = len(y)
GEN = np.array(sorted(set(gene)))
assert len(GEN) == 4847 and int(y.sum()) == 4847 and N == 48394, "not the analysis set"
CHS = sorted(set(chrom))
log(f"[scope] {N} analysis peaks, {len(GEN)} genes, {int(y.sum())} real / "
    f"{int((y==0).sum())} competitor, {len(CHS)} chromosomes")

TBL = np.zeros((256, 4), np.float32); TBL[255] = 0.25
for i in range(4): TBL[i, i] = 1.0
TBLt = torch.from_numpy(TBL).to(dev)
Xt = torch.from_numpy(X.astype(np.int64))
yt = torch.from_numpy(y)

def onehot(codes_t):
    return TBLt[codes_t.to(dev)].permute(0, 2, 1).contiguous()

from cnn_model_def import Net

def groups(ii):
    by = {}
    for k, g in enumerate(gene[ii]):
        r, c = by.setdefault(g, ([], []))
        (r if y[ii][k] == 1 else c).append(k)
    return [(np.array(r), np.array(c)) for r, c in by.values() if r and c]

def winrate(s, grp):
    return float(np.mean([((s[r][:, None] > s[c][None, :]).sum()
                           + 0.5 * (s[r][:, None] == s[c][None, :]).sum()) / (len(r) * len(c))
                          for r, c in grp]))

SC = np.full(N, np.nan)
SH = np.full((A.ksh, N), np.nan)
ROWS = []
WANT = [c for c in CHS if not A.folds or c in A.folds.split(",")]
t00 = time.time()

for fi, ch in enumerate(WANT, 1):
    te = np.flatnonzero(chrom == ch); tr = np.flatnonzero(chrom != ch)
    trg = np.array(sorted(set(gene[tr])))
    rs = np.random.default_rng(A.seed); sh = trg.copy(); rs.shuffle(sh)
    ivg = set(sh[:int(round(A.inner_frac * len(sh)))])
    iv = tr[np.array([g in ivg for g in gene[tr]])]
    it = tr[np.array([g not in ivg for g in gene[tr]])]
    assert not (set(gene[iv]) & set(gene[it])) and not (set(gene[te]) & set(gene[tr]))
    GIV = groups(iv)

    torch.manual_seed(A.seed); np.random.seed(A.seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    net = Net(A.dropout).to(dev)
    npos = float(y[it].sum()); nneg = len(it) - npos
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(nneg / npos, device=dev))
    opt = torch.optim.Adam(net.parameters(), lr=A.lr, weight_decay=A.wd)

    def scores(ii, codes=None, bs=1024):
        net.eval(); out = np.empty(len(ii), np.float32)
        src = Xt[ii] if codes is None else torch.from_numpy(codes.astype(np.int64))
        with torch.no_grad():
            for k in range(0, len(ii), bs):
                out[k:k + bs] = net(onehot(src[k:k + bs])).float().cpu().numpy()
        return out

    best = (-1.0, -1, None); t0 = time.time()
    for ep in range(1, A.epochs + 1):
        net.train(); perm = np.random.permutation(it)
        for k in range(0, len(perm), A.batch):
            j = perm[k:k + A.batch]
            opt.zero_grad(set_to_none=True)
            loss = crit(net(onehot(Xt[j])), yt[j].to(dev))
            loss.backward(); opt.step()
        w = winrate(scores(iv), GIV)
        if w > best[0]:
            best = (w, ep, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
        elif ep - best[1] >= A.patience:
            break
    net.load_state_dict(best[2])
    SC[te] = scores(te)

    rng = np.random.default_rng([A.seed, fi])
    for j in range(A.ksh):
        cd = np.stack([dinuc_shuffle(X[i], rng) for i in te])
        SH[j, te] = scores(te, codes=cd)

    gte = groups(te)
    wte = winrate(SC[te], gte)
    ROWS.append(dict(chrom=ch, n_peaks=len(te), n_genes=len(gte), best_epoch=best[1],
                     inner_val_wg=best[0], fold_wg=wte, n_train=len(it), n_inner=len(iv)))
    log(f"  [{fi:2d}/{len(WANT)}] {ch:>5}  train {len(it):5d} inner-val {len(iv):5d} "
        f"held-out {len(te):5d} ({len(gte):4d} genes)  best ep {best[1]:2d} "
        f"inner-val {best[0]:.4f}  fold within-gene {wte:.4f}  "
        f"{time.time()-t0:.0f}s  [{(time.time()-t00)/60:.1f} min total]")

if not A.folds:
    assert np.isfinite(SC).all() and np.isfinite(SH).all(), "LOCO left peaks unscored"

np.savez_compressed("cnn_loco_scores.npz", logit=SC, logit_shuffled=SH, y=y, gene=gene,
                    chrom=chrom, split=split, k_shuffles=A.ksh, seed=A.seed)
with open("cnn_loco_folds.tsv", "w") as fh:
    fh.write("chrom\tn_peaks\tn_genes\tn_train\tn_inner_val\tbest_epoch\tinner_val_winrate\t"
             "fold_winrate\n")
    for r in ROWS:
        fh.write(f"{r['chrom']}\t{r['n_peaks']}\t{r['n_genes']}\t{r['n_train']}\t{r['n_inner']}\t"
                 f"{r['best_epoch']}\t{r['inner_val_wg']:.5f}\t{r['fold_wg']:.5f}\n")
log(f"[out] cnn_loco_scores.npz cnn_loco_folds.tsv  ({(time.time()-t00)/60:.1f} min)")

# headline + sanity
fin = np.isfinite(SC)
GA = groups(np.flatnonzero(fin))
idx = np.flatnonzero(fin)
WG = winrate(SC[fin], GA)
rk = np.array([1 + int((SC[idx][c] >= SC[idx][r[0]]).sum()) for r, c in GA])
P1S = float((rk == 1).mean())
p1t = np.array([float(SC[idx][r].max() > SC[idx][c].max())
                + 0.5 * float(SC[idx][r].max() == SC[idx][c].max()) for r, c in GA])
log(f"\n[LOCO] all {len(GA)} analysis genes: within-gene win-rate {WG:.5f}  "
    f"strict P@1 {P1S:.5f}  tie-aware P@1 {p1t.mean():.5f}")

if not A.folds:
    tm = split == "test"
    ii = np.flatnonzero(tm); GT = groups(ii)
    wt = winrate(SC[ii], GT)
    log(f"[sanity] restricted to the {len(GT)} genes that were the single-split TEST set, the "
        f"LOCO score gives within-gene {wt:.5f} against the reported single-split 0.82110")
    log(f"[sanity] these are not the same quantity: the LOCO fold for a test chromosome trains on "
        f"every other chromosome, a superset of the single split's train+val, so a small gain is "
        f"expected. Difference {wt-0.82110:+.5f}")
    assert abs(wt - 0.82110) < 0.05, f"LOCO on the old test genes is {wt:.4f}, too far from 0.821"

for j in range(A.ksh):
    log(f"[shuffle] replicate {j+1}: within-gene {winrate(SH[j][fin], GA):.5f}")
