#!/usr/bin/env python3
# panel b with the uncertainty it was missing, and both controls
import sys, argparse, itertools
import numpy as np, torch
def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser(); P.add_argument("--model", default="cnn_model.pt")
P.add_argument("--nshuf", type=int, default=20); P.add_argument("--nboot", type=int, default=2000)
P.add_argument("--pos", type=int, default=-230); P.add_argument("--seed", type=int, default=42)
A = P.parse_args()
NT = "ACGT"; UP = 2000
UNITS = [("acc_model", "ACGGTGTACTTTTGTTGCAGGTG", 4), ("acc_canon", "TTTAACTTTCTTTTTTACAGACG", 4),
         ("acc_agdep", "CCATGCCTTTGATCCTGCAGATG", 4), ("acc_s3max", "TTTCTTCTCTCTCCCTGCAGGTG", 4),
         ("acc_best", "TTTCCTCTCTCTCCCCGCAGGTG", 4), ("acc_noC", "TTTTTTTTTGTGTGTTTTAGGTT", 4),
         ("donor", "CAGGTGAGT", 8)]
dev = "cuda" if torch.cuda.is_available() else "cpu"
from cnn_model_def import Net
ck = torch.load(A.model, map_location=dev, weights_only=False)
net = Net(ck["args"]["dropout"]).to(dev); net.load_state_dict(ck["state"]); net.eval()
for p in net.parameters(): p.requires_grad_(False)
D = np.load("cnn_data.npz", allow_pickle=True); ok = D["ok"]
X = D["X"][ok]; y = D["y"][ok]
te = np.flatnonzero((D["split"][ok] == "test") & (X.max(1) <= 3))
BG = {"real_competitor": X[te[y[te] == 0]], "real_canonical": X[te[y[te] == 1]]}
TBL = np.zeros((256, 4), np.float32); TBL[255] = .25
for i in range(4): TBL[i, i] = 1.
TBLt = torch.from_numpy(TBL).to(dev)
def oh(a): return TBLt[torch.from_numpy(np.ascontiguousarray(a).astype(np.int64)).to(dev)].permute(0,2,1).contiguous()
def fwd(a, bs=8192):
    o = np.empty(len(a), np.float64)
    with torch.inference_mode():
        for k in range(0, len(a), bs): o[k:k+bs] = net(oh(a[k:k+bs])).float().cpu().numpy()
    return o
def tok(s): return np.array([NT.index(c) for c in s], np.uint8)
def probs(s, B):
    m = tok(s); s0 = A.pos+UP-len(m)//2
    q = B.copy(); q[:, s0:s0+len(m)] = m
    return 1/(1+np.exp(-fwd(q)))
DER = [dict(zip(NT, p)) for p in itertools.permutations(NT) if all(a != b for a, b in zip(NT, p))]
def amap(s, m): return "".join(m[c] for c in s)
def boot_med(v, r, n):
    idx = r.integers(0, len(v), size=(n, len(v)))
    return np.percentile(np.median(v[idx], axis=1), [2.5, 97.5])
rows = []
for bn, B in BG.items():
    p0 = 1/(1+np.exp(-fwd(B)))
    rb = np.random.default_rng(A.seed)
    b0 = boot_med(p0, rb, A.nboot)
    log(f"\n=== {bn}  {len(B):,} windows  baseline median {np.median(p0):.4f} "
        f"[{b0[0]:.4f},{b0[1]:.4f}] ===")
    log(f"  {'unit':<11}{'n':>3}{'median p':>10}{'95% CI':>18}{'shuffle':>9}{'sd':>7}"
        f"{'derang':>9}{'sd':>7}")
    for name, unit, maxn in UNITS:
        for n in range(1, maxn+1):
            ins = unit*n
            pv = probs(ins, B); pm = float(np.median(pv))
            lo, hi = boot_med(pv, np.random.default_rng(A.seed+n), A.nboot)
            r = np.random.default_rng(A.seed+n)
            sh = np.array([float(np.median(probs("".join(r.permutation(list(ins))), B)))
                           for _ in range(A.nshuf)])
            dg = np.array([float(np.median(probs(amap(ins, m), B))) for m in DER])
            rows.append((bn, name, unit, n, len(ins), float(np.median(p0)), float(b0[0]),
                         float(b0[1]), pm, float(lo), float(hi), float(sh.mean()),
                         float(sh.std(ddof=1)), float(dg.mean()), float(dg.std(ddof=1)),
                         pm-float(sh.mean()), pm-float(dg.mean()), len(B)))
            log(f"  {name:<11}{n:>3}{pm:>10.4f}  [{lo:.4f},{hi:.4f}]{sh.mean():>9.4f}"
                f"{sh.std(ddof=1):>7.4f}{dg.mean():>9.4f}{dg.std(ddof=1):>7.4f}")
with open("panelB_prob_ci.tsv", "w") as f:
    f.write("background\tname\tunit\tn_copies\tinsert_len\tbase_median_p\tbase_lo\tbase_hi\t"
            "motif_p\tmotif_lo\tmotif_hi\tshuffle_p\tshuffle_sd\tderange_p\tderange_sd\t"
            "motif_minus_shuffle\tmotif_minus_derange\tn_bg\n")
    for r in rows: f.write("\t".join(str(x) for x in r)+"\n")
log("\n[out] panelB_prob_ci.tsv")
