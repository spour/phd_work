#!/usr/bin/env python3
import sys, argparse, math
import numpy as np
import torch

def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser()
P.add_argument("--model", default="cnn_model.pt")
P.add_argument("--npeak", type=int, default=400)
P.add_argument("--min_s3", type=float, default=10.0)
P.add_argument("--nbg", type=int, default=256)
P.add_argument("--seed", type=int, default=42)
A = P.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
UP, WIN = 2000, 2500
rng = np.random.default_rng(A.seed)
from cnn_interp_lib import NT
from maxentpy import maxent
from maxentpy.maxent import load_matrix3
M3 = load_matrix3()
def s3(s):
    try: return float(maxent.score3(s, matrix=M3))
    except Exception: return -99.0

from cnn_model_def import Net
ck = torch.load(A.model, map_location=dev, weights_only=False)
net = Net(ck["args"]["dropout"]).to(dev); net.load_state_dict(ck["state"]); net.eval()
for p in net.parameters(): p.requires_grad_(False)
D = np.load("cnn_data.npz", allow_pickle=True); ok = D["ok"]
X = D["X"][ok]; y = D["y"][ok]
te = np.flatnonzero((D["split"][ok] == "test") & (X.max(1) <= 3))
TBL = np.zeros((256, 4), np.float32); TBL[255] = 0.25
for i in range(4): TBL[i, i] = 1.0
TBLt = torch.from_numpy(TBL).to(dev)
def oh(a): return TBLt[torch.from_numpy(np.ascontiguousarray(a).astype(np.int64)).to(dev)].permute(0,2,1).contiguous()
def first(a, bs=64):
    out = []
    with torch.inference_mode():
        for k in range(0, len(a), bs): out.append(net.first(oh(a[k:k+bs])).float().cpu().numpy())
    return np.concatenate(out)
def logit(a, bs=512):
    o = np.empty(len(a), np.float64)
    with torch.inference_mode():
        for k in range(0, len(a), bs): o[k:k+bs] = net(oh(a[k:k+bs])).float().cpu().numpy()
    return o

# a: real windows
cand = te[y[te] == 1]
sel = cand[rng.permutation(len(cand))][:A.npeak * 3]
POSN, S3V, KEEP = [], [], []
for i in sel:
    w = "".join(NT[b] for b in X[i])
    v = np.array([s3(w[p:p+23]) for p in range(0, UP)])
    j = int(np.argmax(v))
    if v[j] >= A.min_s3:
        KEEP.append(i); POSN.append(j); S3V.append(v[j])
    if len(KEEP) >= A.npeak: break
KEEP = np.array(KEEP); POSN = np.array(POSN); S3V = np.array(S3V)
log(f"[A] {len(KEEP)} held-out real 3' ends whose strongest upstream acceptor scores >= "
    f"{A.min_s3} (median score3 {np.median(S3V):.2f}, max {S3V.max():.2f})")
ACT = first(X[KEEP])
NF = ACT.shape[1]
def at(a, p, half=11): return a[:, :, max(0,p-half):p+half+1].max(2)
AC = np.stack([at(ACT[i:i+1], POSN[i]+18)[0] for i in range(len(KEEP))])
r2 = rng.integers(200, UP-200, size=(len(KEEP), 40))
BG = np.stack([np.stack([at(ACT[i:i+1], int(q))[0] for q in r2[i]]).mean(0) for i in range(len(KEEP))])
rows = []
enr = np.log2((AC.mean(0)+1e-9)/(BG.mean(0)+1e-9))
cor = np.array([np.corrcoef(AC[:, f], S3V)[0,1] for f in range(NF)])
o = np.argsort(-enr)
log(f"\n[A] per-filter activation at the acceptor against its own background in the same windows")
log(f"  {'filter':>7} {'at acceptor':>12} {'background':>11} {'log2 enrich':>12} "
    f"{'r with score3':>14}")
for f_ in o[:6]:
    log(f"  {f'filter{f_:02d}':>7} {AC[:,f_].mean():>12.4f} {BG[:,f_].mean():>11.4f} "
        f"{enr[f_]:>+12.3f} {cor[f_]:>+14.3f}")
log(f"  ... {NF} filters; max log2 enrichment {enr.max():+.3f} (filter{int(np.argmax(enr)):02d}), "
    f"max |r| with score3 {np.abs(cor).max():.3f} (filter{int(np.argmax(np.abs(cor))):02d})")
for f_ in range(NF):
    rows.append(("A", f"filter{f_:02d}", AC[:,f_].mean(), BG[:,f_].mean(), enr[f_], cor[f_]))

# b: synthetic vs its shuffle
ACCSEQ = "TTTCCTCTCTCTCCCCGCAGGTG"
assert abs(s3(ACCSEQ) - 17.172) < 0.01
def mono(s, k): return "".join(np.random.default_rng(k).permutation(list(s)))
B0 = X[rng.choice(te, A.nbg, replace=False)]
p0 = UP - 400; s0 = p0 - len(ACCSEQ)//2
def implant(seq):
    q = B0.copy(); q[:, s0:s0+len(seq)] = np.array([NT.index(c) for c in seq], np.uint8); return q
Wa = implant(ACCSEQ)
SH = [implant(mono(ACCSEQ, [A.seed, j])) for j in range(8)]
Aa = at(first(Wa), s0+18).mean(0)
As = np.mean([at(first(w), s0+18).mean(0) for w in SH], axis=0)
d = np.log2((Aa+1e-9)/(As+1e-9))
la = logit(Wa).mean(); ls = np.mean([logit(w).mean() for w in SH])
log(f"\n[B] the MaxEnt optimum against 8 shuffles of its own bases, implanted at -400 in "
    f"{A.nbg} windows")
log(f"  {'filter':>7} {'acceptor':>10} {'shuffled':>10} {'log2 ratio':>11}")
for f_ in np.argsort(-np.abs(d))[:6]:
    log(f"  {f'filter{f_:02d}':>7} {Aa[f_]:>10.4f} {As[f_]:>10.4f} {d[f_]:>+11.3f}")
log(f"  largest |log2 ratio| over {NF} filters: {np.abs(d).max():.3f} "
    f"(filter{int(np.argmax(np.abs(d))):02d}); {int((np.abs(d)>=1).sum())} filters differ by 2-fold")
log(f"  the model's OUTPUT on the same windows: acceptor {la:+.4f}, shuffled {ls:+.4f}, "
    f"difference {la-ls:+.4f}")
for f_ in range(NF):
    rows.append(("B", f"filter{f_:02d}", Aa[f_], As[f_], d[f_], np.nan))
with open("acceptor_conv1_probe.tsv", "w") as fh:
    fh.write("test\tfilter\tacceptor\tbackground\tlog2\tr_with_score3\n")
    for r_ in rows: fh.write("\t".join(str(x) for x in r_) + "\n")
log(f"\n[read] detection lives in A and B; usage is the output line in B. A filter bank that "
    f"separates the acceptor from its own shuffle while the output does not is a signal lost "
    f"downstream, not a detector that was never learned.")
log("[out] acceptor_conv1_probe.tsv")
