#!/usr/bin/env python3
# implant every 6-mer and rank them, so the most repressive short sequences are found
import sys, time, argparse, itertools
import numpy as np
import torch
from cnn_interp_lib import NT

def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser()
P.add_argument("--model", default="cnn_model.pt")
P.add_argument("--k", type=int, default=6)
P.add_argument("--nbg", type=int, default=256)
P.add_argument("--positions", default="-150,-20,100")
P.add_argument("--chunk", type=int, default=256)
P.add_argument("--bs", type=int, default=768)
P.add_argument("--seed", type=int, default=42)
A = P.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
UP, WIN, K = 2000, 2500, A.k
POS = [int(x) for x in A.positions.split(",")]
rng = np.random.default_rng(A.seed)

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
def oh(a):
    return TBLt[torch.from_numpy(np.ascontiguousarray(a).astype(np.int64)).to(dev)].permute(0, 2, 1).contiguous()
def fwd(a):
    out = np.empty(len(a), np.float64)
    with torch.inference_mode():
        for k in range(0, len(a), A.bs):
            out[k:k+A.bs] = net(oh(a[k:k+A.bs])).float().cpu().numpy()
    return out

KM = np.array(list(itertools.product(range(4), repeat=K)), np.uint8)
TXT = ["".join(NT[i] for i in r) for r in KM]
CNT = np.stack([(KM == i).sum(1) for i in range(4)], 1)
CLS = {}
for i, c in enumerate(map(tuple, CNT)): CLS.setdefault(c, []).append(i)
log(f"[scan] all {len(KM):,} {K}-mers, {len(CLS)} permutation classes "
    f"(sizes {min(map(len,CLS.values()))} to {max(map(len,CLS.values()))}), "
    f"{len(POS)} positions, 2 backgrounds x {A.nbg}")

BGS = (("real_canonical", X[rng.choice(te[y[te] == 1], A.nbg, replace=False)]),
       ("real_competitor", X[rng.choice(te[y[te] == 0], A.nbg, replace=False)]))
rows = []; t0 = time.time()
for bn, B in BGS:
    base = fwd(B)
    for p in POS:
        s0 = p + UP - K // 2
        d = np.empty(len(KM)); se = np.empty(len(KM))
        for c0 in range(0, len(KM), A.chunk):
            c1 = min(c0 + A.chunk, len(KM)); nk = c1 - c0
            M = np.repeat(B[None], nk, 0).reshape(nk, A.nbg, WIN)
            M[:, :, s0:s0+K] = KM[c0:c1, None, :]
            v = fwd(M.reshape(-1, WIN)).reshape(nk, A.nbg) - base[None]
            d[c0:c1] = v.mean(1); se[c0:c1] = v.std(1, ddof=1) / np.sqrt(A.nbg)
        cm = np.full(len(KM), np.nan)
        for c, idx in CLS.items():
            if len(idx) < 2: continue
            tot = d[idx].sum()
            cm[idx] = (tot - d[idx]) / (len(idx) - 1)
        rows.append((bn, p, d, se, cm))
        log(f"[scan] {bn} at {p:+5d}: range {d.min():+.4f} to {d.max():+.4f}   {time.time()-t0:.0f}s")

with open("kmer_scan.tsv", "w") as fh:
    fh.write("background\tposition\tkmer\tat\tgc\tdelta_logit\tse\tclass_mean\tarrangement\tclass_n\n")
    for bn, p, d, se, cm in rows:
        for i in range(len(KM)):
            fh.write(f"{bn}\t{p}\t{TXT[i]}\t{(CNT[i,0]+CNT[i,3])/K:.4f}\t{(CNT[i,1]+CNT[i,2])/K:.4f}\t"
                     f"{d[i]:.5f}\t{se[i]:.5f}\t"
                     f"{'' if np.isnan(cm[i]) else f'{cm[i]:.5f}'}\t"
                     f"{'' if np.isnan(cm[i]) else f'{d[i]-cm[i]:.5f}'}\t"
                     f"{len(CLS[tuple(CNT[i])])}\n")
log("[out] kmer_scan.tsv")

KNOWN = {"AATAAA": "canonical PAS", "ATTAAA": "variant PAS", "TGTAAA": "CFIm UGUA in context",
         "GTAAGT": "U1 5' splice site", "TTTTTT": "poly-T", "TGTGTG": "GU-rich DSE",
         "CCCCCC": "poly-C", "GGGGGG": "poly-G", "GAACCG": "evolved 6-mer"}
for bn, p, d, se, cm in rows:
    o = np.argsort(d)
    log(f"\n[{bn} at {p:+d}]  most repressive 12 of {len(KM):,}")
    log(f"  {'kmer':>7} {'delta':>9} {'se':>7} {'A+T':>5} {'arrangement':>12} {'class n':>8}")
    for i in o[:12]:
        log(f"  {TXT[i]:>7} {d[i]:>+9.4f} {se[i]:>7.4f} {(CNT[i,0]+CNT[i,3])/K:>5.2f} "
            f"{'undefined' if np.isnan(cm[i]) else f'{d[i]-cm[i]:>+12.4f}'} "
            f"{len(CLS[tuple(CNT[i])]):>8}")
    log(f"[{bn} at {p:+d}]  most activating 6")
    for i in o[::-1][:6]:
        log(f"  {TXT[i]:>7} {d[i]:>+9.4f} {se[i]:>7.4f} {(CNT[i,0]+CNT[i,3])/K:>5.2f} "
            f"{'undefined' if np.isnan(cm[i]) else f'{d[i]-cm[i]:>+12.4f}'} "
            f"{len(CLS[tuple(CNT[i])]):>8}")
    r = {t: i for i, t in enumerate(TXT)}
    pct = np.empty(len(KM)); pct[o] = np.arange(len(KM)) / (len(KM) - 1) * 100
    log(f"[{bn} at {p:+d}]  where the named elements land")
    for m, nm in KNOWN.items():
        if m not in r: continue
        i = r[m]
        log(f"  {m} {nm:<22} delta {d[i]:>+8.4f}  percentile {pct[i]:>5.1f}  "
            f"arrangement {'undefined' if np.isnan(cm[i]) else f'{d[i]-cm[i]:>+7.4f}'}")
    atv = (CNT[:, 0] + CNT[:, 3]) / K
    ar = ~np.isnan(cm)
    log(f"[{bn} at {p:+d}]  delta vs A+T r = {np.corrcoef(atv, d)[0,1]:+.3f}; "
        f"A+T explains {np.corrcoef(atv,d)[0,1]**2*100:.0f}% of the variance across all {len(KM):,}; "
        f"the arrangement residual spans {(d-cm)[ar].min():+.4f} to {(d-cm)[ar].max():+.4f}, "
        f"sd {(d-cm)[ar].std():.4f} against a total sd of {d.std():.4f}")
