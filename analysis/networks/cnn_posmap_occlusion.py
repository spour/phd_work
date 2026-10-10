#!/usr/bin/env python3
# the signed filter x position contribution map
import sys, time, argparse
import numpy as np
import torch, torch.nn.functional as F

def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser()
P.add_argument("--model", default="cnn_model.pt")
P.add_argument("--data", default="cnn_data.npz")
P.add_argument("--chunk", type=int, default=512)
P.add_argument("--out", default="cnn_posmap_occlusion.tsv")
A = P.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
NBINS, BINW, UP = 25, 100, 2000

from cnn_model_def import Net
ck = torch.load(A.model, map_location=dev, weights_only=False)
net = Net(ck["args"]["dropout"]).to(dev); net.load_state_dict(ck["state"]); net.eval()
for p in net.parameters(): p.requires_grad_(False)

D = np.load(A.data, allow_pickle=True); ok = D["ok"]
X = D["X"][ok]; y = D["y"][ok]; gene = D["gene"][ok]; split = D["split"][ok]
te = np.flatnonzero(split == "test"); yte = y[te]
NF = net.c1.out_channels
log(f"test {len(te)} sites, {len(set(gene[te]))} genes; {NF} filters x {NBINS} bins = "
    f"{NF*NBINS} occlusions")

TBL = np.zeros((256, 4), np.float32); TBL[255] = 0.25
for i in range(4): TBL[i, i] = 1.0
TBLt = torch.from_numpy(TBL).to(dev)
Xte = torch.from_numpy(X[te].astype(np.int64))
def oh(sl): return TBLt[Xte[sl].to(dev)].permute(0, 2, 1).contiguous()

def head_from_pooled(p):
    h = F.max_pool1d(F.relu(net.b2(net.c2(p))), 5)
    h = F.max_pool1d(F.relu(net.b3(net.c3(h))), 2)
    h = h.flatten(1)
    return net.fc2(F.relu(net.fc1(h))).squeeze(1)

# equivalence check against literal input-level occlusion
with torch.no_grad():
    x = oh(slice(0, 64)); a = net.first(x); p = F.max_pool1d(a, 10)
    ref = head_from_pooled(p)
    assert float((ref - net(x)).abs().max()) < 1e-3, "pooled re-entry != full forward"
    worst = 0.0
    for f, b in ((0, 0), (7, 12), (27, 20), (31, 24)):
        a2 = a.clone(); a2[:, f, b*BINW:(b+1)*BINW] = 0.0
        lit = head_from_pooled(F.max_pool1d(a2, 10))
        p2 = p.clone(); p2[:, f, b*10:(b+1)*10] = 0.0
        fast = head_from_pooled(p2)
        worst = max(worst, float((lit - fast).abs().max()))
    log(f"[check] pooled-level occlusion == input-level occlusion, max |diff| {worst:.2e}")
    assert worst < 1e-3

# gene groups on the test set
by = {}
for k, g in enumerate(gene[te]):
    r, c = by.setdefault(g, ([], []))
    (r if yte[k] == 1 else c).append(k)
GRP = [(np.array(r), np.array(c)) for r, c in by.values() if r and c]
def winrate(s):
    return float(np.mean([((s[r][:, None] > s[c][None, :]).sum()
                           + 0.5 * (s[r][:, None] == s[c][None, :]).sum()) / (len(r) * len(c))
                          for r, c in GRP]))

# sweep
t0 = time.time()
S = np.zeros((NF, NBINS, len(te)), np.float32)
intact = np.empty(len(te), np.float32)
with torch.no_grad():
    for k in range(0, len(te), A.chunk):
        sl = slice(k, min(k + A.chunk, len(te)))
        p = F.max_pool1d(net.first(oh(sl)), 10)
        intact[sl] = head_from_pooled(p).float().cpu().numpy()
        for f in range(NF):
            keep = p[:, f, :].clone()
            for b in range(NBINS):
                p[:, f, b*10:(b+1)*10] = 0.0
                S[f, b, sl] = head_from_pooled(p).float().cpu().numpy()
                p[:, f, b*10:(b+1)*10] = keep[:, b*10:(b+1)*10]
        if (k // A.chunk) % 5 == 0:
            log(f"  {min(k+A.chunk,len(te))}/{len(te)} sites  {time.time()-t0:.0f}s")
log(f"sweep done in {time.time()-t0:.0f}s")

wg0 = winrate(intact)
log(f"[intact] within-gene {wg0:.4f}  logit mean {intact.mean():+.3f}")
mreal = yte == 1; mcomp = yte == 0
rows = []
for f in range(NF):
    for b in range(NBINS):
        d = intact - S[f, b]
        contrib = float(d[mreal].mean() - d[mcomp].mean())
        rows.append(dict(filt=f"filter{f:02d}", b=b, start=b*BINW-UP, end=(b+1)*BINW-UP,
                         contrib=contrib, d_real=float(d[mreal].mean()),
                         d_comp=float(d[mcomp].mean()),
                         winrate=winrate(S[f, b])))
with open(A.out, "w") as fh:
    fh.write("filter\tbin\tbin_start\tbin_end\tcontrib\tmean_dlogit_real\tmean_dlogit_comp\t"
             "winrate_occluded\td_winrate\tintact_winrate\tn_test\tn_genes\n")
    for r in rows:
        fh.write(f"{r['filt']}\t{r['b']}\t{r['start']}\t{r['end']}\t{r['contrib']:.6f}\t"
                 f"{r['d_real']:.6f}\t{r['d_comp']:.6f}\t{r['winrate']:.6f}\t"
                 f"{r['winrate']-wg0:.6f}\t{wg0:.6f}\t{len(te)}\t{len(GRP)}\n")

C = np.array([[r["contrib"] for r in rows[f*NBINS:(f+1)*NBINS]] for f in range(NF)])
tot = np.abs(C).sum(1); order = np.argsort(-tot)
log(f"\n[posmap] filters ranked by summed |class contribution| over bins:")
for f in order[:8]:
    b = int(np.argmax(np.abs(C[f])))
    log(f"  filter{f:02d}  sum|contrib| {tot[f]:.4f}   peak {C[f,b]:+.4f} at "
        f"[{b*BINW-UP:+d},{(b+1)*BINW-UP:+d})   positive bins {int((C[f]>0).sum())}/{NBINS}")
np.savez_compressed("cnn_posmap_contrib.npz", contrib=C, intact_winrate=wg0,
                    bins=np.arange(NBINS)*BINW-UP)
log(f"[out] {A.out} cnn_posmap_contrib.npz")
