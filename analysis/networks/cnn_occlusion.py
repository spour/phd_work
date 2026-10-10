#!/usr/bin/env python3
# attribution-free positional importance
import sys, time, argparse
import numpy as np
import torch

def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser()
P.add_argument("--model", default="cnn_model_v2.pt")
P.add_argument("--data", default="cnn_data.npz")
P.add_argument("--win", type=int, default=100)
P.add_argument("--out", default="occlusion_profile.tsv")
P.add_argument("--legacy", action="store_true", help="load cnn_model_def.Net instead of Net2")
A = P.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"

ck = torch.load(A.model, map_location=dev, weights_only=False)
if A.legacy:
    from cnn_model_def import Net
    net = Net(ck["args"]["dropout"]).to(dev)
else:
    from cnn_model_def2 import Net2
    c = ck["cfg"]; net = Net2(ck["args"]["dropout"], c["variant"], c["bn_head"], c["logscale"]).to(dev)
net.load_state_dict(ck["state"]); net.eval()
log(f"model {A.model} epoch {ck['epoch']} " + (f"cfg {ck['cfg']['tag']}" if not A.legacy else "(legacy)"))

D = np.load(A.data, allow_pickle=True); ok = D["ok"]
X = D["X"][ok]; y = D["y"][ok]; gene = D["gene"][ok]; split = D["split"][ok]; UP = int(D["up"])
te = np.flatnonzero(split == "test"); L = X.shape[1]
log(f"test {len(te)} sites, {len(set(gene[te]))} genes, window {L} nt, occluding {A.win} nt at a time")

TBL = np.zeros((256, 4), np.float32); TBL[255] = 0.25
for i in range(4): TBL[i, i] = 1.0
TBLt = torch.from_numpy(TBL).to(dev)
Xte = torch.from_numpy(X[te].astype(np.int64))
yte = y[te]

def scores(mask=None, bs=512):
    out = np.empty(len(te), np.float64)
    with torch.no_grad():
        for k in range(0, len(te), bs):
            oh = TBLt[Xte[k:k+bs].to(dev)].permute(0, 2, 1).contiguous()
            if mask is not None: oh[:, :, mask[0]:mask[1]] = 0.25
            out[k:k+bs] = net(oh).float().cpu().numpy()
    return out

by = {}
for k, g in enumerate(gene[te]):
    r, c = by.setdefault(g, ([], []))
    (r if yte[k] == 1 else c).append(k)
GRP = [(np.array(r), np.array(c)) for r, c in by.values() if r and c]
def winrate(s):
    return float(np.mean([((s[r][:, None] > s[c][None, :]).sum()
                           + 0.5 * (s[r][:, None] == s[c][None, :]).sum()) / (len(r) * len(c))
                          for r, c in GRP]))

t0 = time.time()
base = scores(); wg0 = winrate(base)
log(f"[intact] within-gene {wg0:.4f}  logit sd {base.std():.3f}")

rows = []
for a in range(0, L - A.win + 1, A.win):
    b = a + A.win
    s = scores((a, b)); d = s - base
    r = dict(start=a - UP, end=b - UP, centre=(a + b) / 2 - UP,
             winrate=winrate(s), d_winrate=winrate(s) - wg0,
             mean_abs_dlogit=float(np.abs(d).mean()),
             mean_dlogit_real=float(d[yte == 1].mean()), mean_dlogit_comp=float(d[yte == 0].mean()))
    rows.append(r)
    log(f"  [{r['start']:+6d},{r['end']:+6d})  within-gene {r['winrate']:.4f} "
        f"(d {r['d_winrate']:+.4f})  mean|dlogit| {r['mean_abs_dlogit']:.3f}  "
        f"d_real {r['mean_dlogit_real']:+.3f}  d_comp {r['mean_dlogit_comp']:+.3f}")

with open(A.out, "w") as fh:
    fh.write("start\tend\tcentre\twinrate\td_winrate\tmean_abs_dlogit\tmean_dlogit_real\t"
             "mean_dlogit_comp\tintact_winrate\tn_test\tn_genes\n")
    for r in rows:
        fh.write(f"{r['start']}\t{r['end']}\t{r['centre']:.0f}\t{r['winrate']:.5f}\t"
                 f"{r['d_winrate']:.5f}\t{r['mean_abs_dlogit']:.5f}\t{r['mean_dlogit_real']:.5f}\t"
                 f"{r['mean_dlogit_comp']:.5f}\t{wg0:.5f}\t{len(te)}\t{len(GRP)}\n")

dd = np.array([-r["d_winrate"] for r in rows])
tot = dd[dd > 0].sum()
o = np.argsort(-dd)
log(f"\n[occlusion] intact within-gene {wg0:.4f}; total lift above chance {wg0-0.5:.4f}")
log(f"[occlusion] largest single-window drop {dd[o[0]]:.4f} at "
    f"[{rows[o[0]]['start']:+d},{rows[o[0]]['end']:+d}) = "
    f"{100*dd[o[0]]/max(tot,1e-9):.1f}% of the summed positive drop, and "
    f"{100*dd[o[0]]/(wg0-0.5):.1f}% of the lift above chance")
log(f"[occlusion] top 5 windows: " + "  ".join(
    f"[{rows[i]['start']:+d},{rows[i]['end']:+d}) {dd[i]:+.4f}" for i in o[:5]))
k = int(np.searchsorted(np.cumsum(np.sort(dd)[::-1]), 0.5 * tot)) + 1
log(f"[occlusion] {k} of {len(rows)} windows account for half the summed drop "
    f"(uniform would be {len(rows)//2})")
log(f"[out] {A.out}  ({time.time()-t0:.0f}s)")
