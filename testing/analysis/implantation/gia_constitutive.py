#!/usr/bin/env python3
# the third gia background: real 3' ends (constitutive sites)
import sys, time
import numpy as np
import torch
from cnn_interp_lib import NT, dinuc_shuffle

def log(*a): print(*a, file=sys.stderr, flush=True)
SEED, NBG, UP = 42, 256, 2000
rng = np.random.default_rng(SEED); torch.manual_seed(SEED)
dev = "cuda" if torch.cuda.is_available() else "cpu"

from cnn_model_def import Net
ck = torch.load("cnn_model.pt", map_location=dev, weights_only=False)
net = Net(ck["args"]["dropout"]).to(dev); net.load_state_dict(ck["state"]); net.eval()
for p in net.parameters(): p.requires_grad_(False)
D = np.load("cnn_data.npz", allow_pickle=True); ok = D["ok"]
X = D["X"][ok]; y = D["y"][ok]; split = D["split"][ok]
te = np.flatnonzero(split == "test")
canon = te[y[te] == 1]
log(f"constitutive background: {len(canon)} real 3' ends in test; sampling {NBG}")

TBL = np.zeros((256, 4), np.float32); TBL[255] = 0.25
for i in range(4): TBL[i, i] = 1.0
TBLt = torch.from_numpy(TBL).to(dev)
def onehot(a): return TBLt[torch.from_numpy(a.astype(np.int64)).to(dev)].permute(0,2,1).contiguous()
def fwd(o, bs=256):
    out = []
    with torch.no_grad():
        for k in range(0, len(o), bs): out.append(net(o[k:k+bs]).float().cpu())
    return torch.cat(out).numpy()

GIA = {"donor_AGGTAAGT": "AGGTAAGT", "donor_CAGGTAAGT": "CAGGTAAGT",
       "donor_perm_ctrl_TAGAGTAG": "TAGAGTAG", "GU_rich_GTGTGTGT": "GTGTGTGT",
       "acceptor_pyAG": "TTTTTTTTTTTTTCAGG",
       "PAS_AATAAA": "AATAAA", "PAS_perm_ctrl_AAAATA": "AAAATA"}
def implant(seqs, motif, pos):
    m = np.array([NT.index(c) for c in motif], np.uint8)
    out = seqs.copy(); s = pos - len(m) // 2
    if s < 0 or s + len(m) > out.shape[1]: return None
    out[:, s:s+len(m)] = m
    return out

B = X[rng.choice(canon, NBG, replace=False)]
positions = sorted(set(list(range(-300, 101, 10)) + list(range(-UP + 100, 500, 100))))
t0 = time.time()
with open("gia_donor_test_constitutive.tsv", "w") as fh:
    fh.write("background\tmotif\tposition\tdelta_logit\tse\tn\n")
    base = fwd(onehot(B))
    log(f"[GIA] constitutive background logit mean {base.mean():+.3f} sd {base.std():.3f}")
    for name, m in GIA.items():
        best = (None, -1e9)
        for pos in positions:
            seqs = implant(B, m, pos + UP)
            if seqs is None: continue
            d = fwd(onehot(seqs)) - base
            fh.write(f"constitutive\t{name}\t{pos}\t{d.mean():.5f}\t"
                     f"{d.std(ddof=1)/np.sqrt(len(d)):.5f}\t{len(d)}\n")
            if d.mean() > best[1]: best = (pos, float(d.mean()))
        log(f"[GIA] constitutive {name:26s} max delta-logit {best[1]:+.4f} at {best[0]:+d} nt")
log(f"[out] gia_donor_test_constitutive.tsv ({time.time()-t0:.0f}s)")
