#!/usr/bin/env python3
# the missing link
import sys, math, glob
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

def log(*a): print(*a, file=sys.stderr, flush=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
D = np.load("cnn_data.npz", allow_pickle=True); ok = D["ok"]
X = D["X"][ok]; y = D["y"][ok]; gene = D["gene"][ok]; split = D["split"][ok]
S = np.load("acceptor_scalar.npz"); ACC = S["acc"].astype(np.float32)
te = np.flatnonzero(split == "test")
mu, sd = ACC[split == "train"].mean(), ACC[split == "train"].std()
ACCz = ((ACC - mu) / sd).astype(np.float32)
TBL = np.zeros((256, 4), np.float32); TBL[255] = 0.25
for i in range(4): TBL[i, i] = 1.0
TBLt = torch.from_numpy(TBL).to(dev)
Xt = torch.from_numpy(X.astype(np.int64)); At = torch.from_numpy(ACCz)
def bo(ii): return TBLt[Xt[ii].to(dev)].permute(0, 2, 1).contiguous()

class V(nn.Module):
    def __init__(s, drop=0.4, pools=(10,5,2), aux=False):
        super().__init__()
        s.c1 = nn.Conv1d(4,32,19,padding=9,bias=False); s.b1 = nn.BatchNorm1d(32)
        s.c2 = nn.Conv1d(32,64,7,padding=3,bias=False); s.b2 = nn.BatchNorm1d(64)
        s.c3 = nn.Conv1d(64,96,5,padding=2,bias=False); s.b3 = nn.BatchNorm1d(96)
        s.p = pools; s.aux = aux; nb = 2500//(pools[0]*pools[1]*pools[2]); s.nb = nb
        s.do = nn.Dropout(drop); s.fc1 = nn.Linear(96*nb+(1 if aux else 0),64); s.fc2 = nn.Linear(64,1)
    def forward(s, x, a=None):
        h = F.max_pool1d(torch.exp(torch.clamp(s.b1(s.c1(x)),max=8.0)), s.p[0])
        h = F.max_pool1d(F.relu(s.b2(s.c2(h))), s.p[1])
        h = F.max_pool1d(F.relu(s.b3(s.c3(h))), s.p[2])
        h = h.flatten(1)
        if s.aux: h = torch.cat([h, a[:,None]], 1)
        return s.fc2(s.do(F.relu(s.fc1(s.do(h))))).squeeze(1)

def pear(x, z):
    m = np.isfinite(x)&np.isfinite(z); x,z = x[m],z[m]; n=len(x)
    r = float(np.corrcoef(x,z)[0,1]); f = 0.5*math.log((1+r)/(1-r)); se = 1/math.sqrt(n-3)
    return r, math.tanh(f-1.96*se), math.tanh(f+1.96*se), n
rows = []
log(f"correlation of each model's TEST score with real upstream acceptor strength "
    f"({len(te):,} peaks)")
log(f"  {'model':<18} {'all':>22} {'real 3 ends only':>26} {'competitors':>22}")
for ckp in sorted(glob.glob("acc_arch_*_s42.pt")) + ["cnn_model.pt"]:
    ck = torch.load(ckp, map_location=dev, weights_only=False)
    if ckp == "cnn_model.pt":
        from cnn_model_def import Net
        net = Net(ck["args"]["dropout"]).to(dev); nm = "reported CNN"; aux = False
        net.load_state_dict(ck["state"])
        fwd1 = lambda j: net(bo(j))
    else:
        nm = ck["variant"]; aux = ck["aux"]
        net = V(0.4, tuple(ck["pools"]), aux).to(dev); net.load_state_dict(ck["state"])
        fwd1 = lambda j: net(bo(j), At[j].to(dev))
    net.eval()
    sc = np.empty(len(te), np.float64)
    with torch.no_grad():
        for k in range(0, len(te), 512):
            j = te[k:k+512]; sc[k:k+512] = fwd1(j).float().cpu().numpy()
    a = ACC[te]; c = y[te] == 1
    out = []
    for lab, m in (("all", np.ones(len(te), bool)), ("real", c), ("comp", ~c)):
        r, lo, hi, n = pear(a[m], sc[m]); out.append(f"{r:+.4f} [{lo:+.4f},{hi:+.4f}]")
        rows.append((nm, lab, n, r, lo, hi))
    log(f"  {nm:<18} {out[0]:>22} {out[1]:>26} {out[2]:>22}")
with open("acceptor_arch_followup.tsv", "w") as fh:
    fh.write("model\tscope\tn\tr\tlo\thi\n")
    for r_ in rows: fh.write("\t".join(str(x) for x in r_) + "\n")
log("[out] acceptor_arch_followup.tsv")
