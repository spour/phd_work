#!/usr/bin/env python3
# is u1 repressive in motifnet2, and does it mark competitors?
import sys, math, glob
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.argv = [sys.argv[0]]
exec(open("motifnet2.py").read().split("D = np.load(\"cnn_data.npz\"")[0]
     .replace('A = P.parse_args()', 'A = P.parse_args([])'))
def log(*a): print(*a, file=sys.stderr, flush=True)
D = np.load("cnn_data.npz", allow_pickle=True); ok = D["ok"]
X = D["X"][ok]; y = D["y"][ok]; split = D["split"][ok]
te = np.flatnonzero(split == "test")
U1S = np.load("u1_scalar.npz")["u1"].astype(float)
ACCS = np.load("acceptor_scalar.npz")["acc"].astype(float)
TBL = np.zeros((256,4), np.float32); TBL[255] = 0.25
for i in range(4): TBL[i,i] = 1.0
TBLt = torch.from_numpy(TBL).to(dev)
Xt = torch.from_numpy(X.astype(np.int64))
def bo(ii): return TBLt[Xt[ii].to(dev)].permute(0,2,1).contiguous()
def pear(a,b):
    m=np.isfinite(a)&np.isfinite(b); a,b=a[m],b[m]; n=len(a)
    r=float(np.corrcoef(a,b)[0,1]); z=0.5*math.log((1+r)/(1-r)); se=1/math.sqrt(n-3)
    return r, math.tanh(z-1.96*se), math.tanh(z+1.96*se)
log(f"{'seed':>5} {'U1 head vs classifier score':>30} {'within real ends':>20} "
    f"{'within competitors':>20}   real U1 mean: real vs comp")
for p in sorted(glob.glob("motifnet2_s*.pt")):
    ck = torch.load(p, map_location=dev, weights_only=False)
    net = MotifNet2(ck["args"]["nbank"], ck["args"]["width"], ck["args"]["dropout"]).to(dev)
    net.load_state_dict(ck["state"]); net.eval()
    S = np.empty(len(te)); Uh = np.empty(len(te))
    with torch.no_grad():
        for k in range(0, len(te), 512):
            j = te[k:k+512]; a,_,u,_ = net(bo(j))
            S[k:k+512] = a.float().cpu().numpy(); Uh[k:k+512] = u.float().cpu().numpy()
    c = y[te] == 1
    r0 = pear(Uh, S); r1 = pear(Uh[c], S[c]); r2 = pear(Uh[~c], S[~c])
    seed = p.split("_s")[1].split(".")[0]
    log(f"{seed:>5} {f'{r0[0]:+.4f} [{r0[1]:+.4f},{r0[2]:+.4f}]':>30} "
        f"{f'{r1[0]:+.4f}':>20} {f'{r2[0]:+.4f}':>20}   "
        f"{Uh[c].mean():+.3f} vs {Uh[~c].mean():+.3f}")
log(f"\nfor reference the REAL U1 feature: mean at real 3' ends {U1S[te][y[te]==1].mean():.3f}, "
    f"at competitors {U1S[te][y[te]==0].mean():.3f}")
log("a NEGATIVE correlation means the model treats predicted U1 as evidence AGAINST a real 3' end")
