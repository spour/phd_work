#!/usr/bin/env python3
# is +27 a big effect or an off-scale one?
import sys, argparse
import numpy as np, torch
def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser(); P.add_argument("--model", default="cnn_model.pt")
P.add_argument("--pos", type=int, default=-230); A = P.parse_args()
NT = "ACGT"; UP = 2000
CON = {"free_max_x1": "GCCGACGGAGGACCGACCGATTC",
       "free_max_x7": "GCCGACGGAGGACCGACCGATTC"*7,
       "model_me3_x1": "ACGGTGTACTTTTGTTGCAGGTG",
       "maxent_optimum_x1": "TTTCCTCTCTCTCCCCGCAGGTG",
       "u1_donor_x1": "CAGGTGAGT", "u1_donor_x14": "CAGGTGAGT"*14}
dev = "cuda" if torch.cuda.is_available() else "cpu"
from cnn_model_def import Net
ck = torch.load(A.model, map_location=dev, weights_only=False)
net = Net(ck["args"]["dropout"]).to(dev); net.load_state_dict(ck["state"]); net.eval()
for p in net.parameters(): p.requires_grad_(False)
D = np.load("cnn_data.npz", allow_pickle=True); ok = D["ok"]
X = D["X"][ok]; y = D["y"][ok]; sp = D["split"][ok]
te = np.flatnonzero((sp == "test") & (X.max(1) <= 3))
TBL = np.zeros((256, 4), np.float32); TBL[255] = .25
for i in range(4): TBL[i, i] = 1.
TBLt = torch.from_numpy(TBL).to(dev)
def oh(a): return TBLt[torch.from_numpy(np.ascontiguousarray(a).astype(np.int64)).to(dev)].permute(0,2,1).contiguous()
def fwd(a, bs=4096):
    o = np.empty(len(a), np.float64)
    with torch.inference_mode():
        for k in range(0, len(a), bs): o[k:k+bs] = net(oh(a[k:k+bs])).float().cpu().numpy()
    return o
Lte = fwd(X[te]); yte = y[te]
def q(v, p): return float(np.percentile(v, p))
log(f"[test] {len(te):,} held-out windows, {int((yte==1).sum()):,} real 3' ends, "
    f"{int((yte==0).sum()):,} competitors")
log(f"\nTHE MODEL'S OWN LOGIT RANGE ON REAL SEQUENCE")
for lab, v in (("all test windows", Lte), ("real 3' ends", Lte[yte == 1]), ("competitors", Lte[yte == 0])):
    log(f"  {lab:<18} min {v.min():+7.2f}  p1 {q(v,1):+6.2f}  median {q(v,50):+6.2f}  "
        f"p99 {q(v,99):+6.2f}  max {v.max():+7.2f}")
OBSMAX = float(Lte.max()); OBSMIN = float(Lte.min())
B = X[te[yte == 1]]; base = fwd(B)
log(f"\n  the implantation background is the {len(B):,} real 3' ends, baseline logit "
    f"median {q(base,50):+.2f}, max {base.max():+.2f}")
def tok(s): return np.array([NT.index(c) for c in s], np.uint8)
rows = []
log(f"\nWHERE EACH CONSTRUCT LANDS")
log(f"  {'construct':<20}{'delta':>9}{'median logit':>14}{'max logit':>11}"
    f"{'% past obs max':>16}{'median p':>10}")
for name, s in CON.items():
    m = tok(s); s0 = A.pos+UP-len(m)//2
    qy = B.copy(); qy[:, s0:s0+len(m)] = m
    v = fwd(qy); d = float((v-base).mean())
    frac = float((v > OBSMAX).mean())*100
    pmed = 1/(1+np.exp(-q(v, 50)))
    rows.append((name, s, d, q(v, 50), float(v.max()), frac, pmed, OBSMAX, OBSMIN))
    log(f"  {name:<20}{d:>+9.3f}{q(v,50):>+14.2f}{v.max():>+11.2f}{frac:>15.1f}%{pmed:>10.4f}")
log(f"\n  observed range on real sequence: {OBSMIN:+.2f} to {OBSMAX:+.2f}")
with open("logit_scale_check.tsv", "w") as f:
    f.write("construct\tsequence\tdelta_logit\tmedian_logit\tmax_logit\tpct_past_observed_max\t"
            "median_prob\tobserved_max\tobserved_min\tn_bg\n")
    for r in rows: f.write("\t".join(str(x) for x in r)+f"\t{len(B)}\n")
log(f"\nHEADROOM. From the background median {q(base,50):+.2f}, the distance to the highest logit")
log(f"the model emits on real sequence is {OBSMAX-q(base,50):+.2f} and to the lowest is "
    f"{OBSMIN-q(base,50):+.2f}.")
log("[out] logit_scale_check.tsv")
