#!/usr/bin/env python3
# the reported model's first-layer filters under the count-matched threshold
import sys, numpy as np, torch
from cnn_interp_lib import NT, REF_MOTIFS, consensus_pwm, pwm_match, match_threshold

def log(*a): print(*a, file=sys.stderr, flush=True)
SEED, W, NF, NSEQ = 42, 19, 32, 4000
rng = np.random.default_rng(SEED)
from cnn_model_def import Net
ck = torch.load("cnn_model.pt", map_location="cpu", weights_only=False)
net = Net(ck["args"]["dropout"]); net.load_state_dict(ck["state"]); net.eval()
for p in net.parameters(): p.requires_grad_(False)
D = np.load("cnn_data.npz", allow_pickle=True); ok = D["ok"]
X = D["X"][ok]; te = np.flatnonzero(D["split"][ok] == "test")
TBL = np.zeros((256, 4), np.float32); TBL[255] = 0.25
for i in range(4): TBL[i, i] = 1.0
TBLt = torch.from_numpy(TBL)
def oh(c): return TBLt[torch.from_numpy(c.astype(np.int64))].permute(0, 2, 1).contiguous()

sub = rng.choice(te, min(NSEQ, len(te)), replace=False)
acc = []
for k in range(0, 400, 128):
    with torch.no_grad(): acc.append(net.first(oh(X[sub[k:k+128]])).numpy())
a0 = np.concatenate(acc, 0)
thr = np.quantile(a0.transpose(1, 0, 2).reshape(NF, -1), 0.999, axis=1)
del a0, acc
counts = np.zeros((NF, W, 4)); nsite = np.zeros(NF, int)
for k in range(0, len(sub), 128):
    ii = sub[k:k+128]
    with torch.no_grad(): a = net.first(oh(X[ii])).numpy()
    for f in range(NF):
        b, p = np.where(a[:, f, :] >= thr[f])
        for bi, pi in zip(b, p):
            s = pi - W // 2
            if s < 0 or s + W > X.shape[1]: continue
            sq = X[ii[bi], s:s+W]; g = sq < 4
            counts[f, np.arange(W)[g], sq[g]] += 1; nsite[f] += 1
PWM = np.where(counts.sum(2, keepdims=True) > 0,
               counts / np.maximum(counts.sum(2, keepdims=True), 1), 0.25)
IC = np.nansum(PWM * np.log2(np.maximum(PWM, 1e-9) / 0.25), (1, 2)); ICC = IC / W
refs = {k: consensus_pwm(v) for k, v in REF_MOTIFS.items()}
thr95 = match_threshold(refs, np.random.default_rng(SEED))
with open("cnn_filters_matched.tsv", "w") as fh:
    fh.write("filter\tic_bits\tic_per_col\tn_sites\tbest_match\tr\tstrand\tbeats_null\t"
             "consensus\tfrac_AT\twidth\n")
    nm = 0
    for f in range(NF):
        cands = [(pwm_match(PWM[f], q)[0], pwm_match(PWM[f], q)[1], k) for k, q in refs.items()]
        r_, st, k_ = max(cands); beats = r_ >= thr95[k_]; nm += int(beats)
        fh.write(f"filter{f:02d}\t{IC[f]:.3f}\t{ICC[f]:.4f}\t{nsite[f]}\t"
                 f"{k_ if beats else 'none'}\t{r_:.3f}\t{st}\t{int(beats)}\t"
                 f"{''.join(NT[i] for i in PWM[f].argmax(1))}\t"
                 f"{float(PWM[f][:,[0,3]].sum()/W):.4f}\t{W}\n")
log(f"[cnn] count-matched: IC {ICC.min():.3f}-{ICC.max():.3f} bits/col "
    f"(median {np.median(ICC):.3f}), median {int(np.median(nsite))} sites/filter, "
    f"{nm}/{NF} beat the random-PWM null")
log("[out] cnn_filters_matched.tsv")
