#!/usr/bin/env python3
# a negative control built by substitution, not by shuffling
import sys, argparse, itertools
import numpy as np, torch
def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser(); P.add_argument("--model", default="cnn_model.pt")
P.add_argument("--ndraw", type=int, default=100); P.add_argument("--nperm", type=int, default=500)
P.add_argument("--pos", type=int, default=-230); P.add_argument("--seed", type=int, default=42)
A = P.parse_args()
NT = "ACGT"; UP = 2000
SEQ = {"model_me3": "ACGGTGTACTTTTGTTGCAGGTG",
       "maxent_optimum": "TTTCCTCTCTCTCCCCGCAGGTG",
       "t_rich_noC": "TTTTTTTTTGTGTGTTTTAGGTT",
       "robust_agdep": "CCATGCCTTTGATCCTGCAGATG"}
dev = "cuda" if torch.cuda.is_available() else "cpu"
from maxentpy import maxent
from maxentpy.maxent import load_matrix3
M3 = load_matrix3()
def s3(s):
    try: return float(maxent.score3(s, matrix=M3))
    except Exception: return float("nan")
from cnn_model_def import Net
ck = torch.load(A.model, map_location=dev, weights_only=False)
net = Net(ck["args"]["dropout"]).to(dev); net.load_state_dict(ck["state"]); net.eval()
for p in net.parameters(): p.requires_grad_(False)
D = np.load("cnn_data.npz", allow_pickle=True); ok = D["ok"]
X = D["X"][ok]; y = D["y"][ok]
te = np.flatnonzero((D["split"][ok] == "test") & (X.max(1) <= 3))
B = X[te[y[te] == 1]]
TBL = np.zeros((256, 4), np.float32); TBL[255] = .25
for i in range(4): TBL[i, i] = 1.
TBLt = torch.from_numpy(TBL).to(dev)
def oh(a): return TBLt[torch.from_numpy(np.ascontiguousarray(a).astype(np.int64)).to(dev)].permute(0,2,1).contiguous()
def fwd(a, bs=4096):
    o = np.empty(len(a), np.float64)
    with torch.inference_mode():
        for k in range(0, len(a), bs): o[k:k+bs] = net(oh(a[k:k+bs])).float().cpu().numpy()
    return o
base = fwd(B)
def tok(s): return np.array([NT.index(c) for c in s], np.uint8)
def deltas(seqs, chunk=48):
    out = np.empty(len(seqs))
    for i in range(0, len(seqs), chunk):
        cs = seqs[i:i+chunk]
        q = np.empty((len(cs)*len(B), B.shape[1]), B.dtype)
        for j, s in enumerate(cs):
            m = tok(s); s0 = A.pos+UP-len(m)//2
            blk = B.copy(); blk[:, s0:s0+len(m)] = m
            q[j*len(B):(j+1)*len(B)] = blk
        out[i:i+chunk] = (fwd(q).reshape(len(cs), len(B))-base[None, :]).mean(1)
    return out
SHIFT = [{a: NT[(NT.index(a)+k) % 4] for a in NT} for k in (1, 2, 3)]
DERANGE = [dict(zip(NT, p)) for p in itertools.permutations(NT)
           if all(a != b for a, b in zip(NT, p))]
def apply_map(s, m): return "".join(m[c] for c in s)
log(f"[bg] {len(B):,} real 3' end windows; fixed position {A.pos}; no maximum")
log(f"[maps] {len(SHIFT)} cyclic shifts, {len(DERANGE)} derangements")
rows, lad = [], []
for name, s in SEQ.items():
    d0 = float(deltas([s])[0])
    sh = deltas([apply_map(s, m) for m in SHIFT])
    dg = deltas([apply_map(s, m) for m in DERANGE])
    pr = np.random.default_rng(A.seed)
    pv = deltas(["".join(pr.permutation(list(s))) for _ in range(A.nperm)])
    rows.append((name, s, s3(s), d0, float(sh.mean()), float(sh.std(ddof=1)),
                 float(dg.mean()), float(dg.std(ddof=1)), float(dg.max()),
                 float(pv.mean()), d0-float(sh.mean()), d0-float(dg.mean()), d0-float(pv.mean()),
                 (d0-float(dg.mean()))/float(dg.std(ddof=1))))
    log(f"\n{name}  {s}  score3 {s3(s):+.2f}")
    log(f"  intact                          {d0:+.4f}")
    for k, (m, v) in enumerate(zip(SHIFT, sh), 1):
        log(f"  cyclic shift {k}  {apply_map(s,m)}  {v:+.4f}")
    log(f"  MEAN OF THE THREE SHIFTS        {sh.mean():+.4f}   intact is {d0-sh.mean():+.4f} above")
    log(f"  mean of all 9 derangements      {dg.mean():+.4f} sd {dg.std(ddof=1):.4f} "
        f"max {dg.max():+.4f}   intact is {d0-dg.mean():+.4f} above, z {(d0-dg.mean())/dg.std(ddof=1):+.1f}")
    log(f"  mean of {A.nperm} shuffles           {pv.mean():+.4f}   intact is {d0-pv.mean():+.4f} above")
    r = np.random.default_rng(A.seed)
    for k in range(1, len(s)+1):
        var = []
        for _ in range(A.ndraw):
            idx = r.choice(len(s), k, replace=False); t = list(s)
            for i in idx: t[i] = r.choice([b for b in NT if b != s[i]])
            var.append("".join(t))
        v = deltas(var)
        lad.append((name, k, float(v.mean()), float(v.std(ddof=1)/np.sqrt(A.ndraw))))
with open("acceptor_subcontrol.tsv", "w") as f:
    f.write("name\tsequence\tscore3\tintact\tshift_mean\tshift_sd\tderange_mean\tderange_sd\t"
            "derange_max\tperm_mean\tgain_vs_shift\tgain_vs_derange\tgain_vs_perm\tz_vs_derange\tn_bg\n")
    for r_ in rows: f.write("\t".join(str(x) for x in r_)+f"\t{len(B)}\n")
with open("acceptor_subcontrol_ladder.tsv", "w") as f:
    f.write("name\tn_substituted\tmean_delta\tse\n")
    for r_ in lad: f.write("\t".join(str(x) for x in r_)+"\n")
log(f"\n{'name':<16}{'intact':>9}{'3 shifts':>10}{'9 derang':>10}{'shuffle':>9}"
    f"{'vs shift':>10}{'vs derang':>11}{'vs shuf':>9}")
for r_ in rows:
    log(f"{r_[0]:<16}{r_[3]:>+9.3f}{r_[4]:>+10.3f}{r_[6]:>+10.3f}{r_[9]:>+9.3f}"
        f"{r_[10]:>+10.3f}{r_[11]:>+11.3f}{r_[12]:>+9.3f}")
log("[out] acceptor_subcontrol.tsv / acceptor_subcontrol_ladder.tsv")
