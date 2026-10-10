#!/usr/bin/env python3
# dose-response for implanted 5' splice sites, one to seven tandem copies
import sys, time, argparse
import numpy as np
import torch
from cnn_interp_lib import NT, dinuc_shuffle

def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser()
P.add_argument("--model", default="cnn_model.pt")
P.add_argument("--gia_bg", type=int, default=256)
P.add_argument("--max_copies", type=int, default=7)
P.add_argument("--nscram", type=int, default=10, help="dinucleotide shuffles per copy number")
P.add_argument("--seed", type=int, default=42)
A = P.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
UP = 2000
rng = np.random.default_rng(A.seed)

sys.path.insert(0, "scripts")
import u1_5ss_landscape_data as L
from maxentpy import maxent
from maxentpy.maxent import load_matrix5
M5 = load_matrix5()
def u1dg(s):
    if len(s) != 9 or any(c not in "ACGT" for c in s): return np.nan
    return float(L.U1[sum(int(L.POW[p]) * "ACGT".index(s[p]) for p in range(9))])

U1X = "".join("ACGT"[(int(np.nanargmax(L.U1)) // int(L.POW[p])) % 4] for p in range(9))
assert U1X == "CAGGTGAGT", U1X
SPACER = "CTCTCT"
ACC = "TTTTTTTTTTTTTCAGG"
log(f"[env] device {dev}; donor unit {U1X} (-dG {u1dg(U1X):.2f}), spacer {SPACER}, "
    f"acceptor {ACC}")

def construct(n): return (U1X + SPACER) * (n - 1) + U1X
def dishuf(s, tag, j):
    tok = np.array(["ACGT".index(c) for c in s], np.uint8)
    r = np.random.default_rng([A.seed, sum(map(ord, tag)), j])
    return "".join("ACGT"[i] for i in dinuc_shuffle(tok, r))

def gtn(s): return sum(1 for i in range(len(s) - 1) if s[i:i+2] == "GT")
SEQ = [("acceptor_pyAG", ACC, "3' splice site (pyrimidine tract + AG)", "motif", 0)]
for n in range(1, A.max_copies + 1):
    c = construct(n)
    SEQ.append((f"donor_x{n}", c, f"{n} tandem 5' splice site" + ("" if n == 1 else "s"),
                "motif", n))
    for j in range(A.nscram):
        sc = dishuf(c, f"donor_x{n}", j)
        assert gtn(sc) == gtn(c), f"dinucleotide shuffle changed the GT count for n={n}"
        SEQ.append((f"donor_x{n}_ctrl{j}", sc, f"{n} tandem, dinucleotide shuffled",
                    "control", n))
intact = {n: sum(sum(1 for i in range(len(s_) - 8) if s_[i:i+9] == U1X)
                 for k, s_, _, kind, nn in SEQ if kind == "control" and nn == n)
          for n in range(1, A.max_copies + 1)}
log(f"[check] intact {U1X} copies surviving across all shuffles, by n: {intact}")
log(f"\n[panel] {len(SEQ)} sequences ({A.nscram} shuffles per copy number)")
for k, s_, lab, kind, n in SEQ:
    if kind == "control" and not k.endswith("_ctrl0"): continue
    log(f"  {k:>20} {kind:>7} n={n} len={len(s_):>3}  GT x{gtn(s_):<3} {s_[:52]}"
        + ("..." if len(s_) > 52 else ""))

from cnn_model_def import Net
ck = torch.load(A.model, map_location=dev, weights_only=False)
net = Net(ck["args"]["dropout"]).to(dev); net.load_state_dict(ck["state"]); net.eval()
for p in net.parameters(): p.requires_grad_(False)
D = np.load("cnn_data.npz", allow_pickle=True); ok = D["ok"]
X = D["X"][ok]; y = D["y"][ok]; te = np.flatnonzero(D["split"][ok] == "test")
TBL = np.zeros((256, 4), np.float32); TBL[255] = 0.25
for i in range(4): TBL[i, i] = 1.0
TBLt = torch.from_numpy(TBL).to(dev)
def oh(a): return TBLt[torch.from_numpy(a.astype(np.int64)).to(dev)].permute(0, 2, 1).contiguous()
def fwd(a, bs=256):
    out = []
    with torch.no_grad():
        for k in range(0, len(a), bs): out.append(net(oh(a[k:k+bs])).float().cpu().numpy())
    return np.concatenate(out)
def implant(seqs, motif, pos):
    m = np.array([NT.index(c) for c in motif], np.uint8)
    o = seqs.copy(); s = pos - len(m) // 2
    if s < 0 or s + len(m) > o.shape[1]: return None
    o[:, s:s+len(m)] = m
    return o

BG = np.stack([dinuc_shuffle(X[i], rng) for i in rng.choice(te, A.gia_bg, replace=False)])
BGR = X[rng.choice(te[y[te] == 0], A.gia_bg, replace=False)]
POS = sorted(set(list(range(-300, 101, 10)) + list(range(-UP + 100, 500, 100))))
log(f"\n[gia] {len(SEQ)} x {len(POS)} positions x 2 backgrounds x {A.gia_bg}")

BEST = {}; rows = []; t0 = time.time()
for bn, B, bl in (("dinuc_shuffled", BG, "Dinucleotide-shuffled background"),
                  ("real_competitor", BGR, "Real competitor background")):
    base = fwd(B)
    for k, s_, lab, kind, n in SEQ:
        bst = (None, -1e9, 0.0)
        for p_ in POS:
            q = implant(B, s_, p_ + UP)
            if q is None: continue
            d = fwd(q) - base
            m_, se_ = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
            rows.append((bn, bl, k, lab, kind, n, len(s_), p_, m_, se_, A.gia_bg))
            if m_ > bst[1]: bst = (p_, m_, se_)
        BEST[(bn, k)] = bst
    log(f"[gia] {bn} done  {time.time()-t0:.0f}s")
with open("gia_donor_dose.tsv", "w") as fh:
    fh.write("background\tbg_label\tseq_id\tlabel\tkind\tn_copies\tinsert_len\tposition\t"
             "delta_logit\tse\tn_bg\tbest_position\tbest_delta\tbest_se\n")
    for r in rows:
        b = BEST[(r[0], r[2])]
        fh.write("\t".join(str(x) if not isinstance(x, float) else f"{x:.5f}" for x in r)
                 + f"\t{b[0]}\t{b[1]:.5f}\t{b[2]:.5f}\n")
log("[out] gia_donor_dose.tsv")

acc = BEST[("dinuc_shuffled", "acceptor_pyAG")][1]
log(f"\n[check] acceptor {acc:+.4f} against the published +0.27169")
assert abs(acc - 0.27169) < 0.08
def ctrl_stats(bn, n):
    v = np.array([BEST[(bn, f"donor_x{n}_ctrl{j}")][1] for j in range(A.nscram)])
    return float(v.mean()), float(v.std(ddof=1))
log(f"\n[dose] delta-logit at each sequence's most favourable position; control is the mean of "
    f"{A.nscram} dinucleotide shuffles")
log(f"{'copies':>7} {'len':>4} {'shuf motif':>11} {'shuf ctrl':>16} {'shuf diff':>10}"
    f" {'real motif':>11} {'real ctrl':>16} {'real diff':>10}")
d1, d2 = [], []
for n in range(1, A.max_copies + 1):
    a1 = BEST[("dinuc_shuffled", f"donor_x{n}")][1]; c1, s1 = ctrl_stats("dinuc_shuffled", n)
    a2 = BEST[("real_competitor", f"donor_x{n}")][1]; c2, s2 = ctrl_stats("real_competitor", n)
    d1.append(a1 - c1); d2.append(a2 - c2)
    log(f"{n:>7} {len(construct(n)):>4} {a1:>+11.4f} {c1:>+9.4f} +/-{s1:<5.3f} {a1-c1:>+10.4f}"
        f" {a2:>+11.4f} {c2:>+9.4f} +/-{s2:<5.3f} {a2-c2:>+10.4f}")
nn = np.arange(1, A.max_copies + 1)
log(f"\n[dose] motif minus its own scramble, trend with copy number: "
    f"shuffled r = {np.corrcoef(nn, d1)[0,1]:+.3f}, real competitor r = {np.corrcoef(nn, d2)[0,1]:+.3f}")
log(f"[dose] most negative: shuffled {min(d1):+.4f} at n={int(nn[int(np.argmin(d1))])}, "
    f"real competitor {min(d2):+.4f} at n={int(nn[int(np.argmin(d2))])}")
log(f"[dose] acceptor for scale: {acc:+.4f}")
