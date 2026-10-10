#!/usr/bin/env python3
# widen the implantation panel on the 5' splice site / u1 axis
import sys, time, argparse, itertools
import numpy as np
import torch
from cnn_interp_lib import NT, dinuc_shuffle

def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser()
P.add_argument("--model", default="cnn_model.pt")
P.add_argument("--gia_bg", type=int, default=256)
P.add_argument("--seed", type=int, default=42)
A = P.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
UP = 2000
rng = np.random.default_rng(A.seed)
log(f"[env] device {dev}")

sys.path.insert(0, "scripts")
import u1_5ss_landscape_data as L
from maxentpy import maxent
from maxentpy.maxent import load_matrix5
M5 = load_matrix5()

def u1dg(s):
    if len(s) != 9 or any(c not in "ACGT" for c in s): return np.nan
    idx = sum(int(L.POW[p]) * "ACGT".index(s[p]) for p in range(9))
    return float(L.U1[idx])
def me5(s):
    if len(s) != 9 or any(c not in "ACGT" for c in s): return np.nan
    try: return float(maxent.score5(s, matrix=M5))
    except Exception: return np.nan

# the extremes, derived not assumed
u1best = max((("".join("ACGT"[(i // int(L.POW[p])) % 4] for p in range(9)), L.U1[i])
              for i in np.argsort(-np.nan_to_num(L.U1, nan=-99))[:1]), key=lambda t: t[1])[0]
m5best = max((("".join(a) + "GT" + "".join(b), me5("".join(a) + "GT" + "".join(b)))
              for a in itertools.product("ACGT", repeat=3)
              for b in itertools.product("ACGT", repeat=4)), key=lambda t: t[1])[0]
log(f"[max] strongest U1 duplex 9-mer {u1best} (-dG {u1dg(u1best):.2f}); "
    f"MaxEnt-optimal 5'ss {m5best} (score5 {me5(m5best):.2f})")
assert u1best == "CAGGTGAGT" and m5best == "ACGGTAAGT"

def scramble(s, tag):
    r = np.random.default_rng([A.seed, sum(map(ord, tag))])
    base = u1dg(s) if len(s) == 9 else 99
    for _ in range(500):
        c = "".join(r.permutation(list(s)))
        if c == s: continue
        if len(c) == 9 and (c[3:5] == "GT" or u1dg(c) > base - 4): continue
        if len(c) != 9 and "GTAAG" in c: continue
        return c
    raise RuntimeError(f"no scramble for {s}")

U1X, M5X = u1best, m5best
EXT = "CAGGTAAGTATC"
X3 = (U1X + "CTCTCT") * 2 + U1X
MOT = [
    ("acceptor_pyAG", "TTTTTTTTTTTTTCAGG", "3' splice site (pyrimidine tract + AG)", "motif", None),
    ("donor_AGGTAAGT", "AGGTAAGT", "5' splice site, AGGTAAGT", "motif", "donor_AGGTAAGT_ctrl"),
    ("donor_CAGGTAAGT", "CAGGTAAGT", "5' splice site, CAGGTAAGT", "motif", "donor_CAGGTAAGT_ctrl"),
    ("donor_U1max", U1X, f"5' splice site, strongest U1 duplex ({U1X})", "motif", "donor_U1max_ctrl"),
    ("donor_ME5max", M5X, f"5' splice site, MaxEnt-optimal ({M5X})", "motif", "donor_ME5max_ctrl"),
    ("donor_ext", EXT, "5' splice site, consensus extended to +9", "motif", "donor_ext_ctrl"),
    ("donor_x3", X3, "5' splice site, three tandem copies", "motif", "donor_x3_ctrl"),
    ("GU_rich", "GTGTGTGTGTGT", "GU-rich run", "motif", "GU_rich_ctrl"),
    ("PAS_AATAAA", "AATAAA", "Polyadenylation signal (AATAAA)", "motif", "PAS_AATAAA_ctrl"),
]
GIA = {}
for k, seq, lab, kind, ctrl in MOT:
    GIA[k] = (seq, lab, "motif")
    if ctrl:
        c = scramble(seq, k)
        GIA[ctrl] = (c, lab + ", scrambled", "control")
log("\n[panel] implanted sequences")
for k, (seq, lab, kind) in GIA.items():
    log(f"  {k:>22} {kind:>7}  {seq:<27} U1 -dG {u1dg(seq) if len(seq)==9 else float('nan'):>6.2f}"
        f"  score5 {me5(seq) if len(seq)==9 else float('nan'):>7.2f}")

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
log(f"\n[gia] {len(GIA)} sequences x {len(POS)} positions x 2 backgrounds x {A.gia_bg}")

BEST = {}; t0 = time.time()
with open("gia_donor_panel.tsv", "w") as fh:
    fh.write("background\tbg_label\tmotif\tlabel\tkind\tsequence\tu1_dg\tme5\tposition\t"
             "delta_logit\tse\tn\tbest_position\tbest_delta\n")
    rows = []
    for bn, B, bl in (("dinuc_shuffled", BG, "Dinucleotide-shuffled background"),
                      ("real_competitor", BGR, "Real competitor background")):
        base = fwd(B)
        for k, (seq, lab, kind) in GIA.items():
            bst = (None, -1e9)
            buf = []
            for p_ in POS:
                s_ = implant(B, seq, p_ + UP)
                if s_ is None: continue
                d = fwd(s_) - base
                buf.append((p_, float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))))
                if d.mean() > bst[1]: bst = (p_, float(d.mean()))
            BEST[(bn, k)] = bst
            for p_, m_, se_ in buf:
                rows.append((bn, bl, k, lab, kind, seq, u1dg(seq), me5(seq), p_, m_, se_,
                             A.gia_bg, bst[0], bst[1]))
        log(f"[gia] {bn} done  {time.time()-t0:.0f}s")
    for r in rows:
        fh.write("\t".join(str(x) if not isinstance(x, float) else f"{x:.5f}" for x in r) + "\n")
log("[out] gia_donor_panel.tsv")

acc = BEST[("dinuc_shuffled", "acceptor_pyAG")][1]
log(f"\n[check] acceptor in shuffled background {acc:+.4f} against the published +0.27169")
assert abs(acc - 0.27169) < 0.08, "acceptor positive control did not reproduce"
log("\n[panel] delta-logit at each sequence's most favourable position")
log(f"{'sequence':>22} {'shuffled':>10} {'real comp':>10}")
for k in GIA:
    log(f"{k:>22} {BEST[('dinuc_shuffled',k)][1]:>+10.4f} {BEST[('real_competitor',k)][1]:>+10.4f}")
log("\n[verdict] every 5' splice site variant against its own scramble")
worst = 0.0
for k, (seq, lab, kind) in GIA.items():
    if kind != "motif" or not k.startswith(("donor", "GU")): continue
    c = k + "_ctrl"
    if c not in GIA: continue
    for bn in ("dinuc_shuffled", "real_competitor"):
        d = BEST[(bn, k)][1] - BEST[(bn, c)][1]
        worst = max(worst, d)
        log(f"  {bn:16s} {k:>16} minus scramble {d:+.4f}")
log(f"[verdict] the largest excess of any 5' splice site variant over its own scramble is "
    f"{worst:+.4f}, against {acc:+.4f} for the acceptor")
