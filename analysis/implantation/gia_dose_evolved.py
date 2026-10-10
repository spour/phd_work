#!/usr/bin/env python3
# the tandem dose series for the two units that bracket the model
import sys, time, argparse, csv
import numpy as np
import torch
from cnn_interp_lib import NT, dinuc_shuffle

def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser()
P.add_argument("--model", default="cnn_model.pt")
P.add_argument("--gia_bg", type=int, default=256)
P.add_argument("--max_copies", type=int, default=7)
P.add_argument("--nscram", type=int, default=10)
P.add_argument("--seed", type=int, default=42)
A = P.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
UP, WIN = 2000, 2500
rng = np.random.default_rng(A.seed)

EV = {r["unit"]: r["sequence"] for r in csv.DictReader(open("evolve_contrast.tsv"), delimiter="\t")}
EV23 = EV["evolved_23"]
assert EV23 == "TTCCGACGGAGGACCGACCGAGT" and len(EV23) == 23, EV23
sys.path.insert(0, "scripts")
import u1_5ss_landscape_data as L
U1X = "".join("ACGT"[(int(np.nanargmax(L.U1)) // int(L.POW[p])) % 4] for p in range(9))
assert U1X == "CAGGTGAGT"

SPACER = "CTCTCT"
UNIT = {"evolved_23": EV23, "donor": U1X}
def at(s): return (s.count("A") + s.count("T")) / len(s)
def comp(s): return tuple(s.count(b) for b in "ACGT")
log(f"[unit] evolved 23-mer, maximises the model    {EV23}  A+T {at(EV23):.2f}")
log(f"[unit] 5' splice site, maximises U1 duplex    {U1X}              A+T {at(U1X):.2f}")

def construct(k, n): return (UNIT[k] + SPACER) * (n - 1) + UNIT[k]
def mono(s, tag, j):
    return "".join(np.random.default_rng([A.seed, sum(map(ord, tag)), 9000 + j]).permutation(list(s)))
def dinuc(s, tag, j):
    t = np.array([NT.index(c) for c in s], np.uint8)
    return "".join(NT[i] for i in dinuc_shuffle(t, np.random.default_rng([A.seed, sum(map(ord, tag)), j])))

SEQ = []; IDENT = {}
for k in UNIT:
    for n in range(1, A.max_copies + 1):
        c = construct(k, n)
        SEQ.append((f"{k}_x{n}", c, k, "motif", n))
        same = 0
        for j in range(A.nscram):
            sc = mono(c, f"{k}_x{n}", j); same += int(sc == c)
            assert comp(sc) == comp(c)
            SEQ.append((f"{k}_x{n}_mono{j}", sc, k, "mono", n))
        IDENT[(k, n)] = same
log(f"[panel] {len(SEQ)} sequences; mononucleotide controls identical to their motif, out of "
    f"{A.nscram}: " + ", ".join(f"{k} {[IDENT[(k,n)] for n in range(1,A.max_copies+1)]}" for k in UNIT))
DG = sum(int(dinuc(U1X, "donor_x1", j) == U1X) for j in range(A.nscram))
log(f"[why mono] a dinucleotide shuffle returns CAGGTGAGT unchanged in {DG} of {A.nscram} draws, "
    f"so it cannot control a 9-mer; the mononucleotide shuffle returns it in {IDENT[('donor',1)]}")

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
def fwd(a, bs=512):
    out = []
    with torch.no_grad():
        for k_ in range(0, len(a), bs): out.append(net(oh(a[k_:k_+bs])).float().cpu().numpy())
    return np.concatenate(out)
def implant(seqs, motif, pos):
    m = np.array([NT.index(c) for c in motif], np.uint8)
    o = seqs.copy(); s = pos - len(m) // 2
    if s < 0 or s + len(m) > o.shape[1]: return None
    o[:, s:s+len(m)] = m
    return o

BGC = X[rng.choice(te[y[te] == 1], A.gia_bg, replace=False)]
BGX = X[rng.choice(te[y[te] == 0], A.gia_bg, replace=False)]
POS = sorted(set(list(range(-300, 101, 10)) + list(range(-UP + 100, 500, 100))))
log(f"\n[gia] {len(SEQ)} x {len(POS)} positions x 2 real backgrounds x {A.gia_bg}")

BEST = {}; rows = []; t0 = time.time()
for bn, B, bl in (("real_canonical", BGC, "Real 3' end background"),
                  ("real_competitor", BGX, "Real competitor background")):
    base = fwd(B)
    for k_, s_, site, arm, n in SEQ:
        bst = (None, -1e9, 0.0)
        for p_ in POS:
            q = implant(B, s_, p_ + UP)
            if q is None: continue
            d = fwd(q) - base
            m_, se_ = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
            rows.append((bn, bl, k_, site, arm, n, len(s_), p_, m_, se_, A.gia_bg))
            if m_ > bst[1]: bst = (p_, m_, se_)
        BEST[(bn, k_)] = bst
    log(f"[gia] {bn} done  {time.time()-t0:.0f}s")
with open("gia_dose_evolved.tsv", "w") as fh:
    fh.write("background\tbg_label\tseq_id\tsite\tarm\tn_copies\tinsert_len\tposition\t"
             "delta_logit\tse\tn_bg\tbest_position\tbest_delta\tbest_se\tunit\tunit_at\n")
    for r in rows:
        b = BEST[(r[0], r[2])]
        fh.write("\t".join(str(x) if not isinstance(x, float) else f"{x:.5f}" for x in r)
                 + f"\t{b[0]}\t{b[1]:.5f}\t{b[2]:.5f}\t{UNIT[r[3]]}\t{at(UNIT[r[3]]):.4f}\n")
log("[out] gia_dose_evolved.tsv")

def ctl(bn, k, n):
    v = np.array([BEST[(bn, f"{k}_x{n}_mono{j}")][1] for j in range(A.nscram)])
    return float(v.mean()), float(v.std(ddof=1))
for bn, bl in (("real_canonical", "Real 3' end background"),
               ("real_competitor", "Real competitor background")):
    log(f"\n[{bl}]  implanted, its own base composition, and the difference")
    log(f"  {'unit':>11} {'n':>2} {'implanted':>10} {'composition':>18} {'difference':>11}")
    for k in UNIT:
        for n in range(1, A.max_copies + 1):
            m_ = BEST[(bn, f"{k}_x{n}")][1]; c_, s_ = ctl(bn, k, n)
            log(f"  {k:>11} {n:>2} {m_:>+10.4f} {c_:>+10.4f} +/-{s_:<5.3f} {m_-c_:>+11.4f}")
    for k in UNIT:
        nn = np.arange(1, A.max_copies + 1)
        d = np.array([BEST[(bn, f"{k}_x{n}")][1] - ctl(bn, k, n)[0] for n in nn])
        log(f"  {k:>11}: difference vs dose r = {np.corrcoef(nn,d)[0,1]:+.3f}, "
            f"n=1 {d[0]:+.4f}, n=7 {d[-1]:+.4f}")
