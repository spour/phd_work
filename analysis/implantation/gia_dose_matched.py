#!/usr/bin/env python3
# the two splice sites against each other at matched implanted length
import sys, time, argparse
import numpy as np
import torch
from cnn_interp_lib import NT

def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser()
P.add_argument("--model", default="cnn_model.pt")
P.add_argument("--gia_bg", type=int, default=256)
P.add_argument("--all_bg", action="store_true")
P.add_argument("--suffix", default="")
P.add_argument("--units", default="acc_best,acc_noC,donor")
P.add_argument("--nscram", type=int, default=10)
P.add_argument("--seed", type=int, default=42)
A = P.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
UP, WIN = 2000, 2500
rng = np.random.default_rng(A.seed)

sys.path.insert(0, "scripts")
import u1_5ss_landscape_data as L
U1X = "".join("ACGT"[(int(np.nanargmax(L.U1)) // int(L.POW[p])) % 4] for p in range(9))
assert U1X == "CAGGTGAGT"
from maxentpy import maxent
from maxentpy.maxent import load_matrix3
M3 = load_matrix3()
ACC = "TTTCCTCTCTCTCCCCGCAGGTG"
assert abs(float(maxent.score3(ACC, matrix=M3)) - 17.172) < 0.01
ACC_NC = "TTTTTTTTTGTGTGTTTTAGGTT"
assert abs(float(maxent.score3(ACC_NC, matrix=M3)) - 14.38) < 0.05 and "C" not in ACC_NC
ACC_PPT = "TTTTTTTTTTTTTCAGG"
ACC_MODEL = "ACGGTGTACTTTTGTTGCAGGTG"
assert abs(float(maxent.score3(ACC_MODEL, matrix=M3)) - 10.04) < 0.05
ACC_AGDEP = "CCATGCCTTTGATCCTGCAGATG"
assert abs(float(maxent.score3(ACC_AGDEP, matrix=M3)) - 10.38) < 0.05
SPACER = "CTCTCT"
_ALL = {"acc_best": ACC, "acc_noC": ACC_NC, "acc_ppt": ACC_PPT, "donor": U1X,
        "acc_model": ACC_MODEL, "acc_agdep": ACC_AGDEP}
_NC = {"acc_best": 7, "acc_noC": 7, "acc_ppt": 7, "donor": 14,
       "acc_model": 7, "acc_agdep": 7}
UNIT = {k: _ALL[k] for k in A.units.split(",")}
NCOP = {k: _NC[k] for k in A.units.split(",")}
def at(s): return (s.count("A") + s.count("T")) / len(s)
def comp(s): return tuple(s.count(b) for b in "ACGT")
log(f"[unit] 3' splice site, score3 17.17, C-rich  {ACC}  C {ACC.count(chr(67))/23:.2f} "
    f"A+T {at(ACC):.2f}")
log(f"[unit] 3' splice site, score3 14.38, no C    {ACC_NC}  C 0.00 A+T {at(ACC_NC):.2f}")
log(f"[unit] 3' splice site, hand-built T tract    {ACC_PPT}        A+T {at(ACC_PPT):.2f}")
log(f"[unit] 5' splice site, strongest U1 duplex  {U1X}              A+T {at(U1X):.2f}")

def construct(k, n): return (UNIT[k] + SPACER) * (n - 1) + UNIT[k]
def mono(s, tag, j):
    return "".join(np.random.default_rng([A.seed, sum(map(ord, tag)), 9000 + j]).permutation(list(s)))
SEQ = []
for k in UNIT:
    for n in range(1, NCOP[k] + 1):
        c = construct(k, n)
        SEQ.append((f"{k}_x{n}", c, k, "motif", n))
        for j in range(A.nscram):
            sc = mono(c, f"{k}_x{n}", j); assert comp(sc) == comp(c)
            SEQ.append((f"{k}_x{n}_mono{j}", sc, k, "mono", n))
for k in UNIT:
    log(f"[len] {k:>9}: " + " ".join(str(len(construct(k, n))) for n in range(1, NCOP[k] + 1)) + " nt")

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
def fwd(a, bs=1024):
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

if A.all_bg:
    BGC = X[te[y[te] == 1]]; BGX = X[te[y[te] == 0]]
else:
    BGC = X[rng.choice(te[y[te] == 1], A.gia_bg, replace=False)]
    BGX = X[rng.choice(te[y[te] == 0], A.gia_bg, replace=False)]
log(f"[bg] {len(BGC):,} real 3' end and {len(BGX):,} real competitor held-out windows"
    + ("  (ALL of them)" if A.all_bg else f"  (sampled)"))
POS = sorted(set(list(range(-300, 101, 10)) + list(range(-UP + 100, 500, 100))))
log(f"\n[gia] {len(SEQ)} sequences x {len(POS)} positions x ({len(BGC):,} + {len(BGX):,}) backgrounds")

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
            rows.append((bn, bl, k_, site, arm, n, len(s_), p_, m_, se_, len(B)))
            if m_ > bst[1]: bst = (p_, m_, se_)
        BEST[(bn, k_)] = bst
    log(f"[gia] {bn} done  {time.time()-t0:.0f}s")
with open(f"gia_dose_matched{A.suffix}.tsv", "w") as fh:
    fh.write("background\tbg_label\tseq_id\tsite\tarm\tn_copies\tinsert_len\tposition\t"
             "delta_logit\tse\tn_bg\tbest_position\tbest_delta\tbest_se\tunit\tunit_at\n")
    for r in rows:
        b = BEST[(r[0], r[2])]
        fh.write("\t".join(str(x) if not isinstance(x, float) else f"{x:.5f}" for x in r)
                 + f"\t{b[0]}\t{b[1]:.5f}\t{b[2]:.5f}\t{UNIT[r[3]]}\t{at(UNIT[r[3]]):.4f}\n")
log(f"[out] gia_dose_matched{A.suffix}.tsv")

def ctl(bn, k, n):
    v = np.array([BEST[(bn, f"{k}_x{n}_mono{j}")][1] for j in range(A.nscram)])
    return float(v.mean()), float(v.std(ddof=1))
for bn, bl in (("real_canonical", "Real 3' end background"),
               ("real_competitor", "Real competitor background")):
    log(f"\n[{bl}]  implanted, its own base composition, and the difference")
    log(f"  {'unit':>9} {'nt':>4} {'implanted':>10} {'composition':>18} {'difference':>11}")
    for k in UNIT:
        for n in range(1, NCOP[k] + 1):
            m_ = BEST[(bn, f"{k}_x{n}")][1]; c_, s_ = ctl(bn, k, n)
            log(f"  {k:>9} {len(construct(k,n)):>4} {m_:>+10.4f} {c_:>+10.4f} +/-{s_:<5.3f} "
                f"{m_-c_:>+11.4f}")
    log("")
    for k in UNIT:
        nn = np.array([len(construct(k, n)) for n in range(1, NCOP[k] + 1)])
        d = np.array([BEST[(bn, f"{k}_x{n}")][1] - ctl(bn, k, n)[0] for n in range(1, NCOP[k] + 1)])
        log(f"  {k:>9}: difference vs implanted length r = {np.corrcoef(nn,d)[0,1]:+.3f}")
log(f"\n[matched] the two units at comparable implanted length, real 3' end background")
log(f"  {'nt':>4} {'3ss':>9} {'nt':>4} {'5ss':>9}")
for na in (range(1, NCOP["acc_best"] + 1)
           if {"acc_best", "donor"} <= set(UNIT) else []):
    la = len(construct("acc_best", na))
    nb = min(range(1, NCOP["donor"] + 1), key=lambda m: abs(len(construct("donor", m)) - la))
    lb = len(construct("donor", nb))
    log(f"  {la:>4} {BEST[('real_canonical', f'acc_best_x{na}')][1]:>+9.4f} "
        f"{lb:>4} {BEST[('real_canonical', f'donor_x{nb}')][1]:>+9.4f}")
