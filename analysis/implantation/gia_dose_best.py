#!/usr/bin/env python3
# implant the best 3' splice site, derived the way the best 5' splice site was
import sys, time, argparse
import numpy as np
import torch
from cnn_interp_lib import NT, dinuc_shuffle

def log(*a): print(*a, file=sys.stderr, flush=True)
P = argparse.ArgumentParser()
P.add_argument("--model", default="cnn_model.pt")
P.add_argument("--gia_bg", type=int, default=256)
P.add_argument("--max_copies", type=int, default=7)
P.add_argument("--nscram", type=int, default=10)
P.add_argument("--nrestart", type=int, default=80)
P.add_argument("--seed", type=int, default=42)
A = P.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
UP = 2000
rng = np.random.default_rng(A.seed)

sys.path.insert(0, "scripts")
import u1_5ss_landscape_data as L
from maxentpy import maxent
from maxentpy.maxent import load_matrix3
M3 = load_matrix3()
def s3(s):
    try: return float(maxent.score3(s, matrix=M3))
    except Exception: return -99.0

def best_acceptor():
    r = np.random.default_rng(A.seed); best = (-99.0, None)
    for _ in range(A.nrestart):
        s = list("".join(r.choice(list("ACGT"), 23))); s[18], s[19] = "A", "G"
        cur = s3("".join(s))
        for _sweep in range(8):
            moved = False
            for i in range(23):
                if i in (18, 19): continue
                bb, bv = s[i], cur
                for b in "ACGT":
                    s[i] = b; v = s3("".join(s))
                    if v > bv: bb, bv = b, v
                s[i] = bb
                if bv > cur + 1e-12: cur = bv; moved = True
            if not moved: break
        if cur > best[0]: best = (cur, "".join(s))
    return best[1], best[0]

ACC_BEST, ACC_SC = best_acceptor()
assert ACC_BEST == "TTTCCTCTCTCTCCCCGCAGGTG" and abs(ACC_SC - 17.172) < 0.01, (ACC_BEST, ACC_SC)
ACC_PPT = "TTTTTTTTTTTTTCAGG"
U1X = "".join("ACGT"[(int(np.nanargmax(L.U1)) // int(L.POW[p])) % 4] for p in range(9))
assert U1X == "CAGGTGAGT"
SPACER = "CTCTCT"
UNIT = {"acc_best": ACC_BEST, "acc_ppt": ACC_PPT, "donor": U1X}
def at(s): return (s.count("A") + s.count("T")) / len(s)
log(f"[opt] MaxEnt-optimal 3'ss {ACC_BEST}  score3 {ACC_SC:.3f}  A+T {at(ACC_BEST):.2f}")
log(f"[opt] hand-built tract    {ACC_PPT}          A+T {at(ACC_PPT):.2f}")
log(f"[opt] U1-optimal 5'ss     {U1X}                  A+T {at(U1X):.2f}")
C0 = np.load("analysis_columns.npz", allow_pickle=True)["spl_upstream_max_me3"]
C0 = C0[np.isfinite(C0)]
log(f"[opt] the optimum exceeds every acceptor observed in the data: max {C0.max():.2f} over "
    f"{len(C0):,} peaks, p99 {np.percentile(C0,99):.2f}, median {np.median(C0):.2f}")

def construct(k, n): return (UNIT[k] + SPACER) * (n - 1) + UNIT[k]
def dinuc(s, tag, j):
    tok = np.array(["ACGT".index(c) for c in s], np.uint8)
    return "".join("ACGT"[i] for i in
                   dinuc_shuffle(tok, np.random.default_rng([A.seed, sum(map(ord, tag)), j])))
def mono(s, tag, j):
    return "".join(np.random.default_rng([A.seed, sum(map(ord, tag)), 9000 + j])
                   .permutation(list(s)))
def comp(s): return tuple(s.count(b) for b in "ACGT")

SEQ = []; IDENT = {}
for k in UNIT:
    for n in range(1, A.max_copies + 1):
        c = construct(k, n)
        SEQ.append((f"{k}_x{n}", c, k, "motif", n))
        for nm, fn in (("mono", mono), ("dinuc", dinuc)):
            same = 0
            for j in range(A.nscram):
                sc = fn(c, f"{k}_x{n}", j); same += int(sc == c)
                assert comp(sc) == comp(c)
                SEQ.append((f"{k}_x{n}_{nm}{j}", sc, k, nm, n))
            IDENT[(k, n, nm)] = same
log(f"\n[panel] {len(SEQ)} sequences; controls identical to their motif, out of {A.nscram}:")
for k in UNIT:
    for nm in ("mono", "dinuc"):
        log(f"   {k:>9} {nm:>6}: {[IDENT[(k,n,nm)] for n in range(1, A.max_copies+1)]}")

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
log(f"\n[gia] {len(SEQ)} x {len(POS)} x 2 real backgrounds x {A.gia_bg}")

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
with open("gia_dose_best.tsv", "w") as fh:
    fh.write("background\tbg_label\tseq_id\tsite\tarm\tn_copies\tinsert_len\tposition\t"
             "delta_logit\tse\tn_bg\tbest_position\tbest_delta\tbest_se\tunit\tunit_at\n")
    for r in rows:
        b = BEST[(r[0], r[2])]
        fh.write("\t".join(str(x) if not isinstance(x, float) else f"{x:.5f}" for x in r)
                 + f"\t{b[0]}\t{b[1]:.5f}\t{b[2]:.5f}\t{UNIT[r[3]]}\t{at(UNIT[r[3]]):.4f}\n")
log("[out] gia_dose_best.tsv")

def ctl(bn, k, n, nm="mono"):
    v = np.array([BEST[(bn, f"{k}_x{n}_{nm}{j}")][1] for j in range(A.nscram)])
    return float(v.mean()), float(v.std(ddof=1))
for bn, bl in (("real_canonical", "Real 3' end background"),
               ("real_competitor", "Real competitor background")):
    log(f"\n[{bl}]  motif, its mononucleotide-shuffled control, and the difference")
    log(f"  {'unit':>9} {'n':>2} {'motif':>9} {'composition':>18} {'motif-comp':>11}")
    for k in UNIT:
        for n in (1, 3, 7):
            m_ = BEST[(bn, f"{k}_x{n}")][1]; c_, s_ = ctl(bn, k, n)
            log(f"  {k:>9} {n:>2} {m_:>+9.4f} {c_:>+10.4f} +/-{s_:<5.3f} {m_-c_:>+11.4f}")
    log("")
    for k in UNIT:
        nn = np.arange(1, A.max_copies + 1)
        d = np.array([BEST[(bn, f"{k}_x{n}")][1] - ctl(bn, k, n)[0] for n in nn])
        log(f"  {k:>9}: motif minus composition, r with dose = {np.corrcoef(nn,d)[0,1]:+.3f}, "
            f"n=1 {d[0]:+.4f}, n=7 {d[-1]:+.4f}")
log("\n[read] the question is whether ANY unit beats a random arrangement of its own bases; the "
    "optimal acceptor is the case most favourable to a motif, since it is stronger than any "
    "acceptor in the data and is not AT-rich.")
