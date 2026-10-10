#!/usr/bin/env python3
# dose-response for implanted 3' and 5' splice sites in real backgrounds
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
P.add_argument("--seed", type=int, default=42)
A = P.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
UP = 2000
rng = np.random.default_rng(A.seed)

sys.path.insert(0, "scripts")
import u1_5ss_landscape_data as L
U1X = "".join("ACGT"[(int(np.nanargmax(L.U1)) // int(L.POW[p])) % 4] for p in range(9))
assert U1X == "CAGGTGAGT"
ACC = "TTTTTTTTTTTTTCAGG"
BROKEN = "TTTCTTTATTTGTTTGT"
assert sorted(BROKEN) == sorted(ACC) and BROKEN != ACC
SPACER = "CTCTCT"
UNIT = {"acceptor": ACC, "donor": U1X}
def comp(s): return tuple(s.count(b) for b in "ACGT")
log(f"[env] device {dev}")
log(f"[units] acceptor {ACC} {comp(ACC)}; donor {U1X} {comp(U1X)}; "
    f"broken-tract control {BROKEN} {comp(BROKEN)}")

def construct(kind, n): return (UNIT[kind] + SPACER) * (n - 1) + UNIT[kind]
def con_broken(n): return (BROKEN + SPACER) * (n - 1) + BROKEN
def dinuc(s, tag, j):
    tok = np.array(["ACGT".index(c) for c in s], np.uint8)
    return "".join("ACGT"[i] for i in
                   dinuc_shuffle(tok, np.random.default_rng([A.seed, sum(map(ord, tag)), j])))
def mono(s, tag, j):
    r = np.random.default_rng([A.seed, sum(map(ord, tag)), 9000 + j])
    return "".join(r.permutation(list(s)))
def cnt(s, d): return sum(1 for i in range(len(s) - 1) if s[i:i+2] == d)

SEQ = []; IDENT = {}
for kind in ("acceptor", "donor"):
    for n in range(1, A.max_copies + 1):
        c = construct(kind, n)
        SEQ.append((f"{kind}_x{n}", c, kind, "motif", n))
        for nm, fn in (("dinuc", dinuc), ("mono", mono)):
            same = 0
            for j in range(A.nscram):
                sc = fn(c, f"{kind}_x{n}", j)
                same += int(sc == c)
                assert comp(sc) == comp(c), "shuffle changed base composition"
                if nm == "dinuc":
                    assert cnt(sc, "GT") == cnt(c, "GT") and cnt(sc, "AG") == cnt(c, "AG")
                SEQ.append((f"{kind}_x{n}_{nm}{j}", sc, kind, nm, n))
            IDENT[(kind, n, nm)] = same
        if kind == "acceptor":
            b = con_broken(n)
            assert comp(b) == comp(c) and b != c
            SEQ.append((f"{kind}_x{n}_broken", b, kind, "broken", n))
log(f"[panel] {len(SEQ)} sequences")
log("[degeneracy] shuffles identical to their own motif, out of "
    f"{A.nscram}, by site and control:")
for kind in ("acceptor", "donor"):
    for nm in ("dinuc", "mono"):
        v = [IDENT[(kind, n, nm)] for n in range(1, A.max_copies + 1)]
        log(f"   {kind:>8} {nm:>6}: {v}")

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
        for k in range(0, len(a), bs): out.append(net(oh(a[k:k+bs])).float().cpu().numpy())
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
    for k, s_, kind, arm, n in SEQ:
        bst = (None, -1e9, 0.0)
        for p_ in POS:
            q = implant(B, s_, p_ + UP)
            if q is None: continue
            d = fwd(q) - base
            m_, se_ = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
            rows.append((bn, bl, k, kind, arm, n, len(s_), p_, m_, se_, A.gia_bg))
            if m_ > bst[1]: bst = (p_, m_, se_)
        BEST[(bn, k)] = bst
    log(f"[gia] {bn} done  {time.time()-t0:.0f}s")
with open("gia_dose_controls.tsv", "w") as fh:
    fh.write("background\tbg_label\tseq_id\tsite\tarm\tn_copies\tinsert_len\tposition\t"
             "delta_logit\tse\tn_bg\tbest_position\tbest_delta\tbest_se\n")
    for r in rows:
        b = BEST[(r[0], r[2])]
        fh.write("\t".join(str(x) if not isinstance(x, float) else f"{x:.5f}" for x in r)
                 + f"\t{b[0]}\t{b[1]:.5f}\t{b[2]:.5f}\n")
log("[out] gia_dose_controls.tsv")

def arm(bn, kind, n, nm):
    v = np.array([BEST[(bn, f"{kind}_x{n}_{nm}{j}")][1] for j in range(A.nscram)])
    return float(v.mean()), float(v.std(ddof=1))
for bn, bl in (("real_canonical", "Real 3' end background"),
               ("real_competitor", "Real competitor background")):
    log(f"\n[{bl}]")
    for kind in ("acceptor", "donor"):
        log(f"  {kind}")
        hdr = f"  {'n':>3} {'motif':>9} {'dinuc':>17} {'mono':>17} {'m-dinuc':>9} {'m-mono':>9}"
        log(hdr + ("   broken  m-broken" if kind == "acceptor" else ""))
        for n in range(1, A.max_copies + 1):
            m_ = BEST[(bn, f"{kind}_x{n}")][1]
            d_, ds = arm(bn, kind, n, "dinuc"); o_, os_ = arm(bn, kind, n, "mono")
            line = (f"  {n:>3} {m_:>+9.4f} {d_:>+10.4f}+/-{ds:<4.3f} {o_:>+10.4f}+/-{os_:<4.3f} "
                    f"{m_-d_:>+9.4f} {m_-o_:>+9.4f}")
            if kind == "acceptor":
                b_ = BEST[(bn, f"{kind}_x{n}_broken")][1]
                line += f"  {b_:>+8.4f} {m_-b_:>+9.4f}"
            log(line)
        nn = np.arange(1, A.max_copies + 1)
        dm = np.array([BEST[(bn, f"{kind}_x{n}")][1] - arm(bn, kind, n, "mono")[0] for n in nn])
        log(f"      motif minus mononucleotide-shuffled control: trend r = "
            f"{np.corrcoef(nn, dm)[0,1]:+.3f}, at n=1 {dm[0]:+.4f}, at n=7 {dm[-1]:+.4f}")
log("\n[read] a motif that beats its MONONUCLEOTIDE shuffle is doing something its base "
    "composition alone does not; one that does not is composition.")
