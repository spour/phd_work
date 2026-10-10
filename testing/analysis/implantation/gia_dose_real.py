#!/usr/bin/env python3
# dose-response for implanted 3' and 5' splice sites, one to seven tandem copies each
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
assert U1X == "CAGGTGAGT", U1X
ACC = "TTTTTTTTTTTTTCAGG"
SPACER = "CTCTCT"
UNIT = {"acceptor": ACC, "donor": U1X}
log(f"[env] device {dev}; acceptor unit {ACC} ({len(ACC)} nt), donor unit {U1X} ({len(U1X)} nt), "
    f"spacer {SPACER}")

def construct(kind, n): return (UNIT[kind] + SPACER) * (n - 1) + UNIT[kind]
def dinuc(s, tag, j):
    tok = np.array(["ACGT".index(c) for c in s], np.uint8)
    return "".join("ACGT"[i] for i in
                   dinuc_shuffle(tok, np.random.default_rng([A.seed, sum(map(ord, tag)), j])))
def cnt(s, d): return sum(1 for i in range(len(s) - 1) if s[i:i+2] == d)

SEQ = []
for kind in ("acceptor", "donor"):
    for n in range(1, A.max_copies + 1):
        c = construct(kind, n)
        SEQ.append((f"{kind}_x{n}", c, kind, "motif", n))
        for j in range(A.nscram):
            sc = dinuc(c, f"{kind}_x{n}", j)
            assert cnt(sc, "GT") == cnt(c, "GT") and cnt(sc, "AG") == cnt(c, "AG"), \
                f"dinucleotide shuffle changed a dinucleotide count for {kind} n={n}"
            SEQ.append((f"{kind}_x{n}_ctrl{j}", sc, kind, "control", n))
log(f"[panel] {len(SEQ)} sequences; GT and AG counts preserved in every control")
for kind in ("acceptor", "donor"):
    for n in (1, 7):
        c = construct(kind, n)
        log(f"  {kind:>8} n={n} len={len(c):>3} GT x{cnt(c,'GT'):<2} AG x{cnt(c,'AG'):<2}  "
            f"{c[:56]}" + ("..." if len(c) > 56 else ""))

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

CAN = te[y[te] == 1]; CMP = te[y[te] == 0]
BGC = X[rng.choice(CAN, A.gia_bg, replace=False)]
BGX = X[rng.choice(CMP, A.gia_bg, replace=False)]
log(f"[bg] real canonical: {A.gia_bg} of {len(CAN)} used 3' ends; "
    f"real competitor: {A.gia_bg} of {len(CMP)} competitors; both held-out test sequence")
POS = sorted(set(list(range(-300, 101, 10)) + list(range(-UP + 100, 500, 100))))

BEST = {}; rows = []; t0 = time.time()
for bn, B, bl in (("real_canonical", BGC, "Real 3' end background"),
                  ("real_competitor", BGX, "Real competitor background")):
    base = fwd(B)
    log(f"[gia] {bl}: background logit mean {base.mean():+.3f} sd {base.std():.3f}")
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
with open("gia_dose_real.tsv", "w") as fh:
    fh.write("background\tbg_label\tseq_id\tsite\tarm\tn_copies\tinsert_len\tposition\t"
             "delta_logit\tse\tn_bg\tbest_position\tbest_delta\tbest_se\n")
    for r in rows:
        b = BEST[(r[0], r[2])]
        fh.write("\t".join(str(x) if not isinstance(x, float) else f"{x:.5f}" for x in r)
                 + f"\t{b[0]}\t{b[1]:.5f}\t{b[2]:.5f}\n")
log("[out] gia_dose_real.tsv")

def cst(bn, kind, n):
    v = np.array([BEST[(bn, f"{kind}_x{n}_ctrl{j}")][1] for j in range(A.nscram)])
    return float(v.mean()), float(v.std(ddof=1))
log(f"\n[dose] delta-logit at the most favourable position; control = mean of {A.nscram} "
    f"dinucleotide shuffles of the same construct")
for bn, bl in (("real_canonical", "Real 3' end background"),
               ("real_competitor", "Real competitor background")):
    log(f"\n  {bl}")
    log(f"  {'copies':>7} | {'acceptor':>9} {'ctrl':>15} {'diff':>8} | {'donor':>9} "
        f"{'ctrl':>15} {'diff':>8}")
    for n in range(1, A.max_copies + 1):
        a = BEST[(bn, f"acceptor_x{n}")][1]; ac, as_ = cst(bn, "acceptor", n)
        d = BEST[(bn, f"donor_x{n}")][1]; dc, ds_ = cst(bn, "donor", n)
        log(f"  {n:>7} | {a:>+9.4f} {ac:>+8.4f} +/-{as_:<4.3f} {a-ac:>+8.4f} | "
            f"{d:>+9.4f} {dc:>+8.4f} +/-{ds_:<4.3f} {d-dc:>+8.4f}")
    nn = np.arange(1, A.max_copies + 1)
    da = np.array([BEST[(bn, f"acceptor_x{n}")][1] - cst(bn, "acceptor", n)[0] for n in nn])
    dd = np.array([BEST[(bn, f"donor_x{n}")][1] - cst(bn, "donor", n)[0] for n in nn])
    log(f"  trend with copy number, minus control: acceptor r = {np.corrcoef(nn,da)[0,1]:+.3f}, "
        f"donor r = {np.corrcoef(nn,dd)[0,1]:+.3f}")
    log(f"  at seven copies: acceptor {da[-1]:+.4f}, donor {dd[-1]:+.4f}, "
        f"separation {da[-1]-dd[-1]:.4f}")
