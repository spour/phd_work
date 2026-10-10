#!/usr/bin/env python3
# controls for the protection-window sweep: is it u1, or compartment, or composition?
import csv, os, re, sys, numpy as np, pandas as pd
from collections import defaultdict
def log(*a): print(*a, file=sys.stderr, flush=True)
SEED, NBOOT, NPERM = 42, 400, 200
rng = np.random.default_rng(SEED)
GTF = "../train/data/annotation/gencode.v49.annotation.gtf"
FASTA = "../train/data/hg38.fa"; ME5_LU = "../train/splice_train/data/me5_lookup.npy"
SPAN, PAD, DTHR = 10000, 30, 8.46
WS = [100, 200, 500, 1000, 2000]

M = list(csv.DictReader(open("fig3_new_positional_features.tsv"), delimiter="\t"))
D = pd.read_parquet("shortlist_feat_all.parquet", engine="fastparquet")
D = D[D.gene.isin(set(r["gene"] for r in M))].reset_index(drop=True)
gene = D.gene.to_numpy(); chrom = D.chrom.to_numpy(); y = D.is_canonical.to_numpy().astype(float)
cen = D.center.to_numpy().astype(int); strand = D.strand.to_numpy()
km = {(r["chrom"], int(r["center"]), r["strand"], r["gene"]): i for i, r in enumerate(M)}
ridx = np.array([km[(c, int(ce), st, g)] for c, ce, st, g in zip(D.chrom, D.center, D.strand, D.gene)])
ROW = np.load("upstream_rowmask.npz")["ROW"]
GC = np.load("termexon_gc_cache.npz")["gc500"][ridx]
SCAN = np.load("almada_scan.npz")["X"].astype(float)
IDX = np.where(ROW)[0]
gi = defaultdict(list)
for i in range(len(gene)): gi[gene[i]].append(i)
gi = {g: np.array(v) for g, v in gi.items()}
GEN = [(g, ix[ROW[ix]]) for g, ix in gi.items() if ROW[ix].sum() >= 2 and (y[ix[ROW[ix]]] == 1).sum() == 1]

# gencode donors and introns
TID = re.compile(r'transcript_id "([^"]+)"')
tx = defaultdict(lambda: {"chrom": None, "strand": None, "exon": []})
with open(GTF) as fh:
    for line in fh:
        if line[0] == "#": continue
        f = line.split("\t")
        if f[2] != "exon": continue
        m = TID.search(f[8])
        if not m: continue
        t = tx[m.group(1)]; t["chrom"] = f[0]; t["strand"] = f[6]
        t["exon"].append((int(f[3]) - 1, int(f[4])))
don = defaultdict(list); intr = defaultdict(list)
for t in tx.values():
    ex = sorted(t["exon"])
    if len(ex) < 2: continue
    st = t["strand"]; key = (t["chrom"], st)
    for i in range(len(ex) - 1):
        lo, hi = ex[i][1], ex[i + 1][0]
        if hi <= lo: continue
        intr[key].append((lo, hi))
        don[key].append(lo if st == "+" else hi - 1)
DON = {k: np.array(sorted(set(v))) for k, v in don.items()}
def merge(ivs):
    ivs = sorted(ivs); out = [list(ivs[0])]
    for a, b in ivs[1:]:
        if a <= out[-1][1]: out[-1][1] = max(out[-1][1], b)
        else: out.append([a, b])
    return np.array([a for a, _ in out]), np.array([b for _, b in out])
INTR = {k: merge(v) for k, v in intr.items() if v}
log(f"annotated donors on {len(DON)} keys")

# annotated-donor distances, and host intron donor
d_up_ann = np.full(len(y), np.nan); d_dn_ann = np.full(len(y), np.nan)
host_don_dist = np.full(len(y), np.nan)
in_intron = np.zeros(len(y), bool)
for i in IDX:
    key = (chrom[i], strand[i]); arr = DON.get(key)
    c = cen[i]; st = strand[i]
    if arr is not None and arr.size:
        if st == "+":
            j = np.searchsorted(arr, c)
            if j > 0: d_up_ann[i] = c - arr[j - 1]
            if j < len(arr): d_dn_ann[i] = arr[j] - c
        else:
            j = np.searchsorted(arr, c)
            if j < len(arr): d_up_ann[i] = arr[j] - c
            if j > 0: d_dn_ann[i] = c - arr[j - 1]
    iv = INTR.get(key)
    if iv is not None:
        s_, e_ = iv
        j = np.searchsorted(s_, c, "right") - 1
        if j >= 0 and c < e_[j]:
            in_intron[i] = True
            host_don_dist[i] = (c - s_[j]) if st == "+" else (e_[j] - 1 - c)
log("annotated distances done")

# control-motif scan
if not os.path.exists("almada_ctrl_scan.npz"):
    _DONOR = {0: {"A": .3, "C": .1, "G": .4, "T": .2}, 1: {"A": .6, "C": .1, "G": .1, "T": .2},
              2: {"A": .1, "C": .0, "G": .8, "T": .1}, 3: {"G": 1.0}, 4: {"T": 1.0},
              5: {"A": .6, "G": .3, "C": .05, "T": .05}, 6: {"A": .7, "G": .1, "C": .1, "T": .1},
              7: {"G": .8, "A": .1, "C": .05, "T": .05}, 8: {"T": .5, "C": .2, "A": .2, "G": .1}}
    W9 = np.zeros((9, 4))
    for j in range(9):
        for b, ch in enumerate("ACGT"): W9[j, b] = _DONOR[j].get(ch, 0.0)
    perm = np.random.default_rng(0).permutation(9)
    WC = W9[perm]
    class Faidx:
        _T = str.maketrans("ACGTNacgtn", "TGCANTGCAN")
        def __init__(s, fa):
            s.fh = open(fa, "rb"); s.idx = {}
            for L in open(fa + ".fai"):
                p = L.split(); s.idx[p[0]] = (int(p[1]), int(p[2]), int(p[3]), int(p[4]))
        def fetch(s, c, a, b, st="+"):
            c = c if c in s.idx else c.lstrip("chr")
            if c not in s.idx: return None
            Ln, off, lb, lw = s.idx[c]
            if a < 0 or b > Ln: return None
            bp = lambda p: off + (p // lb) * lw + (p % lb)
            s.fh.seek(bp(a)); raw = s.fh.read(bp(b) - bp(a))
            q = raw.replace(b"\n", b"").replace(b"\r", b"")[:b - a].decode("ascii").upper()
            return q.translate(s._T)[::-1] if st == "-" else q
    _MAP = np.full(256, -1, np.int64)
    for i, ch in enumerate("ACGT"): _MAP[ord(ch)] = i
    def code(seq): return _MAP[np.frombuffer(seq.encode("ascii"), np.uint8)]
    G = Faidx(FASTA); L = SPAN + PAD
    dens_target = 1.0 / np.nanmedian(SCAN[IDX, 0])
    samp = []
    for i in IDX[:1500]:
        a, b = (cen[i] - L, cen[i] + L) if strand[i] == "+" else (cen[i] - L + 1, cen[i] + L + 1)
        q = G.fetch(chrom[i], a, b, strand[i])
        if q is None or len(q) != 2 * L: continue
        c = code(q); n = len(c) - 8
        w = np.lib.stride_tricks.sliding_window_view(c, 9)[:n]
        ok = (w >= 0).all(1) & ~((c[3:3 + n] == 2) & (c[4:4 + n] == 3))
        sc = WC[np.arange(9)[None, :], np.where(w < 0, 0, w)].mean(1)
        samp.append(sc[ok])
    samp = np.concatenate(samp)
    thr = float(np.quantile(samp, 1.0 - dens_target))
    log(f"control threshold {thr:.4f} for density {dens_target:.2e}/nt "
        f"(hit rate {np.mean(samp >= thr):.2e})")
    OUT = np.full((len(y), 2), np.nan)
    for n_, i in enumerate(IDX):
        a, b = (cen[i] - L, cen[i] + L) if strand[i] == "+" else (cen[i] - L + 1, cen[i] + L + 1)
        q = G.fetch(chrom[i], a, b, strand[i])
        if q is None or len(q) != 2 * L: continue
        c = code(q); S = L; n = len(c) - 8
        w = np.lib.stride_tricks.sliding_window_view(c, 9)[:n]
        ok = (w >= 0).all(1) & ~((c[3:3 + n] == 2) & (c[4:4 + n] == 3))
        sc = WC[np.arange(9)[None, :], np.where(w < 0, 0, w)].mean(1)
        p = np.where(ok & (sc >= thr))[0] + 3
        up = p[p < S]; dn = p[p > S]
        if up.size: OUT[i, 0] = S - up[-1]
        if dn.size: OUT[i, 1] = dn[0] - S
        if (n_ + 1) % 10000 == 0: log(f"  ctrl site {n_+1:,}/{len(IDX):,}")
    np.savez_compressed("almada_ctrl_scan.npz", X=OUT.astype(np.float32), thr=np.array([thr]))
CTRL = np.load("almada_ctrl_scan.npz")["X"].astype(float)

# statistics
def winrates(v):
    out = np.full(len(GEN), np.nan)
    for t, (g, ix) in enumerate(GEN):
        s = v[ix]; c = int(np.argmax(y[ix])); o = np.delete(np.arange(len(ix)), c)
        if not np.isfinite(s[c]): continue
        d = s[o][np.isfinite(s[o])]
        if len(d): out[t] = float(np.mean((s[c] > d) + 0.5 * (s[c] == d)))
    return out
BOOTG = rng.integers(0, len(GEN), size=(NBOOT, len(GEN)))
def wstats(v):
    w = winrates(v); b = np.nanmean(w[BOOTG], axis=1)
    return float(np.nanmean(w)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
def prot(d, W): return np.where(np.isfinite(d), (d <= W).astype(float), 0.0)

print(f"\n===== 1. ANNOTATED DONORS ONLY ({len(GEN)} genes) =====")
mm = np.isfinite(SCAN[IDX, 0]) & np.isfinite(d_up_ann[IDX])
for tol in (0, 2, 10):
    print(f"  scan strong-donor hits (8.46) that coincide with an annotated 5'ss within {tol:>2} nt: "
          f"{np.mean(np.abs(SCAN[IDX, 0][mm] - d_up_ann[IDX][mm]) <= tol):.4f}")
print(f"  median distance: scan hits {np.nanmedian(SCAN[IDX,0]):.0f} nt, "
      f"annotated donors {np.nanmedian(d_up_ann[IDX]):.0f} nt")
with open("almada_controls.tsv", "w") as fh:
    fh.write("analysis\tdirection\tfeature\twinrate\tlo\thi\tfrac_canonical\tfrac_competitor\n")
    for nm, du, dd in [("annotated donors", d_up_ann, d_dn_ann),
                       ("control motif", CTRL[:, 0], CTRL[:, 1]),
                       ("scan donors 8.46", SCAN[:, 0], SCAN[:, 1])]:
        print(f"\n--- {nm} ---")
        print(f"{'feature':<26} {'upstream win-rate':>18} {'95% CI':>17} | "
              f"{'downstream':>11} {'95% CI':>17}")
        for W in WS:
            vu = prot(du, W); vd = prot(dd, W)
            a, alo, ahi = wstats(vu); b, blo, bhi = wstats(vd)
            print(f"W = {W:<5} nt{'':<12} {a:>18.4f} [{alo:.4f},{ahi:.4f}] | {b:>11.4f} [{blo:.4f},{bhi:.4f}]")
            fh.write(f"{nm}\tupstream\tW{W}\t{a:.5f}\t{alo:.5f}\t{ahi:.5f}\t"
                     f"{vu[IDX][y[IDX]==1].mean():.5f}\t{vu[IDX][y[IDX]==0].mean():.5f}\n")
            fh.write(f"{nm}\tdownstream\tW{W}\t{b:.5f}\t{blo:.5f}\t{bhi:.5f}\t"
                     f"{vd[IDX][y[IDX]==1].mean():.5f}\t{vd[IDX][y[IDX]==0].mean():.5f}\n")

    print("\n===== 3. GC CONTROL (upstream donors, cutoff 8.46) =====")
    for W in (500, 1000):
        v = prot(SCAN[:, 0], W)
        m = np.isfinite(GC)
        A = np.column_stack([np.ones(int((ROW & m).sum())), GC[ROW & m]])
        bb = np.linalg.lstsq(A, v[ROW & m], rcond=None)[0]
        r = np.full(len(y), np.nan); r[ROW & m] = v[ROW & m] - A @ bb
        a, lo, hi = wstats(v); ar, rlo, rhi = wstats(r)
        print(f"  W = {W:>4}: raw {a:.4f} [{lo:.4f},{hi:.4f}]  ->  GC-residualised {ar:.4f} [{rlo:.4f},{rhi:.4f}]")
        fh.write(f"gc residualised\tupstream\tW{W}\t{ar:.5f}\t{rlo:.5f}\t{rhi:.5f}\t\t\n")
        q = np.full(len(y), -1)
        qs = np.quantile(GC[ROW & m], [.2, .4, .6, .8])
        q[ROW & m] = np.searchsorted(qs, GC[ROW & m])
        n_pair = 0; n_disc = 0; comp_prot = 0
        for g, ix in GEN:
            c = ix[np.argmax(y[ix])]
            for o in ix[y[ix] == 0]:
                if q[c] < 0 or q[o] < 0 or q[c] != q[o]: continue
                n_pair += 1
                if v[c] != v[o]:
                    n_disc += 1
                    if v[o] > v[c]: comp_prot += 1
        fr = comp_prot / n_disc if n_disc else np.nan
        b = np.array([np.mean(rng.random(n_disc) < fr) for _ in range(200)])
        print(f"           same-GC-quintile pairs {n_pair}, discordant {n_disc}; "
              f"competitor is the protected one in {fr:.4f} of them (0.5 = no effect)")
        fh.write(f"gc quintile pairs\tupstream\tW{W}\t{fr:.5f}\t\t\t{n_disc}\t{n_pair}\n")

    print("\n===== 4. IS THE PROTECTING DONOR THE HOST INTRON'S OWN 5'ss? =====")
    for W in (500, 1000, 2000):
        v = prot(SCAN[:, 0], W).astype(bool)
        for cls, nm in [(1, "canonical"), (0, "competitor")]:
            sel = IDX[(y[IDX] == cls) & v[IDX]]
            if len(sel) == 0: continue
            same = np.abs(SCAN[sel, 0] - host_don_dist[sel]) <= 2
            print(f"  W = {W:>4}, {nm:<11}: protected n {len(sel):>6}, in an annotated intron "
                  f"{np.mean(in_intron[sel]):.3f}, protecting donor is the host intron's own 5'ss "
                  f"{np.nanmean(same):.3f}")
            fh.write(f"host donor\t{nm}\tW{W}\t{float(np.nanmean(same)):.5f}\t\t\t"
                     f"{float(np.mean(in_intron[sel])):.5f}\t{len(sel)}\n")
log("DONE_ALMADA_CONTROLS")
