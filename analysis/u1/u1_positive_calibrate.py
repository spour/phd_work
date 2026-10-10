#!/usr/bin/env python3
# task a: calibrate the u1 'strong site' cutoff against annotated 5'ss (a positive control)
import sys, csv
import numpy as np, pandas as pd

FASTA = "../train/data/hg38.fa"
U1NPY = "../train/splice_train/data/u1_lookup.npy"
EXONS = "/tmp/claude-3106084/-scratch-spour98-remake-model-updated-HumanGeneEnds-reanal-clean-pub/a71690b6-1403-4fb1-a357-cc6e95da6b0d/scratchpad/exons.tsv"
N_MAX = 60000
SEED  = 42
CHROMS = {f"chr{c}" for c in list(range(1, 23)) + ["X", "Y"]}

def log(*a): print(*a, file=sys.stderr, flush=True)
rng = np.random.default_rng(SEED)

# u1 score array indexed by base-4 9-mer code (a=0 c=1 g=2 t=3)
B = np.full(256, -1, np.int64)
for ch, v in zip(b"ACGT", (0, 1, 2, 3)): B[ch] = v
POW = (4 ** np.arange(8, -1, -1)).astype(np.int64)
arr = np.load(U1NPY, allow_pickle=True)
d = arr.item() if arr.dtype == object else dict(arr)
kb = np.frombuffer("".join(d.keys()).encode("ascii"), np.uint8).reshape(-1, 9)
kc = B[kb]; kidx = kc @ POW
SCORE = np.empty(4 ** 9, np.float64); SCORE[kidx] = -np.fromiter(d.values(), np.float64, len(d))
log(f"[u1] {len(d):,} 9-mers ; score mean {SCORE.mean():.3f} sd {SCORE.std():.3f}")

def score9(seq):
    if seq is None or len(seq) != 9: return np.nan
    t = B[np.frombuffer(seq.encode('ascii'), np.uint8)]
    if (t < 0).any(): return np.nan
    return float(SCORE[int(t @ POW)])

class Faidx:
    _T = str.maketrans("ACGTNacgtn", "TGCANTGCAN")
    def __init__(self, fa):
        self.fh = open(fa, "rb"); self.idx = {}
        for line in open(fa + ".fai"):
            p = line.split()
            self.idx[p[0]] = (int(p[1]), int(p[2]), int(p[3]), int(p[4]))
    def fetch(self, chrom, start, end, strand="+"):
        c = chrom if chrom in self.idx else chrom.lstrip("chr")
        if c not in self.idx: return None
        L, off, lb, lw = self.idx[c]
        start = max(0, start); end = min(end, L)
        if end <= start: return None
        bp = lambda p: off + (p // lb) * lw + (p % lb)
        self.fh.seek(bp(start)); raw = self.fh.read(bp(end) - bp(start))
        seq = raw.replace(b"\n", b"").replace(b"\r", b"")[:end - start].decode("ascii").upper()
        if strand == "-": seq = seq.translate(self._T)[::-1]
        return seq
G = Faidx(FASTA)

# donors + introns from the exon table
E = pd.read_csv(EXONS, sep="\t", header=None,
                names=["tx", "enum", "chrom", "strand", "start", "end", "gene"])
E = E[E.chrom.isin(CHROMS)]
log(f"[gtf] {len(E):,} protein_coding exons on standard chroms")

donors = {}
introns = []
for tx, g in E.groupby("tx", sort=False):
    g = g.sort_values("enum")
    rows = g[["chrom", "strand", "start", "end", "gene"]].values
    if len(rows) < 2: continue
    strand = rows[0][1]
    for i in range(len(rows) - 1):
        ch, st, s0, e0, gn = rows[i]
        _, _, s1, e1, _ = rows[i + 1]
        if strand == "+":
            donors[(ch, e0, "+")] = gn
            lo, hi = e0 + 1, s1 - 1
        else:
            donors[(ch, s0, "-")] = gn
            lo, hi = e1 + 1, s0 - 1
        if hi - lo + 1 >= 60:
            introns.append((ch, lo, hi, strand, gn))
log(f"[gtf] {len(donors):,} unique donors ; {len(introns):,} introns (>=60nt)")

# sample donors, extract [-3,+6] 9-mer, confirm register
dk = list(donors.keys())
if len(dk) > N_MAX:
    dk = [dk[i] for i in rng.choice(len(dk), N_MAX, replace=False)]
don_seq, don_sc, gt_ok = [], [], 0
for ch, pos, strand in dk:
    if strand == "+":
        seq = G.fetch(ch, pos - 3, pos + 6, "+")
    else:
        seq = G.fetch(ch, pos - 7, pos + 2, "-")
    if seq and len(seq) == 9:
        don_seq.append(seq)
        if seq[3:5] == "GT": gt_ok += 1
don_seq = np.array(don_seq)
reg_frac = gt_ok / len(don_seq)
log(f"[register] {len(don_seq):,} donor 9-mers ; GT at indices 3-4 in {reg_frac*100:.1f}%")
gt_mask = np.array([s[3:5] == "GT" for s in don_seq])
don_all = np.array([score9(s) for s in don_seq])
don_gt  = don_all[gt_mask]
log(f"[donors] all {np.isfinite(don_all).sum():,} ; GT-canonical {np.isfinite(don_gt).sum():,}")

# count-matched random intronic 9-mers (same genes), length-weighted
N = int(np.isfinite(don_gt).sum())
ilen = np.array([hi - lo + 1 - 40 for (_, lo, hi, _, _) in introns], float)
ilen = np.clip(ilen, 1, None); p = ilen / ilen.sum()
pick = rng.choice(len(introns), size=int(N * 1.15), replace=True, p=p)
intr_sc = []
for j in pick:
    ch, lo, hi, strand, gn = introns[j]
    a = lo + 20; b = hi - 20 - 8
    if b <= a: continue
    s = int(rng.integers(a, b + 1))
    if strand == "+":
        seq = G.fetch(ch, s - 1, s + 8, "+")
    else:
        seq = G.fetch(ch, s - 1, s + 8, "-")
    v = score9(seq)
    if np.isfinite(v): intr_sc.append(v)
    if len(intr_sc) >= N: break
intr_sc = np.array(intr_sc)
log(f"[intron] {len(intr_sc):,} random intronic 9-mers")

SWEEP = np.round(np.arange(0.0, 12.0001, 0.25), 3)
def tail(x, t): return float(np.mean(x >= t))
uni_tail = np.array([tail(SCORE, t) for t in SWEEP])
sens     = np.array([tail(don_gt, t) for t in SWEEP])
intr_tl  = np.array([tail(intr_sc, t) for t in SWEEP])
enr      = np.where(intr_tl > 0, sens / intr_tl, np.nan)
enr_uni  = np.where(uni_tail > 0, sens / uni_tail, np.nan)

def score_at_sens(target):
    j = np.argmin(np.abs(sens - target))
    ok = np.where(sens >= target)[0]
    return float(SWEEP[ok[-1]]) if ok.size else float("nan")
S50, S75, S90 = score_at_sens(0.50), score_at_sens(0.75), score_at_sens(0.90)
def at(t):
    j = int(np.argmin(np.abs(SWEEP - t)))
    return sens[j], enr[j], intr_tl[j]

print("\n================  U1 POSITIVE-CONTROL CALIBRATION  ================")
print(f"register: 9-mer = 3 exonic + 6 intronic [-3..-1|+1..+6]; GT at idx 3-4 in {reg_frac*100:.1f}% of donors")
print(f"positive set: {N:,} canonical GT donors (GENCODE v49, protein_coding)")
print(f"donor U1 score  : median {np.median(don_gt):.2f}  mean {don_gt.mean():.2f}")
print(f"intronic U1     : median {np.median(intr_sc):.2f}  mean {intr_sc.mean():.2f}")
print(f"all-9mer U1     : median {np.median(SCORE):.2f}  mean {SCORE.mean():.2f}\n")
print(f"{'sensitivity':>12} {'score>=':>8} {'enrich_vs_intron':>17} {'enrich_vs_uniform':>18} {'intron_tail':>12}")
for lab, t in [("50%", S50), ("75%", S75), ("90%", S90)]:
    s, e, it = at(t)
    print(f"{lab:>12} {t:8.2f} {e:17.2f} {(s/uni_tail[int(np.argmin(np.abs(SWEEP-t)))]):18.2f} {it:12.5f}")
for lab, t in [("old 4.0", 4.0), ("6.5", 6.5), ("7.0", 7.0), ("8.0", 8.0)]:
    s, e, it = at(t)
    print(f"  [ladder {lab:>7}]  score>={t:.1f}  sensitivity {s*100:5.1f}%  enrich_vs_intron {e:6.2f}")

with open("u1_positive_calibration.tsv", "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t")
    w.writerow(["threshold", "sensitivity", "intron_tail", "uniform_tail",
                "enrich_vs_intron", "enrich_vs_uniform"])
    for j, t in enumerate(SWEEP):
        w.writerow([f"{t:.3f}", f"{sens[j]:.6f}", f"{intr_tl[j]:.6f}", f"{uni_tail[j]:.6f}",
                    f"{enr[j]:.4f}", f"{enr_uni[j]:.4f}"])
    w.writerow([])
    w.writerow([f"# register_GT_frac={reg_frac:.4f}", f"n_donors_GT={N}", f"n_intron={len(intr_sc)}"])
    w.writerow([f"# score_at_sens50={S50}", f"score_at_sens75={S75}", f"score_at_sens90={S90}"])
    for lab, t in [("50",S50),("75",S75),("90",S90)]:
        s,e,it = at(t); w.writerow([f"# sens{lab}", f"thr={t}", f"enrich_intron={e:.3f}", f"sens={s:.4f}"])

edges = np.arange(-6, 12.001, 0.25)
ctr = (edges[:-1] + edges[1:]) / 2
def hist(x):
    h,_ = np.histogram(x[np.isfinite(x)], bins=edges); return h / max(h.sum(),1)
with open("u1_positive_dist.tsv", "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t"); w.writerow(["score","donor_gt","intronic","all_9mer"])
    for c,a,b_,cc in zip(ctr, hist(don_gt), hist(intr_sc), hist(SCORE)):
        w.writerow([f"{c:.3f}", f"{a:.6f}", f"{b_:.6f}", f"{cc:.6f}"])
print("\n[out] u1_positive_calibration.tsv ; u1_positive_dist.tsv")
