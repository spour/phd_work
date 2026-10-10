#!/usr/bin/env python3
# everything measurable about real 3' ends versus the internal sites the models mistake
import argparse, os, sys, json, gzip, math, re, itertools
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

# setup
RC = str.maketrans("ACGTN", "TGCAN")

try:
    from scipy import stats as sps
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

try:
    from maxentpy import maxent
    from maxentpy.maxent import load_matrix5, load_matrix3
    _M5, _M3 = load_matrix5(), load_matrix3()
    HAVE_MAXENT = True
except Exception:
    HAVE_MAXENT = False

PAS_CANON = ["AATAAA", "ATTAAA"]
PAS_VAR = ["TATAAA", "AGTAAA", "AAGAAA", "AATATA", "AATACA", "CATAAA",
           "GATAAA", "AATGAA", "TTTAAA", "ACTAAA", "AATAGA", "AAAAAG",
           "AAAACA", "GGGGCT"]

DG_NN = {"AA": -1.00, "TT": -1.00, "AT": -0.88, "TA": -0.58,
         "CA": -1.45, "TG": -1.45, "GT": -1.44, "AC": -1.44,
         "CT": -1.28, "AG": -1.28, "GA": -1.30, "TC": -1.30,
         "CG": -2.17, "GC": -2.24, "GG": -1.84, "CC": -1.84}

BEND = {"AA": -0.274, "AC": -0.205, "AG": -0.081, "AT": -0.280,
        "CA": 0.018, "CC": -0.032, "CG": 0.043, "CT": -0.081,
        "GA": -0.007, "GC": 0.075, "GG": -0.032, "GT": -0.205,
        "TA": 0.182, "TC": -0.007, "TG": 0.018, "TT": -0.274}

PROP = {"AA": -17.3, "AC": -6.7, "AG": -14.3, "AT": -16.9,
        "CA": -8.6, "CC": -12.1, "CG": -11.1, "CT": -14.3,
        "GA": -15.1, "GC": -11.8, "GG": -12.1, "GT": -6.7,
        "TA": -11.1, "TC": -15.1, "TG": -8.6, "TT": -17.3}

U1_TEMPLATE = "ACUUACCUG"
NN_RNA = {"AA": -0.93, "AU": -1.10, "AC": -2.24, "AG": -2.08,
          "UA": -1.33, "UU": -0.93, "UC": -2.35, "UG": -2.11,
          "CA": -2.11, "CU": -2.08, "CC": -3.26, "CG": -2.36,
          "GA": -2.35, "GU": -2.24, "GC": -3.42, "GG": -3.26}
WC = {"A": "U", "U": "A", "G": "C", "C": "G"}

G4_RE = re.compile(r"G{3,}\w{1,7}G{3,}\w{1,7}G{3,}\w{1,7}G{3,}")
GCLUST_RE = re.compile(r"G{4,}")

WINDOWS = [
    ("pas",      -60,   -10,  False),
    ("cleave",   -10,    10,  False),
    ("dse",       10,    60,  False),
    ("core",    -250,   250,  False),
    ("nearup",  -500,  -100,  False),
    ("neardn",   100,   500,  False),
    ("midup",  -2000,  -500,  True),
    ("middn",    500,  2000,  True),
    ("farup",  -5000, -2000,  True),
    ("fardn",   2000,  5000,  True),
]


class Genome:

    def __init__(self, path):
        fai = path + ".fai"
        if not os.path.exists(fai):
            sys.exit("need " + fai + "; run `samtools faidx " + path + "`")
        self.idx = {}
        for line in open(fai):
            f = line.split("\t")
            self.idx[f[0]] = (int(f[1]), int(f[2]), int(f[3]), int(f[4]))
        self.fh = open(path, "rb")

    def fetch(self, chrom, start, end):
        if chrom not in self.idx or end <= start:
            return ""
        ln, off, lb, lw = self.idx[chrom]
        start, end = max(0, start), min(end, ln)
        if end <= start:
            return ""
        def seek(p):
            return off + p // lb * lw + p % lb
        self.fh.seek(seek(start))
        return self.fh.read(seek(end) - seek(start)).decode().replace("\n", "").upper()


def oriented(gen, chrom, pos, strand, up, dn):
    if strand == "-":
        s = gen.fetch(chrom, int(pos) - dn, int(pos) + up)
        return s.translate(RC)[::-1] if len(s) >= up + dn else ""
    s = gen.fetch(chrom, int(pos) - up, int(pos) + dn)
    return s if len(s) >= up + dn else ""


# primitives
def comp_stats(s):
    n = len(s)
    out = {}
    if n < 10:
        return out
    cnt = Counter(s)
    acgt = sum(cnt[b] for b in "ACGT")
    if acgt < 10:
        return out
    for b in "ACGT":
        out[b] = cnt[b] / acgt
    g, c, a, t = cnt["G"], cnt["C"], cnt["A"], cnt["T"]
    out["GC"] = (g + c) / acgt
    out["GCskew"] = (g - c) / (g + c) if (g + c) else np.nan
    out["ATskew"] = (a - t) / (a + t) if (a + t) else np.nan
    pur, pyr = a + g, c + t
    out["PURskew"] = (pur - pyr) / (pur + pyr) if (pur + pyr) else np.nan
    keto, amino = g + t, a + c
    out["KETOskew"] = (keto - amino) / (keto + amino) if (keto + amino) else np.nan

    di = Counter(s[i:i + 2] for i in range(n - 1))
    dtot = sum(v for k, v in di.items() if "N" not in k)
    if dtot:
        obs_cg = di["CG"]
        exp_cg = c * g / acgt
        out["CpG_oe"] = obs_cg / exp_cg if exp_cg > 0 else np.nan
        out["dG"] = sum(DG_NN.get(k, 0) * v for k, v in di.items()) / dtot
        out["bend"] = sum(BEND.get(k, 0) * v for k, v in di.items()) / dtot
        out["proptwist"] = sum(PROP.get(k, 0) * v for k, v in di.items()) / dtot
        out["WW"] = sum(di[k] for k in ("AA", "AT", "TA", "TT")) / dtot
        out["SS"] = sum(di[k] for k in ("GG", "GC", "CG", "CC")) / dtot

    for k in (1, 2, 3, 4):
        out["ent%d" % k] = kmer_entropy(s, k)
    out["maxA"] = longest_run(s, "A")
    out["maxT"] = longest_run(s, "T")
    out["maxG"] = longest_run(s, "G")
    out["maxAT"] = longest_run(s, "AT")
    out["maxGC"] = longest_run(s, "GC")
    out["A5tracts"] = len(re.findall(r"A{5,}", s)) / (n / 100.0)
    out["T5tracts"] = len(re.findall(r"T{5,}", s)) / (n / 100.0)
    out["G4"] = len(G4_RE.findall(s)) / (n / 1000.0)
    out["Gclust"] = len(GCLUST_RE.findall(s)) / (n / 1000.0)
    return out


def kmer_entropy(s, k):
    if len(s) < k + 1:
        return np.nan
    c = Counter(s[i:i + k] for i in range(len(s) - k + 1) if "N" not in s[i:i + k])
    tot = sum(c.values())
    if not tot:
        return np.nan
    return -sum((v / tot) * math.log2(v / tot) for v in c.values())


def longest_run(s, chars):
    best = run = 0
    for ch in s:
        run = run + 1 if ch in chars else 0
        if run > best:
            best = run
    return best


def periodicity(s, periods=(3, 10, 11, 200)):
    out = {}
    n = len(s)
    if n < 3 * max(periods):
        return {("per%d" % p): np.nan for p in periods}
    v = np.zeros(n - 1)
    for i in range(n - 1):
        d = s[i:i + 2]
        if d in ("AA", "TT", "AT", "TA"):
            v[i] = 1.0
        elif d in ("GG", "CC", "GC", "CG"):
            v[i] = -1.0
    v -= v.mean()
    if not np.any(v):
        return {("per%d" % p): np.nan for p in periods}
    f = np.abs(np.fft.rfft(v)) ** 2
    freqs = np.fft.rfftfreq(len(v))
    tot = f[1:].sum()
    for p in periods:
        j = int(np.argmin(np.abs(freqs - 1.0 / p)))
        out["per%d" % p] = float(f[j] / tot) if tot > 0 else np.nan
    return out


def palindrome_score(s, minstem=6, maxloop=30, maxstem=15):
    n = len(s)
    if n < 2 * minstem + 3:
        return dict(hairpin_len=0, palindrome_frac=np.nan)
    best = 0
    hits = 0
    step = max(1, n // 400)
    for i in range(0, n - 2 * minstem, step):
        for loop in range(3, min(maxloop, n - i - 2 * minstem) + 1, 3):
            j = i + minstem + loop
            if j + minstem > n:
                break
            L = 0
            while (L < maxstem and i + L < n and j + minstem - 1 - L >= 0 and
                   i - L >= 0):
                a = s[i + minstem - 1 - L] if minstem - 1 - L >= 0 else None
                b = s[j + L] if j + L < n else None
                if a is None or b is None:
                    break
                if WC.get(a.replace("T", "U"), "") != b.replace("T", "U"):
                    break
                L += 1
            if L >= minstem:
                hits += 1
                best = max(best, L)
    return dict(hairpin_len=best, palindrome_frac=hits / max(n / 100.0, 1))


def motif_features(seq, centre, w0, w1, tag):
    s = seq[centre + w0:centre + w1]
    n = max(len(s), 1)
    out = {}
    out[tag + " PAScanon"] = sum(s.count(h) for h in PAS_CANON)
    out[tag + " PASvar"] = sum(s.count(h) for h in PAS_VAR)
    out[tag + " UGUA"] = s.count("TGTA")
    out[tag + " Urich"] = len(re.findall(r"T{4,}", s))
    out[tag + " GUrich"] = len(re.findall(r"(?:TG|GT){3,}", s))
    out[tag + " CA"] = s.count("CA")
    for k in list(out):
        out[k] = out[k] / (n / 100.0)
    return out


def hexamer_geometry(seq, centre, up=100):
    s = seq[centre - up:centre]
    out = dict(hex_n=0, hex_best_off=np.nan, hex_spread=np.nan,
               hex_is_canon=np.nan, hex_nearest_off=np.nan)
    offs, canon = [], []
    for h in PAS_CANON + PAS_VAR:
        i = s.find(h)
        while i != -1:
            offs.append(i - up)
            canon.append(h in PAS_CANON)
            i = s.find(h, i + 1)
    if not offs:
        return out
    offs = np.array(offs, float)
    out["hex_n"] = len(offs)
    j = int(np.argmax(offs))
    out["hex_nearest_off"] = float(offs[j])
    out["hex_is_canon"] = float(canon[j])
    out["hex_spread"] = float(offs.max() - offs.min()) if len(offs) > 1 else 0.0
    out["hex_best_off"] = float(offs[np.argmin(np.abs(offs + 21))])
    return out


def u1_energy(seq9):
    s = seq9.replace("T", "U")
    if len(s) != 9:
        return np.nan
    paired = [WC.get(a) == b for a, b in zip(s, U1_TEMPLATE)]
    e = 0.0
    for i in range(8):
        if paired[i] and paired[i + 1]:
            e += NN_RNA.get(s[i:i + 2], 0.0)
    return e


def splice_features(seq, centre, span, lam=500.0):
    out = {}
    i0, i1 = max(0, centre - span), min(len(seq), centre + span)
    d5, s5, d3, s3, e5 = [], [], [], [], []
    for i in range(i0, max(i0, i1 - 9)):
        if seq[i + 3:i + 5] != "GT":
            continue
        w = seq[i:i + 9]
        if "N" in w:
            continue
        try:
            sc = float(maxent.score5(w, matrix=_M5))
        except Exception:
            continue
        if np.isfinite(sc):
            d5.append(i + 3 - centre); s5.append(sc); e5.append(u1_energy(w))
    for i in range(i0, max(i0, i1 - 23)):
        if seq[i + 18:i + 20] != "AG":
            continue
        w = seq[i:i + 23]
        if "N" in w:
            continue
        try:
            sc = float(maxent.score3(w, matrix=_M3))
        except Exception:
            continue
        if np.isfinite(sc):
            d3.append(i + 20 - centre); s3.append(sc)

    d5 = np.array(d5, float); s5 = np.array(s5, float)
    d3 = np.array(d3, float); s3 = np.array(s3, float)
    e5 = np.array(e5, float)

    for thr, tag in ((-99, "any"), (0, "t0"), (6, "t6"), (8, "t8")):
        m5 = s5 >= thr
        m3 = s3 >= thr
        out["n5ss_%s" % tag] = int(m5.sum())
        out["n3ss_%s" % tag] = int(m3.sum())
        dn = m5 & (d5 > 0)
        out["near5ss_%s" % tag] = float(d5[dn].min()) if dn.any() else np.nan
        up3 = m3 & (d3 < 0)
        out["near3ss_%s" % tag] = float(-d3[up3].max()) if up3.any() else np.nan

    out["best5ss"] = float(s5.max()) if len(s5) else np.nan
    out["best3ss_up"] = float(s3[d3 < 0].max()) if (d3 < 0).any() else np.nan
    out["best3ss_dn"] = float(s3[d3 > 0].max()) if (d3 > 0).any() else np.nan
    out["best5ss_dG"] = float(np.nanmin(e5)) if len(e5) else np.nan

    def wbest(d, s, side):
        m = (d < 0) if side == "up" else (d > 0)
        if not m.any():
            return np.nan, np.nan
        w = s[m] - np.log1p(np.abs(d[m]) / lam)
        j = int(np.argmax(w))
        return float(w[j]), float(d[m][j])

    wu, du = wbest(d3, s3, "up")
    wd, dd = wbest(d5, s5, "dn")
    out["exon_acc_w"] = wu
    out["exon_don_w"] = wd
    out["exon_acc_dist"] = du
    out["exon_don_dist"] = dd
    out["exon_balance"] = (wu - wd) if np.isfinite(wu) and np.isfinite(wd) else np.nan
    out["exon_min"] = (float(np.nanmin([wu, wd]))
                       if np.isfinite([wu, wd]).any() else np.nan)
    return out


def notch_shape(gcv, centre, half=1000, flank_in=2000, flank_out=3000):
    flank = float(np.nanmean(np.r_[gcv[centre - flank_out:centre - flank_in],
                                   gcv[centre + flank_in:centre + flank_out]]))
    inner = gcv[centre - half:centre + half]
    below = flank - inner
    out = dict(notch_flank=flank,
               notch_depth=float(np.nanmax(below)),
               notch_width=float(np.sum(below > 0.02)),
               notch_area=float(np.nansum(np.clip(below, 0, None))),
               notch_fracbelow=float(np.nanmean(below > 0)),
               notch_min=float(np.nanmin(inner)))
    j = int(np.nanargmin(inner))
    out["notch_minoff"] = float(j - half)
    left, right = below[:j], below[j:]
    tot = float(np.nansum(below))
    out["notch_asym"] = (float(np.nansum(left) - np.nansum(right)) / tot
                         if abs(tot) > 1e-6 else np.nan)
    return out


# repeats
def load_rmsk(path, want):
    cols = defaultdict(list)
    op = gzip.open if path.endswith(".gz") else open
    n = 0
    with op(path, "rt") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 13 or f[5] not in want:
                continue
            cols[f[5]].append((int(f[6]), int(f[7]), f[9], f[10], f[11], f[12]))
            n += 1
    out = {}
    for c, v in cols.items():
        v.sort(key=lambda x: x[0])
        out[c] = dict(start=np.array([x[0] for x in v]),
                      end=np.array([x[1] for x in v]),
                      strand=np.array([x[2] for x in v]),
                      name=np.array([x[3] for x in v]),
                      cls=np.array([x[4] for x in v]),
                      fam=np.array([x[5] for x in v]))
    print(f"[rmsk] {n:,} intervals, {len(out)} chromosomes", file=sys.stderr)
    return out


REP_FAMS = ("Alu", "L1", "L2", "MIR", "ERV1", "ERVL", "ERVL-MaLR", "ERVK",
            "hAT-Charlie", "TcMar-Tigger", "Simple_repeat", "Low_complexity")


def repeat_features(rm, chrom, pos, strand):
    out = dict(rep_hit=0.0, rep_sense=np.nan, rep_frac=np.nan,
               rep_len=np.nan, rep_family="none", rep_class="none")
    e = rm.get(chrom)
    if e is not None and len(e["start"]):
        j = np.searchsorted(e["start"], pos, side="right") - 1
        if j >= 0 and pos <= e["end"][j]:
            ln = int(e["end"][j] - e["start"][j])
            out.update(rep_hit=1.0, rep_family=str(e["fam"][j]),
                       rep_class=str(e["cls"][j]),
                       rep_sense=float(e["strand"][j] == strand),
                       rep_frac=float((pos - e["start"][j]) / max(ln, 1)),
                       rep_len=float(ln))
    lo, hi = pos - 1000, pos + 1000
    cov = dict(rep_cov=0.0)
    for f in REP_FAMS:
        cov["cov_" + f] = 0.0
        cov["covS_" + f] = 0.0
    if e is not None and len(e["start"]):
        i0 = np.searchsorted(e["end"], lo, side="left")
        i1 = np.searchsorted(e["start"], hi, side="right")
        span = max(hi - lo, 1)
        tot = 0
        for k in range(i0, i1):
            a, b = max(lo, e["start"][k]), min(hi, e["end"][k])
            if b <= a:
                continue
            tot += b - a
            fam = e["fam"][k]
            if fam in REP_FAMS:
                cov["cov_" + fam] += (b - a) / span
                if (e["strand"][k] == strand):
                    cov["covS_" + fam] += (b - a) / span
        cov["rep_cov"] = tot / span
    out.update(cov)
    return out


# feature assembly
def measure(gen, rm, chrom, pos, strand, cfg):
    UP, DN = cfg["up"], cfg["dn"]
    seq = oriented(gen, chrom, pos, strand, UP, DN)
    if len(seq) != UP + DN:
        return None
    c = UP
    rec = {}

    for name, w0, w1, outside in WINDOWS:
        s = seq[c + w0:c + w1]
        for k, v in comp_stats(s).items():
            rec[f"{name}.{k}"] = v
        if (w1 - w0) >= 600:
            for k, v in periodicity(s).items():
                rec[f"{name}.{k}"] = v

    for name, w0, w1, _ in WINDOWS[:6]:
        rec.update(motif_features(seq, c, w0, w1, name))
    rec.update(hexamer_geometry(seq, c, up=cfg["hex_up"]))

    arr = np.frombuffer(seq.encode(), dtype=np.uint8)
    v = ((arr == ord("G")) | (arr == ord("C"))).astype(float)
    k = np.ones(cfg["gc_win"])
    gcv = np.convolve(v, k, "same") / np.convolve(np.ones(len(v)), k, "same")
    rec.update(notch_shape(gcv, c))

    rec.update({("core." + k_): v_
                for k_, v_ in palindrome_score(seq[c - 150:c + 150]).items()})

    if cfg["splice"] and HAVE_MAXENT:
        rec.update(splice_features(seq, c, cfg["span"]))

    if rm is not None:
        rec.update(repeat_features(rm, chrom, pos, strand))
    return rec


def bh_fdr(p):
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    q = np.full(len(p), np.nan)
    idx = np.where(ok)[0]
    if not len(idx):
        return q
    pv = p[idx]
    o = np.argsort(pv)
    n = len(pv)
    adj = pv[o] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.clip(adj, 0, 1)
    q[idx] = out
    return q


def paired_test(v):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if len(v) < 20:
        return dict(n=len(v))
    sd = v.std(ddof=1)
    se = sd / np.sqrt(len(v))
    dz = v.mean() / sd if sd > 0 else np.nan
    if HAVE_SCIPY and sd > 0:
        t, p = sps.ttest_1samp(v, 0.0)
        try:
            _, pw = sps.wilcoxon(v, zero_method="zsplit")
        except Exception:
            pw = np.nan
    else:
        t = v.mean() / se if se > 0 else np.nan
        p = 2 * math.erfc(abs(t) / math.sqrt(2)) / 2 if np.isfinite(t) else np.nan
        pw = np.nan
    return dict(n=len(v), mean=float(v.mean()),
                lo=float(v.mean() - 1.96 * se), hi=float(v.mean() + 1.96 * se),
                dz=float(dz), p=float(p), p_wilcoxon=float(pw)
                if np.isfinite(pw) else np.nan,
                frac_pos=float((v > 0).mean()))


def two_sample(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    if len(x) < 20 or len(y) < 20:
        return dict(n_a=len(x), n_b=len(y))
    sd = math.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
    d = (x.mean() - y.mean()) / sd if sd > 0 else np.nan
    if HAVE_SCIPY:
        t, p = sps.ttest_ind(x, y, equal_var=False)
        try:
            u, pu = sps.mannwhitneyu(x, y, alternative="two-sided")
            auc = u / (len(x) * len(y))
        except Exception:
            pu, auc = np.nan, np.nan
    else:
        se = math.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
        t = (x.mean() - y.mean()) / se if se > 0 else np.nan
        p = 2 * math.erfc(abs(t) / math.sqrt(2)) / 2 if np.isfinite(t) else np.nan
        pu, auc = np.nan, np.nan
    return dict(n_a=len(x), n_b=len(y), mean_a=float(x.mean()),
                mean_b=float(y.mean()), d=float(d), p=float(p),
                auc=float(auc) if np.isfinite(auc) else np.nan,
                var_ratio=float(y.var(ddof=1) / max(x.var(ddof=1), 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peaks", required=True)
    ap.add_argument("--genome", required=True)
    ap.add_argument("--rmsk", default=None)
    ap.add_argument("--up", type=int, default=5000)
    ap.add_argument("--dn", type=int, default=5000)
    ap.add_argument("--gc_win", type=int, default=201)
    ap.add_argument("--hex_up", type=int, default=100)
    ap.add_argument("--span", type=int, default=2500,
                    help="MaxEnt splice search half-width")
    ap.add_argument("--no_splice", action="store_true",
                    help="skip MaxEnt; roughly halves the runtime")
    ap.add_argument("--max_ccpa", type=int, default=8,
                    help="false positives kept per gene. Genes carry 1 to ~900; "
                         "without a cap a handful dominate every pooled mean")
    ap.add_argument("--n_control", type=int, default=2)
    ap.add_argument("--kmer_k", type=int, default=6)
    ap.add_argument("--n_perm", type=int, default=200,
                    help="label permutations for the family-wise null")
    ap.add_argument("--n_clusters", type=int, default=5)
    ap.add_argument("--out", default="master")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    W = lambda n, df: df.to_csv(os.path.join(a.out, n), sep="\t", index=False)
    rng = np.random.default_rng(0)
    cfg = dict(up=a.up, dn=a.dn, gc_win=a.gc_win, hex_up=a.hex_up,
               span=a.span, splice=not a.no_splice)

    print(f"[cfg]  scipy {HAVE_SCIPY}, maxent {HAVE_MAXENT}, "
          f"splice {'on' if cfg['splice'] else 'off'}", file=sys.stderr)

    pk = pd.read_parquet(a.peaks)
    if "in_analysis_set" in pk.columns:
        pk = pk[pk.in_analysis_set.astype(bool) | pk.is_canonical.astype(bool)]
    pos = "center" if "center" in pk.columns else "pos"
    d = pd.DataFrame({"gene": pk.gene.astype(str), "chrom": pk.chrom.astype(str),
                      "pos": pk[pos].astype(int), "strand": pk.strand.astype(str),
                      "is_real": pk.is_canonical.astype(bool)})
    canon = d[d.is_real].groupby("gene").pos.first().to_dict()
    if a.max_ccpa > 0:
        keep = []
        for g, sub in d.groupby("gene", sort=False):
            r = sub[sub.is_real]; c = sub[~sub.is_real]
            if len(c) > a.max_ccpa:
                c = c.sample(a.max_ccpa, random_state=0)
            keep.append(pd.concat([r, c]))
        d = pd.concat(keep, ignore_index=True)
    print(f"[set]  {int(d.is_real.sum()):,} real, "
          f"{int((~d.is_real).sum()):,} false positives, "
          f"{d.gene.nunique():,} genes", file=sys.stderr)

    gen = Genome(a.genome)
    rm = load_rmsk(a.rmsk, set(d.chrom.unique())) if a.rmsk else None

    rows = []
    kmer_sites = {"real": Counter(), "fp": Counter(), "control": Counter()}
    kmer_n = {"real": 0, "fp": 0, "control": 0}
    n_bad = 0
    for gi, (g, sub) in enumerate(d.groupby("gene", sort=False)):
        if gi and gi % 200 == 0:
            print(f"[run]  {gi:,} genes, {len(rows):,} positions...",
                  file=sys.stderr)
        cpos = canon.get(g)
        todo = [(r.chrom, r.pos, r.strand, "real" if r.is_real else "fp")
                for r in sub.itertuples()]
        if cpos is not None:
            for r in sub.itertuples():
                if r.is_real:
                    continue
                off = abs(r.pos - cpos)
                if off < 200:
                    continue
                for _ in range(a.n_control):
                    f = float(rng.uniform(0.5, 2.0))
                    cp = cpos - int(off * f) if r.pos < cpos else cpos + int(off * f)
                    todo.append((r.chrom, cp, r.strand, "control"))
        for chrom, p_, st, cls in todo:
            rec = measure(gen, rm, chrom, p_, st, cfg)
            if rec is None:
                n_bad += 1
                continue
            rec.update(gene=g, chrom=chrom, pos=p_, strand=st, cls=cls)
            rows.append(rec)
            s = oriented(gen, chrom, p_, st, a.up, a.dn)
            core = s[a.up - 250:a.up + 250]
            seen = {core[i:i + a.kmer_k]
                    for i in range(len(core) - a.kmer_k + 1)
                    if "N" not in core[i:i + a.kmer_k]}
            kmer_sites[cls].update(seen)
            kmer_n[cls] += 1

    if n_bad:
        print(f"[seq]  {n_bad:,} positions dropped at contig edges",
              file=sys.stderr)
    P = pd.DataFrame(rows)
    W("per_site.tsv", P)
    print(f"[seq]  {len(P):,} positions, {P.shape[1]:,} columns",
          file=sys.stderr)

    META = {"gene", "chrom", "pos", "strand", "cls", "rep_family", "rep_class"}
    FEATS = [c for c in P.columns if c not in META and
             pd.api.types.is_numeric_dtype(P[c])]
    print(f"[feat] {len(FEATS):,} numeric features", file=sys.stderr)

    RF = P[P.cls.isin(["real", "fp"])]
    CT = P[P.cls == "control"]
    FP = P[P.cls == "fp"]

    print("\n[cmp]  computing paired, pooled and control comparisons...",
          file=sys.stderr)
    real_by_gene = RF[RF.cls == "real"].set_index("gene")
    fp_mean_by_gene = RF[RF.cls == "fp"].groupby("gene")[FEATS].mean()
    common = real_by_gene.index.intersection(fp_mean_by_gene.index)
    DIFF = (real_by_gene.loc[common, FEATS].astype(float) -
            fp_mean_by_gene.loc[common, FEATS].astype(float))
    W("paired_differences.tsv", DIFF.reset_index())

    res = []
    for c in FEATS:
        pr = paired_test(DIFF[c].to_numpy())
        po = two_sample(RF.loc[RF.cls == "real", c], RF.loc[RF.cls == "fp", c])
        cc = two_sample(FP[c], CT[c]) if len(CT) else {}
        outside = not any(c.startswith(w[0] + ".") or c.startswith(w[0] + " ")
                          for w in WINDOWS if not w[3])
        res.append(dict(
            feature=c, n_genes=pr.get("n", 0),
            paired_diff=pr.get("mean", np.nan), paired_dz=pr.get("dz", np.nan),
            paired_lo=pr.get("lo", np.nan), paired_hi=pr.get("hi", np.nan),
            paired_p=pr.get("p", np.nan), frac_pos=pr.get("frac_pos", np.nan),
            pooled_real=po.get("mean_a", np.nan), pooled_fp=po.get("mean_b", np.nan),
            pooled_d=po.get("d", np.nan), pooled_auc=po.get("auc", np.nan),
            pooled_p=po.get("p", np.nan), var_ratio=po.get("var_ratio", np.nan),
            fp_vs_control_d=cc.get("d", np.nan),
            fp_vs_control_p=cc.get("p", np.nan),
            control_mean=cc.get("mean_b", np.nan),
            outside_cnn_window=outside,
            sign_agrees=bool(np.sign(po.get("d", np.nan)) ==
                             np.sign(pr.get("dz", np.nan)))
            if np.isfinite(po.get("d", np.nan)) and
               np.isfinite(pr.get("dz", np.nan)) else False))
    R = pd.DataFrame(res)
    R["paired_q"] = bh_fdr(R.paired_p)
    R["pooled_q"] = bh_fdr(R.pooled_p)
    R["control_q"] = bh_fdr(R.fp_vs_control_d * 0 + R.fp_vs_control_p)
    R = R.sort_values("paired_dz", key=abs, ascending=False)
    W("all_features.tsv", R)

    print("\n[top]  strongest paired effects, real vs false positive",
          file=sys.stderr)
    print(f"       {'feature':<28}{'dz':>8}{'pooled d':>10}{'ctrl d':>9}"
          f"{'q':>10}  out", file=sys.stderr)
    for r_ in R.head(30).itertuples():
        print(f"       {r_.feature:<28}{r_.paired_dz:>+8.3f}"
              f"{r_.pooled_d:>+10.3f}{r_.fp_vs_control_d:>+9.3f}"
              f"{r_.paired_q:>10.2e}  {'y' if r_.outside_cnn_window else 'n'}",
              file=sys.stderr)

    sig = R[(R.paired_q < 0.05) & (R.paired_dz.abs() > 0.2)]
    print(f"\n[top]  {len(sig)} features with q < 0.05 and |dz| > 0.2; "
          f"{int(sig.outside_cnn_window.sum())} of them outside the CNN window",
          file=sys.stderr)

    if a.n_perm > 0:
        top_feats = R.head(60).feature.tolist()
        obs_max = float(R.paired_dz.abs().max())
        nulls = []
        gg = RF.groupby("gene")
        for it in range(a.n_perm):
            if it and it % 50 == 0:
                print(f"[perm] {it}/{a.n_perm}", file=sys.stderr)
            reals, fps = [], []
            for g, sub in gg:
                if len(sub) < 2:
                    continue
                idx = rng.permutation(len(sub))
                reals.append(sub.iloc[idx[0]][top_feats])
                fps.append(sub.iloc[idx[1:]][top_feats].mean())
            if not reals:
                break
            Rp = pd.DataFrame(reals).astype(float).reset_index(drop=True)
            Fp = pd.DataFrame(fps).astype(float).reset_index(drop=True)
            Dp = Rp - Fp
            dz = Dp.mean() / Dp.std(ddof=1)
            nulls.append(float(np.nanmax(np.abs(dz))))
        if nulls:
            nulls = np.array(nulls)
            pfw = (1 + int((nulls >= obs_max).sum())) / (len(nulls) + 1)
            print(f"\n[perm] largest observed |dz| {obs_max:.3f}; permutation "
                  f"null median {np.median(nulls):.3f}, 95th "
                  f"{np.percentile(nulls, 95):.3f}, family-wise p {pfw:.3g}",
                  file=sys.stderr)
            W("permutation_null.tsv",
              pd.DataFrame(dict(max_abs_dz=nulls)))

    krows = []
    for k_ in set(kmer_sites["real"]) | set(kmer_sites["fp"]):
        nr, nf = kmer_n["real"], kmer_n["fp"]
        nc = max(kmer_n["control"], 1)
        fr = kmer_sites["real"][k_] / max(nr, 1)
        ff = kmer_sites["fp"][k_] / max(nf, 1)
        fc = kmer_sites["control"][k_] / nc
        if kmer_sites["real"][k_] + kmer_sites["fp"][k_] < 100:
            continue
        krows.append(dict(kmer=k_, frac_real=round(fr, 5), frac_fp=round(ff, 5),
                          frac_control=round(fc, 5),
                          log2_real_vs_fp=round(float(np.log2((fr + 1e-4) /
                                                              (ff + 1e-4))), 4),
                          log2_fp_vs_control=round(float(np.log2((ff + 1e-4) /
                                                                 (fc + 1e-4))), 4),
                          n_real=kmer_sites["real"][k_],
                          n_fp=kmer_sites["fp"][k_]))
    K = pd.DataFrame(krows)
    if len(K):
        K = K.sort_values("log2_real_vs_fp", ascending=False)
        W("kmers.tsv", K)
        print(f"\n[kmer] {a.kmer_k}-mers by presence in the core +/-250 nt, "
              f"one vote per site", file=sys.stderr)
        print("       enriched at real 3' ends:", file=sys.stderr)
        for r_ in K.head(8).itertuples():
            print(f"         {r_.kmer}  real {100*r_.frac_real:5.1f}%  fp "
                  f"{100*r_.frac_fp:5.1f}%  log2 {r_.log2_real_vs_fp:+.2f}",
                  file=sys.stderr)
        print("       enriched at false positives:", file=sys.stderr)
        for r_ in K.tail(8).iloc[::-1].itertuples():
            print(f"         {r_.kmer}  real {100*r_.frac_real:5.1f}%  fp "
                  f"{100*r_.frac_fp:5.1f}%  log2 {r_.log2_real_vs_fp:+.2f}",
                  file=sys.stderr)

    if rm is not None and len(CT):
        rr = []
        for fam in REP_FAMS:
            for sense, lab in ((1.0, "sense"), (0.0, "antisense")):
                x = ((FP.rep_family == fam) & (FP.rep_sense == sense)).astype(float)
                y = ((CT.rep_family == fam) & (CT.rep_sense == sense)).astype(float)
                z = ((RF[RF.cls == "real"].rep_family == fam) &
                     (RF[RF.cls == "real"].rep_sense == sense)).astype(float)
                if x.sum() + y.sum() < 40:
                    continue
                rr.append(dict(family=fam, orientation=lab,
                               real_rate=round(float(z.mean()), 5),
                               fp_rate=round(float(x.mean()), 5),
                               control_rate=round(float(y.mean()), 5),
                               fp_over_control=round(float((x.mean() + 1e-6) /
                                                           (y.mean() + 1e-6)), 3),
                               n_fp=int(x.sum()), n_control=int(y.sum())))
        RRep = pd.DataFrame(rr).sort_values("fp_over_control", ascending=False)
        W("repeats.tsv", RRep)
        print("\n[rep]  false positives vs matched intronic controls, by family "
              "and orientation", file=sys.stderr)
        print(RRep.head(14).to_string(index=False), file=sys.stderr)

    top = R[(R.paired_q < 0.05)].head(40).feature.tolist()
    if len(top) > 2:
        C = DIFF[top].corr()
        W("feature_correlation.tsv", C.reset_index().rename(
            columns={"index": "feature"}))
        kept = []
        for f in top:
            if all(abs(C.loc[f, k]) < 0.8 for k in kept):
                kept.append(f)
        print(f"\n[red]  {len(top)} significant features collapse to "
              f"{len(kept)} independent axes at |r| < 0.8:", file=sys.stderr)
        for f in kept[:15]:
            dz = float(R.loc[R.feature == f, "paired_dz"].iloc[0])
            print(f"         {f:<30} dz {dz:+.3f}", file=sys.stderr)
        W("independent_features.tsv",
          R[R.feature.isin(kept)][["feature", "paired_dz", "pooled_d",
                                   "fp_vs_control_d", "paired_q",
                                   "outside_cnn_window"]])

    axes = [f for f in (kept if len(top) > 2 else top)][:8]
    if len(axes) >= 3:
        rb = RF[RF.cls == "real"].drop_duplicates("gene").set_index("gene")
        fpk = FP[FP.gene.isin(rb.index)].copy()
        Dv = (fpk[axes].astype(float).to_numpy() -
              rb.loc[fpk.gene, axes].astype(float).to_numpy())
        D = pd.DataFrame(Dv, columns=axes)
        D.insert(0, "pos", fpk.pos.to_numpy())
        D.insert(0, "gene", fpk.gene.to_numpy())
        Z = D[axes].apply(lambda s: (s - s.mean()) / (s.std(ddof=1) or 1))
        Z = Z.fillna(0.0)
        W("fp_deviations.tsv", D)

        X = Z.to_numpy(float)
        k = int(a.n_clusters)
        cen = X[rng.choice(len(X), 1)]
        for _ in range(k - 1):
            d2 = ((X[:, None, :] - cen[None, :, :]) ** 2).sum(-1).min(1)
            probs = d2 / max(d2.sum(), 1e-12)
            cen = np.vstack([cen, X[rng.choice(len(X), p=probs)]])
        for _ in range(60):
            lab = ((X[:, None, :] - cen[None, :, :]) ** 2).sum(-1).argmin(1)
            new = np.vstack([X[lab == j].mean(0) if (lab == j).any() else cen[j]
                             for j in range(k)])
            if np.allclose(new, cen):
                break
            cen = new
        D["cluster"] = lab
        W("fp_clusters.tsv", D[["gene", "pos", "cluster"]])
        prof = pd.DataFrame(cen, columns=axes)
        prof.insert(0, "n", [int((lab == j).sum()) for j in range(k)])
        prof.insert(0, "cluster", range(k))
        W("fp_cluster_profiles.tsv", prof)
        print(f"\n[clu]  {k} clusters of false positives, in units of SD from "
              f"their own gene's real end", file=sys.stderr)
        print(prof.round(2).to_string(index=False), file=sys.stderr)

        ext = np.zeros(len(Z), int)
        for f in axes:
            v = Z[f].to_numpy()
            ext += (np.abs(v) >= np.quantile(np.abs(v), 0.90)).astype(int)
        print(f"\n[clu]  false positives extreme on k of {len(axes)} axes:",
              file=sys.stderr)
        for j in range(min(len(axes), 5) + 1):
            print(f"         k={j}: {int((ext == j).sum()):>7,} "
                  f"({100*(ext == j).mean():5.1f}%)", file=sys.stderr)

        D["unexplained"] = (ext == 0)
        un = D[D.unexplained]
        print(f"\n[unx]  {len(un):,} false positives ({100*len(un)/len(D):.1f}%) "
              f"are extreme on none of the {len(axes)} axes.", file=sys.stderr)
        mask = D.unexplained.to_numpy()
        UNF = fpk[mask]
        EXPF = fpk[~mask]
        rows2 = []
        for c in FEATS:
            r_ = RF.loc[RF.cls == "real", c]
            u_ = UNF[c]
            o_ = EXPF[c]
            st = two_sample(r_, u_)
            st2 = two_sample(o_, u_)
            rows2.append(dict(feature=c,
                              d_real_vs_unexplained=st.get("d", np.nan),
                              p_real_vs_unexplained=st.get("p", np.nan),
                              d_explained_vs_unexplained=st2.get("d", np.nan),
                              mean_unexplained=st.get("mean_b", np.nan),
                              mean_real=st.get("mean_a", np.nan)))
        U = pd.DataFrame(rows2)
        U["q"] = bh_fdr(U.p_real_vs_unexplained)
        U = U.sort_values("d_real_vs_unexplained", key=abs, ascending=False)
        W("unexplained_profile.tsv", U)
        print("       how the unexplained ones differ from real 3' ends:",
              file=sys.stderr)
        for r_ in U.head(12).itertuples():
            print(f"         {r_.feature:<30} d {r_.d_real_vs_unexplained:+.3f}"
                  f"  q {r_.q:.2e}", file=sys.stderr)
        nsm = int((U.d_real_vs_unexplained.abs() < 0.1).sum())
        print(f"       {nsm} of {len(U)} features differ by |d| < 0.1 between "
              f"real 3' ends and the\n       unexplained false positives.",
              file=sys.stderr)

    with open(os.path.join(a.out, "counts.json"), "w") as fh:
        json.dump(dict(n_real=int((P.cls == "real").sum()),
                       n_fp=int((P.cls == "fp").sum()),
                       n_control=int((P.cls == "control").sum()),
                       n_genes=int(P.gene.nunique()),
                       n_features=len(FEATS), up=a.up, dn=a.dn,
                       splice=cfg["splice"], kmer_k=a.kmer_k,
                       n_perm=a.n_perm), fh, indent=2)

    print(f"\n[out]  {a.out}/", file=sys.stderr)
    print("\nRead all_features.tsv sorted by paired_dz, filtered to "
          "outside_cnn_window and paired_q < 0.05. Then independent_features."
          "tsv, which collapses the correlated ones. Then the [unx] block: if "
          "the unexplained half matches real 3' ends on nearly everything, the "
          "distinguishing information is not in the sequence around these "
          "positions.", file=sys.stderr)


if __name__ == "__main__":
    main() 