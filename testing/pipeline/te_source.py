#!/usr/bin/env python3
# are transposable elements a disproportionate source of cryptic poly(a) architecture?
import argparse, os, sys, json, gzip, re
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

RC = str.maketrans("ACGTN", "TGCAN")
PAS = ["AATAAA", "ATTAAA"]
DSE = ["TTTT", "TGTG"]

NOT_TE = {"Simple_repeat", "Low_complexity", "Satellite", "rRNA", "scRNA",
          "snRNA", "srpRNA", "tRNA", "RNA", "Unknown", "Other"}


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


def load_rmsk(path, chroms):
    op = gzip.open if path.endswith(".gz") else open
    by = defaultdict(list)
    n = 0
    with op(path, "rt") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 13 or f[5] not in chroms:
                continue
            by[f[5]].append((int(f[6]), int(f[7]), f[9], f[11], f[12]))
            n += 1
    out = {}
    for c, v in by.items():
        v.sort(key=lambda x: x[0])
        out[c] = dict(start=np.array([x[0] for x in v]),
                      end=np.array([x[1] for x in v]),
                      strand=np.array([x[2] for x in v]),
                      cls=np.array([x[3] for x in v]),
                      fam=np.array([x[4] for x in v]))
    print(f"[rmsk] {n:,} intervals over {len(out)} chromosomes", file=sys.stderr)
    return out


def find_all(seq, pats):
    out = []
    for p in pats:
        i = seq.find(p)
        while i != -1:
            out.append(i)
            i = seq.find(p, i + 1)
    return out


def expected_hexamer_rate(counts):
    tot = sum(counts[b] for b in "ACGT")
    if tot < 100:
        return np.nan
    f = {b: counts[b] / tot for b in "ACGT"}
    p1 = f["A"]**5 * f["T"]
    p2 = f["A"]**4 * f["T"]**2
    return p1 + p2


def expected_dse_rate(counts):
    tot = sum(counts[b] for b in "ACGT")
    if tot < 100:
        return np.nan
    f = {b: counts[b] / tot for b in "ACGT"}
    return f["T"]**4 + f["T"]**2 * f["G"]**2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peaks", required=True)
    ap.add_argument("--rmsk", required=True)
    ap.add_argument("--genome", required=True)
    ap.add_argument("--pad", type=int, default=5000,
                    help="pad each gene's peak extent by this much on each side")
    ap.add_argument("--dse_lo", type=int, default=15,
                    help="downstream element search starts this far past the "
                         "hexamer 3' end (10 nt minimum spacing + 5 nt)")
    ap.add_argument("--dse_hi", type=int, default=70,
                    help="and ends here (30 nt maximum spacing + 40 nt)")
    ap.add_argument("--min_seg", type=int, default=100,
                    help="segments shorter than this are pooled as 'fragment'; "
                         "composition cannot be estimated from a few dozen bases")
    ap.add_argument("--max_genes", type=int, default=0,
                    help="cap for a quick run; 0 uses all")
    ap.add_argument("--out", default="tesrc")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    W = lambda n, df: df.to_csv(os.path.join(a.out, n), sep="\t", index=False)

    pk = pd.read_parquet(a.peaks)
    if "in_analysis_set" in pk.columns:
        pk = pk[pk.in_analysis_set.astype(bool) | pk.is_canonical.astype(bool)]
    POS = "center" if "center" in pk.columns else "pos"
    d = pd.DataFrame({"gene": pk.gene.astype(str), "chrom": pk.chrom.astype(str),
                      "pos": pk[POS].astype(int), "strand": pk.strand.astype(str),
                      "is_real": pk.is_canonical.astype(bool)})

    span = (d.groupby(["gene", "chrom", "strand"], as_index=False)
              .agg(lo=("pos", "min"), hi=("pos", "max")))
    span["lo"] = (span.lo - a.pad).clip(lower=0)
    span["hi"] = span.hi + a.pad
    if a.max_genes:
        span = span.head(a.max_genes)
    print(f"[set]  {len(span):,} genes, "
          f"{(span.hi - span.lo).sum()/1e6:,.0f} Mb of span", file=sys.stderr)

    fp_by_chrom = defaultdict(set)
    for r in d[~d.is_real].itertuples():
        fp_by_chrom[r.chrom].add(r.pos)

    rm = load_rmsk(a.rmsk, set(span.chrom.unique()))
    gen = Genome(a.genome)

    bases = defaultdict(Counter)
    npos = Counter()
    nhex = Counter()
    narch = Counter()
    nfp = Counter()
    nelem = Counter()

    for gi, r in enumerate(span.itertuples()):
        if gi and gi % 250 == 0:
            print(f"[run]  {gi:,}/{len(span):,} genes...", file=sys.stderr)
        L = r.hi - r.lo
        if L < 1000:
            continue
        gseq = gen.fetch(r.chrom, r.lo, r.hi)
        if len(gseq) != L:
            continue

        lab = np.zeros(L, dtype=np.int32)
        segs = [None]
        e = rm.get(r.chrom)
        if e is not None and len(e["start"]):
            i0 = np.searchsorted(e["end"], r.lo, side="left")
            i1 = np.searchsorted(e["start"], r.hi, side="right")
            for k in range(i0, i1):
                if e["cls"][k] in NOT_TE:
                    continue
                s0 = max(r.lo, int(e["start"][k])) - r.lo
                s1 = min(r.hi, int(e["end"][k])) - r.lo
                if s1 <= s0:
                    continue
                sense = "sense" if e["strand"][k] == r.strand else "antisense"
                segs.append((str(e["fam"][k]), sense))
                lab[s0:s1] = len(segs) - 1
                nelem[(str(e["fam"][k]), sense)] += 1

        if r.strand == "-":
            gseq = gseq.translate(RC)[::-1]
            lab = lab[::-1]

        arr = np.frombuffer(gseq.encode(), dtype=np.uint8)
        for key_idx in np.unique(lab):
            m = lab == key_idx
            if m.sum() < a.min_seg:
                continue
            key = ("non-TE", "n/a") if key_idx == 0 else segs[key_idx]
            sub = arr[m]
            for b in "ACGT":
                bases[key][b] += int((sub == ord(b)).sum())
            npos[key] += int(m.sum())

        for i in find_all(gseq, PAS):
            key_idx = lab[i + 3] if i + 3 < L else 0
            key = ("non-TE", "n/a") if key_idx == 0 else segs[key_idx]
            nhex[key] += 1
            w0, w1 = i + 6 + a.dse_lo, min(L, i + 6 + a.dse_hi)
            if w1 > w0 and any(p in gseq[w0:w1] for p in DSE):
                narch[key] += 1
                for dd in range(10, 31):
                    off = i + 6 + dd
                    gpos = (r.lo + off) if r.strand != "-" else (r.hi - off - 1)
                    if gpos in fp_by_chrom[r.chrom]:
                        nfp[key] += 1
                        break

    # assemble
    rows = []
    for key in npos:
        fam, orient = key
        n = npos[key]
        if n < 10000:
            continue
        eh = expected_hexamer_rate(bases[key])
        ed = expected_dse_rate(bases[key])
        obs_h = nhex[key]
        exp_h = eh * n if np.isfinite(eh) else np.nan
        gc = ((bases[key]["G"] + bases[key]["C"]) /
              max(sum(bases[key][b] for b in "ACGT"), 1))
        rows.append(dict(
            family=fam, orientation=orient,
            kb=round(n / 1000, 1), gc=round(gc, 4),
            n_copies=nelem.get(key, 0),
            hexamers=obs_h,
            hex_per_kb=round(1000 * obs_h / n, 4),
            hex_expected=round(exp_h, 1) if np.isfinite(exp_h) else np.nan,
            hex_obs_over_exp=round(obs_h / exp_h, 3) if (np.isfinite(exp_h)
                                                        and exp_h > 0) else np.nan,
            architectures=narch[key],
            arch_per_kb=round(1000 * narch[key] / n, 4),
            arch_frac_of_hex=round(narch[key] / max(obs_h, 1), 3),
            n_false_positive=nfp[key],
            fp_per_architecture=round(nfp[key] / max(narch[key], 1), 4)))
    R = pd.DataFrame(rows).sort_values("hex_obs_over_exp", ascending=False)
    W("te_source.tsv", R)

    ref = R[R.family == "non-TE"]
    ref_oe = float(ref.hex_obs_over_exp.iloc[0]) if len(ref) else np.nan
    ref_arch = float(ref.arch_per_kb.iloc[0]) if len(ref) else np.nan
    ref_conv = float(ref.fp_per_architecture.iloc[0]) if len(ref) else np.nan

    print("\n[ref]  non-TE intronic sequence:", file=sys.stderr)
    if len(ref):
        print(f"       {ref.kb.iloc[0]:,.0f} kb, GC {ref.gc.iloc[0]:.3f}, "
              f"{ref.hex_per_kb.iloc[0]:.3f} hexamers/kb, "
              f"observed/expected {ref_oe:.3f}", file=sys.stderr)
        print(f"       {ref.arch_per_kb.iloc[0]:.3f} architectures/kb, "
              f"{100*ref_conv:.2f}% of them a model false positive",
              file=sys.stderr)

    print("\n[src]  families by hexamer observed/expected, composition-corrected",
          file=sys.stderr)
    print(f"       {'family':<16}{'orient':>10}{'kb':>10}{'GC':>7}"
          f"{'hex/kb':>9}{'O/E':>7}{'arch/kb':>9}{'fp/arch':>9}", file=sys.stderr)
    for r_ in R.head(24).itertuples():
        flag = ""
        if np.isfinite(ref_oe) and r_.family != "non-TE":
            if r_.hex_obs_over_exp > ref_oe * 1.2:
                flag = "  <-"
        print(f"       {r_.family:<16}{r_.orientation:>10}{r_.kb:>10,.0f}"
              f"{r_.gc:>7.3f}{r_.hex_per_kb:>9.3f}{r_.hex_obs_over_exp:>7.2f}"
              f"{r_.arch_per_kb:>9.3f}{r_.fp_per_architecture:>9.4f}{flag}",
              file=sys.stderr)

    # orientation asymmetry
    asym = []
    for fam in sorted(set(R.family) - {"non-TE"}):
        s = R[R.family == fam].set_index("orientation")
        if "sense" not in s.index or "antisense" not in s.index:
            continue
        asym.append(dict(
            family=fam,
            sense_hex_per_kb=float(s.loc["sense", "hex_per_kb"]),
            antisense_hex_per_kb=float(s.loc["antisense", "hex_per_kb"]),
            hex_ratio=round(float(s.loc["sense", "hex_per_kb"] /
                                  max(s.loc["antisense", "hex_per_kb"], 1e-9)), 3),
            sense_arch_per_kb=float(s.loc["sense", "arch_per_kb"]),
            antisense_arch_per_kb=float(s.loc["antisense", "arch_per_kb"]),
            arch_ratio=round(float(s.loc["sense", "arch_per_kb"] /
                                   max(s.loc["antisense", "arch_per_kb"], 1e-9)), 3),
            sense_fp_per_arch=float(s.loc["sense", "fp_per_architecture"]),
            antisense_fp_per_arch=float(s.loc["antisense", "fp_per_architecture"])))
    A = pd.DataFrame(asym).sort_values("arch_ratio", ascending=False)
    W("orientation_asymmetry.tsv", A)
    print("\n[asym] sense over antisense, within family", file=sys.stderr)
    print("       composition is nearly strand-symmetric, so a ratio far from "
          "1.0 is\n       not explained by base content", file=sys.stderr)
    print(A.head(14).to_string(index=False), file=sys.stderr)

    # the summary claim
    te = R[R.family != "non-TE"]
    if len(te) and np.isfinite(ref_arch):
        tot_kb = te.kb.sum()
        tot_arch = te.architectures.sum()
        te_rate = 1000 * tot_arch / (tot_kb * 1000)
        print(f"\n[sum]  all transposable elements pooled: "
              f"{tot_kb:,.0f} kb, {tot_arch:,} architectures, "
              f"{te_rate:.3f}/kb", file=sys.stderr)
        print(f"       non-TE intron: {ref_arch:.3f}/kb", file=sys.stderr)
        print(f"       ratio {te_rate/ref_arch:.2f}x", file=sys.stderr)
        share = tot_arch / (tot_arch + int(ref.architectures.iloc[0]))
        share_kb = tot_kb / (tot_kb + float(ref.kb.iloc[0]))
        print(f"       TEs hold {100*share_kb:.1f}% of the intronic sequence "
              f"scanned and carry {100*share:.1f}% of the architectures",
              file=sys.stderr)

    with open(os.path.join(a.out, "counts.json"), "w") as fh:
        json.dump(dict(n_genes=len(span),
                       total_kb=float(R.kb.sum()),
                       total_hexamers=int(R.hexamers.sum()),
                       total_architectures=int(R.architectures.sum()),
                       non_te_oe=ref_oe, non_te_arch_per_kb=ref_arch,
                       dse_window=[a.dse_lo, a.dse_hi],
                       excluded_classes=sorted(NOT_TE)), fh, indent=2)
    print(f"\n[out]  {a.out}/", file=sys.stderr)
    print("\nRead the O/E column against the non-TE row. A family at the same "
          "ratio as\nnon-TE intron produces architectures at the rate its base "
          "composition predicts\nand is not a source; one well above it is. "
          "Then [asym]: a source of USABLE\nsignals should be asymmetric "
          "between orientations, since composition is not.",
          file=sys.stderr)


if __name__ == "__main__":
    main()