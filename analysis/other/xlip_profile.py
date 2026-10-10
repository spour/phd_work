#!/usr/bin/env python3
# is u1/cpaf machinery bound at our computationally predicted sites?

import argparse
import os
import sys

import numpy as np
import pandas as pd

try:
    import pyBigWig
except ImportError:
    sys.exit("pyBigWig needed")

FACTORS = [
    ("U1A",     "GSM3989602_U1A_XLIP_cAMO",     "U1 snRNP"),
    ("U1C",     "GSM3989603_U1C_XLIP_cAMO",     "U1 snRNP"),
    ("CFIm25",  "GSM3989606_CFIm25_XLIP_cAMO",  "CPA"),
    ("Fip1",    "GSM3989608_Fip1_XLIP_cAMO",    "CPA"),
    ("CstF64",  "GSM3989610_CstF64_XLIP_cAMO",  "CPA"),
    ("SF3B1",   "GSM3989604_SF3B1_XLIP_cAMO",   "negative (spliceosome)"),
    ("hnRNPC",  "GSM3989612_hnRNPC_XLIP_cAMO",  "negative (hnRNP)"),
]
INPUT_STEM = "GSM3989614_Bio-cAMO_Input"

STRAND_PATS = [(".fwd.", ".rev."), ("_Fwd.", "_Rev."), ("_fwd.", "_rev."),
               ("_forward", "_reverse"), ("_F.", "_R."), ("_plus", "_minus"),
               ("_pos.", "_neg."), (".plus.", ".minus."), ("Plus", "Minus")]

SHUF = {"competitors": "competitors shuffled",
        "canonical": "canonical shuffled"}


def open_pair(bw_dir, stem):
    cands = [f for f in os.listdir(bw_dir) if f.startswith(stem)]
    if not cands:
        return None
    fwd = None
    for f in sorted(cands):
        for a, _ in STRAND_PATS:
            if a in f:
                fwd = os.path.join(bw_dir, f)
                break
        if fwd:
            break
    if fwd is None:
        h = pyBigWig.open(os.path.join(bw_dir, cands[0]))
        print("     unstranded: " + cands[0])
        return {"+": h, "-": h, ".": h}
    rev = None
    for a, b in STRAND_PATS:
        if a in os.path.basename(fwd):
            c = fwd.replace(a, b)
            if os.path.exists(c):
                rev = c
            break
    if rev is None:
        print("     WARNING: " + os.path.basename(fwd) + " looks stranded but "
              "no mate found; minus-strand sites will read the plus file")
    hf = pyBigWig.open(fwd)
    hr = pyBigWig.open(rev) if rev else hf
    return {"+": hf, "-": hr, ".": hf}


def profile(handles, sites, half, nbins):
    acc = np.zeros(nbins)
    n = 0
    for c, p, s in zip(sites["chrom"], sites["pos"], sites["strand"]):
        bw = handles.get(s if s in "+-" else ".")
        ch = bw.chroms()
        if c not in ch:
            continue
        a, b = int(p) - half, int(p) + half
        if a < 0 or b > ch[c]:
            continue
        try:
            v = bw.stats(c, a, b, nBins=nbins, type="mean")
        except RuntimeError:
            continue
        v = np.array([0.0 if x is None else abs(x) for x in v])
        if s == "-":
            v = v[::-1]
        acc += v
        n += 1
    return (acc / n if n else acc), n


def centre_signal(handles, sites, half):
    out = np.zeros(len(sites))
    for i, (c, p, s) in enumerate(zip(sites["chrom"], sites["pos"],
                                      sites["strand"])):
        bw = handles.get(s if s in "+-" else ".")
        ch = bw.chroms()
        if c not in ch:
            continue
        a, b = max(0, int(p) - half), min(ch[c], int(p) + half)
        if b <= a:
            continue
        try:
            v = bw.values(c, a, b, numpy=True)
        except RuntimeError:
            continue
        v = np.abs(v[np.isfinite(v)])
        out[i] = float(v.max()) if v.size else 0.0
    return out


def per_site_ratio(sig, inp):
    if inp is None:
        return None
    pos = inp[inp > 0]
    floor = np.percentile(pos, 10) if pos.size else 1e-9
    return sig / np.maximum(inp, floor)


def shuffle_in_own_gene(sites, spans, rng):
    out = sites.copy().reset_index(drop=True)
    key = dict(zip(spans.gene, zip(spans.gstart, spans.gend)))
    newp = out.pos.to_numpy().copy()
    n_moved = 0
    for i, g in enumerate(out.gene.to_numpy()):
        if g in key:
            a, b = key[g]
            if b > a:
                newp[i] = rng.integers(a, b)
                n_moved += 1
    out["pos"] = newp
    if n_moved < len(out):
        print("     note: " + f"{len(out) - n_moved:,}" + " sites had no gene "
              "span and were not moved")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bw-dir", required=True)
    ap.add_argument("--peaks", required=True)
    ap.add_argument("--half", type=int, default=1000, help="profile half-width")
    ap.add_argument("--nbins", type=int, default=100)
    ap.add_argument("--centre-half", type=int, default=100,
                    help="half-width for the per-site enrichment number")
    ap.add_argument("--max-sites", type=int, default=20000)
    ap.add_argument("--gene-pad", type=int, default=5000)
    ap.add_argument("--out", default="xlip_out")
    ap.add_argument("--use-input", action="store_true",
                    help="compute folds on signal/input. OFF by default: the "
                         "GSE135140 Bio-cAMO input carries 3'-end structure "
                         "itself (~2.7x higher at canonical sites than at "
                         "their within-gene shuffle), so dividing by it "
                         "removes the signal being measured and inverts the "
                         "positive control. Both columns are written either "
                         "way.")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(0)

    pk = (pd.read_parquet(a.peaks) if a.peaks.endswith(".parquet")
          else pd.read_csv(a.peaks, sep="\t", low_memory=False))
    if "in_analysis_set" in pk.columns:
        pk = pk[pk.in_analysis_set.astype(bool)]
    pos = "center" if "center" in pk.columns else "pos"
    base = pd.DataFrame({"chrom": pk.chrom.astype(str),
                         "pos": pk[pos].astype(int),
                         "strand": pk.strand.astype(str),
                         "gene": pk.gene.astype(str),
                         "is_canonical": pk.is_canonical.astype(bool)})
    spans = (base.groupby("gene", as_index=False)
             .agg(gstart=("pos", "min"), gend=("pos", "max")))
    spans["gstart"] = (spans.gstart - a.gene_pad).clip(lower=0)
    spans["gend"] = spans.gend + a.gene_pad
    w = spans.gend - spans.gstart
    print("gene spans from peak extents, padded " + f"{a.gene_pad:,}" + " nt: "
          + f"{len(spans):,}" + " genes, median width "
          + f"{np.median(w):,.0f}" + " nt")
    print("  LOWER bound on gene bodies: peak extents do not reach the TSS.")

    comp = base[~base.is_canonical].reset_index(drop=True)
    canon = base[base.is_canonical].reset_index(drop=True)
    if len(comp) > a.max_sites:
        comp = comp.sample(a.max_sites, random_state=0).reset_index(drop=True)

    groups = {
        "competitors": comp,
        "competitors shuffled": shuffle_in_own_gene(comp, spans, rng),
        "canonical": canon,
        "canonical shuffled": shuffle_in_own_gene(canon, spans, rng),
    }
    print()
    for k, v in groups.items():
        print("  " + k.ljust(24) + f"{len(v):>8,}" + " sites")

    inp = open_pair(a.bw_dir, INPUT_STEM)
    if inp is None:
        print("\nno input track (" + INPUT_STEM + "); folds use raw signal")
    elif a.use_input:
        print("\ninput track loaded; --use-input given, folds use signal/input")
    else:
        print("\ninput track loaded but NOT used for folds (see --use-input); "
              "it is still measured and written to the TSV")

    prof_rows, enr_rows = [], []
    for label, stem, role in FACTORS:
        h = open_pair(a.bw_dir, stem)
        if h is None:
            print("  missing " + stem + ", skipping " + label)
            continue
        print("\n== " + label + "  (" + role + ")")
        for gname, g in groups.items():
            pr, n = profile(h, g, a.half, a.nbins)
            if inp is not None:
                pi, _ = profile(inp, g, a.half, a.nbins)
                pos_i = pi[pi > 0]
                floor = np.percentile(pos_i, 10) if pos_i.size else 1e-9
                pr_n = pr / np.maximum(pi, floor)
            else:
                pr_n = pr
            x = np.linspace(-a.half, a.half, a.nbins)
            prof_rows.append(pd.DataFrame(dict(
                factor=label, role=role, group=gname, x=x,
                signal=pr, signal_over_input=pr_n, n=n)))

            cs = centre_signal(h, g, a.centre_half)
            ci = centre_signal(inp, g, a.centre_half) if inp is not None else None
            cn = per_site_ratio(cs, ci)
            enr_rows.append(dict(
                factor=label, role=role, group=gname, n=len(g),
                mean_signal=float(np.mean(cs)),
                median_signal=float(np.median(cs)),
                frac_nonzero=float((cs > 0).mean()),
                mean_input=float(np.mean(ci)) if ci is not None else np.nan,
                mean_over_input=(float(np.mean(cs)) / float(np.mean(ci))
                                 if ci is not None and np.mean(ci) > 0
                                 else np.nan),
                median_over_input=(float(np.median(cn))
                                   if cn is not None else np.nan)))
            print("   " + gname.ljust(24)
                  + " raw " + f"{np.mean(cs):7.3f}"
                  + ("   input " + f"{np.mean(ci):7.3f}"
                     if ci is not None else "")
                  + "   nonzero " + f"{100 * (cs > 0).mean():5.1f}" + "%")

    P = pd.concat(prof_rows, ignore_index=True)
    E = pd.DataFrame(enr_rows)

    # fold: each group against its own shuffle
    use_norm = bool(a.use_input) and E.mean_over_input.notna().any()
    col = "mean_over_input" if use_norm else "mean_signal"
    print("\nfolds computed on "
          + ("signal/input" if use_norm else
             "RAW signal vs the within-gene shuffle"))

    if E.mean_input.notna().any():
        mi = E.groupby("group").mean_input.mean()
        for g, sh in SHUF.items():
            if g in mi and sh in mi and mi[sh] > 0:
                print("  input at " + g + " is " + f"{mi[g] / mi[sh]:.2f}"
                      + "x its shuffle" +
                      ("   <- not flat; it tracks position, so dividing by it "
                       "removes signal" if mi[g] / mi[sh] > 1.5 else ""))

    den = {(r.factor, r.group): getattr(r, col) for r in E.itertuples()}
    folds, denoms = [], []
    for r in E.itertuples():
        sh = SHUF.get(r.group)
        d = den.get((r.factor, sh)) if sh else np.nan
        folds.append(getattr(r, col) / d if (sh and d and d > 0) else np.nan)
        denoms.append(sh if sh else "")
    E["shuffle_used"] = denoms
    E["fold_over_own_shuffle"] = folds
    E["fold_basis"] = col
    E["input_normalised"] = use_norm

    P.to_csv(a.out + "/xlip_profiles.tsv", sep="\t", index=False)
    E.to_csv(a.out + "/xlip_enrichment.tsv", sep="\t", index=False)

    for gname in ("canonical", "competitors"):
        sub = E[E.group == gname].sort_values("fold_over_own_shuffle",
                                              ascending=False)
        if not len(sub):
            continue
        head = ("positive control: real 3' ends" if gname == "canonical"
                else "the question: model false positives")
        print("\nfold over " + SHUF[gname] + "  --  " + head)
        for r in sub.itertuples():
            flag = ""
            if np.isfinite(r.fold_over_own_shuffle):
                if r.role.startswith("negative") and r.fold_over_own_shuffle > 1.3:
                    flag = "   <- negative is enriched"
                elif r.frac_nonzero < 0.15:
                    flag = "   <- sparse, mostly zeros"
            print("  " + r.factor.ljust(9) + r.role.ljust(24)
                  + f"{r.fold_over_own_shuffle:6.2f}" + "x"
                  + "   nonzero " + f"{100 * r.frac_nonzero:5.1f}" + "%" + flag)

    print("\nRead the canonical block first. If the CPA subunits are not above "
          "1 there, the coordinates or strand are wrong and the competitor "
          "block means nothing. Then read the negatives: SF3B1 and hnRNPC "
          "should not be enriched. If every factor is enriched equally the "
          "signal is accessibility rather than binding.")
    print("Rows flagged sparse have most sites at zero, so the mean is an "
          "average over mostly-empty windows: read those as 'no detectable "
          "occupancy at this depth', not as a measured absence. U1 at gene 3' "
          "ends is expected to be low -- U1 is a 5' splice site factor -- so a "
          "low canonical U1A/U1C fold is biology, not a failure.")
    print("\nwrote " + a.out + "/xlip_profiles.tsv and xlip_enrichment.tsv")


if __name__ == "__main__":
    main()