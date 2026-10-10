#!/usr/bin/env python3
# (v3) turn the cpa-featurized shortlist plus the raw shortlist into the tsvs
import argparse, os, sys
import numpy as np, pandas as pd

MOTIF_COLS = ["pas_hits", "ugua", "dse_urun", "dse_gu"]
KEY = ["gene", "chrom", "center", "strand"]
FP_LAB = "Model false positive"


def auroc(labels, scores):
    m = np.isfinite(scores)
    y = np.asarray(labels)[m].astype(int); s = np.asarray(scores)[m]
    npos = int(y.sum()); nneg = int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return np.nan, npos, nneg
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); start = csum - cnt
    ranks = ((start + csum + 1) / 2.0)[inv]
    aucv = (ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)
    return float(aucv), npos, nneg


def auprc(y, s):
    npos = int(y.sum())
    if npos == 0 or npos == len(y):
        return np.nan, np.nan
    o = np.argsort(-s, kind="mergesort")
    ys = y[o]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / npos
    ap = float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))
    return ap, npos / len(y)


def within_gene_rank(sub, score_col="aparent2_prob"):
    can = sub[sub.is_canonical]
    fps = sub[~sub.is_canonical]
    if len(can) != 1 or len(fps) == 0:
        return None
    cs = float(can[score_col].iloc[0])
    v = pd.to_numeric(fps[score_col], errors="coerce").dropna().to_numpy()
    if not np.isfinite(cs) or v.size == 0:
        return None
    return float(((v < cs).sum() + 0.5 * (v == cs).sum()) / v.size), int(v.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shortlist", required=True)
    ap.add_argument("--cpa", required=True)
    ap.add_argument("--peaks", default=None,
                    help="cpa_peaks_final.parquet; merged on "
                         "gene/chrom/center/strand for in_analysis_set and "
                         "flag_apa_annotated")
    ap.add_argument("--no_filter", action="store_true",
                    help="skip the analysis-set filter")
    ap.add_argument("--cpa_bg_auroc", type=float, default=None,
                    help="Fig-1 CPA-vs-background AUROC, for the reference row")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--boot_pr", type=int, default=200,
                    help="bootstrap reps for AUROC/AUPRC CIs (each is O(n log n))")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    W = lambda n, d: d.to_csv(os.path.join(a.out, n), sep="\t", index=False)
    rng = np.random.default_rng(0)

    sl = pd.read_parquet(a.shortlist)
    cpa = pd.read_parquet(a.cpa)

    # merge the analysis-set flags
    filtered = False
    if a.peaks and os.path.exists(a.peaks) and not a.no_filter:
        pk = pd.read_parquet(a.peaks, columns=KEY + ["in_analysis_set",
                                                     "flag_apa_annotated"])
        cpa = cpa.drop(columns=[c for c in ("in_analysis_set", "flag_apa_annotated")
                                if c in cpa.columns])
        n0 = len(cpa)
        cpa = cpa.merge(pk, on=KEY, how="left")
        nmatch = int(cpa.in_analysis_set.notna().sum())
        print(f"[filter] merged peak flags on {'/'.join(KEY)}: "
              f"{nmatch:,}/{n0:,} rows matched", file=sys.stderr)
        if nmatch < n0:
            print(f"[filter] WARNING {n0-nmatch:,} rows unmatched -> treated as "
                  f"OUTSIDE the analysis set", file=sys.stderr)
        cpa["in_analysis_set"] = cpa.in_analysis_set.fillna(False).astype(bool)
        cpa["flag_apa_annotated"] = cpa.flag_apa_annotated.fillna(False).astype(bool)
        filtered = True
        n_apa = int(((~cpa.is_canonical) & cpa.flag_apa_annotated).sum())
        keep = cpa.in_analysis_set | cpa.is_canonical
        print(f"[filter] {int((~keep).sum()):,} non-canonical peaks outside the "
              f"analysis set dropped here ({n_apa:,} APA-annotated); a count of "
              f"0 means featurize_shortlist_cpa.py already removed them",
              file=sys.stderr)
        cpa_f = cpa[keep].reset_index(drop=True)
    else:
        if not a.no_filter:
            print("[filter] no --peaks given; NOT filtered", file=sys.stderr)
        cpa["in_analysis_set"] = True
        cpa["flag_apa_annotated"] = False
        cpa_f = cpa

    n_fp = int((~cpa_f.is_canonical).sum())
    n_can = int(cpa_f.is_canonical.sum())

    # overview: contest-size distribution
    per_gene = sl.groupby("gene").agg(n_internal=("n_internal", "first")).reset_index()
    bins = [("1", 1, 1), ("2", 2, 2), ("3-4", 3, 4), ("5-8", 5, 8), ("9+", 9, 10**9)]
    ov = pd.DataFrame([dict(bucket=lab,
                            genes=int(((per_gene.n_internal >= lo) &
                                       (per_gene.n_internal <= hi)).sum()))
                       for lab, lo, hi in bins])
    ov["frac"] = ov.genes / max(len(per_gene), 1)
    W("fig2_overview.tsv", ov)
    n_genes = len(per_gene)
    n_contested = int((per_gene.n_internal > 0).sum())

    # inter-model peak dispersion (panel a)
    cc = [c for c in ("rf_center", "lr_center", "cnn_center") if c in sl.columns]
    if len(cc) == 3:
        comp = sl[(sl.n_internal > 0) & (~sl.is_canonical)]
        if filtered:
            ok = cpa_f[~cpa_f.is_canonical][KEY]
            comp = comp.merge(ok, on=KEY, how="inner")
        sp = (comp[cc].max(axis=1) - comp[cc].min(axis=1)).dropna()
        W("fig2_dispersion.tsv", pd.DataFrame({"spread_nt": sp.values.astype(int)}))
        print(f"[disp] n={len(sp):,} median={np.median(sp):.0f}nt  "
              f"75%={np.percentile(sp,75):.0f}nt  <=50nt: {(sp<=50).mean():.3f}  "
              f">60nt: {(sp>60).mean():.3f}", file=sys.stderr)

    # aparent2
    rows = []
    if "aparent2_prob" in cpa_f.columns:
        recs = []
        for g, sub in cpa_f.groupby("gene"):
            r = within_gene_rank(sub)
            if r is not None:
                recs.append(dict(gene=g, pct=r[0], n_fp=r[1]))
        R = pd.DataFrame(recs)
        W("fig2_rank.tsv", R)

        BINS = [(1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 5, "4-5"),
                (6, 10, "6-10"), (11, 10**9, "11+")]
        brows = []
        for lo, hi, lab in BINS:
            s_ = R[(R.n_fp >= lo) & (R.n_fp <= hi)]
            if not len(s_):
                continue
            v = s_.pct.to_numpy()
            bm = [rng.choice(v, len(v), replace=True).mean() for _ in range(a.boot)]
            brows.append(dict(bin=lab, lo=lo, n_genes=len(s_),
                              mean=round(float(v.mean()), 4),
                              ci_lo=round(float(np.percentile(bm, 2.5)), 4),
                              ci_hi=round(float(np.percentile(bm, 97.5)), 4)))
        W("fig2_rank_binned.tsv", pd.DataFrame(brows))
        for b in brows:
            flag = "" if (b["ci_lo"] <= 0.5 <= b["ci_hi"]) else "   <- excludes 0.5"
            print(f"[C] n_fp {b['bin']:<5} genes {b['n_genes']:>5}  "
                  f"mean {b['mean']:.3f} [{b['ci_lo']:.3f}, {b['ci_hi']:.3f}]{flag}",
                  file=sys.stderr)

        v = R.pct.to_numpy()
        bm = [rng.choice(v, len(v), replace=True).mean() for _ in range(a.boot)]
        rows.append(dict(kind="within_gene_rank", subset="analysis_set",
                         value=round(float(v.mean()), 4),
                         lo=round(float(np.percentile(bm, 2.5)), 4),
                         hi=round(float(np.percentile(bm, 97.5)), 4),
                         baseline=0.5, n_genes=len(R)))

        sub = cpa_f.dropna(subset=["aparent2_prob"])
        y = sub.is_canonical.astype(int).to_numpy()
        s = sub.aparent2_prob.to_numpy()
        au, npos, nneg = auroc(y, s)
        apv, base = auprc(y, s)

        bo, bp = [], []
        idx = np.arange(len(y))
        for _ in range(a.boot_pr):
            b = rng.choice(idx, len(idx), replace=True)
            yb, sb = y[b], s[b]
            if yb.sum() == 0 or (yb == 0).sum() == 0:
                continue
            ab, _, _ = auroc(yb, sb)
            if np.isfinite(ab):
                bo.append(ab)
            pb, _ = auprc(yb, sb)
            if np.isfinite(pb):
                bp.append(pb)

        rows.append(dict(kind="pooled_auroc", subset="analysis_set",
                         value=round(au, 4),
                         lo=round(float(np.percentile(bo, 2.5)), 4) if bo else np.nan,
                         hi=round(float(np.percentile(bo, 97.5)), 4) if bo else np.nan,
                         baseline=0.5, n_canon=npos, n_fp=nneg))
        rows.append(dict(kind="pooled_auprc", subset="analysis_set",
                         value=round(apv, 4),
                         lo=round(float(np.percentile(bp, 2.5)), 4) if bp else np.nan,
                         hi=round(float(np.percentile(bp, 97.5)), 4) if bp else np.nan,
                         baseline=round(base, 4), n_canon=npos, n_fp=nneg))
        print(f"[C] pooled AUROC {au:.3f} (chance 0.500), "
              f"AUPRC {apv:.3f} (chance {base:.3f}), "
              f"n={npos:,} canonical vs {nneg:,} false positives", file=sys.stderr)

        o = np.argsort(-s, kind="mergesort"); ys = y[o]
        tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
        tpr = tp / max(tp[-1], 1); fpr = fp / max(fp[-1], 1)
        k = np.linspace(0, len(tpr) - 1, min(400, len(tpr))).astype(int)
        W("fig2_roc.tsv", pd.DataFrame(dict(fpr=fpr[k], tpr=tpr[k])))
        prec = tp / np.maximum(tp + fp, 1)
        rec = tp / max(npos, 1)
        k2 = np.linspace(0, len(rec) - 1, min(400, len(rec))).astype(int)
        W("fig2_pr.tsv", pd.DataFrame(dict(recall=rec[k2], precision=prec[k2],
                                           baseline=base)))

        srows = []
        for lab, m in [("Canonical", cpa_f.is_canonical), (FP_LAB, ~cpa_f.is_canonical)]:
            vv = cpa_f.loc[m, "aparent2_prob"].dropna().to_numpy()
            if len(vv) > 20000:
                vv = rng.choice(vv, 20000, replace=False)
            srows.append(pd.DataFrame(dict(cls=lab, score=vv)))
        pd.concat(srows).to_csv(os.path.join(a.out, "fig2_scores.tsv"),
                                sep="\t", index=False)

    if a.cpa_bg_auroc is not None:
        rows.append(dict(kind="cpa_vs_background_ref", subset="figure1",
                         value=round(a.cpa_bg_auroc, 4)))
    W("fig2_aparent.tsv", pd.DataFrame(rows))

    mrows = []
    for col in MOTIF_COLS:
        for lab, sub2 in [("Canonical", cpa_f[cpa_f.is_canonical]),
                          (FP_LAB, cpa_f[~cpa_f.is_canonical])]:
            vv = pd.to_numeric(sub2[col], errors="coerce").dropna().values
            if len(vv) == 0:
                continue
            bm = [rng.choice(vv, len(vv), replace=True).mean() for _ in range(200)]
            mrows.append(dict(motif=col, cls=lab, mean=float(vv.mean()),
                              lo=float(np.percentile(bm, 2.5)),
                              hi=float(np.percentile(bm, 97.5)), n=len(vv)))
    W("fig2_motifs.tsv", pd.DataFrame(mrows))

    nrows = []
    for lab, sub2 in [("Canonical", cpa_f[cpa_f.is_canonical]),
                      (FP_LAB, cpa_f[~cpa_f.is_canonical])]:
        for src, col in [("Real", "has_pas"), ("Dinuc null", "has_pas_null")]:
            vv = pd.to_numeric(sub2[col], errors="coerce").dropna()
            if len(vv):
                nrows.append(dict(cls=lab, source=src, rate=float(vv.mean()),
                                  se=float(vv.std(ddof=1)/np.sqrt(len(vv)))
                                  if len(vv) > 1 else 0.0, n=len(vv)))
    W("fig2_null.tsv", pd.DataFrame(nrows))

    nrank = len(pd.read_csv(os.path.join(a.out, "fig2_rank.tsv"), sep="\t")) \
        if os.path.exists(os.path.join(a.out, "fig2_rank.tsv")) else 0
    W("fig2_counts.tsv", pd.DataFrame([
        dict(panel="A", n=n_fp, unit="model false positives"),
        dict(panel="B", n=n_fp, unit="model false positives"),
        dict(panel="C", n=nrank, unit="genes"),
        dict(panel="D", n=int(len(cpa_f)),
             unit=f"peaks: {n_fp:,} false positive + {n_can:,} canonical"),
    ]))

    print(f"[fig2] genes={n_genes:,} contested={n_contested:,} "
          f"false_positives={n_fp:,} canonical={n_can:,} "
          f"({'analysis set' if filtered else 'UNFILTERED'})", file=sys.stderr)
    for r in rows:
        print("   ", r, file=sys.stderr)
    print(f"[out] -> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()