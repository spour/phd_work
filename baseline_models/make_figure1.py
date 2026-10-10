#!/usr/bin/env python3
# publication figures for the cpa model comparison + cpccpa suppression story
import argparse
import glob
import json
import os
import subprocess
import tempfile
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, geom_col, geom_violin, geom_boxplot,
    geom_hline, geom_vline, geom_text, geom_tile, geom_point, geom_abline,
    scale_color_manual, scale_fill_manual, scale_fill_gradient,
    scale_x_log10, scale_y_continuous, scale_x_continuous,
    position_dodge, labs, theme, theme_bw, element_text, element_rect,
    element_line, element_blank, facet_wrap, coord_equal,
)
from mizani.formatters import percent_format


def _save_tsv(df, tsv_path):
    if df is None or (hasattr(df, "__len__") and len(df) == 0):
        return
    df.to_csv(tsv_path, sep="\t", index=False, na_rep="NA",
              float_format="%.6g")
    print(f"    [tsv] wrote {tsv_path}  ({len(df):,} rows)")


def _tsv_path(png_path, piece=None):
    base = png_path[:-4] if png_path.endswith(".png") else png_path
    if piece:
        return f"{base}_{piece}.tsv"
    return f"{base}.tsv"


PAL = {
    "purple":    "#7F77DD",
    "amber":     "#BA7517",
    "teal":      "#1D9E75",
    "blue":      "#378ADD",
    "coral":     "#D85A30",
    "gray":      "#888780",
    "purple_dk": "#3C3489",
}

MODEL_REGISTRY = {
    "rf":       {"display": "RF",       "color": PAL["purple"],
                 "prob_col": "rf_prob",       "top_off_col": "rf_top_offset"},
    "lr":       {"display": "LR",       "color": PAL["amber"],
                 "prob_col": "lr_prob",       "top_off_col": "lr_top_offset"},
    "cnn":      {"display": "CNN",      "color": PAL["teal"],
                 "prob_col": "cnn_prob",      "top_off_col": "cnn_top_offset"},
    "aparent2": {"display": "APARENT2", "color": PAL["coral"],
                 "prob_col": "aparent2_prob", "top_off_col": "aparent2_top_offset"},
}


def display_names(models):
    return [MODEL_REGISTRY[m]["display"] for m in models]


def model_colors(models):
    return {MODEL_REGISTRY[m]["display"]: MODEL_REGISTRY[m]["color"]
            for m in models}


_CANON_RED = "#DC2626"

CPCCPA_CLASS_COLORS = {
    "Constitutive":        PAL["teal"],
    "tier3_exactly_three": "#7E2C12",
    "tier2_exactly_two":   "#D85A30",
    "tier1_exactly_one":   "#F0997B",
    "Random":              PAL["gray"],
}
CPCCPA_CLASS_LABEL = {
    "Constitutive":        "Constitutive\n(pos. control)",
    "tier3_exactly_three": "All 3 models",
    "tier2_exactly_two":   "Exactly 2 models",
    "tier1_exactly_one":   "Exactly 1 model",
    "Random":              "Random\n(neg. control)",
}
CPCCPA_CLASS_ORDER = [
    "Constitutive",
    "tier3_exactly_three",
    "tier2_exactly_two",
    "tier1_exactly_one",
    "Random",
]
CPCCPA_REF_COLORS = {
    "PolyASite":     "#85B7EB",
    "PolyA_DB Main": "#378ADD",
    "PolyA_DB Max":  "#0C447C",
}
CPCCPA_REF_ORDER = ["PolyASite", "PolyA_DB Main", "PolyA_DB Max"]


def detect_models(scan_dir, requested=None):
    import pyarrow.parquet as pq
    parquets = sorted(glob.glob(os.path.join(scan_dir, "per_gene", "*.parquet")))
    if not parquets:
        raise SystemExit(f"No parquets under {scan_dir}/per_gene")
    sampled = parquets[:5] if len(parquets) >= 5 else parquets
    cols_intersect = None
    for p in sampled:
        cols = set(pq.ParquetFile(p).schema_arrow.names)
        cols_intersect = cols if cols_intersect is None else (cols_intersect & cols)
    present = [m for m, info in MODEL_REGISTRY.items()
               if info["prob_col"] in cols_intersect]
    if requested:
        present = [m for m in requested if m in present]
        missing = [m for m in requested if m not in present]
        if missing:
            print(f"  WARN: requested models {missing} not present in parquets, "
                  f"skipping them")
    print(f"  detected/using models: {present}")
    return present


def pub_theme():
    return (
        theme_bw(base_size=11, base_family="DejaVu Sans")
        + theme(
            figure_size=(7, 4),
            dpi=500,
            plot_background=element_rect(fill="white", color="white"),
            panel_background=element_rect(fill="white", color="white"),
            panel_border=element_rect(fill="none", color="black", size=0.8),
            panel_grid_major=element_line(color="#D3D1C7", size=0.3),
            panel_grid_minor=element_line(color="#E8E6DD", size=0.2),
            axis_line=element_blank(),
            axis_ticks=element_line(color="black", size=0.4),
            axis_text=element_text(color="#444441", size=9),
            axis_title=element_text(color="#2C2C2A", size=11),
            plot_title=element_text(color="#2C2C2A", size=13, weight="normal",
                                    ha="left", margin={"b": 10}),
            plot_subtitle=element_text(color="#5F5E5A", size=10,
                                       ha="left", margin={"b": 8}),
            plot_caption=element_text(color="#5F5E5A", size=8, ha="left"),
            legend_position="right",
            legend_background=element_rect(fill="white", color="none"),
            legend_key=element_rect(fill="white", color="none"),
            legend_title=element_text(color="#2C2C2A", size=10),
            legend_text=element_text(color="#444441", size=9),
            strip_background=element_rect(fill="#F1EFE8", color="black", size=0.5),
            strip_text=element_text(color="#2C2C2A", size=10),
        )
    )


def read_bed(path):
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            rows.append({
                "chrom":  p[0],
                "start":  int(p[1]),
                "end":    int(p[2]),
                "name":   p[3] if len(p) > 3 else f"{p[0]}:{p[1]}-{p[2]}",
                "strand": p[5] if len(p) > 5 else "+",
            })
    return rows


def build_union_atlas(atlas_beds, out_dir):
    if isinstance(atlas_beds, str):
        atlas_beds = [atlas_beds]
    union_path = os.path.join(out_dir, "_atlas_union.bed")
    combined = os.path.join(out_dir, "_atlas_concat.bed")
    with open(combined, "w") as fout:
        for bp in atlas_beds:
            for line in open(bp):
                if not line.strip() or line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 6:
                    parts = parts + ["0"] * (5 - len(parts)) + ["+"]
                strand = parts[5] if parts[5] in ("+", "-") else "+"
                fout.write(f"{parts[0]}\t{parts[1]}\t{parts[2]}\t"
                           f".\t0\t{strand}\n")
    sorted_p = os.path.join(out_dir, "_atlas_sorted.bed")
    subprocess.run(["sort", "-k1,1", "-k2,2n", combined],
                   stdout=open(sorted_p, "w"), check=True)
    subprocess.run(["bedtools", "merge", "-s",
                    "-c", "6", "-o", "distinct", "-i", sorted_p],
                   stdout=open(union_path, "w"), check=True)
    final = os.path.join(out_dir, "_atlas_union_6col.bed")
    with open(union_path) as fin, open(final, "w") as fout:
        for line in fin:
            p = line.rstrip("\n").split("\t")
            if len(p) < 4: continue
            chrom, start, end, strand = p[0], p[1], p[2], p[3]
            fout.write(f"{chrom}\t{start}\t{end}\t.\t0\t{strand}\n")
    os.remove(combined); os.remove(sorted_p); os.remove(union_path)
    os.rename(final, union_path)
    n = sum(1 for _ in open(union_path))
    print(f"  [atlas] unioned {len(atlas_beds)} BED(s) -> {union_path} "
          f"({n:,} merged intervals)")
    return union_path


def get_mean_threshold(scan_dir, canonical_bed, models, test_chroms_path=None,
                       test_chroms_only=False, vicinity_nt=500):
    canon = pd.DataFrame(read_bed(canonical_bed))
    canon["gene"] = canon["name"].str.split("::").str[0]
    if test_chroms_only:
        if not test_chroms_path:
            raise ValueError("test_chroms_only requires --test_chroms")
        test_chroms = set(open(test_chroms_path).read().split())
        canon = canon[canon["chrom"].isin(test_chroms)]
        print(f"  using TEST-chrom canonical sites only: {len(canon):,}")
    else:
        print(f"  using ALL canonical sites: {len(canon):,}")
    print(f"  vicinity for max-search: +/- {vicinity_nt} nt around canonical "
          f"window offset")

    parquets = {os.path.basename(p).replace(".parquet", ""): p
                for p in glob.glob(os.path.join(scan_dir, "per_gene", "*.parquet"))}
    prob_cols = [MODEL_REGISTRY[m]["prob_col"] for m in models]
    probs = {m: [] for m in models}
    diag_exact_vs_max = {m: [] for m in models}

    for _, row in canon.iterrows():
        pq = parquets.get(row["gene"])
        if pq is None: continue
        try:
            df = pd.read_parquet(pq, columns=prob_cols +
                                  ["is_constitutive_cpa", "window_offset"])
        except Exception:
            continue
        canon_idx = df.index[df["is_constitutive_cpa"]]
        if len(canon_idx) == 0: continue
        ci = canon_idx[0]
        canon_off = int(df["window_offset"].iloc[ci])
        offsets = df["window_offset"].values

        near = np.abs(offsets - canon_off) <= vicinity_nt
        if not near.any():
            continue

        for m in models:
            col = MODEL_REGISTRY[m]["prob_col"]
            probs_all = df[col].values
            exact_val = float(probs_all[ci])
            max_val   = float(probs_all[near].max())
            probs[m].append(max_val)
            diag_exact_vs_max[m].append((exact_val, max_val))

    thresholds = {m: float(np.mean(v)) if v else 0.5 for m, v in probs.items()}
    print(f"  thresholds (mean MAX prob within +/- {vicinity_nt} nt of "
          f"each canonical site):")
    for m in models:
        disp = MODEL_REGISTRY[m]["display"]
        n = len(probs[m])
        if n == 0:
            print(f"    {disp:<10}: {thresholds[m]:.4f}  (n=0)")
            continue
        deltas = [(mx - ex) for ex, mx in diag_exact_vs_max[m]]
        median_delta = float(np.median(deltas))
        p95_delta    = float(np.percentile(deltas, 95))
        print(f"    {disp:<10}: {thresholds[m]:.4f}  (n={n:,})   "
              f"  shift from exact-window: median +{median_delta:.4f}, "
              f"p95 +{p95_delta:.4f}")
    return thresholds


def fig_training_stats(rf_dir, lr_dir, cnn_dir, out_path):
    print(f"\n[Figure 0] Training stats from metadata.json (RF/LR/CNN only)")
    rows_pr, rows_roc, rows_summary = [], [], []
    for m, run_dir in [("RF", rf_dir), ("LR", lr_dir), ("CNN", cnn_dir)]:
        meta_path = os.path.join(run_dir, "metadata.json")
        if not os.path.exists(meta_path):
            print(f"  WARN: {meta_path} not found, skipping {m}")
            continue
        meta = json.load(open(meta_path))
        winner = meta.get("winner_meta", meta)
        tm = winner.get("test_metrics", winner)
        def get_first(d, keys):
            for k in keys:
                if k in d and d[k] is not None:
                    try: return float(d[k])
                    except (TypeError, ValueError): continue
            return np.nan
        auprc = get_first(tm, ["auprc", "AUPRC", "average_precision"])
        auroc = get_first(tm, ["auroc", "AUROC", "auc", "AUC", "roc_auc"])
        f1    = get_first(tm, ["f1",    "F1"])
        rows_summary += [
            {"Model": m, "Metric": "AUPRC", "value": auprc},
            {"Model": m, "Metric": "AUROC", "value": auroc},
            {"Model": m, "Metric": "F1",    "value": f1}]
        pr = tm.get("pr_curve"); roc = tm.get("roc_curve")
        if pr and "precision" in pr and "recall" in pr:
            prec = np.asarray(pr["precision"], float); rec = np.asarray(pr["recall"], float)
            if len(prec) > 500:
                idx = np.linspace(0, len(prec)-1, 500).astype(int)
                prec = prec[idx]; rec = rec[idx]
            for p_, r_ in zip(prec, rec):
                rows_pr.append({"Model": m, "recall": float(r_), "precision": float(p_)})
        if roc and "fpr" in roc and "tpr" in roc:
            fpr = np.asarray(roc["fpr"], float); tpr = np.asarray(roc["tpr"], float)
            if len(fpr) > 500:
                idx = np.linspace(0, len(fpr)-1, 500).astype(int)
                fpr = fpr[idx]; tpr = tpr[idx]
            for f_, t_ in zip(fpr, tpr):
                rows_roc.append({"Model": m, "fpr": float(f_), "tpr": float(t_)})

    base_colors = {"RF": PAL["purple"], "LR": PAL["amber"], "CNN": PAL["teal"]}
    if rows_summary:
        sdf = pd.DataFrame(rows_summary)
        _save_tsv(sdf[["Model", "Metric", "value"]],
                  _tsv_path(out_path, "summary"))
        sdf["Model"]  = pd.Categorical(sdf["Model"],  ["RF", "LR", "CNN"])
        sdf["Metric"] = pd.Categorical(sdf["Metric"], ["AUPRC", "AUROC", "F1"])
        sdf["label"]  = sdf["value"].map(lambda v: f"{v:.3f}" if pd.notna(v) else "n/a")
        y_min_val = float(sdf["value"].min())
        y_lo = max(0.0, np.floor(y_min_val * 10) / 10 - 0.05); y_hi = 1.08
        breaks = [round(b, 2) for b in np.arange(y_lo, 1.01, 0.05)]
        label_y = {"RF": y_hi - 0.06, "LR": y_hi - 0.03, "CNN": y_hi}
        sdf = sdf.assign(label_y=sdf["Model"].astype(str).map(label_y))
        from plotnine import coord_cartesian
        pS = (ggplot(sdf, aes(x="Metric", y="value", fill="Model"))
              + geom_col(position=position_dodge(width=0.8), width=0.7,
                         color="black", size=0.3)
              + geom_text(aes(x="Metric", y="label_y", label="label", color="Model"),
                          position=position_dodge(width=0.8),
                          va="center", size=8, show_legend=False)
              + scale_fill_manual(values=base_colors)
              + scale_color_manual(values=base_colors, guide=None)
              + scale_y_continuous(breaks=breaks)
              + coord_cartesian(ylim=[y_lo, y_hi + 0.02])
              + labs(x="", y="Score", title="Window-level test-set metrics",
                     subtitle=f"From each model's training metadata (y-axis zoomed to {y_lo:.2f}-{y_hi:.2f})")
              + pub_theme())
        pS_path = out_path.replace(".png", "_summary.png")
        pS.save(pS_path, dpi=500, width=7, height=4, verbose=False)
        print(f"    wrote {pS_path}")

    if rows_pr:
        pr_df = pd.DataFrame(rows_pr)
        _save_tsv(pr_df, _tsv_path(out_path, "pr"))
        pr_df["Model"] = pd.Categorical(pr_df["Model"], ["RF", "LR", "CNN"])
        pPR = (ggplot(pr_df, aes(x="recall", y="precision", color="Model"))
               + geom_line(size=1.1)
               + scale_x_continuous(limits=[0,1], breaks=[0,.25,.5,.75,1.0], labels=percent_format())
               + scale_y_continuous(limits=[0,1.02], breaks=[0,.25,.5,.75,1.0], labels=percent_format())
               + scale_color_manual(values=base_colors)
               + labs(x="Recall", y="Precision",
                      title="Test-set precision-recall curves",
                      subtitle="Window-level CPA discrimination")
               + pub_theme())
        pPR.save(out_path.replace(".png", "_pr.png"), dpi=500, width=6, height=5, verbose=False)

    if rows_roc:
        roc_df = pd.DataFrame(rows_roc)
        _save_tsv(roc_df, _tsv_path(out_path, "roc"))
        roc_df["Model"] = pd.Categorical(roc_df["Model"], ["RF", "LR", "CNN"])
        pROC = (ggplot(roc_df, aes(x="fpr", y="tpr", color="Model"))
                + geom_abline(slope=1, intercept=0, color="#888780", linetype="dashed", size=0.4)
                + geom_line(size=1.1)
                + scale_x_continuous(limits=[0,1], breaks=[0,.25,.5,.75,1.0], labels=percent_format())
                + scale_y_continuous(limits=[0,1.02], breaks=[0,.25,.5,.75,1.0], labels=percent_format())
                + scale_color_manual(values=base_colors)
                + labs(x="False positive rate", y="True positive rate",
                       title="Test-set ROC curves", subtitle="Window-level CPA discrimination")
                + pub_theme())
        pROC.save(out_path.replace(".png", "_roc.png"), dpi=500, width=6, height=5, verbose=False)


KS_FOR_TOPK = [1, 3, 5, 10, 100]


def collect_metrics(scan_dir, models, pc_genes_bed=None, test_chroms=None,
                    max_genes=None, Ks=KS_FOR_TOPK, include_combined=True):
    parquets = sorted(glob.glob(os.path.join(scan_dir, "per_gene", "*.parquet")))
    if test_chroms is not None and pc_genes_bed is not None:
        pc_df = pd.DataFrame(read_bed(pc_genes_bed))
        keep_chroms = set(test_chroms)
        keep_genes = set(pc_df[pc_df["chrom"].isin(keep_chroms)]["name"])
        before = len(parquets)
        parquets = [p for p in parquets
                    if os.path.basename(p).replace(".parquet", "") in keep_genes]
        print(f"  test-chrom filter: {before:,} -> {len(parquets):,} genes "
              f"(chroms: {','.join(sorted(keep_chroms))})")
    if max_genes:
        parquets = parquets[:max_genes]

    eff_models = list(models)
    if include_combined and len(models) >= 2:
        eff_models = eff_models + ["combined"]
    rank_strict = {m: [] for m in eff_models}
    topk_hit    = {m: {K: [] for K in Ks} for m in eff_models}
    dists       = {m: [] for m in eff_models}
    n_skipped = 0
    K_max = max(Ks)

    prob_cols = [MODEL_REGISTRY[m]["prob_col"] for m in models]
    needed_cols = prob_cols + ["is_constitutive_cpa", "window_offset"]

    for i, pq in enumerate(parquets):
        try:
            df = pd.read_parquet(pq, columns=needed_cols)
        except Exception:
            continue
        canon_idx = df.index[df["is_constitutive_cpa"]]
        if len(canon_idx) == 0:
            n_skipped += 1; continue
        ci = canon_idx[0]
        canon_off = int(df["window_offset"].iloc[ci])
        offsets = df["window_offset"].values

        prob_by_model = {}
        for m in models:
            prob_by_model[m] = df[MODEL_REGISTRY[m]["prob_col"]].values
        if "combined" in eff_models:
            arr = np.stack([prob_by_model[m] for m in models]).mean(axis=0)
            prob_by_model["combined"] = arr

        for m in eff_models:
            probs = prob_by_model[m]
            p_canon = probs[ci]
            rank_strict[m].append(int((probs > p_canon).sum()) + 1)
            argmax_idx = int(probs.argmax())
            dists[m].append(abs(int(offsets[argmax_idx]) - canon_off))
            top_n = min(K_max, len(probs))
            top_idx = np.argpartition(probs, -top_n)[-top_n:]
            top_idx = top_idx[np.argsort(-probs[top_idx])]
            top_offs = offsets[top_idx]
            offset_diff = np.abs(top_offs - canon_off)
            for K in Ks:
                hit = bool((offset_diff[:K] < 500).any())
                topk_hit[m][K].append(hit)

        if (i + 1) % 2000 == 0:
            print(f"    read {i+1:,}/{len(parquets):,}")

    print(f"  read {len(parquets):,} parquets ({n_skipped:,} no canonical)")
    return rank_strict, topk_hit, dists


def fig_ranks(rank_strict, topk_hit, models, out_path, label_suffix=""):
    print(f"\n[Figure 1] Rank distributions + top-K overlap{label_suffix}")
    eff = [m for m in models if m in rank_strict]
    if "combined" in rank_strict:
        eff = eff + ["combined"]
    disp_for = {**{m: MODEL_REGISTRY[m]["display"] for m in MODEL_REGISTRY},
                "combined": "Combined"}
    name_order = [disp_for[m] for m in eff]
    color_for  = {**model_colors(models), "Combined": PAL["purple_dk"]}

    cdf_rows = []
    max_rank = max(max(rank_strict[m]) for m in eff)
    xs = np.unique(np.round(np.logspace(0, np.log10(max(max_rank, 2)), 200)).astype(int))
    for m in eff:
        r = np.array(rank_strict[m])
        for x in xs:
            cdf_rows.append({"Model": disp_for[m], "rank": int(x),
                             "frac": float((r <= x).mean())})
    cdf_df = pd.DataFrame(cdf_rows)
    _save_tsv(cdf_df, _tsv_path(out_path.replace(".png", "_cdf.png")))
    cdf_df["Model"] = pd.Categorical(cdf_df["Model"], name_order)

    p1 = (ggplot(cdf_df, aes(x="rank", y="frac", color="Model"))
          + geom_line(size=1.1)
          + scale_x_log10(breaks=[1, 10, 100, 1000, 10000])
          + scale_y_continuous(labels=percent_format(), limits=[0, 1.02],
                               breaks=[0, .25, .5, .75, 1.0])
          + scale_color_manual(values=color_for)
          + labs(x="Rank of canonical window (log scale)",
                 y="Genes (cumulative %)",
                 title=f"Rank of the canonical CPA window{label_suffix}",
                 subtitle="Strict per-window rank (no overlap allowance)")
          + pub_theme())
    p1.save(out_path.replace(".png", "_cdf.png"), dpi=500, width=7, height=4, verbose=False)
    print(f"    wrote {out_path.replace('.png', '_cdf.png')}")

    bar_rows = []
    for m in eff:
        for K in KS_FOR_TOPK:
            pct = float(np.mean(topk_hit[m][K]) * 100)
            bar_rows.append({"Model": disp_for[m], "K": f"top-{K}",
                             "K_num": K, "pct": pct})
    bar_df = pd.DataFrame(bar_rows)
    _save_tsv(bar_df, _tsv_path(out_path.replace(".png", "_topk.png")))
    bar_df["Model"] = pd.Categorical(bar_df["Model"], name_order)
    bar_df["K"]     = pd.Categorical(bar_df["K"], [f"top-{K}" for K in KS_FOR_TOPK])
    bar_df["label"] = bar_df["pct"].map(lambda v: f"{v:.0f}%")

    print(f"\n  Top-K overlap hits:")
    hdr = "    " + "K".rjust(8) + "  " + "  ".join(n.rjust(9) for n in name_order)
    print(hdr)
    for K in KS_FOR_TOPK:
        line = "    " + f"top-{K}".rjust(8) + "  "
        line += "  ".join(f"{float(np.mean(topk_hit[m][K]) * 100):6.2f}%".rjust(9)
                          for m in eff)
        print(line)

    n_models = len(eff)
    bar_width = 0.78
    fig_w = max(7, 1.5 + 1.4 * n_models)
    p2 = (ggplot(bar_df, aes(x="K", y="pct", fill="Model"))
          + geom_col(position=position_dodge(width=bar_width),
                     width=bar_width * 0.85, color="black", size=0.3)
          + geom_text(aes(label="label"), position=position_dodge(width=bar_width),
                      va="bottom", ha="center", size=8, color="#2C2C2A",
                      nudge_y=2.0, angle=90)
          + scale_fill_manual(values=color_for)
          + scale_y_continuous(limits=[0, 122], breaks=[0, 25, 50, 75, 100],
                               labels=lambda l: [f"{int(v)}%" for v in l])
          + labs(x="", y="Genes (%)",
                 title=f"Canonical CPA region in top-K predictions{label_suffix}",
                 subtitle="Hit = any of the top-K windows overlaps the canonical 500-nt window")
          + pub_theme()
          + theme(figure_size=(fig_w, 4)))
    p2.save(out_path.replace(".png", "_topk.png"), dpi=500,
            width=fig_w, height=4, verbose=False)
    print(f"    wrote {out_path.replace('.png', '_topk.png')}")


def peaks_cache_path(out_dir, model, threshold, merge_dist, cache_tag=""):
    tag = f"_{cache_tag}" if cache_tag else ""
    return os.path.join(
        out_dir,
        f"_cache_peaks_{model}_thr{threshold:.4f}_merge{merge_dist}{tag}.parquet")


def _scan_dir_tag(scan_dir):
    parquets = sorted(glob.glob(os.path.join(scan_dir, "per_gene", "*.parquet")))
    if not parquets: return ""
    if len(parquets) > 100:
        idx = np.linspace(0, len(parquets) - 1, 100).astype(int)
        sample = [parquets[i] for i in idx]
    else:
        sample = parquets
    max_mtime = max(os.path.getmtime(p) for p in sample)
    return f"scan{int(max_mtime)}"


def collect_high_conf_peaks(scan_dir, pc_genes_bed, thresholds, out_dir,
                             models, merge_dist=0, force=False, cache_tag=""):
    if not force:
        caches = {m: peaks_cache_path(out_dir, m, thresholds[m], merge_dist, cache_tag)
                  for m in models}
        if all(os.path.exists(p) for p in caches.values()):
            print(f"  [cache] all {len(models)} model peak caches exist; loading...")
            result = {}
            for m, p in caches.items():
                df = pd.read_parquet(p)
                result[m] = df
                print(f"    {MODEL_REGISTRY[m]['display']}: {len(df):,} peaks  ({p})")
            return result

    pc_df = pd.DataFrame(read_bed(pc_genes_bed)).rename(columns={"name": "gene"})
    parquets = sorted(glob.glob(os.path.join(scan_dir, "per_gene", "*.parquet")))
    prob_cols = [MODEL_REGISTRY[m]["prob_col"] for m in models]
    needed = prob_cols + ["is_constitutive_cpa", "window_offset"]
    out_records = defaultdict(list)

    for i, pq in enumerate(parquets):
        gene = os.path.basename(pq).replace(".parquet", "")
        gi_rows = pc_df[pc_df["gene"] == gene]
        if len(gi_rows) == 0: continue
        gi = gi_rows.iloc[0]
        try:
            df = pd.read_parquet(pq, columns=needed)
        except Exception:
            continue
        offset = df["window_offset"].values
        gs = (gi["start"] + offset if gi["strand"] == "+"
              else gi["end"] - offset - 500)
        canon_mask = df["is_constitutive_cpa"].values
        for m in models:
            probs = df[MODEL_REGISTRY[m]["prob_col"]].values
            high = probs >= thresholds[m]
            if not high.any(): continue
            idx = np.where(high)[0]
            for j in idx:
                out_records[m].append((
                    gi["chrom"], int(gs[j]), int(gs[j] + 500),
                    gene, gi["strand"], float(probs[j]),
                    bool(canon_mask[j])))
        if (i + 1) % 2000 == 0:
            print(f"    scanned {i+1:,}/{len(parquets):,}")

    merged_per_model = {}
    for m in models:
        recs = out_records[m]
        if not recs:
            merged_per_model[m] = pd.DataFrame(columns=[
                "chrom", "start", "end", "gene", "strand", "max_prob", "is_canonical"])
        else:
            df_recs = pd.DataFrame(recs, columns=[
                "chrom", "start", "end", "gene", "strand", "prob", "is_can"
            ]).sort_values(["gene", "strand", "start"])
            merged = []
            for (g, s), sub in df_recs.groupby(["gene", "strand"]):
                sub = sub.sort_values("start").reset_index(drop=True)
                first = sub.iloc[0]
                env_end       = int(first["end"])
                best_start    = int(first["start"])
                best_end      = int(first["end"])
                best_prob     = float(first["prob"])
                any_canonical = bool(first["is_can"])
                for j in range(1, len(sub)):
                    r = sub.iloc[j]
                    s_j   = int(r["start"]); e_j = int(r["end"])
                    p_j   = float(r["prob"]); c_j = bool(r["is_can"])
                    if s_j - env_end <= merge_dist:
                        env_end = max(env_end, e_j)
                        any_canonical = any_canonical or c_j
                        if p_j > best_prob:
                            best_prob  = p_j
                            best_start = s_j
                            best_end   = e_j
                    else:
                        merged.append({"chrom": first["chrom"], "start": best_start,
                                       "end": best_end, "gene": g, "strand": s,
                                       "max_prob": best_prob,
                                       "is_canonical": any_canonical})
                        env_end       = e_j; best_start = s_j; best_end = e_j
                        best_prob     = p_j; any_canonical = c_j
                merged.append({"chrom": first["chrom"], "start": best_start,
                               "end": best_end, "gene": g, "strand": s,
                               "max_prob": best_prob, "is_canonical": any_canonical})
            merged_per_model[m] = pd.DataFrame(merged)
        path = peaks_cache_path(out_dir, m, thresholds[m], merge_dist, cache_tag)
        merged_per_model[m].to_parquet(path, index=False, compression="snappy")
        print(f"    {MODEL_REGISTRY[m]['display']}: "
              f"{len(merged_per_model[m]):,} merged peaks -> cached at {path}")
    return merged_per_model


def fig_distance_atlas(scan_dir, pc_genes_bed, atlas_bed, dists, models, out_path):
    print(f"\n[Figure 3] Distance from canonical + atlas-by-distance")
    eff = [m for m in models if m in dists]
    cdf_rows = []
    max_dist = max(max(dists[m]) for m in eff)
    xs = np.unique(np.round(np.logspace(0, np.log10(max_dist + 1), 200)).astype(int))
    for m in eff:
        d = np.array(dists[m])
        for x in xs:
            cdf_rows.append({"Model": MODEL_REGISTRY[m]["display"],
                             "distance": int(x), "frac": float((d <= x).mean())})
    cdf_df = pd.DataFrame(cdf_rows)
    _save_tsv(cdf_df, _tsv_path(out_path.replace(".png", "_cdf.png")))
    name_order = display_names(eff)
    cdf_df["Model"] = pd.Categorical(cdf_df["Model"], name_order)
    p1 = (ggplot(cdf_df, aes(x="distance", y="frac", color="Model"))
          + geom_vline(xintercept=[100, 1000, 10000], color="#D3D1C7",
                       linetype="dotted", size=0.3)
          + geom_line(size=1.1)
          + scale_x_log10(breaks=[1, 10, 100, 1000, 10000, 100000])
          + scale_y_continuous(labels=percent_format(), limits=[0, 1.02],
                               breaks=[0, .25, .5, .75, 1.0])
          + scale_color_manual(values=model_colors(eff))
          + labs(x="Distance from canonical CPA (nt, log scale)",
                 y="Genes (cumulative %)",
                 title="How close is each model's top pick to the canonical CPA?",
                 subtitle="Distance of zero means the top pick is the canonical window itself")
          + pub_theme())
    p1.save(out_path.replace(".png", "_cdf.png"), dpi=500, width=7, height=4, verbose=False)
    print(f"    wrote {out_path.replace('.png', '_cdf.png')}")

    pc_df = pd.DataFrame(read_bed(pc_genes_bed)).rename(columns={"name": "gene"})
    manifest_path = os.path.join(scan_dir, "manifest_all.parquet")
    if not os.path.exists(manifest_path):
        print(f"  WARN: {manifest_path} not found; skipping Panel B")
        return
    manifest = pd.read_parquet(manifest_path)
    manifest_cols = set(manifest.columns)
    models_with_offset = [m for m in eff
                          if MODEL_REGISTRY[m]["top_off_col"] in manifest_cols]
    skipped = [m for m in eff if m not in models_with_offset]
    if skipped:
        print(f"  Panel B: skipping models without top_offset column: "
              f"{[MODEL_REGISTRY[m]['display'] for m in skipped]}")
    if not models_with_offset:
        print(f"  Panel B: no models with top_offset; skipping")
        return

    m_all = manifest[manifest["canonical_window_offset"].notna()].merge(
        pc_df, on="gene", how="left")
    bins = [0, 100, 500, 2000, 5000, 20000, 10**9]
    bin_labels = ["0-100", "100-500", "500-2k", "2k-5k", "5k-20k", ">20k"]
    rows = []
    for m in models_with_offset:
        off_col = MODEL_REGISTRY[m]["top_off_col"]
        df = m_all.copy()
        df["dist"] = (df[off_col] - df["canonical_window_offset"]).abs()
        plus = df["strand"] == "+"
        df["gs"] = np.where(plus, df["start"] + df[off_col],
                            df["end"] - df[off_col] - 500).astype(int)
        df["ge"] = df["gs"] + 500
        bed_path = f"/tmp/__{m}_top.bed"
        df[["chrom", "gs", "ge", "gene", "dist", "strand"]].to_csv(
            bed_path, sep="\t", header=False, index=False)
        atlas_out = f"/tmp/__{m}_hits.bed"
        subprocess.run(
            ["bedtools", "intersect", "-u", "-s", "-a", bed_path, "-b", atlas_bed],
            stdout=open(atlas_out, "w"), check=True)
        if os.path.getsize(atlas_out) > 0:
            hits = pd.read_csv(atlas_out, sep="\t", header=None, names=[
                "chrom", "start", "end", "gene", "dist", "strand"])
            hit_set = set(zip(hits["gene"], hits["dist"]))
        else:
            hit_set = set()
        for lo, hi, label in zip(bins[:-1], bins[1:], bin_labels):
            sub = df[(df["dist"] >= lo) & (df["dist"] < hi)]
            if len(sub) == 0: continue
            hit = sum(1 for _, r in sub.iterrows()
                      if (r["gene"], r["dist"]) in hit_set)
            rows.append({"Model": MODEL_REGISTRY[m]["display"], "bin": label,
                         "rate": hit / len(sub) * 100, "n": len(sub)})
        os.remove(bed_path); os.remove(atlas_out)

    df2 = pd.DataFrame(rows)
    _save_tsv(df2, _tsv_path(out_path.replace(".png", "_strat.png")))
    name_order_b = [MODEL_REGISTRY[m]["display"] for m in models_with_offset]
    df2["Model"] = pd.Categorical(df2["Model"], name_order_b)
    df2["bin"]   = pd.Categorical(df2["bin"], bin_labels)
    fig_w = max(8, 1.5 + 1.6 * len(models_with_offset))
    p2 = (ggplot(df2, aes(x="bin", y="rate", fill="Model"))
          + geom_col(position=position_dodge(width=0.8), width=0.75,
                     color="black", size=0.3)
          + geom_hline(yintercept=16.6, linetype="dashed", color="#5F5E5A", size=0.5)
          + scale_fill_manual(values=model_colors(models_with_offset))
          + scale_y_continuous(limits=[0, 100], breaks=[0, 25, 50, 75, 100],
                               labels=lambda l: [f"{int(v)}%" for v in l])
          + labs(x="Distance from canonical CPA (nt)",
                 y="Atlas overlap rate",
                 title="Are 'wrong' picks real atlas CPA sites?",
                 subtitle="Dashed line: random in-gene window baseline (~17%)")
          + pub_theme()
          + theme(figure_size=(fig_w, 4)))
    p2.save(out_path.replace(".png", "_strat.png"), dpi=500,
            width=fig_w, height=4, verbose=False)
    print(f"    wrote {out_path.replace('.png', '_strat.png')}")


def fig_agreement(scan_dir, merged_per_model, models, out_path,
                  corr_sample_per_gene=200, seed=42):
    print(f"\n[Figure 5] Inter-model agreement")
    parquets = sorted(glob.glob(os.path.join(scan_dir, "per_gene", "*.parquet")))
    rng = np.random.default_rng(seed)

    # panel a: correlation matrix
    print(f"  (A) sampling per-window scores for correlation...")
    prob_cols = [MODEL_REGISTRY[m]["prob_col"] for m in models]
    samples = {m: [] for m in models}
    for pq in parquets:
        try:
            pf = pd.read_parquet(pq, columns=prob_cols)
        except Exception:
            continue
        if len(pf) == 0: continue
        k = min(corr_sample_per_gene, len(pf))
        idx = rng.choice(len(pf), size=k, replace=False)
        for m in models:
            samples[m].extend(pf[MODEL_REGISTRY[m]["prob_col"]].values[idx])
    sample_df = pd.DataFrame(samples)
    sample_df.columns = [MODEL_REGISTRY[m]["display"] for m in models]
    print(f"    sampled {len(sample_df):,} per-window scores")
    corr = sample_df.corr(method="pearson")
    print(f"    Pearson correlation matrix:")
    print(corr.round(3))

    name_order = display_names(models)
    corr_rows = []
    for n1 in name_order:
        for n2 in name_order:
            corr_rows.append({"x": n1, "y": n2, "rho": float(corr.loc[n1, n2])})
    corr_df = pd.DataFrame(corr_rows)
    _save_tsv(corr_df, _tsv_path(out_path.replace(".png", "_corr.png")))
    corr_df["x"] = pd.Categorical(corr_df["x"], name_order)
    corr_df["y"] = pd.Categorical(corr_df["y"], list(reversed(name_order)))
    corr_df["label"] = corr_df["rho"].map(lambda v: f"{v:.2f}")
    fig_dim = max(5.5, 1.5 + 1.4 * len(models))
    pA = (ggplot(corr_df, aes(x="x", y="y", fill="rho"))
          + geom_tile(color="black", size=0.3)
          + geom_text(aes(label="label"), color="#2C2C2A", size=11)
          + scale_fill_gradient(low="#F1EFE8", high=PAL["purple_dk"], limits=[0, 1])
          + coord_equal()
          + labs(x="", y="",
                 title="Per-window score agreement between models",
                 subtitle=f"Pearson r on {len(sample_df):,} sampled windows.",
                 fill="Pearson")
          + pub_theme()
          + theme(figure_size=(fig_dim, fig_dim)))
    pA.save(out_path.replace(".png", "_corr.png"), dpi=500,
            width=fig_dim, height=fig_dim, verbose=False)
    print(f"    wrote {out_path.replace('.png', '_corr.png')}")

    # panel b: pairwise jaccard
    print(f"  (B) per-gene Jaccard of high-confidence peak sets...")
    centers_by_gene = {m: defaultdict(list) for m in models}
    for m in models:
        df = merged_per_model.get(m, pd.DataFrame())
        if len(df) == 0: continue
        df = df.copy()
        df["center"] = ((df["start"] + df["end"]) // 2).astype(int)
        for g, sub in df.groupby("gene"):
            centers_by_gene[m][g] = sorted(sub["center"].tolist())

    def match_set(centers_a, centers_b, tol=250):
        a = sorted(centers_a); b = sorted(centers_b)
        used_b = [False] * len(b); inter = 0
        for ca in a:
            for j, cb in enumerate(b):
                if used_b[j]: continue
                if abs(ca - cb) <= tol:
                    inter += 1; used_b[j] = True; break
        return inter, len(a) + len(b) - inter

    all_genes = set()
    for m in models:
        all_genes.update(centers_by_gene[m].keys())

    pair_rows = []
    for m1, m2 in combinations(models, 2):
        d1 = MODEL_REGISTRY[m1]["display"]; d2 = MODEL_REGISTRY[m2]["display"]
        jaccs = []
        for g in all_genes:
            a = centers_by_gene[m1].get(g, [])
            b = centers_by_gene[m2].get(g, [])
            if not a and not b: continue
            inter, union = match_set(a, b)
            if union > 0: jaccs.append(inter / union)
        median_j = float(np.median(jaccs)) if jaccs else 0.0
        mean_j   = float(np.mean(jaccs))   if jaccs else 0.0
        pair_rows.append({"pair": f"{d1} vs {d2}", "median": median_j,
                          "mean": mean_j, "n": len(jaccs)})
        print(f"    {d1} vs {d2}: median Jaccard {median_j:.3f}, "
              f"mean {mean_j:.3f}, n={len(jaccs):,}")
    jacc_df = pd.DataFrame(pair_rows)
    _save_tsv(jacc_df, _tsv_path(out_path.replace(".png", "_jaccard.png")))
    pair_order = [r["pair"] for r in pair_rows]
    jacc_df["pair"] = pd.Categorical(jacc_df["pair"], pair_order)
    jacc_df["label_median"] = jacc_df["median"].map(lambda v: f"{v:.2f}")
    fig_w_b = max(6, 1.5 + 1.0 * len(pair_rows))
    pB = (ggplot(jacc_df, aes(x="pair", y="median"))
          + geom_col(width=0.5, color="black", size=0.3, fill=PAL["blue"])
          + geom_text(aes(label="label_median"), va="bottom", size=10,
                      color="#2C2C2A", nudge_y=0.015)
          + scale_y_continuous(limits=[0, 1.05], breaks=[0, .25, .5, .75, 1.0])
          + labs(x="", y="Median Jaccard index (per gene)",
                 title="Peak-set agreement between model pairs",
                 subtitle="Peaks within 250 nt count as the same site")
          + pub_theme()
          + theme(figure_size=(fig_w_b, 4)))
    pB.save(out_path.replace(".png", "_jaccard.png"), dpi=500,
            width=fig_w_b, height=4, verbose=False)
    print(f"    wrote {out_path.replace('.png', '_jaccard.png')}")

    # panel c: n-way upset
    print(f"  (C) {len(models)}-way overlap upset...")
    cats = display_names(models)
    breakdown = defaultdict(int)
    for g in all_genes:
        all_pks = []
        for m in models:
            disp = MODEL_REGISTRY[m]["display"]
            for c in centers_by_gene[m].get(g, []):
                all_pks.append((c, disp))
        if not all_pks: continue
        all_pks.sort()
        clusters = []
        for c, disp in all_pks:
            placed = False
            for cl in clusters:
                if any(abs(c - cc) <= 250 for cc, _ in cl):
                    cl.append((c, disp)); placed = True; break
            if not placed:
                clusters.append([(c, disp)])
        for cl in clusters:
            sig = tuple(sorted(set(d for _, d in cl)))
            breakdown[sig] += 1

    intersections = sorted([(sig, n) for sig, n in breakdown.items() if n > 0],
                            key=lambda x: -x[1])
    total = sum(n for _, n in intersections)
    print(f"    total clusters: {total:,}")
    print(f"    intersection sizes:")
    for sig, n in intersections:
        print(f"      {' + '.join(sig):<30} {n:>8,}  ({n/total*100:5.1f}%)")
    set_sizes = {c: 0 for c in cats}
    for sig, n in intersections:
        for c in sig: set_sizes[c] += n

    upset_rows = []
    for rank_idx, (sig, n) in enumerate(intersections):
        upset_rows.append({
            "intersection_id": rank_idx + 1,
            "models_in_intersection": "+".join(sig),
            "n_models": len(sig),
            "count": n,
            "frac_of_total": n / max(total, 1),
        })
    _save_tsv(pd.DataFrame(upset_rows),
              _tsv_path(out_path.replace(".png", "_upset.png"),
                        "intersections"))
    set_rows = [{"Model": c, "set_size": set_sizes[c]} for c in cats]
    _save_tsv(pd.DataFrame(set_rows),
              _tsv_path(out_path.replace(".png", "_upset.png"), "setsizes"))
    mat_rows = []
    for rank_idx, (sig, n) in enumerate(intersections):
        for c in cats:
            mat_rows.append({
                "intersection_id": rank_idx + 1,
                "Model": c,
                "in_intersection": int(c in sig),
                "count": n,
            })
    _save_tsv(pd.DataFrame(mat_rows),
              _tsv_path(out_path.replace(".png", "_upset.png"), "matrix"))

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    n_int = len(intersections)
    fig = plt.figure(figsize=(max(7, 1.0 + 0.6 * n_int), 5.0),
                     dpi=500, facecolor="white")
    gs = GridSpec(2, 2, figure=fig,
                  width_ratios=[1.4, max(3.0, 0.55 * n_int)],
                  height_ratios=[2.2, max(1.0, 0.3 * len(cats))],
                  wspace=0.05, hspace=0.08)
    ax_top_left = fig.add_subplot(gs[0, 0]); ax_top_left.axis("off")
    ax_int = fig.add_subplot(gs[0, 1])
    xs_arr = np.arange(n_int)
    ys = np.array([n for _, n in intersections])
    ax_int.bar(xs_arr, ys, color=PAL["teal"], edgecolor="black",
               linewidth=0.5, width=0.75)
    for i, (_, n) in enumerate(intersections):
        ax_int.text(i, n + ys.max() * 0.015,
                    f"{n:,}\n({n/total*100:.0f}%)",
                    ha="center", va="bottom", fontsize=8, color="#2C2C2A")
    ax_int.set_ylabel("Intersection size", fontsize=11, color="#2C2C2A")
    ax_int.set_xticks([])
    ax_int.set_xlim(-0.6, n_int - 0.4)
    ax_int.set_ylim(0, ys.max() * 1.18)
    ax_int.grid(axis="y", color="#D3D1C7", linewidth=0.3, alpha=0.6)
    ax_int.set_axisbelow(True)
    for s in ax_int.spines.values(): s.set_color("black"); s.set_linewidth(0.8)
    ax_int.tick_params(colors="#444441", labelsize=9)

    ax_set = fig.add_subplot(gs[1, 0])
    ys_cat = np.arange(len(cats))
    set_vals = np.array([set_sizes[c] for c in cats])
    ax_set.barh(ys_cat, set_vals, color=PAL["gray"], edgecolor="black",
                linewidth=0.5, height=0.6)
    for i, v in enumerate(set_vals):
        ax_set.text(v + max(set_vals.max(), 1) * 0.02, i,
                    f"{v:,}", ha="left", va="center", fontsize=8, color="#2C2C2A")
    ax_set.set_yticks(ys_cat)
    ax_set.set_yticklabels(cats, fontsize=10, color="#2C2C2A")
    ax_set.invert_xaxis()
    ax_set.set_xlabel("Set size", fontsize=10, color="#2C2C2A")
    ax_set.set_xlim(max(set_vals.max(), 1) * 1.25, 0)
    ax_set.set_ylim(-0.5, len(cats) - 0.5)
    ax_set.grid(axis="x", color="#D3D1C7", linewidth=0.3, alpha=0.6)
    ax_set.set_axisbelow(True)
    for s in ax_set.spines.values(): s.set_color("black"); s.set_linewidth(0.8)
    ax_set.tick_params(colors="#444441", labelsize=9)

    ax_mat = fig.add_subplot(gs[1, 1])
    for i, (sig, _) in enumerate(intersections):
        for j, c in enumerate(cats):
            if c in sig:
                ax_mat.plot([i], [j], "o", color=PAL["teal"], markersize=10,
                            markeredgecolor="black", markeredgewidth=0.5, zorder=3)
            else:
                ax_mat.plot([i], [j], "o", color="#E8E6DD", markersize=10,
                            markeredgecolor="#D3D1C7", markeredgewidth=0.5, zorder=2)
        active = [j for j, c in enumerate(cats) if c in sig]
        if len(active) > 1:
            ax_mat.plot([i, i], [min(active), max(active)],
                         color=PAL["teal"], linewidth=2, zorder=2.5)
    ax_mat.set_xlim(-0.6, n_int - 0.4)
    ax_mat.set_ylim(-0.5, len(cats) - 0.5)
    ax_mat.set_xticks([]); ax_mat.set_yticks([])
    for s in ax_mat.spines.values(): s.set_visible(False)

    fig.suptitle("High-confidence peak overlap across models",
                 fontsize=13, color="#2C2C2A", x=0.02, ha="left", y=1.04)
    fig.text(0.02, 0.99,
             f"Total distinct peak clusters: {total:,}. "
             "Peaks within 250 nt are treated as the same site.",
             fontsize=10, color="#5F5E5A", ha="left")
    pC_path = out_path.replace(".png", "_upset.png")
    fig.savefig(pC_path, dpi=500, bbox_inches="tight", facecolor="white",
                pad_inches=0.3)
    plt.close(fig)
    print(f"    wrote {pC_path}")


def fig_gene_trace(scan_dir, pc_genes_bed, gene, thresholds, models, out_path,
                   atlas_bed=None):
    print(f"\n[Figure 6] Per-gene trace for {gene}")
    pq = os.path.join(scan_dir, "per_gene", f"{gene}.parquet")
    if not os.path.exists(pq):
        print(f"  WARN: no parquet for gene {gene}; skipping")
        return
    pc_df = pd.DataFrame(read_bed(pc_genes_bed)).rename(columns={"name": "gene"})
    gi_rows = pc_df[pc_df["gene"] == gene]
    if len(gi_rows) == 0:
        print(f"  WARN: gene {gene} not in pc_genes_bed"); return
    gi = gi_rows.iloc[0]
    chrom = gi["chrom"]; strand = gi["strand"]
    gene_len = gi["end"] - gi["start"]

    prob_cols = [MODEL_REGISTRY[m]["prob_col"] for m in models]
    df = pd.read_parquet(pq, columns=prob_cols +
                          ["is_constitutive_cpa", "window_offset"])
    offsets = df["window_offset"].values
    canon_off = (int(df["window_offset"][df["is_constitutive_cpa"]].iloc[0])
                 if df["is_constitutive_cpa"].any() else None)
    x_kb = (offsets + 250) / 1000.0

    atlas_x_kb = []
    if atlas_bed is not None and os.path.exists(atlas_bed):
        for row in read_bed(atlas_bed):
            if row["chrom"] != chrom or row["strand"] != strand: continue
            mid = (row["start"] + row["end"]) // 2
            if not (gi["start"] <= mid <= gi["end"]): continue
            tx_off = (mid - gi["start"]) if strand == "+" else (gi["end"] - mid)
            atlas_x_kb.append(tx_off / 1000.0)
        print(f"  atlas sites in gene: {len(atlas_x_kb)}")
    canon_kb = (canon_off + 250) / 1000.0 if canon_off is not None else None

    trace_rows = []
    for m in models:
        col = MODEL_REGISTRY[m]["prob_col"]
        disp = MODEL_REGISTRY[m]["display"]
        probs = df[col].values
        thr = thresholds[m]
        for x_, p_ in zip(x_kb, probs):
            trace_rows.append({
                "Model": disp, "position_kb": float(x_),
                "prob": float(p_), "threshold": float(thr),
            })
    _save_tsv(pd.DataFrame(trace_rows), _tsv_path(out_path))
    marker_rows = []
    if canon_kb is not None:
        marker_rows.append({"kind": "canonical", "position_kb": float(canon_kb)})
    for a in atlas_x_kb:
        marker_rows.append({"kind": "atlas", "position_kb": float(a)})
    for m in models:
        marker_rows.append({
            "kind": f"threshold_{MODEL_REGISTRY[m]['display']}",
            "position_kb": float(thresholds[m]),
        })
    meta_rows = [{
        "gene": gene, "chrom": chrom, "start": int(gi["start"]),
        "end": int(gi["end"]), "strand": strand, "length_kb": gene_len/1000.0,
        "n_atlas_sites": len(atlas_x_kb),
    }]
    _save_tsv(pd.DataFrame(marker_rows), _tsv_path(out_path, "markers"))
    _save_tsv(pd.DataFrame(meta_rows),   _tsv_path(out_path, "meta"))

    import matplotlib.pyplot as plt
    n_panels = len(models)
    fig_h = 2.0 * n_panels + 1.0
    fig, axes = plt.subplots(n_panels, 1, figsize=(9, fig_h), dpi=500,
                              sharex=True, facecolor="white")
    if n_panels == 1: axes = [axes]
    fig.subplots_adjust(left=0.10, right=0.97, top=0.88, bottom=0.10, hspace=0.18)

    for ax, m in zip(axes, models):
        col = MODEL_REGISTRY[m]["prob_col"]
        disp = MODEL_REGISTRY[m]["display"]
        color = MODEL_REGISTRY[m]["color"]
        probs = df[col].values
        thr   = thresholds[m]
        order = np.argsort(x_kb)
        x_sorted = x_kb[order]; p_sorted = probs[order]
        ax.fill_between(x_sorted, 0, p_sorted, color=color, alpha=0.18, linewidth=0)
        ax.plot(x_sorted, p_sorted, color=color, linewidth=0.8)
        above = p_sorted >= thr
        if above.any():
            ax.fill_between(x_sorted, 0, p_sorted, where=above, color=color,
                             alpha=0.55, linewidth=0)
        ax.axhline(thr, color=color, linestyle="--", linewidth=0.7, alpha=0.7)
        ax.text(x_sorted[-1], thr, f" thr={thr:.3f}",
                fontsize=7, color=color, va="center", ha="left")
        if canon_kb is not None:
            ax.axvline(canon_kb, color=_CANON_RED, linewidth=1.4,
                       linestyle="--", alpha=0.9, zorder=2)
        for a in atlas_x_kb:
            ax.plot([a], [1.02], marker="v", color=PAL["gray"],
                     markersize=4, clip_on=False, zorder=3)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(f"{disp}\nprob", fontsize=10, color="#2C2C2A")
        for s in ax.spines.values(): s.set_color("black"); s.set_linewidth(0.8)
        ax.tick_params(colors="#444441", labelsize=8)
        ax.grid(axis="y", color="#D3D1C7", linewidth=0.3, alpha=0.6)
        ax.set_axisbelow(True); ax.set_facecolor("white")

    axes[-1].set_xlabel("Position along gene, 5' -> 3' (kb)",
                         fontsize=10, color="#2C2C2A")
    strand_label = ("+" if strand == "+" else "-")
    subtitle_bits = [f"{chrom}:{gi['start']:,}-{gi['end']:,} "
                     f"(strand {strand_label}, {gene_len/1000:.1f} kb)"]
    if canon_kb is not None:
        subtitle_bits.append("red dashed line = canonical CPA")
    if atlas_x_kb:
        subtitle_bits.append(f"grey v = atlas CPA ({len(atlas_x_kb)} sites)")
    fig.suptitle(f"Score traces across {gene}", fontsize=13, color="#2C2C2A",
                 x=0.02, ha="left", y=0.96)
    fig.text(0.02, 0.915, "  -  ".join(subtitle_bits),
             fontsize=9, color="#5F5E5A", ha="left")
    fig.savefig(out_path, dpi=500, bbox_inches="tight", facecolor="white",
                pad_inches=0.3)
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_genome_coverage(scan_dir, pc_genes_bed, thresholds, merged_per_model,
                         models, out_path):
    print(f"\n[Figure 7] Genome coverage by model")
    pc_df = pd.DataFrame(read_bed(pc_genes_bed))
    pc_df = pc_df.assign(span=pc_df["end"] - pc_df["start"]) \
                 .sort_values("span", ascending=False) \
                 .drop_duplicates("name", keep="first") \
                 .drop(columns=["span"])
    gene_total_bp = int((pc_df["end"] - pc_df["start"]).sum())
    print(f"  total protein-coding gene-body footprint: {gene_total_bp:,} bp "
          f"({len(pc_df):,} unique genes)")
    print(f"  computing per-model above-threshold base counts...")
    above_bp = {m: 0 for m in models}
    parquets = sorted(glob.glob(os.path.join(scan_dir, "per_gene", "*.parquet")))
    pc_by_gene = pc_df.set_index("name")[["start", "end", "strand"]].to_dict("index")
    prob_cols = [MODEL_REGISTRY[m]["prob_col"] for m in models]
    needed = prob_cols + ["window_offset"]
    for i, pq in enumerate(parquets):
        gene = os.path.basename(pq).replace(".parquet", "")
        gi = pc_by_gene.get(gene)
        if gi is None: continue
        try:
            df = pd.read_parquet(pq, columns=needed)
        except Exception:
            continue
        offsets = df["window_offset"].values
        if gi["strand"] == "+":
            starts = gi["start"] + offsets
        else:
            starts = gi["end"] - offsets - 500
        ends = starts + 500
        gene_start = gi["start"]; gene_end = gi["end"]
        for m in models:
            mask = df[MODEL_REGISTRY[m]["prob_col"]].values >= thresholds[m]
            if not mask.any(): continue
            s = np.maximum(starts[mask], gene_start)
            e = np.minimum(ends[mask], gene_end)
            valid = e > s
            s = s[valid]; e = e[valid]
            if len(s) == 0: continue
            order = np.argsort(s)
            s = s[order]; e = e[order]
            cur_s, cur_e = int(s[0]), int(e[0])
            total = 0
            for j in range(1, len(s)):
                if s[j] <= cur_e:
                    cur_e = max(cur_e, int(e[j]))
                else:
                    total += cur_e - cur_s
                    cur_s, cur_e = int(s[j]), int(e[j])
            total += cur_e - cur_s
            above_bp[m] += total
        if (i + 1) % 2000 == 0:
            print(f"    scanned {i+1:,}/{len(parquets):,}")

    panelA_rows = []
    for m in models:
        pct = above_bp[m] / max(gene_total_bp, 1) * 100
        panelA_rows.append({"Model": MODEL_REGISTRY[m]["display"],
                            "Metric": "Above-threshold",
                            "bp": above_bp[m], "pct": pct})
        print(f"    {MODEL_REGISTRY[m]['display']}: "
              f"{above_bp[m]:,} bp / {gene_total_bp:,} = {pct:.3f}%")

    panelB_rows = []
    for m in models:
        df = merged_per_model.get(m)
        disp = MODEL_REGISTRY[m]["display"]
        if df is None or len(df) == 0:
            panelB_rows.append({"Model": disp, "Metric": "Merged peaks",
                                "bp": 0, "pct": 0})
            print(f"    {disp} merged: 0 bp = 0%")
            continue
        bp = int((df["end"] - df["start"]).sum())
        pct = bp / max(gene_total_bp, 1) * 100
        panelB_rows.append({"Model": disp, "Metric": "Merged peaks",
                            "bp": bp, "pct": pct})
        print(f"    {disp} merged: {bp:,} bp = {pct:.3f}%")

    plot_df = pd.DataFrame(panelA_rows + panelB_rows)
    _save_tsv(plot_df, _tsv_path(out_path))
    _save_tsv(pd.DataFrame([{"total_genebody_bp": int(gene_total_bp)}]),
              _tsv_path(out_path, "denominator"))
    name_order = display_names(models)
    plot_df["Model"]  = pd.Categorical(plot_df["Model"], name_order)
    plot_df["Metric"] = pd.Categorical(
        plot_df["Metric"], ["Above-threshold", "Merged peaks"])
    plot_df["label"]  = plot_df["pct"].map(lambda v: f"{v:.2f}%")
    y_hi = max(plot_df["pct"].max() * 1.25, 1.0)
    fig_w = max(8, 2 + 1.5 * len(models))
    p = (ggplot(plot_df, aes(x="Model", y="pct", fill="Model"))
         + geom_col(width=0.6, color="black", size=0.3, show_legend=False)
         + geom_text(aes(label="label"), va="bottom", size=9,
                     color="#2C2C2A", nudge_y=y_hi * 0.02)
         + facet_wrap("Metric", nrow=1)
         + scale_fill_manual(values=model_colors(models))
         + scale_y_continuous(limits=[0, y_hi])
         + labs(x="", y="% of protein-coding gene-body bases",
                title="How much of the protein-coding genome is called positive?",
                subtitle=(f"Total gene-body footprint: {gene_total_bp/1e6:.1f} Mb. "
                          "Left: any above-threshold 500-nt window; "
                          "Right: bedtools-merged representative peaks"))
         + pub_theme()
         + theme(figure_size=(fig_w, 4.2)))
    p.save(out_path, dpi=500, width=fig_w, height=4.2, verbose=False)
    print(f"  wrote {out_path}")


def fig_score_saturation(scan_dir, pc_genes_bed, thresholds, models, out_path,
                          test_chroms=None, high_thr=0.95):
    print(f"\n[Figure 9] Score-saturation diagnostic (high_thr = {high_thr})")
    parquets = sorted(glob.glob(os.path.join(scan_dir, "per_gene", "*.parquet")))
    if test_chroms is not None and pc_genes_bed is not None:
        pc_df = pd.DataFrame(read_bed(pc_genes_bed))
        keep_chroms = set(test_chroms)
        keep_genes = set(pc_df[pc_df["chrom"].isin(keep_chroms)]["name"])
        parquets = [p for p in parquets
                    if os.path.basename(p).replace(".parquet", "") in keep_genes]
        print(f"  test-chrom filter: {len(parquets):,} genes")

    prob_cols = [MODEL_REGISTRY[m]["prob_col"] for m in models]
    needed = prob_cols + ["is_constitutive_cpa", "window_offset"]
    panelA_rows = []
    panelB_rows = []
    rank_rows = []
    miss_count = {m: 0 for m in models}
    near_tie_count = {m: 0 for m in models}

    for i, pq in enumerate(parquets):
        try:
            df = pd.read_parquet(pq, columns=needed)
        except Exception:
            continue
        canon_idx = df.index[df["is_constitutive_cpa"]]
        if len(canon_idx) == 0: continue
        ci = canon_idx[0]
        canon_off = int(df["window_offset"].iloc[ci])
        for m in models:
            probs = df[MODEL_REGISTRY[m]["prob_col"]].values
            disp = MODEL_REGISTRY[m]["display"]
            n_high = int((probs >= high_thr).sum())
            panelA_rows.append({"Model": disp, "n_high": n_high})
            offsets = df["window_offset"].values
            argmax_idx = int(probs.argmax())
            argmax_off = int(offsets[argmax_idx])
            is_miss = abs(argmax_off - canon_off) >= 500
            if is_miss:
                miss_count[m] += 1
                gap = float(probs[argmax_idx] - probs[ci])
                panelB_rows.append({"Model": disp, "gap": gap})
                if gap < 0.01: near_tie_count[m] += 1
            p_canon = probs[ci]
            r_strict = int((probs > p_canon).sum()) + 1
            rank_rows.append({"Model": disp, "rank": r_strict})
        if (i + 1) % 2000 == 0:
            print(f"    scanned {i+1:,}/{len(parquets):,}")

    print(f"  per-model top-1 miss counts:")
    for m in models:
        n_miss = miss_count[m]; n_near = near_tie_count[m]
        if n_miss > 0:
            print(f"    {MODEL_REGISTRY[m]['display']}: {n_miss:,} misses, "
                  f"{n_near:,} ({n_near/n_miss*100:.1f}%) gap < 0.01")
        else:
            print(f"    {MODEL_REGISTRY[m]['display']}: 0 misses")

    name_order = display_names(models)
    panelA_df = pd.DataFrame(panelA_rows)
    _save_tsv(panelA_df, _tsv_path(out_path, "panelA"))
    panelA_df["Model"] = pd.Categorical(panelA_df["Model"], name_order)
    print(f"\n  Panel A (#windows above {high_thr} per gene):")
    for m in models:
        disp = MODEL_REGISTRY[m]["display"]
        sub = panelA_df[panelA_df["Model"] == disp]["n_high"]
        if len(sub) == 0: continue
        print(f"    {disp}: median {sub.median():.0f}, "
              f"p95 {sub.quantile(0.95):.0f}, max {sub.max():.0f}")

    panelB_df = pd.DataFrame(panelB_rows)
    _save_tsv(panelB_df, _tsv_path(out_path, "panelB"))
    if len(panelB_df):
        panelB_df["Model"] = pd.Categorical(panelB_df["Model"], name_order)
    rank_df = pd.DataFrame(rank_rows)
    _save_tsv(rank_df, _tsv_path(out_path, "panelC"))
    rank_df["Model"] = pd.Categorical(rank_df["Model"], name_order)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=500, facecolor="white")
    fig.subplots_adjust(left=0.05, right=0.98, top=0.85, bottom=0.13, wspace=0.25)

    ax = axes[0]
    bin_max = max(50, int(np.ceil(panelA_df["n_high"].quantile(0.99))))
    bins = np.arange(0, bin_max + 2, 1) if bin_max <= 60 else \
           np.linspace(0, bin_max + 2, 60)
    for m in models:
        disp = MODEL_REGISTRY[m]["display"]
        color = MODEL_REGISTRY[m]["color"]
        vals = panelA_df[panelA_df["Model"] == disp]["n_high"].values
        if len(vals) == 0: continue
        ax.hist(vals, bins=bins, color=color, edgecolor="black",
                 linewidth=0.3, alpha=0.55, label=disp)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xlabel(f"# windows per gene with prob >= {high_thr}",
                  fontsize=10, color="#2C2C2A")
    ax.set_ylabel("Genes (log)", fontsize=10, color="#2C2C2A")
    ax.set_title("A. How many windows does each model call confidently?",
                 fontsize=11, color="#2C2C2A", loc="left", pad=8)
    ax.legend(frameon=False, fontsize=9)
    for s in ax.spines.values(): s.set_color("black"); s.set_linewidth(0.8)
    ax.grid(axis="y", color="#D3D1C7", linewidth=0.3, alpha=0.6)
    ax.set_axisbelow(True); ax.set_facecolor("white")
    ax.tick_params(colors="#444441", labelsize=9)

    ax = axes[1]
    if len(panelB_df):
        bins_b = np.linspace(0, 1.0, 41)
        for m in models:
            disp = MODEL_REGISTRY[m]["display"]
            color = MODEL_REGISTRY[m]["color"]
            sub = panelB_df[panelB_df["Model"] == disp]
            if len(sub) == 0: continue
            ax.hist(sub["gap"], bins=bins_b, color=color, edgecolor="black",
                     linewidth=0.3, alpha=0.55,
                     label=f"{disp} (n={len(sub):,})")
        ax.axvline(0.01, color="#5F5E5A", linestyle="--", linewidth=0.7, alpha=0.7)
    ax.set_xlabel("(top-1 prob) - (canonical prob), for top-1-miss genes",
                  fontsize=10, color="#2C2C2A")
    ax.set_ylabel("Genes", fontsize=10, color="#2C2C2A")
    ax.set_title("B. Are 'misses' actually ties?",
                 fontsize=11, color="#2C2C2A", loc="left", pad=8)
    ax.legend(frameon=False, fontsize=9)
    for s in ax.spines.values(): s.set_color("black"); s.set_linewidth(0.8)
    ax.grid(axis="y", color="#D3D1C7", linewidth=0.3, alpha=0.6)
    ax.set_axisbelow(True); ax.set_facecolor("white")
    ax.tick_params(colors="#444441", labelsize=9)
    ax.set_xlim(0, 1.0)

    ax = axes[2]
    max_rank = max(2, int(rank_df["rank"].max()))
    xs_l = np.unique(np.round(np.logspace(0, np.log10(max_rank), 200)).astype(int))
    for m in models:
        disp = MODEL_REGISTRY[m]["display"]
        color = MODEL_REGISTRY[m]["color"]
        r = rank_df[rank_df["Model"] == disp]["rank"].values
        if len(r) == 0: continue
        cdf_y = [float((r <= x).mean()) for x in xs_l]
        ax.plot(xs_l, cdf_y, color=color, linewidth=1.5, label=disp)
    ax.set_xscale("log")
    ax.set_xlabel("Rank of canonical window in gene (log)",
                  fontsize=10, color="#2C2C2A")
    ax.set_ylabel("Cumulative fraction of genes",
                  fontsize=10, color="#2C2C2A")
    ax.set_title("C. Where does the canonical rank?",
                 fontsize=11, color="#2C2C2A", loc="left", pad=8)
    ax.set_ylim(0, 1.02); ax.set_xlim(1, max_rank)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    for s in ax.spines.values(): s.set_color("black"); s.set_linewidth(0.8)
    ax.grid(color="#D3D1C7", linewidth=0.3, alpha=0.6)
    ax.set_axisbelow(True); ax.set_facecolor("white")
    ax.tick_params(colors="#444441", labelsize=9)

    suffix = " (test chroms only)" if test_chroms is not None else ""
    fig.suptitle(f"Score-saturation diagnostic{suffix}",
                 fontsize=14, color="#2C2C2A", x=0.02, ha="left", y=1.02)
    fig.text(0.02, 0.965,
             "If many windows tie at high probability, argmax becomes a "
             "coin flip - even when the canonical IS the strongest signal.",
             fontsize=10, color="#5F5E5A", ha="left")
    fig.savefig(out_path, dpi=500, bbox_inches="tight", facecolor="white", pad_inches=0.3)
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_calibration(scan_dir, pc_genes_bed, atlas_beds, canonical_bed,
                     models, out_path,
                     sample_per_gene=200, seed=42, n_bins=12,
                     extra_positive_beds=None):
    print(f"\n[Figure 10] Calibration plots")
    if extra_positive_beds is None:
        extra_positive_beds = []

    # build a positive-region bed: union of canonical + atlas + extras
    print(f"  building positive-region union BED...")
    pos_concat = tempfile.NamedTemporaryFile(suffix=".bed", delete=False, mode="w")
    for bp in [canonical_bed] + list(atlas_beds) + list(extra_positive_beds):
        if not bp or not os.path.exists(bp):
            if bp:
                print(f"    skip missing {bp}")
            continue
        for line in open(bp):
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            if len(parts) < 6:
                parts = parts + ["0"] * (5 - len(parts)) + ["+"]
            strand = parts[5] if parts[5] in ("+", "-") else "+"
            pos_concat.write(f"{parts[0]}\t{parts[1]}\t{parts[2]}\t"
                              f".\t0\t{strand}\n")
    pos_concat.close()
    pos_sorted = pos_concat.name + ".sorted"
    pos_merged = pos_concat.name + ".merged"
    subprocess.run(["sort", "-k1,1", "-k2,2n", pos_concat.name],
                   stdout=open(pos_sorted, "w"), check=True)
    subprocess.run(["bedtools", "merge", "-s", "-c", "6", "-o", "distinct",
                    "-i", pos_sorted],
                   stdout=open(pos_merged, "w"), check=True)
    print(f"  positive-region union: "
          f"{sum(1 for _ in open(pos_merged)):,} merged intervals")

    print(f"  sampling windows from parquets...")
    pc_df = pd.DataFrame(read_bed(pc_genes_bed)).rename(columns={"name": "gene"})
    n_before = len(pc_df)
    pc_df = (pc_df.assign(span=pc_df["end"] - pc_df["start"])
                   .sort_values("span", ascending=False)
                   .drop_duplicates("gene", keep="first")
                   .drop(columns=["span"]))
    if len(pc_df) < n_before:
        print(f"  deduped pc_genes: {n_before:,} -> {len(pc_df):,} unique genes")
    parquets = sorted(glob.glob(os.path.join(scan_dir, "per_gene", "*.parquet")))
    pc_by_gene = pc_df.set_index("gene")[["chrom", "start", "end", "strand"]].to_dict("index")
    rng = np.random.default_rng(seed)

    prob_cols = [MODEL_REGISTRY[m]["prob_col"] for m in models]
    needed = prob_cols + ["window_offset"]

    sampled_bed_path = tempfile.NamedTemporaryFile(suffix=".bed", delete=False,
                                                     mode="w").name
    sampled_probs = []
    sampled_keys  = []

    n_genes_done = 0
    with open(sampled_bed_path, "w") as fout:
        for pq_path in parquets:
            gene = os.path.basename(pq_path).replace(".parquet", "")
            gi = pc_by_gene.get(gene)
            if gi is None:
                continue
            try:
                df = pd.read_parquet(pq_path, columns=needed)
            except Exception:
                continue
            if len(df) == 0:
                continue
            k = min(sample_per_gene, len(df))
            idx = rng.choice(len(df), size=k, replace=False)
            offsets = df["window_offset"].values[idx]
            if gi["strand"] == "+":
                starts = gi["start"] + offsets
            else:
                starts = gi["end"] - offsets - 500
            ends = starts + 500
            probs_arr = np.stack(
                [df[MODEL_REGISTRY[m]["prob_col"]].values[idx] for m in models],
                axis=1,
            )
            for i in range(k):
                key = f"{gene}_{int(offsets[i])}"
                sampled_keys.append(key)
                sampled_probs.append(probs_arr[i])
                fout.write(f"{gi['chrom']}\t{int(starts[i])}\t{int(ends[i])}\t"
                           f"{key}\t0\t{gi['strand']}\n")
            n_genes_done += 1
            if n_genes_done % 2000 == 0:
                print(f"    sampled {n_genes_done:,}/{len(parquets):,} genes")
    print(f"  total sampled windows: {len(sampled_keys):,}")

    if len(sampled_keys) == 0:
        print(f"  no windows sampled; skipping")
        return

    print(f"  flagging positives via bedtools intersect...")
    hits_path = sampled_bed_path + ".hits"
    subprocess.run(
        ["bedtools", "intersect", "-u", "-s",
         "-a", sampled_bed_path, "-b", pos_merged],
        stdout=open(hits_path, "w"), check=True,
    )
    positive_keys = set()
    if os.path.getsize(hits_path) > 0:
        for line in open(hits_path):
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                positive_keys.add(parts[3])
    print(f"  positive windows: {len(positive_keys):,} / {len(sampled_keys):,} "
          f"({len(positive_keys)/max(len(sampled_keys),1)*100:.2f}%)")

    sampled_probs_arr = np.array(sampled_probs)
    labels = np.array([k in positive_keys for k in sampled_keys], dtype=bool)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for mi, m in enumerate(models):
        disp = MODEL_REGISTRY[m]["display"]
        p = sampled_probs_arr[:, mi]
        bin_idx = np.minimum(np.digitize(p, bin_edges, right=False) - 1, n_bins - 1)
        bin_idx = np.maximum(bin_idx, 0)
        for b in range(n_bins):
            mask = bin_idx == b
            n = int(mask.sum())
            if n < 10:
                continue
            mean_pred = float(p[mask].mean())
            emp_rate  = float(labels[mask].mean())
            rows.append({"Model": disp, "bin": b, "n": n,
                         "mean_pred": mean_pred, "empirical": emp_rate})
    cal_df = pd.DataFrame(rows)
    _save_tsv(cal_df, _tsv_path(out_path))
    if len(cal_df) == 0:
        print(f"  no bins had enough samples; skipping")
        return
    name_order = display_names(models)
    cal_df["Model"] = pd.Categorical(cal_df["Model"], name_order)

    fig_dim = max(6, 1 + 1.0 * len(models))
    p_calib = (ggplot(cal_df, aes(x="mean_pred", y="empirical", color="Model"))
               + geom_abline(slope=1, intercept=0, color="#888780",
                             linetype="dashed", size=0.4)
               + geom_line(size=1.0)
               + geom_point(aes(size="n"), alpha=0.85)
               + scale_color_manual(values=model_colors(models))
               + scale_x_continuous(limits=[0, 1.02],
                                     breaks=[0, .25, .5, .75, 1.0],
                                     labels=percent_format())
               + scale_y_continuous(limits=[0, 1.02],
                                     breaks=[0, .25, .5, .75, 1.0],
                                     labels=percent_format())
               + labs(x="Mean predicted probability (in bin)",
                      y="Empirical positive rate (in bin)",
                      title="Are model probabilities calibrated?",
                      subtitle=("Predicted vs observed positive rate per bin; "
                                "diagonal = perfectly calibrated. "
                                f"{len(sampled_keys):,} sampled windows, "
                                f"{len(positive_keys):,} positives "
                                "(canonical + atlas)."),
                      size="bin count")
               + pub_theme()
               + theme(figure_size=(fig_dim, fig_dim)))
    p_calib.save(out_path, dpi=500, width=fig_dim, height=fig_dim, verbose=False)
    print(f"  wrote {out_path}")

    hist_rows = []
    for mi, m in enumerate(models):
        disp = MODEL_REGISTRY[m]["display"]
        p = sampled_probs_arr[:, mi]
        hist, _ = np.histogram(p, bins=bin_edges)
        for b in range(n_bins):
            hist_rows.append({"Model": disp, "bin_lo": float(bin_edges[b]),
                              "bin_hi": float(bin_edges[b + 1]),
                              "bin_mid": float((bin_edges[b] + bin_edges[b + 1]) / 2),
                              "count": int(hist[b])})
    hist_df = pd.DataFrame(hist_rows)
    _save_tsv(hist_df, _tsv_path(out_path.replace(".png", "_dist.png")))
    hist_df["Model"] = pd.Categorical(hist_df["Model"], name_order)
    hist_path = out_path.replace(".png", "_dist.png")
    fig_w = max(7, 2 + 1.5 * len(models))
    p_hist = (ggplot(hist_df, aes(x="bin_mid", y="count", fill="Model"))
              + geom_col(width=1.0 / n_bins * 0.95, color="black", size=0.2,
                          show_legend=False)
              + facet_wrap("Model", nrow=1, scales="free_y")
              + scale_fill_manual(values=model_colors(models))
              + scale_x_continuous(limits=[0, 1.02], breaks=[0, .25, .5, .75, 1.0],
                                    labels=percent_format())
              + labs(x="Predicted probability", y="Sampled windows",
                     title="Distribution of predicted probabilities per model",
                     subtitle=("How much of the probability mass each model places at low / mid / "
                               "high confidence. Flat = uncertain everywhere; "
                               "bimodal = decisive."))
              + pub_theme()
              + theme(figure_size=(fig_w, 4)))
    p_hist.save(hist_path, dpi=500, width=fig_w, height=4, verbose=False)
    print(f"  wrote {hist_path}")

    for p_ in [pos_concat.name, pos_sorted, pos_merged,
               sampled_bed_path, hits_path]:
        try: os.remove(p_)
        except OSError: pass


def fig_atlas_recovery(merged_per_model, models, ref_paths_with_labels, out_path,
                        include_combined=True, slack=50):
    print(f"\n[Figure 2] Atlas recovery rates per model")

    model_bed_paths = {}
    for m in models:
        df = merged_per_model.get(m, pd.DataFrame())
        if len(df) == 0:
            model_bed_paths[m] = None
            continue
        tmp = tempfile.NamedTemporaryFile(suffix=".bed", delete=False, mode="w").name
        sub = df.copy()
        sub["start"] = (sub["start"] - slack).clip(lower=0)
        sub["end"]   = sub["end"] + slack
        sub["score"] = 0
        sub.assign(idx=np.arange(len(sub)))[
            ["chrom", "start", "end", "idx", "score", "strand"]
        ].to_csv(tmp, sep="\t", header=False, index=False)
        subprocess.run(f"sort -k1,1 -k2,2n {tmp} -o {tmp}",
                       shell=True, check=True)
        model_bed_paths[m] = tmp

    combined_path = None
    if include_combined:
        valid = [p for p in model_bed_paths.values() if p is not None]
        if valid:
            concat = tempfile.NamedTemporaryFile(suffix=".bed", delete=False, mode="w").name
            with open(concat, "w") as fout:
                for p in valid:
                    with open(p) as fin:
                        for line in fin:
                            fout.write(line)
            sorted_p = concat + ".sorted"
            subprocess.run(f"sort -k1,1 -k2,2n {concat} -o {sorted_p}",
                           shell=True, check=True)
            merged_p = concat + ".merged"
            subprocess.run(
                ["bedtools", "merge", "-s", "-c", "6", "-o", "distinct",
                 "-i", sorted_p],
                stdout=open(merged_p, "w"), check=True)
            combined_path = concat + ".final"
            with open(merged_p) as fin, open(combined_path, "w") as fout:
                for line in fin:
                    p = line.rstrip("\n").split("\t")
                    if len(p) >= 4:
                        chrom, s, e, strand = p[0], p[1], p[2], p[3]
                        fout.write(f"{chrom}\t{s}\t{e}\t.\t0\t{strand}\n")
            os.remove(concat); os.remove(sorted_p); os.remove(merged_p)

    rows = []
    for ref_label, ref_path in ref_paths_with_labels:
        if not ref_path or not os.path.exists(ref_path):
            print(f"  skip missing ref {ref_label} ({ref_path})")
            continue
        n_ref = sum(1 for _ in open(ref_path))
        print(f"\n  {ref_label} ({n_ref:,} sites):")
        for m in models:
            disp = MODEL_REGISTRY[m]["display"]
            peak_bed = model_bed_paths[m]
            if peak_bed is None:
                rows.append({"Atlas": ref_label, "Model": disp,
                             "recall": 0.0, "n_hit": 0, "n_total": n_ref})
                continue
            res = subprocess.run(
                ["bedtools", "intersect", "-u", "-s",
                 "-a", ref_path, "-b", peak_bed],
                stdout=subprocess.PIPE, check=True, text=True)
            n_hit = sum(1 for l in res.stdout.splitlines() if l.strip())
            rate = n_hit / max(n_ref, 1) * 100
            rows.append({"Atlas": ref_label, "Model": disp,
                         "recall": rate, "n_hit": n_hit, "n_total": n_ref})
            print(f"    {disp:<10}: {n_hit:>7,}/{n_ref:<7,} = {rate:5.1f}%")
        if combined_path is not None:
            res = subprocess.run(
                ["bedtools", "intersect", "-u", "-s",
                 "-a", ref_path, "-b", combined_path],
                stdout=subprocess.PIPE, check=True, text=True)
            n_hit = sum(1 for l in res.stdout.splitlines() if l.strip())
            rate = n_hit / max(n_ref, 1) * 100
            rows.append({"Atlas": ref_label, "Model": "Combined",
                         "recall": rate, "n_hit": n_hit, "n_total": n_ref})
            print(f"    {'Combined':<10}: {n_hit:>7,}/{n_ref:<7,} = {rate:5.1f}%")

    for p in model_bed_paths.values():
        if p is not None:
            try: os.remove(p)
            except OSError: pass
    if combined_path is not None:
        try: os.remove(combined_path)
        except OSError: pass

    if not rows:
        print("  no data; skipping render")
        return

    df = pd.DataFrame(rows)
    _save_tsv(df, _tsv_path(out_path))
    atlas_order = [l for l, p in ref_paths_with_labels
                   if p and os.path.exists(p)]
    model_order = display_names(models) + (
        ["Combined"] if include_combined else [])
    df["Atlas"] = pd.Categorical(df["Atlas"], atlas_order)
    df["Model"] = pd.Categorical(df["Model"], model_order)
    df["label"] = df["recall"].map(lambda v: f"{v:.0f}%")

    color_for = {**model_colors(models), "Combined": PAL["purple_dk"]}

    fig_w = max(8, 1.5 + 1.6 * len(atlas_order))
    p_plot = (ggplot(df, aes(x="Atlas", y="recall", fill="Model"))
              + geom_col(position=position_dodge(width=0.85), width=0.78,
                          color="black", size=0.3)
              + geom_text(aes(label="label"),
                          position=position_dodge(width=0.85),
                          va="bottom", size=8, color="#2C2C2A", nudge_y=1.5)
              + scale_fill_manual(values=color_for)
              + scale_y_continuous(limits=[0, 112], breaks=[0, 25, 50, 75, 100],
                                    labels=lambda l: [f"{int(v)}%" for v in l])
              + labs(x="", y="Recall (% of reference sites with a model peak)",
                     title="Atlas recovery by each model's peak set",
                     subtitle=(f"Slack +/- {slack} nt around model peaks. "
                                "'Combined' = bedtools-merged union of all model peak BEDs."))
              + pub_theme()
              + theme(figure_size=(fig_w, 5)))
    p_plot.save(out_path, dpi=500, width=fig_w, height=5, verbose=False)
    print(f"\n  wrote {out_path}")


def fig_score_distributions(scan_dir, pc_genes_bed, atlas_bed_path,
                              models, thresholds, out_path,
                              sample_per_gene=100, seed=42, test_chroms=None):
    print(f"\n[Figure 4] Score distributions by window class")

    pc_df = pd.DataFrame(read_bed(pc_genes_bed)).rename(columns={"name": "gene"})
    if test_chroms is not None:
        keep_genes = set(pc_df[pc_df["chrom"].isin(test_chroms)]["gene"])
    else:
        keep_genes = None
    pc_df = (pc_df.assign(span=pc_df["end"] - pc_df["start"])
                   .sort_values("span", ascending=False)
                   .drop_duplicates("gene", keep="first")
                   .drop(columns=["span"]))
    pc_by_gene = pc_df.set_index("gene")[
        ["chrom", "start", "end", "strand"]].to_dict("index")

    parquets = sorted(glob.glob(os.path.join(scan_dir, "per_gene", "*.parquet")))
    if keep_genes is not None:
        parquets = [p for p in parquets
                    if os.path.basename(p).replace(".parquet", "") in keep_genes]

    rng = np.random.default_rng(seed)
    prob_cols = [MODEL_REGISTRY[m]["prob_col"] for m in models]
    needed = prob_cols + ["is_constitutive_cpa", "window_offset"]

    sampled_bed_path = tempfile.NamedTemporaryFile(
        suffix=".bed", delete=False, mode="w").name
    canon_keys = set()
    sampled_keys = []
    sampled_probs = []

    print(f"  sampling ~{sample_per_gene} windows per gene...")
    with open(sampled_bed_path, "w") as fout:
        for pq_path in parquets:
            gene = os.path.basename(pq_path).replace(".parquet", "")
            gi = pc_by_gene.get(gene)
            if gi is None:
                continue
            try:
                df = pd.read_parquet(pq_path, columns=needed)
            except Exception:
                continue
            if len(df) == 0:
                continue

            canon_idx_list = df.index[df["is_constitutive_cpa"]].tolist()
            k = min(sample_per_gene, len(df))
            random_idx = list(rng.choice(len(df), size=k, replace=False))
            all_idx = set(int(i) for i in random_idx)
            for ci in canon_idx_list:
                all_idx.add(int(ci))
            all_idx = sorted(all_idx)

            offsets = df["window_offset"].values
            probs_arr = np.stack(
                [df[MODEL_REGISTRY[m]["prob_col"]].values for m in models],
                axis=1)
            canon_flag = df["is_constitutive_cpa"].values

            for i in all_idx:
                key = f"{gene}_{int(offsets[i])}"
                if gi["strand"] == "+":
                    start = gi["start"] + int(offsets[i])
                else:
                    start = gi["end"] - int(offsets[i]) - 500
                end = start + 500
                if bool(canon_flag[i]):
                    canon_keys.add(key)
                sampled_keys.append(key)
                sampled_probs.append(probs_arr[i])
                fout.write(f"{gi['chrom']}\t{start}\t{end}\t"
                           f"{key}\t0\t{gi['strand']}\n")

    n_sampled = len(sampled_keys)
    print(f"  sampled {n_sampled:,} windows ({len(canon_keys):,} canonical)")
    if n_sampled == 0:
        print(f"  no windows sampled; skipping")
        return

    if not (atlas_bed_path and os.path.exists(atlas_bed_path)):
        print(f"  atlas BED {atlas_bed_path} missing; skipping")
        return
    hits_path = sampled_bed_path + ".hits"
    subprocess.run(
        ["bedtools", "intersect", "-u", "-s",
         "-a", sampled_bed_path, "-b", atlas_bed_path],
        stdout=open(hits_path, "w"), check=True)
    atlas_keys = set()
    if os.path.getsize(hits_path) > 0:
        for line in open(hits_path):
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                atlas_keys.add(parts[3])
    print(f"  atlas-overlapping windows: {len(atlas_keys):,}")

    scores_by_model_class = {
        m: {"Canonical": [], "Atlas": [], "Background": []} for m in models}
    sampled_probs_arr = np.array(sampled_probs)
    for i, key in enumerate(sampled_keys):
        if key in canon_keys:
            cls = "Canonical"
        elif key in atlas_keys:
            cls = "Atlas"
        else:
            cls = "Background"
        for mi, m in enumerate(models):
            scores_by_model_class[m][cls].append(float(sampled_probs_arr[i, mi]))

    for p_ in [sampled_bed_path, hits_path]:
        try: os.remove(p_)
        except OSError: pass

    print(f"  class sizes:")
    for cls in ["Canonical", "Atlas", "Background"]:
        n = len(scores_by_model_class[models[0]][cls])
        print(f"    {cls:<12}: {n:,}")

    long_rows = []
    for m in models:
        disp = MODEL_REGISTRY[m]["display"]
        for cls in ["Canonical", "Atlas", "Background"]:
            for v in scores_by_model_class[m][cls]:
                long_rows.append({"Model": disp, "Class": cls, "score": v})
    if long_rows:
        _save_tsv(pd.DataFrame(long_rows), _tsv_path(out_path))
    thr_rows = [{"Model": MODEL_REGISTRY[m]["display"],
                 "threshold": float(thresholds[m])} for m in models]
    _save_tsv(pd.DataFrame(thr_rows), _tsv_path(out_path, "thresholds"))

    import matplotlib.pyplot as plt
    n_m = len(models)
    fig, axes = plt.subplots(1, n_m, figsize=(3.6 * n_m, 4.5), dpi=500,
                              sharey=True, facecolor="white")
    if n_m == 1:
        axes = [axes]
    fig.subplots_adjust(left=0.07, right=0.97, top=0.83, bottom=0.18,
                         wspace=0.15)
    classes = ["Canonical", "Atlas", "Background"]
    class_colors = {"Canonical": _CANON_RED, "Atlas": PAL["blue"],
                    "Background": PAL["gray"]}

    for ax, m in zip(axes, models):
        disp = MODEL_REGISTRY[m]["display"]
        color = MODEL_REGISTRY[m]["color"]
        data = [scores_by_model_class[m][c] for c in classes]
        ns = [len(d) for d in data]
        data_with_pos = [(i + 1, d) for i, d in enumerate(data) if len(d) > 0]
        if data_with_pos:
            positions = [pos for pos, _ in data_with_pos]
            arrs = [d for _, d in data_with_pos]
            parts = ax.violinplot(arrs, positions=positions,
                                    showmedians=True, showextrema=False,
                                    widths=0.7)
            for pc, (pos, _) in zip(parts["bodies"], data_with_pos):
                cls = classes[int(pos) - 1]
                pc.set_facecolor(class_colors[cls]); pc.set_alpha(0.7)
                pc.set_edgecolor("black"); pc.set_linewidth(0.4)
            parts["cmedians"].set_color("black")

        thr = thresholds[m]
        ax.axhline(thr, color=color, linestyle="--", linewidth=1.0, alpha=0.85)
        ax.text(3.45, thr, f" thr={thr:.3f}", fontsize=7, color=color,
                 va="center", ha="left")

        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(
            [f"{classes[i]}\nn={ns[i]:,}" for i in range(3)], fontsize=8)
        ax.set_title(disp, fontsize=11, color="#2C2C2A", pad=8)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0.4, 3.6)
        for s in ax.spines.values():
            s.set_color("black"); s.set_linewidth(0.6)
        ax.grid(axis="y", color="#D3D1C7", linewidth=0.3, alpha=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors="#444441", labelsize=9)

    axes[0].set_ylabel("Predicted probability", fontsize=10, color="#2C2C2A")
    fig.suptitle("Score distributions by window class",
                 fontsize=13, color="#2C2C2A", x=0.02, ha="left", y=0.97)
    fig.text(0.02, 0.91,
             f"~{sample_per_gene} windows sampled per gene plus every canonical "
             "window. Atlas = non-canonical window overlapping the atlas union. "
             "Dashed line = each model's threshold.",
             fontsize=9, color="#5F5E5A", ha="left")
    fig.savefig(out_path, dpi=500, bbox_inches="tight", facecolor="white",
                 pad_inches=0.3)
    plt.close(fig)
    print(f"  wrote {out_path}")


def cluster_all_models(merged_per_model, models, slack=250):
    by_gene = defaultdict(list)
    for m in models:
        df = merged_per_model.get(m, pd.DataFrame())
        if len(df) == 0:
            continue
        for _, row in df.iterrows():
            key = (row["gene"], row["strand"])
            by_gene[key].append({
                "start": int(row["start"]),
                "end":   int(row["end"]),
                "model": m,
                "chrom": row["chrom"],
            })

    clusters_all = []
    for (gene, strand), items in by_gene.items():
        items.sort(key=lambda x: x["start"])
        running = []
        for it in items:
            center = (it["start"] + it["end"]) // 2
            if running and (it["start"] - running[-1]["env_end"] <= slack):
                running[-1]["models"].add(it["model"])
                running[-1]["centers"].append(center)
                running[-1]["env_end"] = max(running[-1]["env_end"], it["end"])
            else:
                running.append({
                    "env_start": it["start"],
                    "env_end":   it["end"],
                    "models":    {it["model"]},
                    "centers":   [center],
                    "chrom":     it["chrom"],
                })
        for cl in running:
            rep = int(np.median(cl["centers"]))
            clusters_all.append({"chrom": cl["chrom"], "center": rep,
                                 "gene": gene, "strand": strand,
                                 "models": cl["models"]})
    return clusters_all


def extract_agreement_tiers(merged_per_model, models, three_prime_zone_bed,
                              out_dir, halfwidth=250, slack=250):
    print(f"\n[CPCCPA] Extracting agreement tiers")
    print(f"  models: {models}")
    print(f"  slack:  {slack} nt  halfwidth: {halfwidth} nt")
    print(f"  3'UTR zone BED: {three_prime_zone_bed}")
    if not os.path.exists(three_prime_zone_bed):
        raise FileNotFoundError(
            f"--three_prime_zone_bed not found: {three_prime_zone_bed}")

    clusters = cluster_all_models(merged_per_model, models, slack=slack)
    print(f"  total distinct clusters across all models: {len(clusters):,}")

    tier_map = {1: "tier1_exactly_one", 2: "tier2_exactly_two",
                3: "tier3_exactly_three"}
    tiers = defaultdict(list)
    for cl in clusters:
        k = len(cl["models"])
        if k in tier_map:
            tiers[tier_map[k]].append(cl)

    print(f"\n  tier sizes (before 3'UTR filter):")
    for k in sorted(tier_map):
        tname = tier_map[k]
        print(f"    {tname}: {len(tiers.get(tname, [])):,}")

    out = defaultdict(dict)
    for tname, recs in tiers.items():
        if not recs:
            continue
        with_path = os.path.join(out_dir, f"cpccpa_{tname}_with_3utr.bed")
        with open(with_path, "w") as f:
            for i, r in enumerate(recs):
                s = max(0, r["center"] - halfwidth)
                e = r["center"] + halfwidth
                models_str = ",".join(sorted(r["models"]))
                f.write(f"{r['chrom']}\t{s}\t{e}\t"
                        f"{tname}_{i}_{r['gene']}_{models_str}\t"
                        f"0\t{r['strand']}\n")
        subprocess.run(f"sort -k1,1 -k2,2n {with_path} -o {with_path}",
                       shell=True, check=True)

        no_path = os.path.join(out_dir, f"cpccpa_{tname}_no_3utr.bed")
        subprocess.run(
            ["bedtools", "intersect", "-v", "-s",
             "-a", with_path, "-b", three_prime_zone_bed],
            stdout=open(no_path, "w"), check=True)
        n_with = sum(1 for _ in open(with_path))
        n_no   = sum(1 for _ in open(no_path))
        pct_removed = (n_with - n_no) / max(n_with, 1) * 100
        print(f"    {tname}: with_3utr={n_with:,}  no_3utr={n_no:,}  "
              f"({n_with - n_no:,} = {pct_removed:.1f}% removed by 3'UTR filter)")
        out[tname]["with_3utr"] = with_path
        out[tname]["no_3utr"] = no_path

    return dict(out)


def _overlap_with_expression(sites_df, ref_bed, slack=50):
    if len(sites_df) == 0:
        return (np.array([], dtype=bool),
                np.array([], dtype=float),
                np.array([], dtype=float))

    sites_t = tempfile.NamedTemporaryFile(suffix=".bed", delete=False, mode="w")
    s = sites_df.reset_index(drop=True).copy()
    s["start"] = (s["start"] - slack).clip(lower=0)
    s["end"]   = s["end"] + slack
    if "strand" not in s.columns:
        s["strand"] = "+"
    s["score"] = 0
    s.assign(idx=np.arange(len(s)))[[
        "chrom", "start", "end", "idx", "score", "strand"
    ]].to_csv(sites_t.name, sep="\t", header=False, index=False)
    sites_t.close()

    res = subprocess.run(
        ["bedtools", "intersect", "-wa", "-wb", "-s",
         "-a", sites_t.name, "-b", ref_bed],
        stdout=subprocess.PIPE, check=True, text=True)
    os.remove(sites_t.name)

    hit_mask = np.zeros(len(s), dtype=bool)
    best_rpm = np.full(len(s), np.nan)
    best_pse = np.full(len(s), np.nan)
    for line in res.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 6 + 5:
            continue
        idx = int(parts[3])
        hit_mask[idx] = True
        try:
            rpm = float(parts[10])
        except (ValueError, IndexError):
            rpm = np.nan
        try:
            pse = float(parts[12]) if len(parts) > 12 else np.nan
        except (ValueError, IndexError):
            pse = np.nan
        if not np.isnan(rpm):
            if np.isnan(best_rpm[idx]) or rpm > best_rpm[idx]:
                best_rpm[idx] = rpm
                if not np.isnan(pse):
                    best_pse[idx] = pse
    return hit_mask, best_rpm, best_pse


def fig_cpccpa(class_beds, polyasite_bed, polyadb_main_bed, polyadb_max_bed,
                out_path, slack=50, title_suffix=""):
    print(f"\n[fig_cpccpa]{title_suffix}")
    refs = [("PolyASite",     polyasite_bed),
            ("PolyA_DB Main", polyadb_main_bed),
            ("PolyA_DB Max",  polyadb_max_bed)]
    refs = [(name, p) for name, p in refs if p and os.path.exists(p)]
    if not refs:
        print(f"  no reference BEDs available; skipping")
        return

    classes = [c for c in CPCCPA_CLASS_ORDER
               if c in class_beds and os.path.exists(class_beds[c])]
    if not classes:
        print(f"  no class BEDs found; skipping")
        return

    overlap_rows = []
    rpm_long, pse_long, detect_rows = [], [], []
    n_by_class = {}
    for cls in classes:
        sites = pd.DataFrame(read_bed(class_beds[cls]))
        n_by_class[cls] = len(sites)
        print(f"  {cls}: n={len(sites):,}")
        max_hit = None; max_rpm = None; max_pse = None
        for ref_label, ref_path in refs:
            hit, rpm, pse = _overlap_with_expression(sites, ref_path, slack=slack)
            rate = hit.mean() * 100 if len(hit) else 0
            overlap_rows.append({"Class": cls, "Reference": ref_label,
                                  "rate": rate, "n_hit": int(hit.sum()),
                                  "n_total": len(sites)})
            print(f"    vs {ref_label:<14}: {int(hit.sum()):>7,}/{len(sites):<7,} "
                  f"= {rate:5.1f}%")
            if ref_label == "PolyA_DB Max":
                max_hit, max_rpm, max_pse = hit, rpm, pse
        if max_hit is not None:
            rpm_vals = max_rpm[max_hit & ~np.isnan(max_rpm) & (max_rpm > 0)]
            pse_vals = max_pse[max_hit & ~np.isnan(max_pse) & (max_pse > 0)]
            for v in rpm_vals: rpm_long.append({"Class": cls, "rpm": float(v)})
            for v in pse_vals: pse_long.append({"Class": cls, "pse": float(v)})
            n_with_rpm = int((max_hit & ~np.isnan(max_rpm) & (max_rpm > 0)).sum())
            detect_rows.append({
                "Class": cls,
                "pct_with_expr": n_with_rpm / max(len(sites), 1) * 100,
                "n_with_expr": n_with_rpm,
                "n_total": len(sites),
            })

    if not overlap_rows:
        print(f"  no overlap data; skipping")
        return

    df_overlap = pd.DataFrame(overlap_rows)
    df_rpm     = pd.DataFrame(rpm_long, columns=["Class", "rpm"]) \
                    if not rpm_long else pd.DataFrame(rpm_long)
    df_pse     = pd.DataFrame(pse_long, columns=["Class", "pse"]) \
                    if not pse_long else pd.DataFrame(pse_long)
    df_detect  = pd.DataFrame(detect_rows)

    _save_tsv(df_overlap, _tsv_path(out_path, "overlap"))
    _save_tsv(df_rpm,     _tsv_path(out_path, "rpm"))
    _save_tsv(df_pse,     _tsv_path(out_path, "pse"))
    _save_tsv(df_detect,  _tsv_path(out_path, "detect"))

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(13, 14), dpi=500, facecolor="white")
    gs = GridSpec(3, 1, figure=fig, hspace=0.55,
                  height_ratios=[1.15, 1.0, 1.0])

    axA = fig.add_subplot(gs[0])
    n_classes = len(classes)
    n_refs = len(refs)
    bar_w = 0.85 / n_refs
    x = np.arange(n_classes)
    for ri, (ref_label, _) in enumerate(refs):
        vals = []
        for c in classes:
            v = df_overlap[(df_overlap["Class"] == c) &
                            (df_overlap["Reference"] == ref_label)]["rate"].values
            vals.append(float(v[0]) if len(v) else 0)
        offsets = x + (ri - (n_refs - 1) / 2) * bar_w
        axA.bar(offsets, vals, bar_w,
                color=CPCCPA_REF_COLORS.get(ref_label, "#888"),
                edgecolor="black", linewidth=0.4, label=ref_label)
        for xi, vi in zip(offsets, vals):
            axA.text(xi, vi + 1.5, f"{vi:.0f}", ha="center", va="bottom",
                     fontsize=7, color="#2C2C2A")
    axA.set_xticks(x)
    axA.set_xticklabels([CPCCPA_CLASS_LABEL[c] for c in classes], fontsize=9)
    axA.set_ylabel("Sites overlapping reference (%)", fontsize=10)
    axA.set_ylim(0, 112)
    axA.legend(title="Reference (escalating permissiveness)",
                frameon=False, fontsize=9, title_fontsize=9, loc="upper right")
    axA.set_title("A. Atlas overlap by model-agreement tier",
                  fontsize=12, loc="left", color="#2C2C2A", pad=8)
    for s in axA.spines.values(): s.set_color("black"); s.set_linewidth(0.6)
    axA.grid(axis="y", color="#D3D1C7", linewidth=0.3, alpha=0.6)
    axA.set_axisbelow(True)
    for i, c in enumerate(classes):
        axA.text(i, -0.10, f"n={n_by_class[c]:,}", ha="center", va="top",
                  fontsize=7, color="#5F5E5A",
                  transform=axA.get_xaxis_transform())

    axB = fig.add_subplot(gs[1])
    _draw_class_violins(axB, classes, df_rpm, "rpm",
                         transform=np.log10,
                         ylabel="log10(RPM) where overlaps v4 Max",
                         title="B. Expression among detected sites (PolyA_DB v4 Max RPM)",
                         detect_df=df_detect, n_by_class=n_by_class)

    axC = fig.add_subplot(gs[2])
    _draw_class_violins(axC, classes, df_pse, "pse",
                         transform=None,
                         ylabel="PSE (% of samples expressing) where overlaps v4 Max",
                         title="C. Detection breadth among detected sites (PolyA_DB v4 Max PSE)",
                         detect_df=df_detect, n_by_class=n_by_class)

    fig.suptitle(f"CPCCPA experimental usage{title_suffix}",
                 fontsize=14, color="#2C2C2A", x=0.02, ha="left", y=0.995)

    fig.savefig(out_path, dpi=500, bbox_inches="tight", facecolor="white",
                pad_inches=0.3)
    plt.close(fig)
    print(f"  wrote {out_path}")

    print(f"\n  Detection breadth (% of class sites with RPM>0 in v4 Max):")
    for r in detect_rows:
        print(f"    {r['Class']:<22} {r['n_with_expr']:>7,}/{r['n_total']:<7,} "
              f"= {r['pct_with_expr']:5.1f}%")


def _draw_class_violins(ax, classes, long_df, value_col, transform, ylabel,
                         title, detect_df, n_by_class):
    data = []; labels = []; colors = []; ns = []
    for c in classes:
        sub = long_df[long_df["Class"] == c][value_col].values
        if len(sub) == 0:
            data.append(np.array([np.nan]))
        else:
            if transform is not None:
                data.append(transform(sub))
            else:
                data.append(sub)
        labels.append(CPCCPA_CLASS_LABEL[c])
        colors.append(CPCCPA_CLASS_COLORS[c])
        ns.append(len(sub))

    valid_data = [(i, d) for i, d in enumerate(data)
                  if len(d) > 0 and not np.all(np.isnan(d))]
    if valid_data:
        positions = [i + 1 for i, _ in valid_data]
        d_filtered = [d for _, d in valid_data]
        parts = ax.violinplot(d_filtered, positions=positions,
                                showmedians=True, showextrema=False, widths=0.7)
        for pc, (i, _) in zip(parts["bodies"], valid_data):
            pc.set_facecolor(colors[i]); pc.set_alpha(0.75)
            pc.set_edgecolor("black"); pc.set_linewidth(0.5)
        parts["cmedians"].set_color("black")

    n_classes = len(classes)
    ax.set_xticks(np.arange(1, n_classes + 1))
    ax.set_xticklabels(
        [f"{labels[i]}\nn_detected={ns[i]:,}" for i in range(n_classes)],
        fontsize=8)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12, loc="left", color="#2C2C2A", pad=8)
    for s in ax.spines.values(): s.set_color("black"); s.set_linewidth(0.6)
    ax.grid(axis="y", color="#D3D1C7", linewidth=0.3, alpha=0.6)
    ax.set_axisbelow(True)

    detect_lookup = {r["Class"]: r for r in detect_df.to_dict("records")}
    ax_top = ax.get_ylim()[1]
    for i, c in enumerate(classes):
        r = detect_lookup.get(c)
        if r is None:
            continue
        ax.text(i + 1, ax_top * 1.02,
                f"{r['pct_with_expr']:.0f}% have expr",
                ha="center", va="bottom", fontsize=7, color="#5F5E5A")


def extract_3way_intragenic_agreement(merged_per_model, three_prime_zone_bed,
                                        out_dir, halfwidth=250, slack=250,
                                        target_models=("rf", "lr", "cnn")):
    target = list(target_models)
    if not all(m in merged_per_model for m in target):
        print(f"  [3-way Venn] requires {target}; not all present, skipping")
        return None

    print(f"\n[3-way Venn extraction] models: {target}")
    print(f"  filtering each model's peaks to intragenic (excluding 3'UTR zone)")

    def _filter_intragenic(df):
        if len(df) == 0:
            return df
        tmp = tempfile.NamedTemporaryFile(suffix=".bed", delete=False,
                                            mode="w").name
        sub = df.copy().reset_index(drop=True)
        sub["score"] = 0
        sub_out = sub.assign(idx=np.arange(len(sub)))[
            ["chrom", "start", "end", "idx", "score", "strand"]]
        sub_out.to_csv(tmp, sep="\t", header=False, index=False)
        subprocess.run(f"sort -k1,1 -k2,2n {tmp} -o {tmp}",
                       shell=True, check=True)
        out = tmp + ".intragenic"
        subprocess.run(
            ["bedtools", "intersect", "-v", "-s",
             "-a", tmp, "-b", three_prime_zone_bed],
            stdout=open(out, "w"), check=True)
        kept = set()
        if os.path.getsize(out) > 0:
            for line in open(out):
                p = line.rstrip("\n").split("\t")
                if len(p) >= 4:
                    kept.add(int(p[3]))
        os.remove(tmp); os.remove(out)
        return df.iloc[sorted(kept)].reset_index(drop=True)

    intragenic = {}
    for m in target:
        intragenic[m] = _filter_intragenic(merged_per_model[m])
        n_before = len(merged_per_model[m]); n_after = len(intragenic[m])
        print(f"    {MODEL_REGISTRY[m]['display']}: {n_before:,} -> "
              f"{n_after:,} intragenic ({n_before - n_after:,} removed)")

    clusters = cluster_all_models(intragenic, target, slack=slack)
    print(f"  total intragenic clusters across {len(target)} models: "
          f"{len(clusters):,}")

    sigs = defaultdict(list)
    for cl in clusters:
        key = tuple(sorted(cl["models"]))
        sigs[key].append(cl)

    region_order = [
        ("rf",), ("lr",), ("cnn",),
        ("lr", "rf"), ("cnn", "rf"), ("cnn", "lr"),
        ("cnn", "lr", "rf"),
    ]
    region_labels = {
        ("rf",):              "RF only",
        ("lr",):              "LR only",
        ("cnn",):             "CNN only",
        ("lr", "rf"):         "RF & LR only",
        ("cnn", "rf"):        "RF & CNN only",
        ("cnn", "lr"):        "LR & CNN only",
        ("cnn", "lr", "rf"):  "all 3 (RF & LR & CNN)",
    }
    print(f"\n  region sizes:")
    counts = {}
    for sig in region_order:
        counts[sig] = len(sigs.get(sig, []))
        print(f"    {region_labels[sig]:<25}: {counts[sig]:,}")

    all_three_path = os.path.join(out_dir, "cpccpa_3way_rflrcnn_intragenic.bed")
    recs = sigs.get(("cnn", "lr", "rf"), [])
    with open(all_three_path, "w") as f:
        for i, r in enumerate(recs):
            s = max(0, r["center"] - halfwidth)
            e = r["center"] + halfwidth
            f.write(f"{r['chrom']}\t{s}\t{e}\t"
                    f"3way_rflrcnn_{i}_{r['gene']}\t0\t{r['strand']}\n")
    if recs:
        subprocess.run(f"sort -k1,1 -k2,2n {all_three_path} -o {all_three_path}",
                       shell=True, check=True)
    print(f"\n  wrote {all_three_path}  ({len(recs):,} sites)")

    return {
        "counts": counts,
        "region_labels": region_labels,
        "region_order": region_order,
        "all_three_bed": all_three_path,
        "target_models": target,
    }


def fig_3way_venn(venn_dict, out_path):
    if venn_dict is None:
        print(f"\n[Figure 8] no Venn data; skipping")
        return
    print(f"\n[Figure 8] 3-way RF/LR/CNN agreement Venn (intragenic)")
    counts       = venn_dict["counts"]
    region_order = venn_dict["region_order"]
    total = sum(counts.values())

    venn_rows = []
    for sig in region_order:
        n = counts.get(sig, 0)
        venn_rows.append({
            "region": "+".join(sig),
            "n_models": len(sig),
            "count": n,
            "frac_of_total": n / max(total, 1),
        })
    _save_tsv(pd.DataFrame(venn_rows), _tsv_path(out_path))

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8), dpi=500, facecolor="white")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")

    r = 0.25
    centers = {"rf": (0.50, 0.62), "lr": (0.34, 0.36), "cnn": (0.66, 0.36)}
    set_colors = {"rf": MODEL_REGISTRY["rf"]["color"],
                   "lr": MODEL_REGISTRY["lr"]["color"],
                   "cnn": MODEL_REGISTRY["cnn"]["color"]}
    set_label = {"rf": "RF", "lr": "LR", "cnn": "CNN"}
    set_label_pos = {"rf":  (0.50, 0.94),
                     "lr":  (0.08, 0.22),
                     "cnn": (0.92, 0.22)}

    for k, c in centers.items():
        ax.add_patch(Circle(c, r, facecolor=set_colors[k],
                             edgecolor="black", linewidth=1.0, alpha=0.40))
    for k, p in set_label_pos.items():
        ax.text(p[0], p[1], set_label[k], fontsize=15, ha="center", va="center",
                color=set_colors[k], fontweight="bold")

    region_pos = {
        ("rf",):           (0.50, 0.78),
        ("lr",):           (0.18, 0.28),
        ("cnn",):          (0.82, 0.28),
        ("lr", "rf"):       (0.30, 0.50),
        ("cnn", "rf"):      (0.70, 0.50),
        ("cnn", "lr"):      (0.50, 0.26),
        ("cnn", "lr", "rf"): (0.50, 0.45),
    }
    for sig in region_order:
        n = counts.get(sig, 0)
        pct = n / max(total, 1) * 100
        label = f"{n:,}\n({pct:.1f}%)"
        p = region_pos[sig]
        if sig == ("cnn", "lr", "rf"):
            ax.text(p[0], p[1], label, fontsize=12, ha="center", va="center",
                    color="#2C2C2A", fontweight="bold")
        else:
            ax.text(p[0], p[1], label, fontsize=10, ha="center", va="center",
                    color="#2C2C2A")

    fig.suptitle(f"RF / LR / CNN agreement on intragenic CPA peaks\n"
                 f"(total clusters: {total:,})",
                 fontsize=12, color="#2C2C2A", x=0.50, ha="center", y=0.97)
    fig.savefig(out_path, dpi=500, bbox_inches="tight", facecolor="white",
                pad_inches=0.3)
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_motif_density(canonical_bed, cpccpa_bed, reference_fasta, out_path,
                       win_up=100, win_dn=40, smooth_window=5,
                       cpccpa_label="CPCCPA (RF+LR+CNN intragenic)"):
    print(f"\n[Figure 11] Motif density at cut sites (canonical vs CPCCPA)")
    if not (reference_fasta and os.path.exists(reference_fasta)):
        print(f"  reference FASTA missing; skipping")
        return
    if not os.path.exists(canonical_bed):
        print(f"  canonical BED missing; skipping")
        return
    if not os.path.exists(cpccpa_bed):
        print(f"  CPCCPA BED missing; skipping")
        return

    try:
        import pysam
    except ImportError:
        print(f"  pysam not available; skipping")
        return

    fa = pysam.FastaFile(reference_fasta)
    fa_refs = set(fa.references)

    def fetch_window(chrom, center, strand):
        c = chrom if chrom in fa_refs else chrom.lstrip("chr")
        if c not in fa_refs:
            return None
        if strand == "+":
            start = center - win_up
            end   = center + win_dn + 1
        else:
            start = center - win_dn
            end   = center + win_up + 1
        if start < 0:
            return None
        try:
            seq = fa.fetch(c, start, end).upper()
        except Exception:
            return None
        if len(seq) != (win_up + win_dn + 1):
            return None
        if strand == "-":
            seq = seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]
        return seq

    def get_sequences(bed_path):
        seqs = []; skipped = 0
        for row in read_bed(bed_path):
            center = (row["start"] + row["end"]) // 2
            s = fetch_window(row["chrom"], center, row["strand"])
            if s is None:
                skipped += 1; continue
            seqs.append(s)
        return seqs, skipped

    canon_seqs,  canon_skip  = get_sequences(canonical_bed)
    cpccpa_seqs, cpccpa_skip = get_sequences(cpccpa_bed)
    print(f"  canonical: {len(canon_seqs):,} usable seqs "
          f"({canon_skip:,} skipped, edge/missing chrom)")
    print(f"  CPCCPA:    {len(cpccpa_seqs):,} usable seqs "
          f"({cpccpa_skip:,} skipped)")

    if not canon_seqs or not cpccpa_seqs:
        print(f"  not enough sequences; skipping")
        return

    L = win_up + win_dn + 1
    positions = np.arange(L) - win_up

    PAS_VARIANTS = ["AATAAA", "ATTAAA", "AGTAAA", "TATAAA",
                    "CATAAA", "GATAAA", "AATATA", "AATACA",
                    "AATAGA", "ACTAAA", "AATGAA", "AAGAAA"]

    def density_motif(seqs, motifs):
        arr = np.zeros(L)
        if not seqs:
            return arr
        for s in seqs:
            for m in motifs:
                idx = 0
                while True:
                    p = s.find(m, idx)
                    if p == -1 or p >= L:
                        break
                    arr[p] += 1
                    idx = p + 1
        arr /= len(seqs)
        return arr

    def freq_base(seqs, base):
        arr = np.zeros(L)
        if not seqs:
            return arr
        for s in seqs:
            n = min(len(s), L)
            for i in range(n):
                if s[i] == base:
                    arr[i] += 1
        arr /= len(seqs)
        return arr

    def freq_dinuc(seqs, dinucs):
        arr = np.zeros(L)
        if not seqs:
            return arr
        for s in seqs:
            n = min(len(s) - 1, L - 1)
            for i in range(n):
                if s[i:i + 2] in dinucs:
                    arr[i] += 1
        arr /= len(seqs)
        return arr

    def smooth(arr, w):
        if w <= 1:
            return arr
        kernel = np.ones(w) / w
        return np.convolve(arr, kernel, mode="same")

    metrics = [
        ("A. PAS hexamer (AAUAAA + variants)",
         smooth(density_motif(canon_seqs,  PAS_VARIANTS), smooth_window),
         smooth(density_motif(cpccpa_seqs, PAS_VARIANTS), smooth_window)),
        ("B. UGUA (CFIm motif)",
         smooth(density_motif(canon_seqs,  ["TGTA"]), smooth_window),
         smooth(density_motif(cpccpa_seqs, ["TGTA"]), smooth_window)),
        ("C. U content (T frequency)",
         smooth(freq_base(canon_seqs,  "T"), smooth_window),
         smooth(freq_base(cpccpa_seqs, "T"), smooth_window)),
        ("D. GU / UG dinucleotide content",
         smooth(freq_dinuc(canon_seqs,  {"GT", "TG"}), smooth_window),
         smooth(freq_dinuc(cpccpa_seqs, {"GT", "TG"}), smooth_window)),
    ]

    long_rows = []
    for label, canon_arr, cpccpa_arr in metrics:
        for pos, v_can, v_cpc in zip(positions, canon_arr, cpccpa_arr):
            long_rows.append({"panel": label, "set": "Canonical",
                              "n_seqs": len(canon_seqs),
                              "position": int(pos), "value": float(v_can)})
            long_rows.append({"panel": label, "set": cpccpa_label,
                              "n_seqs": len(cpccpa_seqs),
                              "position": int(pos), "value": float(v_cpc)})
    _save_tsv(pd.DataFrame(long_rows), _tsv_path(out_path))

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=500,
                              sharex=True, facecolor="white")
    fig.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.10,
                         wspace=0.20, hspace=0.32)
    color_canon  = _CANON_RED
    color_cpccpa = "#7E2C12"

    for ax, (label, canon, cpccpa) in zip(axes.flat, metrics):
        ax.plot(positions, canon, color=color_canon, lw=1.6,
                 label=f"Canonical (n={len(canon_seqs):,})")
        ax.plot(positions, cpccpa, color=color_cpccpa, lw=1.6,
                 label=f"{cpccpa_label} (n={len(cpccpa_seqs):,})")
        ax.axvline(0, color="#5F5E5A", linestyle="--", linewidth=0.7,
                    alpha=0.7)
        ax.set_title(label, fontsize=11, color="#2C2C2A", loc="left", pad=6)
        ax.set_xlabel("Position relative to cut site (nt)",
                      fontsize=9, color="#2C2C2A")
        ax.set_ylabel("Per-sequence frequency", fontsize=9, color="#2C2C2A")
        ax.set_xlim(-win_up, win_dn)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        for s in ax.spines.values():
            s.set_color("black"); s.set_linewidth(0.6)
        ax.grid(color="#D3D1C7", linewidth=0.3, alpha=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors="#444441", labelsize=8)

    fig.suptitle("Canonical CPA sequence motifs at canonical vs CPCCPA cut sites",
                 fontsize=13, color="#2C2C2A", x=0.02, ha="left", y=0.97)
    fig.text(0.02, 0.925,
             f"Per-position occurrence rate in a [-{win_up}, +{win_dn}] nt "
             f"window around each cut site (vertical line). "
             f"Smoothed with a {smooth_window}-nt running average.",
             fontsize=9, color="#5F5E5A", ha="left")
    fig.savefig(out_path, dpi=500, bbox_inches="tight", facecolor="white",
                 pad_inches=0.3)
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_threshold_robustness(scan_dir, pc_genes_bed, canonical_bed,
                              three_prime_zone_bed, atlas_bed,
                              models, out_dir, out_path,
                              vicinity_nt=500,
                              percentiles=(50, 60, 70, 75, 80, 85, 90, 95, 99),
                              cache_tag="", force=False,
                              atlas_label="PolyA_DB v4 Max"):
    print(f"\n[Figure 13] Threshold-robustness heatmap")
    print(f"  atlas reference: {atlas_bed} ({atlas_label})")
    print(f"  threshold percentiles: {list(percentiles)}")

    canon = pd.DataFrame(read_bed(canonical_bed))
    canon["gene"] = canon["name"].str.split("::").str[0]
    parquets = {os.path.basename(p).replace(".parquet", ""): p
                for p in glob.glob(os.path.join(scan_dir, "per_gene", "*.parquet"))}
    prob_cols = [MODEL_REGISTRY[m]["prob_col"] for m in models]
    canon_probs = {m: [] for m in models}

    print(f"  collecting canonical-vicinity probs from {len(canon):,} canonical sites...")
    for _, row in canon.iterrows():
        pq = parquets.get(row["gene"])
        if pq is None:
            continue
        try:
            df = pd.read_parquet(pq, columns=prob_cols +
                                  ["is_constitutive_cpa", "window_offset"])
        except Exception:
            continue
        ci_arr = df.index[df["is_constitutive_cpa"]]
        if len(ci_arr) == 0:
            continue
        ci = ci_arr[0]
        canon_off = int(df["window_offset"].iloc[ci])
        offsets = df["window_offset"].values
        near = np.abs(offsets - canon_off) <= vicinity_nt
        if not near.any():
            continue
        for m in models:
            col = MODEL_REGISTRY[m]["prob_col"]
            canon_probs[m].append(float(df[col].values[near].max()))

    threshold_grid = {}
    print(f"\n  thresholds (probability values per percentile):")
    for m in models:
        vals = np.array(canon_probs[m])
        if len(vals) == 0:
            print(f"    {MODEL_REGISTRY[m]['display']}: no canonical probs; skipping")
            continue
        threshold_grid[m] = np.percentile(vals, list(percentiles))
        disp = MODEL_REGISTRY[m]["display"]
        line = ", ".join(f"p{p}={t:.3f}"
                          for p, t in zip(percentiles, threshold_grid[m]))
        print(f"    {disp:<10}: {line}")

    models_with_data = [m for m in models if m in threshold_grid]
    if not models_with_data:
        print(f"  no thresholds available; skipping")
        return
    n_models = len(models_with_data)
    n_pct = len(percentiles)
    overlap_grid = np.full((n_models, n_pct), np.nan)
    count_grid   = np.full((n_models, n_pct), 0.0)

    pc_df = pd.DataFrame(read_bed(pc_genes_bed)).rename(columns={"name": "gene"})
    pc_df = (pc_df.assign(span=pc_df["end"] - pc_df["start"])
                   .sort_values("span", ascending=False)
                   .drop_duplicates("gene", keep="first")
                   .drop(columns=["span"]))
    pc_by_gene = pc_df.set_index("gene")[["chrom", "start", "end", "strand"]].to_dict("index")

    for mi, m in enumerate(models_with_data):
        loose_thr = float(threshold_grid[m][0])
        disp = MODEL_REGISTRY[m]["display"]
        cache_path = os.path.join(
            out_dir,
            f"_cache_allwin_{m}_loose{loose_thr:.4f}"
            f"{'_'+cache_tag if cache_tag else ''}.bed")

        if force or not os.path.exists(cache_path):
            print(f"\n  scanning parquets for {disp} (loose thr {loose_thr:.4f})...")
            col = MODEL_REGISTRY[m]["prob_col"]
            n_written = 0
            with open(cache_path, "w") as fout:
                for i, (gene, pq_path) in enumerate(parquets.items()):
                    gi = pc_by_gene.get(gene)
                    if gi is None:
                        continue
                    try:
                        df = pd.read_parquet(pq_path,
                                              columns=[col, "window_offset"])
                    except Exception:
                        continue
                    probs = df[col].values
                    hi = probs >= loose_thr
                    if not hi.any():
                        continue
                    offsets = df["window_offset"].values[hi]
                    hi_probs = probs[hi]
                    if gi["strand"] == "+":
                        starts = gi["start"] + offsets
                    else:
                        starts = gi["end"] - offsets - 500
                    for s, p in zip(starts, hi_probs):
                        fout.write(f"{gi['chrom']}\t{int(s)}\t{int(s)+500}\t"
                                    f".\t{p:.4f}\t{gi['strand']}\n")
                        n_written += 1
                    if (i + 1) % 3000 == 0:
                        print(f"    {i+1:,}/{len(parquets):,} parquets")
            subprocess.run(f"sort -k1,1 -k2,2n {cache_path} -o {cache_path}",
                           shell=True, check=True)
            print(f"    wrote {cache_path} ({n_written:,} windows)")
        else:
            n_written = sum(1 for _ in open(cache_path))
            print(f"\n  [cache hit] {disp}: {cache_path} ({n_written:,} windows)")

        intragenic_path = cache_path + ".intragenic"
        if force or not os.path.exists(intragenic_path):
            subprocess.run(
                ["bedtools", "intersect", "-v", "-s",
                 "-a", cache_path, "-b", three_prime_zone_bed],
                stdout=open(intragenic_path, "w"), check=True)
        n_intragenic = sum(1 for _ in open(intragenic_path))
        print(f"    intragenic windows: {n_intragenic:,}")

        for pi, p_pct in enumerate(percentiles):
            thr = float(threshold_grid[m][pi])
            filt_path = intragenic_path + f".p{p_pct}"
            n_above = 0
            with open(intragenic_path) as fin, open(filt_path, "w") as fout:
                for line in fin:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) >= 5 and float(parts[4]) >= thr:
                        fout.write(line)
                        n_above += 1
            if n_above == 0:
                rate = 0.0
                n_hit = 0
            else:
                res = subprocess.run(
                    ["bedtools", "intersect", "-u", "-s",
                     "-a", filt_path, "-b", atlas_bed],
                    stdout=subprocess.PIPE, check=True, text=True)
                n_hit = sum(1 for l in res.stdout.splitlines() if l.strip())
                rate = n_hit / max(n_above, 1) * 100
            overlap_grid[mi, pi] = rate
            count_grid[mi, pi]   = n_above
            print(f"    p{p_pct:>3} thr={thr:.4f}: {n_above:>9,} intragenic, "
                  f"{n_hit:>7,} overlap atlas = {rate:5.2f}%")
            os.remove(filt_path)

    tr_rows = []
    for mi, m in enumerate(models_with_data):
        disp = MODEL_REGISTRY[m]["display"]
        for pi, p_pct in enumerate(percentiles):
            tr_rows.append({
                "Model": disp,
                "percentile": int(p_pct),
                "threshold": float(threshold_grid[m][pi]),
                "n_intragenic": int(count_grid[mi, pi]),
                "atlas_overlap_pct": float(overlap_grid[mi, pi])
                    if not np.isnan(overlap_grid[mi, pi]) else None,
                "atlas_label": atlas_label,
            })
    _save_tsv(pd.DataFrame(tr_rows), _tsv_path(out_path))

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(max(8, 0.85 * n_pct + 3), 7),
                              dpi=500, facecolor="white")
    fig.subplots_adjust(left=0.13, right=0.97, top=0.86, bottom=0.10,
                         hspace=0.55)

    model_labels = [MODEL_REGISTRY[m]["display"] for m in models_with_data]
    col_labels = [f"p{p}" for p in percentiles]

    vmax_a = max(np.nanmax(overlap_grid), 30.0)
    im1 = axes[0].imshow(overlap_grid, aspect="auto", cmap="RdYlBu_r",
                          vmin=0, vmax=vmax_a)
    axes[0].set_xticks(np.arange(n_pct))
    axes[0].set_xticklabels(col_labels, fontsize=9)
    axes[0].set_yticks(np.arange(n_models))
    axes[0].set_yticklabels(model_labels, fontsize=10)
    axes[0].set_title(f"A. Atlas overlap rate (%) of intragenic "
                       f"above-threshold windows  -  ref: {atlas_label}",
                       fontsize=11, color="#2C2C2A", loc="left", pad=8)
    for i in range(n_models):
        for j in range(n_pct):
            v = overlap_grid[i, j]
            if np.isnan(v):
                continue
            txt_color = "white" if v > vmax_a * 0.55 else "#2C2C2A"
            axes[0].text(j, i, f"{v:.1f}", ha="center", va="center",
                         fontsize=8, color=txt_color)
    cbar1 = plt.colorbar(im1, ax=axes[0], shrink=0.85, pad=0.02)
    cbar1.set_label("% overlapping atlas", fontsize=9, color="#2C2C2A")
    cbar1.ax.tick_params(labelsize=8)

    log_counts = np.log10(np.maximum(count_grid, 1))
    im2 = axes[1].imshow(log_counts, aspect="auto", cmap="Blues",
                          vmin=0, vmax=max(log_counts.max(), 1))
    axes[1].set_xticks(np.arange(n_pct))
    axes[1].set_xticklabels(col_labels, fontsize=9)
    axes[1].set_yticks(np.arange(n_models))
    axes[1].set_yticklabels(model_labels, fontsize=10)
    axes[1].set_title("B. log10(# intragenic above-threshold windows) per (model, percentile)",
                       fontsize=11, color="#2C2C2A", loc="left", pad=8)
    for i in range(n_models):
        for j in range(n_pct):
            n = int(count_grid[i, j])
            if n == 0:
                continue
            txt_color = "white" if log_counts[i, j] > log_counts.max() * 0.65 else "#2C2C2A"
            label = f"{n:,}" if n < 10000 else f"{n/1000:.0f}k"
            axes[1].text(j, i, label, ha="center", va="center",
                         fontsize=8, color=txt_color)
    cbar2 = plt.colorbar(im2, ax=axes[1], shrink=0.85, pad=0.02)
    cbar2.set_label("log10(count)", fontsize=9, color="#2C2C2A")
    cbar2.ax.tick_params(labelsize=8)
    axes[1].set_xlabel("Threshold (percentile of canonical-CPA probabilities)",
                       fontsize=10, color="#2C2C2A")

    fig.suptitle(f"Threshold robustness  -  CPCCPA atlas overlap stays low at any threshold",
                 fontsize=13, color="#2C2C2A", x=0.02, ha="left", y=0.97)
    fig.text(0.02, 0.925,
             "Each row sweeps that model's threshold across percentiles of its "
             "canonical-CPA score distribution (p50 = median, p99 = top 1%). "
             "If CPCCPA were a thresholding artifact, atlas overlap in Panel A "
             "would rise toward 100% as we move right; instead it stays low.",
             fontsize=9, color="#5F5E5A", ha="left")
    fig.savefig(out_path, dpi=500, bbox_inches="tight", facecolor="white",
                pad_inches=0.3)
    plt.close(fig)
    print(f"\n  wrote {out_path}")


def _map_sites_to_genes(sites_bed, pc_genes_bed):
    sorted_sites = tempfile.NamedTemporaryFile(
        suffix=".bed", delete=False, mode="w").name
    subprocess.run(f"sort -k1,1 -k2,2n {sites_bed} > {sorted_sites}",
                   shell=True, check=True)
    tmp = tempfile.NamedTemporaryFile(suffix=".bed", delete=False, mode="w").name
    subprocess.run(
        ["bedtools", "intersect", "-wa", "-wb", "-s",
         "-a", sorted_sites, "-b", pc_genes_bed],
        stdout=open(tmp, "w"), check=True)
    mapping = {}
    with open(tmp) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 10:
                continue
            site_name = parts[3]
            gene_name = parts[9]
            if site_name not in mapping:
                mapping[site_name] = gene_name
    os.remove(tmp); os.remove(sorted_sites)
    return mapping


def fig_score_heatmap(scan_dir, pc_genes_bed, sites_bed, models, out_path,
                       thresholds=None, flank_nt=500, max_sites=None,
                       title="", site_label="sites", cmap="viridis"):
    target_models = [m for m in ["rf", "lr", "cnn"] if m in models]
    if not target_models:
        print(f"  heatmap needs at least one of RF/LR/CNN; skipping")
        return

    print(f"\n[Figure 13] {title}")
    print(f"  flank: +/- {flank_nt} nt")

    site_to_gene = _map_sites_to_genes(sites_bed, pc_genes_bed)
    print(f"  mapped {len(site_to_gene):,} sites to genes")

    sites = read_bed(sites_bed)
    sites_in_genes = [s for s in sites if s["name"] in site_to_gene]
    print(f"  {len(sites):,} total sites; {len(sites_in_genes):,} fall in a gene")
    if max_sites and len(sites_in_genes) > max_sites:
        rng = np.random.default_rng(42)
        idx = sorted(rng.choice(len(sites_in_genes), size=max_sites,
                                 replace=False))
        sites_in_genes = [sites_in_genes[i] for i in idx]
        print(f"  subsampled to {len(sites_in_genes):,} (max_sites cap)")
    if not sites_in_genes:
        print(f"  no usable sites; skipping")
        return

    test_pq = sorted(glob.glob(os.path.join(scan_dir, "per_gene", "*.parquet")))[:1]
    if not test_pq:
        print(f"  no parquets; skipping")
        return
    sample_offs = pd.read_parquet(
        test_pq[0], columns=["window_offset"])["window_offset"].values
    diffs = np.diff(np.sort(np.unique(sample_offs)))
    stride = int(np.median(diffs[diffs > 0])) if len(diffs) > 0 else 50
    print(f"  detected scan stride: {stride} nt")

    positions = np.arange(-flank_nt, flank_nt + 1, stride)
    n_cols    = len(positions)

    pc_df = pd.DataFrame(read_bed(pc_genes_bed)).rename(columns={"name": "gene"})
    pc_df = (pc_df.assign(span=pc_df["end"] - pc_df["start"])
                   .sort_values("span", ascending=False)
                   .drop_duplicates("gene", keep="first")
                   .drop(columns=["span"]))
    pc_by_gene = pc_df.set_index("gene")[
        ["start", "end", "strand"]].to_dict("index")

    sites_by_gene = defaultdict(list)
    for s in sites_in_genes:
        sites_by_gene[site_to_gene[s["name"]]].append(s)

    rows_per_model = {m: [] for m in target_models}
    prob_cols = [MODEL_REGISTRY[m]["prob_col"] for m in target_models]
    needed = prob_cols + ["window_offset"]

    print(f"  reading {len(sites_by_gene):,} per-gene parquets...")
    for i, (gene, gene_sites) in enumerate(sites_by_gene.items()):
        pq_path = os.path.join(scan_dir, "per_gene", f"{gene}.parquet")
        if not os.path.exists(pq_path):
            continue
        try:
            df = pd.read_parquet(pq_path, columns=needed)
        except Exception:
            continue
        if len(df) == 0:
            continue
        gi = pc_by_gene.get(gene)
        if gi is None:
            continue

        offsets = df["window_offset"].values
        offset_to_idx = {int(o): k for k, o in enumerate(offsets)}
        prob_arr = np.stack(
            [df[MODEL_REGISTRY[m]["prob_col"]].values for m in target_models],
            axis=1)

        for s in gene_sites:
            site_center = (s["start"] + s["end"]) // 2
            if gi["strand"] == "+":
                tx_pos = site_center - gi["start"]
            else:
                tx_pos = gi["end"] - site_center
            target_window_offset = tx_pos - 250

            for mi, m in enumerate(target_models):
                row = np.full(n_cols, np.nan)
                for j, rel_off in enumerate(positions):
                    target = target_window_offset + rel_off
                    k = offset_to_idx.get(int(target))
                    if k is not None:
                        row[j] = prob_arr[k, mi]
                rows_per_model[m].append(row)

        if (i + 1) % 2000 == 0:
            print(f"    {i+1:,}/{len(sites_by_gene):,} parquets done")

    matrices = {}
    for m in target_models:
        if not rows_per_model[m]:
            print(f"  no rows for {m}; skipping")
            return
        matrices[m] = np.asarray(rows_per_model[m])

    n_sites_used = matrices[target_models[0]].shape[0]
    print(f"  built matrices: {n_sites_used:,} sites × {n_cols} positions × "
          f"{len(target_models)} models")

    center_idx = n_cols // 2
    cut_probs = np.nanmean(
        np.stack([matrices[m][:, center_idx] for m in target_models], axis=1),
        axis=1)
    sort_order = np.argsort(-cut_probs)
    matrices = {m: matrices[m][sort_order] for m in target_models}

    long_rows = []
    for m in target_models:
        disp = MODEL_REGISTRY[m]["display"]
        mat = matrices[m]
        for row_idx in range(mat.shape[0]):
            for col_idx, pos in enumerate(positions):
                v = mat[row_idx, col_idx]
                if np.isnan(v):
                    continue
                long_rows.append({
                    "Model": disp, "site_idx": int(row_idx),
                    "position": int(pos), "prob": float(v),
                })
    if long_rows:
        _save_tsv(pd.DataFrame(long_rows), _tsv_path(out_path))
    meta_rows = []
    for m in target_models:
        disp = MODEL_REGISTRY[m]["display"]
        mean_sig = np.nanmean(matrices[m], axis=0)
        thr = thresholds[m] if (thresholds and m in thresholds) else None
        for pos, v in zip(positions, mean_sig):
            meta_rows.append({
                "Model": disp, "position": int(pos),
                "mean_prob": float(v) if not np.isnan(v) else None,
                "threshold": float(thr) if thr is not None else None,
            })
    _save_tsv(pd.DataFrame(meta_rows), _tsv_path(out_path, "metaprofile"))

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    n_m = len(target_models)
    fig = plt.figure(figsize=(3 * n_m + 1.6, 10), dpi=300, facecolor="white")
    gs = GridSpec(2, n_m + 1, figure=fig,
                   height_ratios=[1, 7],
                   width_ratios=[1] * n_m + [0.07],
                   hspace=0.05, wspace=0.12)

    im = None
    for i, m in enumerate(target_models):
        ax_meta = fig.add_subplot(gs[0, i])
        ax_heat = fig.add_subplot(gs[1, i])
        mat   = matrices[m]
        color = MODEL_REGISTRY[m]["color"]
        disp  = MODEL_REGISTRY[m]["display"]

        mean_sig = np.nanmean(mat, axis=0)
        ax_meta.plot(positions, mean_sig, color=color, lw=1.5)
        ax_meta.fill_between(positions, 0, mean_sig, color=color, alpha=0.30)
        if thresholds and m in thresholds:
            ax_meta.axhline(thresholds[m], color="#5F5E5A", linestyle="--",
                             linewidth=0.6, alpha=0.7)
        ax_meta.set_xlim(positions[0], positions[-1])
        ax_meta.set_ylim(0, 1.0)
        ax_meta.axvline(0, color="#888780", linewidth=0.5, alpha=0.7)
        ax_meta.set_xticks([])
        ax_meta.set_title(disp, color=color, fontsize=12, fontweight="bold")
        if i == 0:
            ax_meta.set_ylabel("Mean\nprob", fontsize=8, color="#2C2C2A")
        else:
            ax_meta.set_yticklabels([])
        for s in ax_meta.spines.values():
            s.set_color("black"); s.set_linewidth(0.5)
        ax_meta.tick_params(colors="#444441", labelsize=7)

        im = ax_heat.imshow(
            mat, aspect="auto", cmap=cmap, vmin=0, vmax=1,
            extent=[positions[0], positions[-1], n_sites_used, 0],
            interpolation="nearest")
        ax_heat.axvline(0, color="white", linewidth=0.5, alpha=0.6)
        ax_heat.set_xlabel("nt from cut site", fontsize=9, color="#2C2C2A")
        if i == 0:
            ax_heat.set_ylabel(f"{site_label} (sorted by mean center prob)",
                                fontsize=9, color="#2C2C2A")
        else:
            ax_heat.set_yticklabels([])
        ax_heat.tick_params(colors="#444441", labelsize=7)
        for s in ax_heat.spines.values():
            s.set_color("black"); s.set_linewidth(0.5)

    if im is not None:
        cbar_ax = fig.add_subplot(gs[1, n_m])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("Predicted CPA probability", fontsize=9, color="#2C2C2A")
        cbar.ax.tick_params(labelsize=7, colors="#444441")
        if thresholds:
            for m in target_models:
                if m in thresholds:
                    cbar.ax.axhline(thresholds[m],
                                     color=MODEL_REGISTRY[m]["color"],
                                     linewidth=2.0)

    fig.suptitle(title, fontsize=13, color="#2C2C2A", x=0.02, ha="left", y=0.97)
    thr_str = ""
    if thresholds:
        thr_str = "  Thresholds: " + ", ".join(
            f"{MODEL_REGISTRY[m]['display']}={thresholds[m]:.3f}"
            for m in target_models if m in thresholds)
    fig.text(0.02, 0.935,
              f"n={n_sites_used:,} sites • ±{flank_nt} nt window at stride "
              f"{stride} nt • rows sorted high-to-low by mean center-window "
              f"probability." + thr_str,
              fontsize=8, color="#5F5E5A", ha="left")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white",
                 pad_inches=0.3)
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_saliency(scan_dir, pc_genes_bed, sites_bed, reference_fasta,
                  models, genes, out_path,
                  rf_run_dir=None, lr_run_dir=None, cnn_run_dir=None,
                  window_radius=120, site_label="canonical",
                  aparent2_saliency_parquet=None,
                  max_sites_per_gene=1):
    print(f"\n[Figure 12] Saliency at {site_label} CPAs")
    if not genes:
        print(f"  no genes specified; skipping")
        return
    if not reference_fasta or not os.path.exists(reference_fasta):
        print(f"  reference FASTA missing; skipping")
        return

    feasible = []
    if "lr"  in models and lr_run_dir  and os.path.exists(lr_run_dir):  feasible.append("lr")
    if "cnn" in models and cnn_run_dir and os.path.exists(cnn_run_dir): feasible.append("cnn")
    if not feasible:
        print(f"  no feasible models for saliency; skipping")
        return
    skipped = [MODEL_REGISTRY[m]["display"] for m in models if m not in feasible]
    if skipped:
        print(f"  saliency skipped for: {skipped}")

    try:
        import pysam
        fa = pysam.FastaFile(reference_fasta)
        fa_refs = set(fa.references)
        def fetch(chrom, s, e, strand):
            c = chrom if chrom in fa_refs else chrom.lstrip("chr")
            if c not in fa_refs: return None
            seq = fa.fetch(c, s, e).upper()
            if strand == "-":
                seq = seq.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]
            return seq
    except Exception as e:
        print(f"  failed to open FASTA: {e}; skipping")
        return

    sites_all = pd.DataFrame(read_bed(sites_bed))
    sites_all["gene"] = sites_all["name"].str.split("::").str[0]
    if site_label != "canonical":
        def _gene_from_cpccpa_name(n):
            parts = n.split("_")
            if len(parts) >= 6 and parts[0].startswith("tier"):
                return parts[4]
            return parts[0]
        sites_all["gene"] = sites_all["name"].map(_gene_from_cpccpa_name)

    sites_use = sites_all[sites_all["gene"].isin(genes)].copy()
    if max_sites_per_gene > 0:
        sites_use = (sites_use.groupby("gene", group_keys=False)
                                .head(max_sites_per_gene))
    print(f"  using {len(sites_use):,} sites across {sites_use['gene'].nunique()} genes")

    gene_windows = []
    for _, ci in sites_use.iterrows():
        mid = (ci["start"] + ci["end"]) // 2
        strand = ci["strand"]
        chrom = ci["chrom"]
        site_id = f"{ci['gene']}:{chrom}:{mid}:{strand}"
        w_start = max(0, mid - window_radius)
        w_end   = mid + window_radius + 1
        seq = fetch(chrom, w_start, w_end, strand)
        if seq is None or len(seq) != (w_end - w_start):
            print(f"  WARN: bad fetch for {site_id}; skipping")
            continue
        if strand == "+":
            canon_local = mid - w_start
        else:
            canon_local = (w_end - 1) - mid
        gene_windows.append({"gene": ci["gene"], "site_id": site_id,
                             "chrom": chrom, "strand": strand,
                             "seq": seq, "canon_local": canon_local})

    if not gene_windows:
        print(f"  no usable sites among requested genes; skipping")
        return

    L = window_radius * 2 + 1
    BASES = {"A": 0, "C": 1, "G": 2, "T": 3}

    def one_hot(seq):
        arr = np.zeros((L, 4), dtype=np.float32)
        for i, b in enumerate(seq[:L]):
            j = BASES.get(b)
            if j is not None:
                arr[i, j] = 1.0
        return arr

    lr_saliency = {}
    if "lr" in feasible:
        print(f"  computing LR saliency...")
        import pickle, json as _json
        lr_pkl = None
        for cand in ["lr_model.pkl", "model.pkl"]:
            p = os.path.join(lr_run_dir, cand)
            if os.path.exists(p): lr_pkl = p; break
        if lr_pkl is None:
            print(f"    no lr_model.pkl in {lr_run_dir}; skipping LR")
        else:
            try:
                with open(lr_pkl, "rb") as fh:
                    lr_obj = pickle.load(fh)
                meta_p = os.path.join(lr_run_dir, "metadata.json")
                feat_kind = "unknown"
                if os.path.exists(meta_p):
                    md = _json.load(open(meta_p))
                    feat_kind = (md.get("features", {}).get("kind") or
                                  md.get("feature_kind") or "unknown")
                print(f"    LR feature kind: {feat_kind}")
                if feat_kind not in ("one_hot", "onehot", "ohe"):
                    print(f"    LR not one-hot; skipping LR saliency.")
                else:
                    coef = None
                    if hasattr(lr_obj, "coef_"):
                        coef = np.asarray(lr_obj.coef_).reshape(-1)
                    elif isinstance(lr_obj, dict) and "coef_" in lr_obj:
                        coef = np.asarray(lr_obj["coef_"]).reshape(-1)
                    if coef is None or coef.size != L * 4:
                        print(f"    LR coef shape unsuitable; skipping")
                    else:
                        coef = coef.reshape(L, 4)
                        for gw in gene_windows:
                            oh = one_hot(gw["seq"])
                            contrib = (coef * oh).sum(axis=1)
                            lr_saliency[gw["site_id"]] = np.abs(contrib)
            except Exception as e:
                print(f"    LR saliency failed: {e}; skipping LR")

    cnn_saliency = {}
    if "cnn" in feasible:
        print(f"  computing CNN saliency...")
        try:
            import torch
        except ImportError:
            torch = None
            print(f"    torch unavailable; skipping CNN")
        if torch is not None:
            cnn_ckpt = os.path.join(cnn_run_dir, "cnn_model.pt")
            if not os.path.exists(cnn_ckpt):
                print(f"    no cnn_model.pt in {cnn_run_dir}; skipping CNN")
            else:
                try:
                    state = torch.load(cnn_ckpt, map_location="cpu")
                    if isinstance(state, dict) and "state_dict" in state:
                        print(f"    cnn_model.pt is state_dict only; skipping CNN")
                    else:
                        model = state
                        model.eval()
                        for gw in gene_windows:
                            oh = one_hot(gw["seq"])
                            target_len = 500
                            full = np.zeros((target_len, 4), dtype=np.float32)
                            cl = gw["canon_local"]
                            half = target_len // 2
                            src_lo = max(0, cl - half)
                            src_hi = min(L, cl + half)
                            dst_lo = half - (cl - src_lo)
                            dst_hi = dst_lo + (src_hi - src_lo)
                            full[dst_lo:dst_hi] = oh[src_lo:src_hi]
                            x = torch.tensor(full, dtype=torch.float32,
                                              requires_grad=True).unsqueeze(0)
                            try:
                                y = model(x)
                            except Exception:
                                y = model(x.transpose(1, 2))
                            if isinstance(y, (tuple, list)): y = y[0]
                            score = y.squeeze() if y.ndim > 0 else y
                            score.sum().backward()
                            grad = x.grad.detach().cpu().numpy()[0]
                            attr = np.abs((grad * full).sum(axis=1))
                            attr_l = attr[dst_lo:dst_hi]
                            out = np.zeros(L, dtype=np.float32)
                            out_lo = cl - (cl - src_lo)
                            out[out_lo:out_lo + len(attr_l)] = attr_l
                            cnn_saliency[gw["site_id"]] = out
                except Exception as e:
                    print(f"    CNN saliency failed: {e}; skipping CNN")

    aparent2_saliency = {}
    if "aparent2" in models:
        if aparent2_saliency_parquet:
            sal_pq = aparent2_saliency_parquet
        else:
            default_name = (f"_aparent2_saliency_{site_label}.parquet"
                            if site_label != "canonical"
                            else "_aparent2_saliency.parquet")
            sal_pq = os.path.join(os.path.dirname(out_path), default_name)
        if os.path.exists(sal_pq):
            print(f"  loading APARENT2 saliency from {sal_pq}...")
            sal_df = pd.read_parquet(sal_pq)
            has_site_id = "site_id" in sal_df.columns
            for gw in gene_windows:
                if has_site_id:
                    sub = sal_df[sal_df["site_id"] == gw["site_id"]]
                else:
                    sub = sal_df[sal_df["gene"] == gw["gene"]]
                if len(sub) == 0:
                    continue
                arr = np.zeros(L, dtype=np.float32)
                for _, row in sub.iterrows():
                    pos_rel = int(row["position_rel"])
                    idx = window_radius + pos_rel
                    if 0 <= idx < L:
                        arr[idx] = float(row["saliency"])
                aparent2_saliency[gw["site_id"]] = arr
            if aparent2_saliency:
                feasible.append("aparent2")
        else:
            print(f"  no APARENT2 saliency parquet at {sal_pq}")
            print(f"  to add APARENT2: run saliency_aparent2.py with "
                  f"--site_label {site_label} --out {sal_pq}")

    sal_by_model = {"lr": lr_saliency, "cnn": cnn_saliency,
                    "aparent2": aparent2_saliency}
    models_for_plot = [m for m in feasible
                       if m in sal_by_model and sal_by_model[m]]
    if not models_for_plot:
        print(f"  no model produced saliency; skipping render")
        return

    sal_rows = []
    for m in models_for_plot:
        disp = MODEL_REGISTRY[m]["display"]
        sal_map = sal_by_model[m]
        for gw in gene_windows:
            sal = sal_map.get(gw["site_id"])
            if sal is None:
                continue
            norm = sal / max(sal.max(), 1e-9)
            for i, v in enumerate(norm):
                sal_rows.append({
                    "Model": disp, "site_id": gw["site_id"],
                    "gene": gw["gene"], "chrom": gw["chrom"],
                    "strand": gw["strand"],
                    "position": i - window_radius,
                    "attribution_norm": float(v),
                })
    if sal_rows:
        _save_tsv(pd.DataFrame(sal_rows), _tsv_path(out_path))

    import matplotlib.pyplot as plt
    n_g = len(gene_windows)
    n_m = len(models_for_plot)
    fig, axes = plt.subplots(n_m, n_g,
                              figsize=(3.2 * n_g, 1.6 * n_m),
                              dpi=500, facecolor="white", squeeze=False)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.85, bottom=0.12,
                         wspace=0.25, hspace=0.35)

    for mi, m in enumerate(models_for_plot):
        color = MODEL_REGISTRY[m]["color"]
        disp = MODEL_REGISTRY[m]["display"]
        per_gene_counter = {}
        for gi, gw in enumerate(gene_windows):
            ax = axes[mi][gi]
            sal = sal_by_model[m].get(gw["site_id"])
            if sal is None:
                ax.text(0.5, 0.5, "no saliency", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="#888780")
                ax.set_xticks([]); ax.set_yticks([])
            else:
                s_norm = sal / max(sal.max(), 1e-9)
                xs = np.arange(L) - window_radius
                ax.bar(xs, s_norm, width=1.0, color=color, edgecolor="none",
                        linewidth=0)
                ax.axvline(0, color=_CANON_RED, linestyle="--",
                           linewidth=0.8, alpha=0.8, zorder=3)
                ax.set_ylim(0, 1.05)
                ax.set_xlim(-window_radius, window_radius)
                if mi == n_m - 1:
                    ax.set_xlabel(f"nt from {site_label} CPA", fontsize=8,
                                  color="#444441")
                else:
                    ax.set_xticklabels([])
                ax.tick_params(colors="#444441", labelsize=7)
                for s_ in ax.spines.values():
                    s_.set_color("black"); s_.set_linewidth(0.5)
            if gi == 0:
                ax.set_ylabel(disp, fontsize=10, color=color, weight="bold")
            if mi == 0:
                count = per_gene_counter.get(gw["gene"], 0) + 1
                per_gene_counter[gw["gene"]] = count
                n_for_gene = sum(1 for gw2 in gene_windows
                                  if gw2["gene"] == gw["gene"])
                title = gw["gene"] if n_for_gene == 1 else f"{gw['gene']} #{count}"
                ax.set_title(title, fontsize=10, color="#2C2C2A", pad=4)

    fig.suptitle(f"Sequence attribution at {site_label} CPAs",
                 fontsize=13, color="#2C2C2A", x=0.02, ha="left", y=0.97)
    fig.text(0.02, 0.92,
             f"Per-position attribution magnitude in a +/- {window_radius} nt "
             f"window around the {site_label} CPA (red dashed line). "
             "Bars are normalized per track.",
             fontsize=9, color="#5F5E5A", ha="left")
    fig.savefig(out_path, dpi=500, bbox_inches="tight", facecolor="white",
                pad_inches=0.3)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _roc_pr(y_true, y_score):
    y_true = np.asarray(y_true).astype(bool)
    y_score = np.asarray(y_score).astype(float)
    order = np.argsort(-y_score, kind="stable")
    y_true_s = y_true[order]
    tps = np.cumsum(y_true_s).astype(float)
    fps = np.cumsum(~y_true_s).astype(float)
    n_pos = int(y_true.sum())
    n_neg = int((~y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return (np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan"),
                np.array([0.0, 1.0]), np.array([1.0, 0.0]), float("nan"))
    tpr = tps / n_pos
    fpr = fps / n_neg
    fpr_full = np.concatenate(([0.0], fpr))
    tpr_full = np.concatenate(([0.0], tpr))
    auroc = float(np.sum(np.diff(fpr_full) *
                          (tpr_full[:-1] + tpr_full[1:]) / 2.0))
    precision = tps / (tps + fps)
    recall = tps / n_pos
    prec_full = np.concatenate(([1.0], precision))
    rec_full  = np.concatenate(([0.0], recall))
    auprc = float(np.sum(np.diff(rec_full) * precision))
    return fpr_full, tpr_full, auroc, recall, precision, auprc


def _max_in_vicinity_per_site(sites_bed, scan_dir, pc_genes_bed,
                                prob_col, vicinity_nt=500, win_size=500,
                                return_indices=False):
    site_to_gene = _map_sites_to_genes(sites_bed, pc_genes_bed)

    pc_df = pd.DataFrame(read_bed(pc_genes_bed)).rename(columns={"name": "gene"})
    pc_df = (pc_df.assign(span=pc_df["end"] - pc_df["start"])
                   .sort_values("span", ascending=False)
                   .drop_duplicates("gene", keep="first")
                   .drop(columns=["span"]))
    pc_by_gene = pc_df.set_index("gene")[
        ["start", "end", "strand"]].to_dict("index")

    sites = read_bed(sites_bed)
    sites_by_gene = defaultdict(list)
    for bed_idx, s in enumerate(sites):
        g = site_to_gene.get(s["name"])
        if g is not None:
            sites_by_gene[g].append((bed_idx, s))

    max_scores = []
    bed_indices = []
    for gene, gene_sites in sites_by_gene.items():
        pq_path = os.path.join(scan_dir, "per_gene", f"{gene}.parquet")
        if not os.path.exists(pq_path):
            continue
        try:
            df = pd.read_parquet(pq_path, columns=[prob_col, "window_offset"])
        except Exception:
            continue
        if len(df) == 0:
            continue
        gi = pc_by_gene.get(gene)
        if gi is None:
            continue
        offsets = df["window_offset"].values
        probs = df[prob_col].values
        for bed_idx, s in gene_sites:
            site_center = (s["start"] + s["end"]) // 2
            if gi["strand"] == "+":
                tx_pos = site_center - gi["start"]
            else:
                tx_pos = gi["end"] - site_center
            target_offset = tx_pos - win_size // 2
            mask = np.abs(offsets - target_offset) <= vicinity_nt
            if mask.any():
                max_scores.append(float(probs[mask].max()))
                bed_indices.append(bed_idx)
    scores_arr = np.array(max_scores)
    if return_indices:
        return scores_arr, np.array(bed_indices, dtype=int)
    return scores_arr


def _ip_mask_per_site(sites_bed, reference_fasta, run_len=12, a_frac=0.75,
                        flank_nt=200):
    if not (reference_fasta and os.path.exists(reference_fasta)):
        print(f"  WARN: reference_fasta missing; cannot compute IP mask")
        return None
    try:
        import pysam
    except ImportError:
        print(f"  WARN: pysam unavailable; cannot compute IP mask")
        return None
    fa = pysam.FastaFile(reference_fasta)
    fa_refs = set(fa.references)
    sites = read_bed(sites_bed)
    flags = np.zeros(len(sites), dtype=bool)
    for i, s in enumerate(sites):
        chrom = s["chrom"] if s["chrom"] in fa_refs else s["chrom"].lstrip("chr")
        if chrom not in fa_refs:
            continue
        center = (s["start"] + s["end"]) // 2
        try:
            seq = fa.fetch(chrom, max(0, center - flank_nt),
                            center + flank_nt + 1).upper()
        except Exception:
            continue
        if s["strand"] == "-":
            seq = seq.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]
        if len(seq) < run_len:
            continue
        is_A = np.array([c == "A" for c in seq], dtype=np.uint8)
        csum = np.concatenate(([0], np.cumsum(is_A)))
        wsum = csum[run_len:] - csum[:-run_len]
        if (wsum >= a_frac * run_len).any():
            flags[i] = True
    return flags


def fig_external_discriminability(scan_dir, pc_genes_bed, canonical_bed,
                                     cpccpa_bed, random_bed, out_path,
                                     external_model="aparent2",
                                     vicinity_nt=1000, win_size=500,
                                     score_col=None,
                                     reference_fasta=None,
                                     report_ip_filter=True,
                                     ip_run_len=12, ip_a_frac=0.75):
    info = MODEL_REGISTRY.get(external_model)
    if info is None:
        print(f"\n[External discrim] unknown model {external_model!r}; skipping")
        return
    if score_col is None:
        if external_model == "aparent2":
            score_col = "aparent2_cleavage_sum"
        else:
            score_col = info["prob_col"]
    disp = info["display"]
    print(f"\n[External discrim] {disp} vs canonical/3-way/random "
          f"(vicinity ±{vicinity_nt} nt, score column = {score_col!r})")

    sample_pq = sorted(glob.glob(os.path.join(scan_dir, "per_gene", "*.parquet")))
    if not sample_pq:
        print(f"  no parquets found; skipping"); return
    import pyarrow.parquet as pq
    sample_cols = set(pq.ParquetFile(sample_pq[0]).schema_arrow.names)
    if score_col not in sample_cols:
        print(f"  {score_col} not in parquet columns; skipping. "
              f"Available columns: {sorted(sample_cols)}")
        return

    for bed_label, path in [("canonical",  canonical_bed),
                             ("3-way",      cpccpa_bed),
                             ("random",     random_bed)]:
        if not path or not os.path.exists(path):
            print(f"  {bed_label} BED missing ({path}); skipping"); return

    print(f"  collecting max-in-vicinity {score_col} per site for each class...")
    canon_scores, canon_idx   = _max_in_vicinity_per_site(
        canonical_bed, scan_dir, pc_genes_bed, score_col, vicinity_nt, win_size,
        return_indices=True)
    cpccpa_scores, cpccpa_idx = _max_in_vicinity_per_site(
        cpccpa_bed, scan_dir, pc_genes_bed, score_col, vicinity_nt, win_size,
        return_indices=True)
    random_scores, random_idx = _max_in_vicinity_per_site(
        random_bed, scan_dir, pc_genes_bed, score_col, vicinity_nt, win_size,
        return_indices=True)

    def desc(a):
        if len(a) == 0: return "n=0"
        return (f"n={len(a):,}  median={np.median(a):.3f}  "
                f"mean={np.mean(a):.3f}  p90={np.percentile(a, 90):.3f}")
    print(f"    canonical: {desc(canon_scores)}")
    print(f"    3-way:     {desc(cpccpa_scores)}")
    print(f"    random:    {desc(random_scores)}")

    if len(canon_scores) == 0 or len(random_scores) == 0 or len(cpccpa_scores) == 0:
        print(f"  one of the classes has zero scored sites; cannot compute curves")
        return

    ip_summary_rows = []
    ip_canon = ip_3way = ip_rand = None
    if report_ip_filter and reference_fasta and os.path.exists(reference_fasta):
        print(f"\n  computing internal-priming masks "
              f"(>={ip_run_len} nt run with >={int(ip_a_frac*100)}% A, ±200 nt flank)...")
        ip_canon  = _ip_mask_per_site(canonical_bed, reference_fasta,
                                        run_len=ip_run_len, a_frac=ip_a_frac)
        ip_3way   = _ip_mask_per_site(cpccpa_bed, reference_fasta,
                                        run_len=ip_run_len, a_frac=ip_a_frac)
        ip_rand   = _ip_mask_per_site(random_bed, reference_fasta,
                                        run_len=ip_run_len, a_frac=ip_a_frac)
        for label, m in [("Canonical", ip_canon),
                          ("3-way",     ip_3way),
                          ("Random",    ip_rand)]:
            if m is None: continue
            f = float(m.mean()) * 100 if len(m) else 0.0
            print(f"    {label:<10}: {int(m.sum())}/{len(m)} sites flagged "
                  f"({f:.2f}%) for internal-priming signature")
            ip_summary_rows.append({
                "class": label, "n_total": int(len(m)),
                "n_ip_flagged": int(m.sum()),
                "pct_ip_flagged": float(m.mean() * 100) if len(m) else 0.0,
                "run_len": ip_run_len,
                "a_frac_threshold": ip_a_frac,
            })

        if ip_canon is not None and ip_3way is not None and ip_rand is not None:
            ip_canon_aligned = ip_canon[canon_idx]
            ip_3way_aligned  = ip_3way[cpccpa_idx]
            ip_rand_aligned  = ip_rand[random_idx]
            canon_scores_filt  = canon_scores[~ip_canon_aligned]
            cpccpa_scores_filt = cpccpa_scores[~ip_3way_aligned]
            random_scores_filt = random_scores[~ip_rand_aligned]
            print(f"\n  AFTER IP filter (sites with internal-priming "
                  f"signature excluded):")
            print(f"    canonical: n={len(canon_scores_filt):,} "
                  f"(dropped {int(ip_canon_aligned.sum())})")
            print(f"    3-way:     n={len(cpccpa_scores_filt):,} "
                  f"(dropped {int(ip_3way_aligned.sum())})")
            print(f"    random:    n={len(random_scores_filt):,} "
                  f"(dropped {int(ip_rand_aligned.sum())})")
        else:
            canon_scores_filt = cpccpa_scores_filt = random_scores_filt = None
    else:
        canon_scores_filt = cpccpa_scores_filt = random_scores_filt = None

    comparisons = [
        ("Canonical vs Random",    canon_scores, random_scores, PAL["teal"],   False),
        ("Canonical vs 3-way",     canon_scores, cpccpa_scores, PAL["coral"],  False),
    ]
    if canon_scores_filt is not None and len(canon_scores_filt) > 0 and \
       len(cpccpa_scores_filt) > 0 and len(random_scores_filt) > 0:
        comparisons += [
            ("Canonical vs Random (IP-filtered)",
                canon_scores_filt, random_scores_filt, PAL["teal"],   True),
            ("Canonical vs 3-way (IP-filtered)",
                canon_scores_filt, cpccpa_scores_filt, PAL["coral"],  True),
        ]
    roc_rows = []
    pr_rows = []
    summary_rows = []
    curve_data = {}
    for label, pos, neg, color, is_ip_filtered in comparisons:
        y_true = np.concatenate([np.ones(len(pos), dtype=bool),
                                  np.zeros(len(neg), dtype=bool)])
        y_score = np.concatenate([pos, neg])
        fpr, tpr, auroc, rec, prec, auprc = _roc_pr(y_true, y_score)
        if len(fpr) > 2000:
            idx = np.linspace(0, len(fpr) - 1, 2000).astype(int)
            fpr_p, tpr_p = fpr[idx], tpr[idx]
        else:
            fpr_p, tpr_p = fpr, tpr
        if len(rec) > 2000:
            idx = np.linspace(0, len(rec) - 1, 2000).astype(int)
            rec_p, prec_p = rec[idx], prec[idx]
        else:
            rec_p, prec_p = rec, prec
        for f, t in zip(fpr_p, tpr_p):
            roc_rows.append({"comparison": label, "fpr": float(f),
                              "tpr": float(t), "ip_filtered": is_ip_filtered})
        for r, p_ in zip(rec_p, prec_p):
            pr_rows.append({"comparison": label, "recall": float(r),
                             "precision": float(p_), "ip_filtered": is_ip_filtered})
        summary_rows.append({
            "comparison": label, "auroc": auroc, "auprc": auprc,
            "n_pos": len(pos), "n_neg": len(neg),
            "external_model": disp,
            "score_col": score_col,
            "vicinity_nt": vicinity_nt,
            "ip_filtered": is_ip_filtered,
        })
        curve_data[label] = (fpr_p, tpr_p, rec_p, prec_p, auroc, auprc, color)
        print(f"    {label:<50}  AUROC={auroc:.3f}  AUPRC={auprc:.3f}")

    _save_tsv(pd.DataFrame(roc_rows),     _tsv_path(out_path, "roc"))
    _save_tsv(pd.DataFrame(pr_rows),      _tsv_path(out_path, "pr"))
    _save_tsv(pd.DataFrame(summary_rows), _tsv_path(out_path, "summary"))
    if ip_summary_rows:
        _save_tsv(pd.DataFrame(ip_summary_rows),
                   _tsv_path(out_path, "ip_summary"))
    score_rows = (
        [{"Class": "Canonical",         "score": float(s)} for s in canon_scores] +
        [{"Class": "3-way intragenic",  "score": float(s)} for s in cpccpa_scores] +
        [{"Class": "Random",            "score": float(s)} for s in random_scores]
    )
    _save_tsv(pd.DataFrame(score_rows),   _tsv_path(out_path, "scores"))

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=500, facecolor="white")
    fig.subplots_adjust(left=0.07, right=0.97, top=0.83, bottom=0.13, wspace=0.25)
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color="#888780", linestyle="--",
             linewidth=0.6, zorder=1)
    for label, (fpr, tpr, _, _, auroc, _, color) in curve_data.items():
        ax.plot(fpr, tpr, color=color, linewidth=1.6,
                 label=f"{label} (AUROC={auroc:.3f})", zorder=2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("A. ROC", loc="left", color="#2C2C2A", pad=6)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    for s in ax.spines.values(): s.set_color("black"); s.set_linewidth(0.6)
    ax.grid(color="#D3D1C7", linewidth=0.3, alpha=0.6); ax.set_axisbelow(True)
    ax = axes[1]
    for label, (_, _, rec, prec, _, auprc, color) in curve_data.items():
        ax.plot(rec, prec, color=color, linewidth=1.6,
                 label=f"{label} (AUPRC={auprc:.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("B. Precision-Recall", loc="left", color="#2C2C2A", pad=6)
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    for s in ax.spines.values(): s.set_color("black"); s.set_linewidth(0.6)
    ax.grid(color="#D3D1C7", linewidth=0.3, alpha=0.6); ax.set_axisbelow(True)

    fig.suptitle(f"{disp} cannot distinguish 3-way intragenic from canonical CPA "
                  f"sites",
                  x=0.02, ha="left", y=0.97, fontsize=13, color="#2C2C2A")
    fig.text(0.02, 0.92,
              f"Per-site MAX {score_col} within ±{vicinity_nt} nt of each "
              f"site's center offset. Canonical-vs-Random is the positive "
              f"control (expected high AUROC); Canonical-vs-3-way is the key "
              f"test (near-diagonal ROC means {disp} treats the two classes "
              "as equivalent).",
              fontsize=9, color="#5F5E5A", ha="left")
    fig.savefig(out_path, dpi=500, bbox_inches="tight", facecolor="white",
                pad_inches=0.3)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan_dir",      required=True)
    ap.add_argument("--pc_genes_bed",  required=True)
    ap.add_argument("--atlas_bed",     required=False, nargs="+", default=None,
                    help="One or more atlas BEDs; unioned for Figs 3/4/6/10. "
                         "OPTIONAL: if not given, defaults to the union of "
                         "--polyasite_bed + --polyadb_main_bed + --polyadb_max_bed.")
    ap.add_argument("--canonical_bed", required=True)
    ap.add_argument("--three_prime_zone_bed", required=True,
                    help="BED of terminal-exon + 3'UTR + downstream slop. "
                         "Peaks overlapping this zone are excluded from the "
                         "no_3utr tier BEDs. Produce via extract_3utr.py.")
    ap.add_argument("--polyasite_bed", required=True,
                    help="PolyASite 2.0 clusters BED (independent reference 1)")
    ap.add_argument("--polyadb_main_bed", required=True,
                    help="PolyA_DB v4 Main BED (independent reference 2). "
                         "Must have RPM in col 5 and PSE in col 7.")
    ap.add_argument("--polyadb_max_bed", required=True,
                    help="PolyA_DB v4 Max BED (independent reference 3). "
                         "Must have RPM in col 5 and PSE in col 7.")
    ap.add_argument("--random_bed", required=True,
                    help="Random intragenic windows (size-matched to CPCCPA). "
                         "Used as negative control in fig_cpccpa.")
    ap.add_argument("--test_chroms",   default=None)
    ap.add_argument("--test_chroms_only", action="store_true")
    ap.add_argument("--rf_dir",  default=None)
    ap.add_argument("--lr_dir",  default=None)
    ap.add_argument("--cnn_dir", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--peak_merge_dist", type=int, default=0)
    ap.add_argument("--trace_gene", default=None, nargs="+")
    ap.add_argument("--force_peaks", action="store_true")
    ap.add_argument("--skip_fig", default="",
                    help="Comma-separated figure IDs to skip. Valid IDs: "
                         "0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, cpccpa, "
                         "external, saliency.")
    ap.add_argument("--saturation_thr", type=float, default=0.95)
    ap.add_argument("--models", nargs="+", default=["rf", "lr", "cnn"],
                    help=f"Primary models used in all main figures (consensus, "
                         f"recovery, agreement, etc). Default: rf lr cnn. "
                         f"APARENT2 is intentionally excluded here; it's used "
                         f"only by the external-validation figure (see "
                         f"--external_model).")
    ap.add_argument("--external_model", default="aparent2",
                    help="External-validation model used by the discriminability "
                         "figure ONLY. Default: aparent2. The external model is "
                         "scored at canonical/3-way/random sites to test whether "
                         "it can tell canonical from 3-way intragenic.")
    ap.add_argument("--external_vicinity_nt", type=int, default=1000,
                    help="Vicinity (nt) for the external-discrim max-in-vicinity "
                         "search. Default 1000 (more generous than the threshold-"
                         "computation vicinity of 500); larger = more leeway to "
                         "the external model. Reporting the same figure at "
                         "multiple vicinities (e.g. 500, 1000, 2500) shows the "
                         "conclusion is robust.")
    ap.add_argument("--external_score_col", default=None,
                    help="Override the parquet column used to score the external "
                         "model. Defaults: aparent2 -> aparent2_cleavage_sum "
                         "(APARENT2's per-position cleavage head, integrated over "
                         "the 11-nt cut window; the right metric for chemistry-"
                         "level CPA prediction on isolated genomic windows). "
                         "Run additionally with 'aparent2_prob' as a "
                         "supplementary check, noting the iso head is "
                         "off-distribution on isolated windows.")
    ap.add_argument("--external_skip_ip_filter", action="store_true",
                    help="Skip the APARENT2 internal-priming filter check.")
    ap.add_argument("--external_ip_run_len", type=int, default=12,
                    help="Internal-priming run length (nt) per Linder et al. "
                         "Default 12 nt.")
    ap.add_argument("--external_ip_a_frac", type=float, default=0.75,
                    help="Internal-priming A-fraction threshold per Linder. "
                         "Default 0.75 (75% A within the run window).")
    ap.add_argument("--vicinity_nt", type=int, default=500)
    ap.add_argument("--reference_fasta", default=None,
                    help="Reference FASTA (needed for saliency)")
    ap.add_argument("--enable_saliency", action="store_true")
    ap.add_argument("--cpccpa_saliency_tier", default="tier3_exactly_three",
                    choices=["tier1_exactly_one", "tier2_exactly_two",
                             "tier3_exactly_three"],
                    help="Which CPCCPA tier's no_3utr BED to use for saliency.")
    ap.add_argument("--cpccpa_saliency_sites_per_gene", type=int, default=3)
    ap.add_argument("--cluster_slack", type=int, default=250,
                    help="Single-link cluster slack (nt) when combining peaks "
                         "across models for tier extraction. Default 250.")
    ap.add_argument("--cluster_halfwidth", type=int, default=250,
                    help="Halfwidth (nt) of the emitted cluster-representative "
                         "BED intervals. Default 250 (so each site is 501 nt).")
    ap.add_argument("--motif_win_up", type=int, default=100,
                    help="Fig 11 motif-density window: nt upstream of cut site.")
    ap.add_argument("--motif_win_dn", type=int, default=40,
                    help="Fig 11 motif-density window: nt downstream of cut site.")
    ap.add_argument("--heatmap_flank", type=int, default=500,
                    help="Fig 13 heatmap: nt flank around cut site. Default 500.")
    ap.add_argument("--heatmap_max_sites", type=int, default=None,
                    help="Fig 13 heatmap: cap on sites used (default: all).")

    args = ap.parse_args()

    skip = set(s.strip() for s in args.skip_fig.split(",")) if args.skip_fig else set()
    os.makedirs(args.out_dir, exist_ok=True)

    print("[Detecting active models]")
    models = detect_models(args.scan_dir, requested=args.models)
    if not models:
        raise SystemExit("ERROR: no recognized model columns found in parquets")

    print("\n[Computing thresholds from constitutive sites]")
    thresholds = get_mean_threshold(
        args.scan_dir, args.canonical_bed, models,
        test_chroms_path=args.test_chroms,
        test_chroms_only=args.test_chroms_only,
        vicinity_nt=args.vicinity_nt)

    print("\n[Building atlas union]")
    if args.atlas_bed:
        atlas_inputs = list(args.atlas_bed)
        print(f"  using --atlas_bed: {atlas_inputs}")
    else:
        atlas_inputs = [p for p in [args.polyasite_bed,
                                      args.polyadb_main_bed,
                                      args.polyadb_max_bed] if p]
        print(f"  --atlas_bed not given; unioning labeled refs: {atlas_inputs}")
    atlas_path = build_union_atlas(atlas_inputs, args.out_dir)

    if "0" not in skip:
        if args.rf_dir and args.lr_dir and args.cnn_dir:
            fig_training_stats(args.rf_dir, args.lr_dir, args.cnn_dir,
                               os.path.join(args.out_dir, "fig0_training_stats.png"))
        else:
            print("\n[Figure 0] Skipping: provide --rf_dir --lr_dir --cnn_dir")

    test_chrom_set = None
    if args.test_chroms and os.path.exists(args.test_chroms):
        test_chrom_set = set(open(args.test_chroms).read().split())
        print(f"\n[Test chromosomes] loaded {len(test_chrom_set)} chroms: "
              f"{', '.join(sorted(test_chrom_set))}")

    need_metrics_all = ("1" not in skip) or ("3" not in skip)
    rank_strict = topk_hit = dists = None
    if need_metrics_all:
        print("\n[Loading ranks/top-K/distances - ALL genes]")
        rank_strict, topk_hit, dists = collect_metrics(
            args.scan_dir, models, pc_genes_bed=args.pc_genes_bed,
            test_chroms=None, include_combined=True)

    if "1" not in skip:
        fig_ranks(rank_strict, topk_hit, models,
                  os.path.join(args.out_dir, "fig1_ranks.png"), label_suffix="")
        if test_chrom_set is not None:
            print("\n[Loading ranks/top-K/distances - TEST chroms only]")
            rs_t, th_t, _ = collect_metrics(
                args.scan_dir, models, pc_genes_bed=args.pc_genes_bed,
                test_chroms=test_chrom_set, include_combined=True)
            fig_ranks(rs_t, th_t, models,
                      os.path.join(args.out_dir, "fig1_ranks_testonly.png"),
                      label_suffix=" (test chromosomes only)")

    merged_per_model = None
    need_peaks = (("2" not in skip) or ("5" not in skip) or ("7" not in skip) or
                  ("8" not in skip) or ("11" not in skip) or
                  ("13" not in skip) or ("external" not in skip) or
                  ("cpccpa" not in skip))
    if need_peaks:
        print("\n[Collecting high-confidence peaks (cached)]")
        cache_tag = _scan_dir_tag(args.scan_dir)
        merged_per_model = collect_high_conf_peaks(
            args.scan_dir, args.pc_genes_bed, thresholds, args.out_dir,
            models, merge_dist=args.peak_merge_dist,
            force=args.force_peaks, cache_tag=cache_tag)

    if "2" not in skip:
        fig_atlas_recovery(
            merged_per_model, models,
            ref_paths_with_labels=[
                ("Canonical",     args.canonical_bed),
                ("PolyASite",     args.polyasite_bed),
                ("PolyA_DB Main", args.polyadb_main_bed),
                ("PolyA_DB Max",  args.polyadb_max_bed),
            ],
            out_path=os.path.join(args.out_dir, "fig2_atlas_recovery.png"),
            include_combined=True, slack=50)
    if "3" not in skip:
        fig_distance_atlas(args.scan_dir, args.pc_genes_bed, atlas_path,
                           dists, models,
                           os.path.join(args.out_dir, "fig3_distance_atlas.png"))
    if "4" not in skip:
        fig_score_distributions(
            args.scan_dir, args.pc_genes_bed, atlas_path,
            models, thresholds,
            os.path.join(args.out_dir, "fig4_score_distributions.png"),
            sample_per_gene=100)
    if "5" not in skip:
        fig_agreement(args.scan_dir, merged_per_model, models,
                      os.path.join(args.out_dir, "fig5_agreement.png"))
    if "6" not in skip and args.trace_gene:
        for gene in args.trace_gene:
            fig_gene_trace(args.scan_dir, args.pc_genes_bed, gene, thresholds,
                           models,
                           os.path.join(args.out_dir, f"fig6_trace_{gene}.png"),
                           atlas_bed=atlas_path)
    elif "6" not in skip:
        print("\n[Figure 6] Skipping: provide --trace_gene <NAME> ...")
    if "7" not in skip:
        fig_genome_coverage(
            args.scan_dir, args.pc_genes_bed, thresholds, merged_per_model,
            models, os.path.join(args.out_dir, "fig7_genome_coverage.png"))
    if "9" not in skip:
        fig_score_saturation(
            args.scan_dir, args.pc_genes_bed, thresholds, models,
            os.path.join(args.out_dir, "fig9_saturation.png"),
            test_chroms=None, high_thr=args.saturation_thr)
        if test_chrom_set is not None:
            fig_score_saturation(
                args.scan_dir, args.pc_genes_bed, thresholds, models,
                os.path.join(args.out_dir, "fig9_saturation_testonly.png"),
                test_chroms=test_chrom_set, high_thr=args.saturation_thr)
    if "10" not in skip:
        fig_calibration(
            args.scan_dir, args.pc_genes_bed, atlas_inputs,
            args.canonical_bed, models,
            os.path.join(args.out_dir, "fig10_calibration.png"))

    tier_beds = None
    if "cpccpa" not in skip:
        tier_beds = extract_agreement_tiers(
            merged_per_model, models, args.three_prime_zone_bed,
            args.out_dir,
            halfwidth=args.cluster_halfwidth, slack=args.cluster_slack)

        for variant in ["with_3utr", "no_3utr"]:
            class_beds = {
                "Constitutive": args.canonical_bed,
                "Random":       args.random_bed,
            }
            for tname, paths in tier_beds.items():
                if variant in paths:
                    class_beds[tname] = paths[variant]
            suffix = " (with 3'UTR)" if variant == "with_3utr" else " (intragenic only, 3'UTR excluded)"
            fig_cpccpa(
                class_beds,
                args.polyasite_bed, args.polyadb_main_bed, args.polyadb_max_bed,
                os.path.join(args.out_dir, f"fig_cpccpa_{variant}.png"),
                slack=50, title_suffix=suffix)

    venn_dict = None
    need_3way_bed = any(s not in skip for s in ("8", "11", "13", "external"))
    if need_3way_bed:
        venn_dict = extract_3way_intragenic_agreement(
            merged_per_model, args.three_prime_zone_bed, args.out_dir,
            halfwidth=args.cluster_halfwidth, slack=args.cluster_slack)

    if "8" not in skip:
        fig_3way_venn(venn_dict,
                       os.path.join(args.out_dir, "fig8_venn_rflrcnn.png"))

    if "external" not in skip:
        if venn_dict is not None and venn_dict.get("all_three_bed") and \
           os.path.exists(venn_dict["all_three_bed"]):
            score_col = args.external_score_col
            tag_bits = [args.external_model]
            if score_col and score_col != MODEL_REGISTRY.get(
                    args.external_model, {}).get("prob_col"):
                tag_bits.append(score_col.replace(
                    f"{args.external_model}_", ""))
            if args.external_vicinity_nt != 1000:
                tag_bits.append(f"v{args.external_vicinity_nt}")
            out_name = "fig_" + "_".join(tag_bits) + "_discrim.png"
            fig_external_discriminability(
                args.scan_dir, args.pc_genes_bed, args.canonical_bed,
                venn_dict["all_three_bed"], args.random_bed,
                os.path.join(args.out_dir, out_name),
                external_model=args.external_model,
                vicinity_nt=args.external_vicinity_nt,
                score_col=score_col,
                reference_fasta=args.reference_fasta,
                report_ip_filter=not args.external_skip_ip_filter,
                ip_run_len=args.external_ip_run_len,
                ip_a_frac=args.external_ip_a_frac)
        else:
            print(f"\n[External discrim] 3-way BED unavailable; skipping")

    if "11" not in skip and venn_dict is not None:
        if args.reference_fasta and os.path.exists(args.reference_fasta):
            fig_motif_density(
                args.canonical_bed, venn_dict["all_three_bed"],
                args.reference_fasta,
                os.path.join(args.out_dir, "fig11_motif_density.png"),
                win_up=args.motif_win_up, win_dn=args.motif_win_dn)
        else:
            print(f"\n[Figure 11] needs --reference_fasta; skipping")

    if "13" not in skip:
        fig_score_heatmap(
            args.scan_dir, args.pc_genes_bed, args.canonical_bed, models,
            os.path.join(args.out_dir, "fig13_heatmap_canonical.png"),
            thresholds=thresholds, flank_nt=args.heatmap_flank,
            max_sites=args.heatmap_max_sites,
            title="Per-site CPA score profile: canonical sites",
            site_label="Canonical sites")
        cpccpa_bed_for_heat = None
        if venn_dict and venn_dict.get("all_three_bed") and \
           os.path.exists(venn_dict["all_three_bed"]):
            cpccpa_bed_for_heat = venn_dict["all_three_bed"]
        if cpccpa_bed_for_heat:
            fig_score_heatmap(
                args.scan_dir, args.pc_genes_bed, cpccpa_bed_for_heat, models,
                os.path.join(args.out_dir, "fig13_heatmap_cpccpa.png"),
                thresholds=thresholds, flank_nt=args.heatmap_flank,
                max_sites=args.heatmap_max_sites,
                title="Per-site CPA score profile: CPCCPA sites (RF+LR+CNN intragenic)",
                site_label="CPCCPA sites")
        else:
            print(f"\n[Figure 13 CPCCPA] no 3-way intragenic BED; skipping")

    if "13" not in skip:
        cache_tag_t = _scan_dir_tag(args.scan_dir)
        fig_threshold_robustness(
            args.scan_dir, args.pc_genes_bed, args.canonical_bed,
            args.three_prime_zone_bed, args.polyadb_max_bed,
            models, args.out_dir,
            os.path.join(args.out_dir, "fig13_threshold_robustness.png"),
            vicinity_nt=args.vicinity_nt,
            cache_tag=cache_tag_t,
            force=args.force_peaks,
            atlas_label="PolyA_DB v4 Max")

    if "saliency" not in skip and args.enable_saliency:
        if not args.trace_gene:
            print("\n[Saliency] Skipping: --enable_saliency requires --trace_gene")
        else:
            fig_saliency(
                args.scan_dir, args.pc_genes_bed, args.canonical_bed,
                args.reference_fasta, models, args.trace_gene,
                os.path.join(args.out_dir, "fig12_saliency_canonical.png"),
                rf_run_dir=args.rf_dir, lr_run_dir=args.lr_dir,
                cnn_run_dir=args.cnn_dir,
                site_label="canonical",
                max_sites_per_gene=1)
            cpccpa_bed = None
            if tier_beds and args.cpccpa_saliency_tier in tier_beds:
                cpccpa_bed = tier_beds[args.cpccpa_saliency_tier].get("no_3utr")
            if cpccpa_bed and os.path.exists(cpccpa_bed):
                fig_saliency(
                    args.scan_dir, args.pc_genes_bed, cpccpa_bed,
                    args.reference_fasta, models, args.trace_gene,
                    os.path.join(args.out_dir, "fig12_saliency_cpccpa.png"),
                    rf_run_dir=args.rf_dir, lr_run_dir=args.lr_dir,
                    cnn_run_dir=args.cnn_dir,
                    site_label="cpccpa",
                    max_sites_per_gene=args.cpccpa_saliency_sites_per_gene)
            else:
                print(f"\n[CPCCPA saliency] BED for tier "
                      f"{args.cpccpa_saliency_tier} not available; skipping")

    print(f"\nAll figures written to {args.out_dir}/")


if __name__ == "__main__":
    main()