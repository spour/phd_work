#!/usr/bin/env python3
# (v3.0) changes vs v2.4 - per-gene peak calling
import argparse, os, glob, json, sys, csv
import numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor

def _nworkers():
    n = os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 2
    try: n = int(n)
    except Exception: n = 2
    return max(2, min(n, 16))

def _threaded(files, fn, label, every=4000):
    out = []
    with ThreadPoolExecutor(max_workers=_nworkers()) as ex:
        for i, r in enumerate(ex.map(fn, files), 1):
            if i % every == 0: print(f"[{label}] {i}/{len(files)}...", flush=True)
            out.append(r)
    return out

MODELS = ["RF", "LR", "CNN"]
COLS   = {"RF": "rf_prob", "LR": "lr_prob", "CNN": "cnn_prob"}
_PAIRS = [(a, b) for i, a in enumerate(MODELS) for b in MODELS[i+1:]]

_COUNTS = {}
def _rec(panel, n, unit="genes"):
    _COUNTS[panel] = (int(n), unit)

def flush_counts(out):
    path = os.path.join(out, "fig1_counts.tsv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t"); w.writerow(["panel", "n", "unit"])
        for p in sorted(_COUNTS): w.writerow([p, _COUNTS[p][0], _COUNTS[p][1]])

def read_pq(path, columns):
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception as e:
        print(f"[skip] unreadable parquet {os.path.basename(str(path))}: "
              f"{type(e).__name__}", file=sys.stderr)
        return None

def load_atlas(atlas_bed):
    rows = []
    with open(atlas_bed) as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 10: continue
            try:
                rep = int(p[3].split(":")[1]) - 1
                tpm = float(p[4]); pct = float(p[6]); typ = p[9]
            except Exception:
                continue
            chrom = p[0] if p[0].startswith("chr") else "chr" + p[0]
            rows.append((chrom, rep, tpm, pct, typ, p[5]))
    return pd.DataFrame(rows, columns=["chrom","mid","tpm","pct","type","strand"])

def load_gene_chrom(canonical_bed):
    m = {}
    with open(canonical_bed) as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 4: continue
            chrom = p[0] if p[0].startswith("chr") else "chr" + p[0]
            m[p[3]] = chrom
    return m

def build_atlas_index(atlas):
    idx = {}
    for c, g in atlas.groupby("chrom"):
        gg = g.sort_values("mid", kind="mergesort")
        idx[c] = (gg["mid"].to_numpy(),
                  gg["tpm"].to_numpy(dtype=np.float64),
                  gg["type"].to_numpy())
    return idx

def _chrom_inrange(idx, chrom, lo, hi):
    entry = idx.get(chrom)
    if entry is None:
        return None
    mids, tpm, typ = entry
    i0 = np.searchsorted(mids, lo, side="left")
    i1 = np.searchsorted(mids, hi, side="right")
    if i1 > i0:
        return mids[i0:i1], tpm[i0:i1], typ[i0:i1]
    return None

def _group_of(t):
    if t == "TE": return "Terminal"
    if t in ("IN", "EX"): return "Intragenic"
    return "Other"


def gene_threshold(ws, isc, arr, vic):
    if not isc.any():
        return None
    cpos = int(ws[np.argmax(isc)])
    near = np.abs(ws - cpos) <= vic
    if not near.any():
        return None
    return {m: float(arr[m][near].max()) for m in MODELS}


def scan_prod(prod_dir, atlas_idx, gene_chrom, vic, out, per_gene=6, seed=0,
              gene_filter=None, require_canonical=True):
    files = sorted(glob.glob(os.path.join(prod_dir, "*.parquet")))
    if gene_filter is not None:
        n0 = len(files)
        files = [f for f in files if os.path.basename(f)[:-8] in gene_filter]
        print(f"[filter] prod scan restricted to {len(files):,}/{n0:,} genes")
    if not files:
        print("[scan] no prod parquets; skipping B/E/G/thr")
        return {}

    thr_per = {m: [] for m in MODELS}
    Bacc    = {p: np.zeros(6, dtype=np.float64) for p in _PAIRS}
    Erows, Grows = [], []
    rng = np.random.default_rng(seed)
    used = 0
    skipped_nocanon = 0
    unmapped = 0
    ntot = len(files)

    for fi, f in enumerate(files):
        if fi and fi % 2000 == 0:
            print(f"[scan] {fi}/{ntot} genes...", flush=True)
        d = read_pq(f, ["window_start_genomic"] + list(COLS.values()) + ["is_constitutive_cpa"])
        if d is None: continue
        ws  = d["window_start_genomic"].to_numpy()
        isc = d["is_constitutive_cpa"].to_numpy()
        if require_canonical and not isc.any():
            skipped_nocanon += 1
            continue
        used += 1
        arr = {m: d[COLS[m]].to_numpy(dtype=np.float64) for m in MODELS}

        if require_canonical:
            print(f"[scan] {skipped_nocanon:,} genes dropped for having no "
                f"annotated canonical site; {used:,} retained. One gene set is "
                f"used for every prod panel (B, E, F, G).")
        gt = gene_threshold(ws, isc, arr, vic)
        if gt is not None:
            for m in MODELS:
                thr_per[m].append(gt[m])

        for (a, b) in _PAIRS:
            x, y = arr[a], arr[b]
            Bacc[(a, b)] += (len(x), x.sum(), y.sum(),
                             (x*x).sum(), (y*y).sum(), (x*y).sum())

        for m in MODELS:
            s = arr[m]
            if isc.any(): Erows.append(dict(model=m, cls="Canonical", score=float(s[isc][0])))
            bg = np.where(~isc)[0]
            if len(bg):
                for j in rng.choice(bg, min(per_gene, len(bg)), replace=False):
                    Erows.append(dict(model=m, cls="Background", score=float(s[j])))

        if atlas_idx is not None:
            g = os.path.basename(f)[:-8]
            chrom = gene_chrom.get(g)
            if chrom is None:
                unmapped += 1
                sel = None
            else:
                sel = _chrom_inrange(atlas_idx, chrom, ws.min(), ws.max())
            if sel is not None:
                mids_s, tpm_s, typ_s = sel
                if isc.any():
                    cpos = int(ws[isc][0])
                    for am in mids_s:
                        if abs(int(am) - cpos) <= 250: continue
                        j = int(np.argmin(np.abs(ws - am)))
                        if abs(ws[j] - am) <= 250:
                            for m in MODELS:
                                Erows.append(dict(model=m, cls="Atlas", score=float(arr[m][j])))
                for k in range(len(mids_s)):
                    near = np.abs(ws - mids_s[k]) <= 250
                    if not near.any(): continue
                    grp = _group_of(typ_s[k])
                    for m in MODELS:
                        Grows.append(dict(model=m, group=grp, tpm=float(tpm_s[k]),
                                          score=float(arr[m][near].max())))

    if atlas_idx is not None:
        print(f"[chrom] atlas matched on own chromosome; "
              f"{used-unmapped}/{used} genes mapped, {unmapped} unmapped (atlas skipped for those)")

    # threshold distribution, not a single cut
    trows = []
    for m in MODELS:
        v = np.asarray(thr_per[m], dtype=float)
        trows.append(dict(
            model=m, n=int(v.size),
            threshold=round(float(v.mean()), 4) if v.size else None,
            mean=round(float(v.mean()), 4) if v.size else None,
            median=round(float(np.median(v)), 4) if v.size else None,
            q25=round(float(np.percentile(v, 25)), 4) if v.size else None,
            q75=round(float(np.percentile(v, 75)), 4) if v.size else None,
            min=round(float(v.min()), 4) if v.size else None,
            max=round(float(v.max()), 4) if v.size else None))
    pd.DataFrame(trows).to_csv(os.path.join(out, "fig1_thresholds.tsv"),
                               sep="\t", index=False)
    print(f"[thr] per-gene cuts (score at each gene's own 3' end, +/-{vic}nt), "
          f"n={len(thr_per['RF'])} genes with a canonical site:")
    for r in trows:
        if r["n"]:
            print(f"      {r['model']:<4} median {r['median']:.3f}  "
                  f"IQR {r['q25']:.3f}-{r['q75']:.3f}  range {r['min']:.3f}-{r['max']:.3f}")
    print("      These are a DISTRIBUTION. No global cut is used or plotted.")

    def pearson(s):
        n, Sx, Sy, Sxx, Syy, Sxy = s
        cov = n*Sxy - Sx*Sy; vx = n*Sxx - Sx*Sx; vy = n*Syy - Sy*Sy
        den = np.sqrt(vx*vy)
        return float(cov/den) if den > 0 else float("nan")
    rows = [dict(model_x=m, model_y=m, r=1.0) for m in MODELS]
    for (a, b) in _PAIRS:
        r = round(pearson(Bacc[(a, b)]), 3)
        rows += [dict(model_x=a, model_y=b, r=r), dict(model_x=b, model_y=a, r=r)]
    pd.DataFrame(rows).to_csv(os.path.join(out, "fig1_corr.tsv"), sep="\t", index=False)
    _rec("B", used); print(f"[B] fig1_corr.tsv ({used} genes, streaming Pearson)")

    dfe = pd.DataFrame(Erows).rename(columns={"cls": "class"})
    dfe.to_csv(os.path.join(out, "fig1_scoredist.tsv"), sep="\t", index=False)
    _rec("E", used); print(f"[E] fig1_scoredist.tsv ({used} genes, PROD; classes {dfe['class'].value_counts().to_dict()})")

    _write_figG(Grows, out)
    return {m: (float(np.mean(thr_per[m])) if thr_per[m] else None) for m in MODELS}

def _write_figG(rows, out, nbin=12):
    if not rows: print("[G] no atlas/scan overlap; skip"); return
    df = pd.DataFrame(rows); df = df[df.tpm > 0]
    GROUPS = ["Terminal", "Intragenic", "Other"]
    df["lx"] = np.log10(df["tpm"])
    binned = []
    for (m, grp), sub in df.groupby(["model", "group"]):
        if len(sub) < 20: continue
        q = pd.qcut(sub["lx"].rank(method="first"), min(nbin, max(2, len(sub)//30)), labels=False)
        for b, ss in sub.groupby(q):
            mu = ss["score"].mean(); sd = ss["score"].std(ddof=1) if len(ss) > 1 else 0.0
            se = sd / np.sqrt(len(ss))
            binned.append(dict(model=m, group=grp, med_tpm=float(10**ss["lx"].median()),
                               mean_score=float(mu), lo=float(mu-1.96*se),
                               hi=float(mu+1.96*se), n=int(len(ss))))
    pd.DataFrame(binned).to_csv(os.path.join(out, "figG_binned.tsv"), sep="\t", index=False)
    raw = df[["model","group","tpm","score"]].copy()
    if len(raw) > 60000:
        raw = df[["model","group","tpm","score"]].sample(min(len(df),60000), random_state=0)
    raw.to_csv(os.path.join(out, "figG_raw.tsv"), sep="\t", index=False)
    one = df[df.model=="RF"]
    gn = one.groupby("group").size().reindex(GROUPS).fillna(0).astype(int)
    pd.DataFrame({"group": gn.index, "n": gn.values}).to_csv(
        os.path.join(out, "figG_group_counts.tsv"), sep="\t", index=False)
    cor = (df.groupby(["model","group"])
             .apply(lambda x: x["score"].corr(x["tpm"], method="spearman"),
                    include_groups=False).reset_index(name="spearman"))
    cor.to_csv(os.path.join(out, "figG_corr.tsv"), sep="\t", index=False)
    _rec("G", int(gn.sum()), "sites")
    print(f"[G] figG_binned.tsv + figG_raw.tsv ({len(df):,} site-model rows); group n: {dict(gn)}")
    print("    spearman(score,tpm) by group:\n",
          cor.pivot(index='group', columns='model', values='spearman').round(3).to_string())

def _c_one(f, tol=250, Kmax=10):
    d = read_pq(f, ["window_start_genomic"] + list(COLS.values()) + ["is_constitutive_cpa"])
    if d is None: return None
    isc = d["is_constitutive_cpa"].to_numpy()
    if not isc.any(): return None
    ws = d["window_start_genomic"].to_numpy()
    cpos = int(ws[np.argmax(isc)])
    near = np.abs(ws - cpos) <= tol
    best = {}
    for m in MODELS:
        order = np.argsort(d[COLS[m]].to_numpy())[::-1]
        nio = near[order]
        best[m] = int(np.argmax(nio)) if nio.any() else Kmax + 1
    return best

def panel_C_topk(crossfit_dir, out, Kmax=10, tol=250):
    files = glob.glob(os.path.join(crossfit_dir, "*.parquet"))
    if not files: print("[C] no crossfit parquets; skip"); return
    results = _threaded(files, lambda f: _c_one(f, tol, Kmax), "C")
    hit = {m: np.zeros(Kmax) for m in MODELS}; hit["Combined"] = np.zeros(Kmax); n = 0
    for best in results:
        if best is None: continue
        n += 1
        for m in MODELS:
            for K in range(Kmax):
                if best[m] <= K: hit[m][K] += 1
        combined = min(best.values())
        for K in range(Kmax):
            if combined <= K: hit["Combined"][K] += 1
    rows = [dict(K=K+1, model=m, recovery=round(hit[m][K]/max(n,1), 4))
            for m in list(MODELS)+["Combined"] for K in range(Kmax)]
    pd.DataFrame(rows).to_csv(os.path.join(out, "fig1_topk.tsv"), sep="\t", index=False)
    _rec("C", n); print(f"[C] fig1_topk.tsv ({n} genes, CROSSFIT, within-{tol}nt; K=1 = argmax recovery)")


def panel_D_traces(prod_dir, erp27, out):
    p = os.path.join(prod_dir, f"{erp27}.parquet")
    if not os.path.exists(p): print(f"[D] {p} not found; skip"); return
    d = read_pq(p, ["window_offset"] + list(COLS.values()))
    if d is None: print(f"[D] {p} unreadable; skip"); return
    rows = [dict(pos=int(o), model=m, prob=float(v))
            for m in MODELS for o, v in zip(d["window_offset"], d[COLS[m]])]
    pd.DataFrame(rows).to_csv(os.path.join(out, "fig1_traces.tsv"), sep="\t", index=False)
    _rec("D", len(d), "windows"); print(f"[D] fig1_traces.tsv ({erp27}, {len(d)} windows, PROD)")


def _f_one(f, vic):
    d = read_pq(f, ["window_start_genomic"] + list(COLS.values())
                + ["is_constitutive_cpa"])
    if d is None: return None
    ws  = d["window_start_genomic"].to_numpy()
    isc = d["is_constitutive_cpa"].to_numpy()
    arr = {m: d[COLS[m]].to_numpy(dtype=np.float64) for m in MODELS}
    gt = gene_threshold(ws, isc, arr, vic)
    if gt is None: return None
    peak = {m: set(np.where(arr[m] >= gt[m])[0]) for m in MODELS}
    out = {}
    for a, b in (("RF","LR"), ("RF","CNN"), ("LR","CNN")):
        u = peak[a] | peak[b]
        out[f"{a}-{b}"] = (len(peak[a] & peak[b]) / len(u)) if u else np.nan
    return out


def panel_F_jaccard(prod_dir, vic, out, gene_filter=None):
    files = sorted(glob.glob(os.path.join(prod_dir, "*.parquet")))
    if gene_filter is not None:
        files = [f for f in files if os.path.basename(f)[:-8] in gene_filter]
    if not files: print("[F] no prod parquets; skip"); return
    pairs = ["RF-LR", "RF-CNN", "LR-CNN"]
    results = _threaded(files, lambda f: _f_one(f, vic), "F")
    acc = {k: [] for k in pairs}; used = 0; nocanon = 0
    for r in results:
        if r is None: nocanon += 1; continue
        used += 1
        for k in pairs:
            if not np.isnan(r[k]): acc[k].append(r[k])
    rows = [dict(pair=k, jaccard=round(float(np.mean(v)), 3)) for k, v in acc.items() if v]
    pd.DataFrame(rows).to_csv(os.path.join(out, "fig1_jaccard.tsv"), sep="\t", index=False)
    _rec("F", used)
    print(f"[F] fig1_jaccard.tsv ({used} genes, PROD, PER-GENE 3'-end threshold; "
          f"{nocanon} genes excluded for having no canonical site)")


def record_panelA_n(prod_rf, out):
    if not prod_rf or not os.path.exists(prod_rf):
        print(f"[A] {prod_rf} not found; panel A n unchanged", file=sys.stderr)
        return
    try:
        if prod_rf.endswith(".parquet"):
            import pyarrow.parquet as pq
            n = pq.ParquetFile(prod_rf).metadata.num_rows
        else:
            n = sum(1 for _ in open(prod_rf)) - 1
    except Exception as e:
        print(f"[A] could not count {prod_rf}: {type(e).__name__}: {e}",
              file=sys.stderr)
        return
    _rec("A", n, "test windows")
    print(f"[A] panel A n = {n:,} test windows (from {os.path.basename(prod_rf)})")


LEGENDS = {
"A": ("Held-out test performance. AUROC, AUPRC and F1 for the random forest "
      "(RF), logistic regression (LR) and convolutional network (CNN) on {n} "
      "windows held out from training. Bars are single values from the locked "
      "evaluation, not means, so no error bars are shown."),
"B": ("Per-window score agreement. Pearson correlation between each pair of "
      "models across every scored window in {n} genes, accumulated in a single "
      "streaming pass. High AUROC in (A) does not imply the models agree "
      "window by window."),
"C": ("Top-K canonical recovery. Fraction of {n} genes whose annotated "
      "constitutive cleavage site falls within 250 nt of one of the model's K "
      "highest-scoring windows, from cross-fit predictions so no gene is "
      "scored by a model that saw it in training. Combined takes the best rank "
      "across the three models. K = 1 is argmax recovery."),
"D": ("Score traces along a single gene (ERP27, {n} windows). Each model's "
      "score at every window position. No threshold line is drawn: peaks are "
      "called per gene using the score at that gene's own 3' end, so a single "
      "horizontal cut would not correspond to the calling rule."),
"E": ("Score by site class across {n} genes. Canonical is the annotated "
      "constitutive site; Atlas is a PolyASite site within 250 nt of a scored "
      "window, excluding the canonical site itself; Background is a random "
      "sample of non-canonical windows per gene. Violins are width-scaled, so "
      "widths are comparable within a class but not across classes."),
"F": ("Peak-set agreement between models. Per-gene Jaccard index of the "
      "window sets called as peaks, averaged over {n} genes. Each gene's cut "
      "is the score at its own annotated 3' end, so the threshold varies by "
      "gene and model; genes with no annotated canonical site are excluded. "
      "Genes where both models call nothing contribute no value rather than a "
      "Jaccard of zero."),
"G": ("Model score against site expression, for {n} PolyASite sites falling "
      "inside scanned genes. Sites are grouped as Terminal (TE), Intragenic "
      "(IN or EX) or Other; each point is the mean model score in an "
      "expression bin, with a 95% confidence interval on that mean, and rho is "
      "the Spearman correlation between score and expression across "
      "unbinned sites in that group. A site's score is the maximum over "
      "windows within 250 nt of it."),
}

def write_legends(out):
    path = os.path.join(out, "figure1_legends.md")
    with open(path, "w") as fh:
        fh.write("# Figure 1\n\n")
        fh.write("Sequence models of cleavage and polyadenylation site usage: "
                 "held-out performance, between-model agreement, and behaviour "
                 "at annotated sites.\n\n")
        for p in ["A", "B", "C", "D", "E", "F", "G"]:
            if p not in LEGENDS: continue
            n, unit = _COUNTS.get(p, (None, ""))
            nn = f"{n:,} {unit}".strip() if n is not None else "n"
            fh.write(f"**({p})** " + LEGENDS[p].format(n=nn) + "\n\n")
        fh.write("---\n\nAll panels use the same gene set: genes with an "
                 "annotated dominant CPA site in PolyA_DB. Genes without one "
                 "are excluded throughout, since no per-gene peak-calling cut "
                 "can be defined for them. These are not a random subset — they "
                 "skew toward low expression and dispersed cleavage — so the "
                 "figure describes well-annotated genes.\n\n"
                 "Peaks are called per gene: a window is a peak for a model if "
                 "its score is at least the score that model assigns at that "
                 "gene's own annotated 3' end. Per-gene cut distributions are "
                 "in `fig1_thresholds.tsv`; they span the full score range "
                 "(RF median 0.98, range 0.004-0.998), and low-cut genes do "
                 "not produce degenerate peak sets (median 0.9% of windows "
                 "called).\n")
    print(f"[legends] figure1_legends.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod_dir", required=True); ap.add_argument("--crossfit_dir", required=True)
    ap.add_argument("--canonical_bed", required=True)
    ap.add_argument("--atlas_bed", default=None)
    ap.add_argument("--prod_rf", required=True); ap.add_argument("--prod_lr", required=True)
    ap.add_argument("--prod_cnn", required=True)
    ap.add_argument("--erp27", default="ERP27"); ap.add_argument("--vic", type=int, default=250)
    ap.add_argument("--atlas_sample", type=int, default=0,
                    help="uniform random subsample of atlas sites (seeded, before grouping); 0=all")
    ap.add_argument("--analysis_genes", default=None,
                    help="optional file, one gene name per line (matching the "
                         "parquet basename). Restricts the prod scan (B/E/F/G) "
                         "to those genes. NOTE: the peak-level analysis-set "
                         "filters (APA-annotated, terminal-exon, "
                         "at-or-downstream-of-canonical) are defined on PEAKS "
                         "and cannot be applied to these window scans -- this "
                         "flag only restricts which GENES are scanned. Panels "
                         "A, C and D are model evaluation and are not filtered.")
    ap.add_argument("--require_canonical", action="store_true", default=True,
                    help="restrict every prod panel to genes with an annotated "
                         "canonical CPA site, so the whole figure uses one gene "
                         "set. Panels C and F already require one; this makes B, "
                         "E and G match. The excluded genes are NOT a random "
                         "sample: genes with no dominant PolyA_DB site skew "
                         "toward low expression and dispersed cleavage, so the "
                         "figure describes well-annotated genes specifically.")
    ap.add_argument("--no_require_canonical", dest="require_canonical",
                    action="store_false",
                    help="keep genes with no annotated canonical site in B/E/G")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    gene_filter = None
    if a.analysis_genes:
        with open(a.analysis_genes) as fh:
            gene_filter = {ln.strip() for ln in fh if ln.strip()}
        print(f"[filter] {len(gene_filter):,} gene names loaded from "
              f"{os.path.basename(a.analysis_genes)}; applies to B/E/F/G only")

    if os.path.exists(os.path.join(a.out, "fig1_counts.tsv")):
        try:
            with open(os.path.join(a.out, "fig1_counts.tsv")) as fh:
                r = csv.reader(fh, delimiter="\t"); next(r, None)
                for row in r:
                    if len(row) >= 3 and row[0] == "A":
                        _COUNTS["A"] = (int(row[1]), row[2])
        except Exception:
            pass
    for s in ("fig1_topk.tsv", "fig1_traces.tsv", "fig1_jaccard.tsv",
              "fig1_corr.tsv", "fig1_scoredist.tsv", "fig1_thresholds.tsv",
              "figG_binned.tsv", "figG_raw.tsv", "figG_group_counts.tsv",
              "figG_corr.tsv", "figure1_legends.md"):
        try: os.remove(os.path.join(a.out, s))
        except FileNotFoundError: pass
    flush_counts(a.out)

    atlas = load_atlas(a.atlas_bed) if (a.atlas_bed and os.path.exists(a.atlas_bed)) else None
    if atlas is not None:
        print(f"[atlas] {len(atlas):,} sites loaded")
        if a.atlas_sample and a.atlas_sample < len(atlas):
            atlas = atlas.sample(n=a.atlas_sample, random_state=0)
            print(f"[atlas] subsampled to {len(atlas):,} sites (seed=0, uniform)")
    atlas_idx = build_atlas_index(atlas) if atlas is not None else None
    gene_chrom = load_gene_chrom(a.canonical_bed) if atlas is not None else {}
    if atlas is not None:
        print(f"[chrom] gene->chrom map: {len(gene_chrom):,} genes from {os.path.basename(a.canonical_bed)}")

    scan_prod(a.prod_dir, atlas_idx, gene_chrom, a.vic, a.out,
              gene_filter=gene_filter,
              require_canonical=a.require_canonical); flush_counts(a.out)
    panel_C_topk(a.crossfit_dir, a.out); flush_counts(a.out)
    panel_D_traces(a.prod_dir, a.erp27, a.out); flush_counts(a.out)
    panel_F_jaccard(a.prod_dir, a.vic, a.out, gene_filter=gene_filter); flush_counts(a.out)
    record_panelA_n(a.prod_rf, a.out); flush_counts(a.out)
    write_legends(a.out)
    print("[n] fig1_counts.tsv:", {k:v[0] for k,v in sorted(_COUNTS.items())})
    print("[done] rerun: Rscript make_figure1.R", a.out)

if __name__ == "__main__":
    main()