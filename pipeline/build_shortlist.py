#!/usr/bin/env python3
# (consensus rebuild for figure 2) per gene, from its crossfit scan parquet (leak-free
import argparse, os, sys
import numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor

def _nworkers(cap=16):
    n = os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 2
    try: n = int(n)
    except Exception: n = 2
    return max(2, min(n, cap))
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k): return x

TEST_CHROMS = {"chr18","chr19","chr20","chr21","chr22","chrX"}
VAL_CHROMS  = {"chr3","chr13"}

MODELS = ["rf", "lr", "cnn"]
PROB   = {"rf": "rf_prob", "lr": "lr_prob", "cnn": "cnn_prob"}
EXTRA_FROM_OLD = ["aparent2_prob"]

def load_canonical(bed):
    df = pd.read_csv(bed, sep="\t", header=None,
                     names=["chrom","start","end","symbol","enst","strand"])
    df["chrom"] = df["chrom"].astype(str)
    df["chrom"] = df["chrom"].where(df["chrom"].str.startswith("chr"), "chr"+df["chrom"])
    by = {}
    for _, r in df.iterrows():
        by.setdefault(r["symbol"], []).append(
            (r["chrom"], int(r["start"]), int(r["end"]), r["strand"]))
    print(f"[canon] {len(df):,} canonical windows, {len(by):,} symbols", file=sys.stderr)
    return by

def nms_peaks(centers, scores, merge_dist, reserved_center=None):
    n = len(centers)
    if n == 0: return []
    oc = np.argsort(centers, kind="mergesort")
    cs = centers[oc]
    suppressed = np.zeros(n, bool)
    kept = []
    def suppress(c):
        lo = np.searchsorted(cs, c - merge_dist, side="left")
        hi = np.searchsorted(cs, c + merge_dist, side="right")
        suppressed[oc[lo:hi]] = True
    if reserved_center is not None:
        lo = np.searchsorted(cs, reserved_center - merge_dist, side="left")
        hi = np.searchsorted(cs, reserved_center + merge_dist, side="right")
        if hi > lo:
            block = oc[lo:hi]
            cpk = int(block[np.argmax(scores[block])])
            kept.append(cpk); suppress(centers[cpk])
    for i in np.argsort(scores, kind="mergesort")[::-1]:
        if suppressed[i]: continue
        kept.append(int(i)); suppress(centers[i])
    return kept

def _span(sorted_arr, x, r):
    return (np.searchsorted(sorted_arr, x - r, side="left"),
            np.searchsorted(sorted_arr, x + r, side="right"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_gene_dir", required=True,
                    help="crossfit per-gene scans (leak-free rf/lr/cnn)")
    ap.add_argument("--canonical_bed", required=True)
    ap.add_argument("--aparent_scan_dir", default=None,
                    help="OLD scan dir carrying aparent2_prob (+other extras), "
                         "joined on window_start_genomic. Optional.")
    ap.add_argument("--pc_genes_bed", default=None, help="optional, sanity only")
    ap.add_argument("--win", type=int, default=500)
    ap.add_argument("--merge_dist", type=int, default=250,
                    help="NMS collapse distance AND canonical-vicinity radius (nt). "
                         "A surviving peak within merge_dist of the annotated "
                         "canonical center is the canonical's shoulder; beyond it "
                         "it is a distinct internal competitor.")
    ap.add_argument("--agree_mode", default="site", choices=["site","window"],
                    help="site: each model has its OWN peak (clearing its OWN bar) within "
                         "merge_dist of the site; per-model scores are that model's peak. "
                         "window: additionally require ONE window to clear all three bars "
                         "simultaneously (stricter; discards offset-peak sites).")
    ap.add_argument("--score_col", default="consensus",
                    choices=["consensus","rf_prob","lr_prob","cnn_prob"],
                    help="consensus = all three models beat canonical; or a single model")
    ap.add_argument("--split", default="all", choices=["all","test","val","train"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--out", default="shortlist_crossfit")
    a = ap.parse_args()
    off = a.win // 2
    merge_dist = a.merge_dist

    canon = load_canonical(a.canonical_bed)
    genes = sorted(canon.keys())
    if a.split != "all":
        sel = TEST_CHROMS if a.split=="test" else VAL_CHROMS if a.split=="val" else None
        def keep(g):
            chl = {c for c,_,_,_ in canon[g]}
            if a.split == "train": return not (chl & TEST_CHROMS) and not (chl & VAL_CHROMS)
            return bool(chl & sel)
        genes = [g for g in genes if keep(g)]
    if a.n_shards > 1:
        genes = genes[a.shard::a.n_shards]
        print(f"[shard] {a.shard}/{a.n_shards}: {len(genes):,} genes", file=sys.stderr)
    print(f"[genes] {len(genes):,} genes in split={a.split}; selection={a.score_col}", file=sys.stderr)

    read_cols = ["window_start_genomic"] + list(PROB.values())
    rows = []; summ = []
    missing = 0; no_overlap = 0
    ap_match_num = 0; ap_match_den = 0; ap_gene_hit = 0; ap_gene_tot = 0

    # i/o prefetch
    def _load(g):
        pth = os.path.join(a.per_gene_dir, f"{g}.parquet")
        if not os.path.exists(pth): return (g, None, None)
        try:
            d = pd.read_parquet(pth, columns=read_cols).dropna(subset=list(PROB.values()))
        except Exception as e:
            print(f"[skip] {g}: {type(e).__name__}", file=sys.stderr); return (g, None, None)
        od = None
        if a.aparent_scan_dir:
            op = os.path.join(a.aparent_scan_dir, f"{g}.parquet")
            if os.path.exists(op):
                try: od = pd.read_parquet(op, columns=["window_start_genomic"] + EXTRA_FROM_OLD)
                except Exception: od = None
        return (g, d, od)

    CHUNK = 512
    def _stream():
        with ThreadPoolExecutor(max_workers=_nworkers()) as ex:
            for i in range(0, len(genes), CHUNK):
                for item in ex.map(_load, genes[i:i+CHUNK]):
                    yield item

    for g, d, od_pre in tqdm(_stream(), total=len(genes), desc="shortlist"):
        if d is None: missing += 1; continue
        if len(d) == 0: continue
        ws = d["window_start_genomic"].to_numpy()
        we = ws + a.win
        S = {m: d[PROB[m]].to_numpy() for m in MODELS}

        is_canon = np.zeros(len(d), bool); canon_centers = []
        for chrom, cs, ce, strand in canon[g]:
            is_canon |= (ws < ce) & (we > cs)
            canon_centers.append((cs + ce)//2)
        if not is_canon.any():
            no_overlap += 1; continue
        m_canon = {m: float(S[m][is_canon].max()) for m in MODELS}

        c_all = ws + off
        cc = np.array(canon_centers)
        canon_idx = np.nonzero(is_canon)[0]

        mean_score = np.mean([S[m] for m in MODELS], axis=0)
        canon_peak_center = int(c_all[canon_idx[np.argmax(mean_score[canon_idx])]])

        # consensus (symmetric, no fallback)
        if a.score_col == "consensus":
            peak_idx = {}; peak_ctr = {}; peak_scr = {}
            for m in MODELS:
                hi = np.nonzero(S[m] >= m_canon[m])[0]
                if len(hi) == 0:
                    peak_idx[m] = np.array([], dtype=int); peak_ctr[m] = np.array([])
                    peak_scr[m] = np.array([]); continue
                keepm = nms_peaks(c_all[hi], S[m][hi], merge_dist,
                                  reserved_center=canon_peak_center)
                pi = hi[np.asarray(keepm, dtype=int)]
                order = np.argsort(c_all[pi], kind="mergesort")
                pi = pi[order]
                peak_idx[m] = pi; peak_ctr[m] = c_all[pi].astype(float); peak_scr[m] = S[m][pi]
            if any(peak_ctr[m].size == 0 for m in MODELS):
                comp_items = []
            else:
                anchors = np.unique(np.concatenate([peak_ctr[m] for m in MODELS]))
                clears_all = (S["rf"] >= m_canon["rf"]) & (S["lr"] >= m_canon["lr"]) & \
                             (S["cnn"] >= m_canon["cnn"])
                clear_idx = np.nonzero(clears_all)[0]
                clear_ctr = c_all[clear_idx]
                seen = {}
                for x in anchors:
                    ok = True; pm_ctr = {}; pm_scr = {}
                    for m in MODELS:
                        arr = peak_ctr[m]
                        j = np.searchsorted(arr, x)
                        best = np.inf; bk = -1
                        for k in (j-1, j):
                            if 0 <= k < arr.size:
                                dd = abs(arr[k] - x)
                                if dd < best: best = dd; bk = k
                        if best > merge_dist: ok = False; break
                        pm_ctr[m] = float(arr[bk]); pm_scr[m] = float(peak_scr[m][bk])
                    if not ok: continue
                    if a.agree_mode == "window":
                        lo, hi2 = _span(clear_ctr, x, merge_dist)
                        if hi2 <= lo: continue
                        block = clear_idx[lo:hi2]
                        rep = int(block[np.argmax(mean_score[block])])
                        rep_scr = {m: float(S[m][rep]) for m in MODELS}
                    else:
                        med = float(np.median([pm_ctr[m] for m in MODELS]))
                        rep = int(np.argmin(np.abs(c_all - med)))
                        rep_scr = dict(pm_scr)
                    if rep not in seen:
                        seen[rep] = (pm_ctr, rep_scr)
                comp_items = [(rep, pm_ctr, rep_scr) for rep, (pm_ctr, rep_scr) in seen.items()
                              if abs(int(c_all[rep]) - canon_peak_center) > merge_dist]
        else:
            m = a.score_col.split("_")[0]
            hi = np.nonzero(S[m] >= m_canon[m])[0]
            if len(hi) == 0: comp_items = []
            else:
                keepm = nms_peaks(c_all[hi], S[m][hi], merge_dist,
                                  reserved_center=canon_peak_center)
                comp_items = [(int(i), {}, {mm: float(S[mm][i]) for mm in MODELS})
                              for i in hi[np.asarray(keepm, dtype=int)]
                              if abs(int(c_all[i]) - canon_peak_center) > merge_dist]

        # canonical is a reference, not a candidate
        canon_rep = int(canon_idx[np.argmax(mean_score[canon_idx])])
        peak_items = [(canon_rep, {m: float(canon_peak_center) for m in MODELS},
                       {m: float(m_canon[m]) for m in MODELS}, True)] + \
                     [(rep, pm, sc, False) for rep, pm, sc in comp_items]

        strand = canon[g][0][3]; chrom = canon[g][0][0]
        peaks = np.array([p[0] for p in peak_items], dtype=int)

        extra = {c: {} for c in EXTRA_FROM_OLD}
        if a.aparent_scan_dir:
            ap_gene_tot += 1
            if od_pre is not None:
                have = [c for c in EXTRA_FROM_OLD]
                try:
                    od = od_pre
                    ap_gene_hit += 1
                    ows = od["window_start_genomic"].to_numpy()
                    oo = np.argsort(ows, kind="mergesort"); ows_s = ows[oo]
                    want = ws[peaks]
                    j = np.searchsorted(ows_s, want)
                    jc = np.clip(j, 0, max(len(ows_s)-1, 0))
                    exact = (len(ows_s) > 0) & (j < len(ows_s)) & (ows_s[jc] == want)
                    for c in have:
                        arr = od[c].to_numpy()[oo]
                        for t, i in enumerate(peaks):
                            if exact[t]: extra[c][i] = float(arr[jc[t]])
                except Exception as e:
                    print(f"[aparent] {g}: {type(e).__name__}", file=sys.stderr)

        peak_rows = []
        for i, pm, sc, is_c in peak_items:
            ctr = int(c_all[i])
            d_can = int(abs(ctr - canon_peak_center))
            peak_rows.append((i, ctr, d_can, bool(is_c), pm, sc))
        n_short = len(peak_rows); n_internal = sum(1 for _,_,_,c,_,_ in peak_rows if not c)

        for i, ctr, d_can, is_c, pm, sc in peak_rows:
            row = dict(gene=g, chrom=chrom, strand=strand,
                       window_start_genomic=int(ws[i]), center=ctr,
                       rf_prob=sc["rf"], lr_prob=sc["lr"], cnn_prob=sc["cnn"],
                       is_canonical=is_c, dist_to_canon=d_can,
                       rf_canonical=m_canon["rf"], lr_canonical=m_canon["lr"],
                       cnn_canonical=m_canon["cnn"],
                       rf_center=pm.get("rf", np.nan), lr_center=pm.get("lr", np.nan),
                       cnn_center=pm.get("cnn", np.nan),
                       n_shortlist=n_short, n_internal=n_internal)
            for c in EXTRA_FROM_OLD:
                v = extra[c].get(i, np.nan)
                row[c] = v
                if c == "aparent2_prob":
                    ap_match_den += 1; ap_match_num += int(np.isfinite(v))
            rows.append(row)
        summ.append(dict(gene=g, chrom=chrom, n_shortlist=n_short,
                         n_internal=n_internal, contested=int(n_internal > 0),
                         **{f"{m}_canonical": m_canon[m] for m in MODELS}))

    R = pd.DataFrame(rows); Ssum = pd.DataFrame(summ)
    suf = f".{a.shard:03d}" if a.n_shards > 1 else ""
    R.to_parquet(f"{a.out}_windows{suf}.parquet", index=False)
    Ssum.to_csv(f"{a.out}_summary{suf}.tsv", sep="\t", index=False)

    print(f"\n[built] {len(R):,} shortlisted peaks over {len(Ssum):,} genes "
          f"({missing} parquets missing, {no_overlap} genes had no window "
          f"overlapping canonical)", file=sys.stderr)
    if a.aparent_scan_dir:
        gfrac = ap_gene_hit/max(ap_gene_tot,1)
        wfrac = ap_match_num/max(ap_match_den,1)
        print(f"[aparent] old-scan covered {ap_gene_hit:,}/{ap_gene_tot:,} genes "
              f"({gfrac:.1%}); APARENT2 attached to {ap_match_num:,}/{ap_match_den:,} "
              f"peaks ({wfrac:.1%}). Peaks w/o APARENT2 have aparent2_prob=NaN.",
              file=sys.stderr)
        if wfrac < 0.5 and ap_gene_hit > 0:
            print("[aparent][WARN] <50% of peaks matched on window_start_genomic — "
                  "possible coordinate-origin mismatch between crossfit and old scan.",
                  file=sys.stderr)
    if len(Ssum):
        contested = Ssum[Ssum.contested == 1]
        print(f"[contest] {len(contested):,}/{len(Ssum):,} genes contested "
              f"({len(contested)/len(Ssum):.1%})", file=sys.stderr)
        for lab, lo, hi in [("1",1,1),("2",2,2),("3-4",3,4),("5-8",5,8),("9+",9,10**9)]:
            n = int(((Ssum.n_shortlist>=lo)&(Ssum.n_shortlist<=hi)).sum())
            print(f"    {lab:<5} {n:>6}  {n/len(Ssum):.3f}", file=sys.stderr)
    print(f"[out] {a.out}_windows.parquet, {a.out}_summary.tsv", file=sys.stderr)

if __name__ == "__main__":
    main()