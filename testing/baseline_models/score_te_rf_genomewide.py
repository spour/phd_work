#!/usr/bin/env python3
# score the terminal-exon rf at every candidate window in the reranker peak table
import argparse, os, sys
import numpy as np, pandas as pd, joblib

AXES = ["upstream_mean_u1","upstream_max_me3","term_3ss","term_ppt",
        "local_me5","pas_prox_u1","dn5ss_has_strong","interv_me5_max"]
U1_5P = "AUACUUACCUG"
U2_BP = "GUAGUA"
try:
    import RNA
    HAVE_RNA = True
except Exception:
    HAVE_RNA = False

def duplex_mfe(target, query):
    if not HAVE_RNA or not target or not query or len(target) < len(query):
        return np.nan
    try:
        return float(RNA.duplexfold(target.replace("T", "U"), query).energy)
    except Exception:
        return np.nan

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rf_joblib", required=True, help="rf_calibrated.joblib")
    ap.add_argument("--metadata", required=True, help="the run's metadata.json (for column order)")
    ap.add_argument("--peak_table", required=True)
    ap.add_argument("--scripts_dir", required=True)
    ap.add_argument("--pc_genes_bed", required=True)
    ap.add_argument("--reference_fasta", required=True)
    ap.add_argument("--u1_lookup", required=True)
    ap.add_argument("--me5_lookup", required=True)
    ap.add_argument("--center_offset", type=int, default=249)
    ap.add_argument("--upstream", type=int, default=3000)
    ap.add_argument("--downstream", type=int, default=1000)
    ap.add_argument("--n_bins", type=int, default=8)
    ap.add_argument("--duplex_down_nt", type=int, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit_genes", type=int, default=0, help="0 = all; else first N genes by name")
    a = ap.parse_args()
    sys.path.insert(0, a.scripts_dir)

    import json
    from extract_splicing_features import GenomeFasta, load_u1_lookup, load_me5_lookup, init_maxentscan, extract_one_site
    from compute_gene_baselines import derive_axes
    try:
        import splice_extra_features as XF; HAVE_XF = True
    except Exception as e:
        print(f"[warn] splice_extra_features not importable ({e})", file=sys.stderr); HAVE_XF = False

    clf = joblib.load(a.rf_joblib)
    cols = json.load(open(a.metadata))["winner_meta"]["columns"]
    print(f"[model] {a.rf_joblib}  expects {len(cols)} features: {cols}")

    fa = GenomeFasta(a.reference_fasta)
    u1 = load_u1_lookup(a.u1_lookup); me5 = load_me5_lookup(a.me5_lookup)
    _, score3, _, m3 = init_maxentscan()
    W = a.upstream + a.downstream

    df = pd.read_parquet(a.peak_table)
    df["chrom"] = df["chrom"].astype(str)
    pc = pd.read_csv(a.pc_genes_bed, sep="\t", header=None,
                     names=["chrom","start","end","name","score","strand"])
    pc["gene"] = pc["name"].astype(str).str.split("_").str[-1]
    if a.limit_genes > 0:
        keep = sorted(df["gene"].unique())[:a.limit_genes]
        df = df[df["gene"].isin(keep)].copy()
        print(f"[limit] first {a.limit_genes} genes by name -> {len(df):,} candidate windows")
    strand = pc.drop_duplicates("gene").set_index("gene")["strand"].to_dict()
    df["strand"] = df["gene"].map(strand).fillna("+")

    def feats(chrom, pos, strand):
        center = pos + a.center_offset if strand == "+" else pos - a.center_offset
        if strand == "+": w0 = max(0, center - a.upstream); w1 = center + a.downstream
        else:             w0 = max(0, center - a.downstream); w1 = center + a.upstream
        seq = fa.fetch(chrom, w0, w1, strand=strand)
        if seq is None or len(seq) < W - 100:
            return None
        feat = extract_one_site(seq, a.upstream, a.downstream, a.n_bins, me5, score3, m3, u1)
        axes = derive_axes(feat, seq, a.upstream)
        row = {ax: (float(axes.get(ax)) if axes.get(ax, None) is not None else np.nan) for ax in AXES}
        if HAVE_XF:
            try:
                acc = XF._find_acceptor(seq, a.upstream, score3=score3, m3=m3)
                row["bp_strength"]         = XF.branchpoint_score(seq, a.upstream, acc_pos=acc, score3=score3, m3=m3)
                row["u2af65_ppt"]          = XF.u2af65_ppt(seq, a.upstream, acc_pos=acc, score3=score3, m3=m3)
                row["u2af35_ag"]           = XF.u2af35_ag(seq, a.upstream, acc_pos=acc)
                row["u1_suppression_down"] = XF.u1_suppression_down(seq, a.upstream, a.downstream)
            except Exception:
                pass
        if HAVE_RNA:
            row["u1_duplex_down"] = duplex_mfe(seq[a.upstream:a.upstream + a.duplex_down_nt], U1_5P)
            row["u2_duplex_bp"]   = duplex_mfe(seq[max(0, a.upstream - 50):a.upstream - 15], U2_BP)
        return row

    te = np.full(len(df), np.nan, np.float32)
    chroms = df["chrom"].to_numpy(); poss = df["position"].to_numpy(); strs = df["strand"].to_numpy()
    BATCH = 4096
    buf_idx, buf_rows = [], []
    def flush():
        if not buf_idx:
            return
        X = pd.DataFrame(buf_rows).reindex(columns=cols).fillna(0.0).to_numpy()
        p = clf.predict_proba(X)[:, 1]
        for j, pr in zip(buf_idx, p):
            te[j] = pr
        buf_idx.clear(); buf_rows.clear()

    n_ok = n_fail = 0
    for i in range(len(df)):
        r = feats(chroms[i], int(poss[i]), strs[i])
        if r is None:
            n_fail += 1; continue
        buf_idx.append(i); buf_rows.append(r); n_ok += 1
        if len(buf_idx) >= BATCH:
            flush()
        if (i + 1) % 25000 == 0:
            print(f"  {i+1:,}/{len(df):,} featurized", flush=True)
    flush()

    df["te_prob"] = te
    df.drop(columns=["strand"], inplace=True)
    df.to_parquet(a.out, index=False)
    print(f"[done] scored {n_ok:,} ({n_fail:,} failed); te_prob NaN={int(np.isnan(te).sum()):,}")
    print(f"[out] {a.out}")

    ok = df.dropna(subset=["te_prob"])
    can = ok[ok.in_zone == 1]["te_prob"]; cry = ok[ok.in_zone == 0]["te_prob"]
    if len(can) and len(cry):
        from scipy.stats import rankdata
        rr = rankdata(np.concatenate([can.values, cry.values]))
        auc = (rr[:len(can)].sum() - len(can)*(len(can)+1)/2) / (len(can)*len(cry))
        print(f"[gate] te_prob canonical-vs-internal AUROC = {auc:.3f} "
              f"(canon {can.mean():.3f} vs internal {cry.mean():.3f}); >0.55 carries reranker signal")

if __name__ == "__main__":
    main()
