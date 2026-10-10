#!/usr/bin/env python3
# positional control for the lambdarank reranker, done correctly
import argparse, sys
import numpy as np, pandas as pd
import lightgbm as lgb

RF_TEST = {"chr18","chr19","chr20","chr21","chr22","chrX"}
AXES = ["upstream_mean_u1","upstream_max_me3","term_3ss","term_ppt",
        "local_me5","pas_prox_u1","dn5ss_has_strong","interv_me5_max"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peak_table", required=True, help="scored table WITH __wg + rf_prob + gene_length")
    ap.add_argument("--model_txt", required=True)
    ap.add_argument("--features", nargs="+", required=True, help="exact training feature order (8 __wg)")
    ap.add_argument("--scripts_dir", required=True)
    ap.add_argument("--pc_genes_bed", required=True)
    ap.add_argument("--reference_fasta", required=True)
    ap.add_argument("--gene_baselines", required=True)
    ap.add_argument("--u1_lookup", required=True)
    ap.add_argument("--me5_lookup", required=True)
    ap.add_argument("--center_offset", type=int, default=249)
    ap.add_argument("--upstream", type=int, default=3000)
    ap.add_argument("--downstream", type=int, default=1000)
    ap.add_argument("--n_bins", type=int, default=8)
    ap.add_argument("--jitters", type=int, nargs="+", default=[0, 250, 500, 1000, 2000])
    a = ap.parse_args()
    sys.path.insert(0, a.scripts_dir)
    from extract_splicing_features import (GenomeFasta, load_u1_lookup, load_me5_lookup,
                                           init_maxentscan, extract_one_site)
    from compute_gene_baselines import derive_axes

    fa = GenomeFasta(a.reference_fasta)
    u1 = load_u1_lookup(a.u1_lookup); me5 = load_me5_lookup(a.me5_lookup)
    _, score3, _, m3 = init_maxentscan()
    booster = lgb.Booster(model_file=a.model_txt)
    base = pd.read_csv(a.gene_baselines, sep="\t").drop_duplicates("gene").set_index("gene")
    W = a.upstream + a.downstream

    df = pd.read_parquet(a.peak_table)
    df["chrom"] = df["chrom"].astype(str)
    df = df[df.chrom.isin(RF_TEST)].copy()
    pc = pd.read_csv(a.pc_genes_bed, sep="\t", header=None,
                     names=["chrom","start","end","name","score","strand"])
    pc["gene"] = pc["name"].astype(str).str.split("_").str[-1]
    strand = pc.drop_duplicates("gene").set_index("gene")["strand"].to_dict()
    df["strand"] = df["gene"].map(strand).fillna("+")

    keep = []
    for gene, g in df.groupby("gene", sort=False):
        iz = g["in_zone"].to_numpy()
        if iz.sum() == 0:
            continue
        crf = float(g.rf_prob[iz == 1].max())
        if (g.rf_prob[iz == 0] >= crf).any():
            keep.append(gene)
    df = df[df.gene.isin(set(keep))].copy()
    print(f"[jitter] contested test genes: {df.gene.nunique():,}")

    ax_of = {ax + "__wg": ax for ax in AXES}
    feat_axes = [ax_of[f] for f in a.features]

    def shifted_canon_wg(gene, chrom, strand, shifted_ws):
        center = shifted_ws + a.center_offset if strand == "+" else shifted_ws - a.center_offset
        if strand == "+": w0 = max(0, center - a.upstream); w1 = center + a.downstream
        else:             w0 = max(0, center - a.downstream); w1 = center + a.upstream
        seq = fa.fetch(chrom, w0, w1, strand=strand)
        if seq is None or len(seq) < W - 100:
            return None
        feat = extract_one_site(seq, a.upstream, a.downstream, a.n_bins, me5, score3, m3, u1)
        axes = derive_axes(feat, seq, a.upstream)
        vals = {}
        for f, ax in zip(a.features, feat_axes):
            v = axes.get(ax, None)
            bcol = f"baseline_{ax}"
            bv = base.at[gene, bcol] if (gene in base.index and bcol in base.columns) else np.nan
            if v is None or not np.isfinite(v) or not pd.notna(bv):
                vals[f] = np.nan
            else:
                vals[f] = float(v) - float(bv)
        return vals

    genes = []
    for gene, g in df.groupby("gene", sort=False):
        g = g.reset_index(drop=True)
        iz = g["in_zone"].to_numpy()
        ci = int(np.where(iz == 1)[0][0])
        genes.append((gene, g["chrom"].iloc[0], g["strand"].iloc[0],
                      int(g["position"].to_numpy()[ci]), ci,
                      g[a.features].fillna(0.0).to_numpy().copy(), iz, len(g)))

    print(f"  {'jitter':>8}  {'recovery':>9}  {'random':>8}  {'delta':>8}  {'n':>6}")
    base0 = None
    for d in a.jitters:
        rec_s = rec_r = n = 0
        for gene, chrom, strand, cpos, ci, X0, iz, k in genes:
            shifted = cpos + (d if strand == "+" else -d)
            wg = shifted_canon_wg(gene, chrom, strand, shifted)
            if wg is None:
                continue
            X = X0.copy()
            for j, f in enumerate(a.features):
                if np.isfinite(wg[f]):
                    X[ci, j] = wg[f]
            s = booster.predict(X)
            rec_s += int(iz[int(np.argmax(s))] == 1)
            rec_r += iz.sum() / k
            n += 1
        rs = rec_s / n; rr = rec_r / n
        if d == 0:
            base0 = rs
            print(f"  {d:>8}  {rs:>9.3f}  {rr:>8.3f}  {rs-rr:>+8.3f}  {n:>6}   <- d=0 must match real recovery")
        else:
            print(f"  {d:>8}  {rs:>9.3f}  {rr:>8.3f}  {rs-rr:>+8.3f}  {n:>6}")
    print("\nDECAY from d=0 => positional/local/real. FLAT => non-local.")
    if base0 is not None and base0 < 0.20:
        print(f"[WARN] d=0 recovery {base0:.3f} is far below your real ~0.28-0.41; "
              f"centering still off - do NOT trust the decay.")
    else:
        print(f"[ok] d=0 recovery {base0:.3f} is in range; decay is interpretable.")

if __name__ == "__main__":
    main()