#!/usr/bin/env python3
# compute the 6 splice features missing from the peak table in within-gene-centered
import argparse, sys
import numpy as np, pandas as pd

U1_5P = "AUACUUACCUG"; U2_BP = "GUAGUA"
NEW = ["bp_strength","u2af65_ppt","u2af35_ag","u1_suppression_down",
       "u1_duplex_down","u2_duplex_bp"]
try:
    import RNA; HAVE_RNA = True
except Exception:
    HAVE_RNA = False

def duplex_mfe(target, query):
    if not HAVE_RNA or not target or len(target) < len(query):
        return np.nan
    try:
        return float(RNA.duplexfold(target.replace("T","U"), query).energy)
    except Exception:
        return np.nan

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peak_table", required=True)
    ap.add_argument("--scripts_dir", required=True)
    ap.add_argument("--pc_genes_bed", required=True)
    ap.add_argument("--reference_fasta", required=True)
    ap.add_argument("--center_offset", type=int, default=249)
    ap.add_argument("--upstream", type=int, default=3000)
    ap.add_argument("--downstream", type=int, default=1000)
    ap.add_argument("--duplex_down_nt", type=int, default=300)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.scripts_dir)
    from extract_splicing_features import GenomeFasta, init_maxentscan
    import splice_extra_features as XF

    fa = GenomeFasta(a.reference_fasta)
    _, score3, _, m3 = init_maxentscan()
    W = a.upstream + a.downstream

    df = pd.read_parquet(a.peak_table)
    df["chrom"] = df["chrom"].astype(str)
    pc = pd.read_csv(a.pc_genes_bed, sep="\t", header=None,
                     names=["chrom","start","end","name","score","strand"])
    pc["gene"] = pc["name"].astype(str).str.split("_").str[-1]
    strand = pc.drop_duplicates("gene").set_index("gene")["strand"].to_dict()
    df["strand"] = df["gene"].map(strand).fillna("+")

    def raw_feats(chrom, pos, strand):
        center = pos + a.center_offset if strand == "+" else pos - a.center_offset
        if strand == "+": w0 = max(0, center - a.upstream); w1 = center + a.downstream
        else:             w0 = max(0, center - a.downstream); w1 = center + a.upstream
        seq = fa.fetch(chrom, w0, w1, strand=strand)
        if seq is None or len(seq) < W - 100:
            return {f: np.nan for f in NEW}
        try:
            acc = XF._find_acceptor(seq, a.upstream, score3=score3, m3=m3)
            r = {"bp_strength":         XF.branchpoint_score(seq, a.upstream, acc_pos=acc, score3=score3, m3=m3),
                 "u2af65_ppt":          XF.u2af65_ppt(seq, a.upstream, acc_pos=acc, score3=score3, m3=m3),
                 "u2af35_ag":           XF.u2af35_ag(seq, a.upstream, acc_pos=acc),
                 "u1_suppression_down": XF.u1_suppression_down(seq, a.upstream, a.downstream)}
        except Exception:
            r = {"bp_strength": np.nan, "u2af65_ppt": np.nan, "u2af35_ag": np.nan,
                 "u1_suppression_down": np.nan}
        r["u1_duplex_down"] = duplex_mfe(seq[a.upstream:a.upstream + a.duplex_down_nt], U1_5P)
        r["u2_duplex_bp"]   = duplex_mfe(seq[max(0, a.upstream - 50):a.upstream - 15], U2_BP)
        return r

    raw = {f: np.full(len(df), np.nan, np.float32) for f in NEW}
    chroms = df["chrom"].to_numpy(); poss = df["position"].to_numpy(); strs = df["strand"].to_numpy()
    for i in range(len(df)):
        r = raw_feats(chroms[i], int(poss[i]), strs[i])
        for f in NEW:
            raw[f][i] = r[f]
        if (i + 1) % 25000 == 0:
            print(f"  {i+1:,}/{len(df):,} featurized", flush=True)
    for f in NEW:
        df[f] = raw[f]

    canon = df[df.in_zone == 1].drop_duplicates("gene").set_index("gene")
    for f in NEW:
        bvals = df["gene"].map(canon[f])
        df[f + "__wg"] = df[f].to_numpy() - bvals.to_numpy()

    df.drop(columns=["strand"] + NEW, inplace=True)
    df.to_parquet(a.out, index=False)
    have = [c for c in df.columns if c.endswith("__wg")]
    print(f"[done] {a.out}")
    print(f"[__wg columns present] {sorted(have)}")

if __name__ == "__main__":
    main()