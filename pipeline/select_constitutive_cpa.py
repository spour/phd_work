#!/usr/bin/env python3
# build the positive (constitutive cpa) training set from polya db 3 (hg38.pas.main.tsv)
import argparse, sys
import numpy as np, pandas as pd

SOURCES = ["3READS", "ENCODE.PB", "GTEx.ONT"]


def parse_pas_id(pid):
    m = str(pid).split(":")
    if len(m) != 3:
        return pd.Series([None, None, np.nan])
    chrom, strand, pos = m
    try:
        pos = int(pos)
    except Exception:
        return pd.Series([None, None, np.nan])
    return pd.Series([chrom, strand, pos])


def score_per_source(df):
    cols = {}
    for s in SOURCES:
        pse = pd.to_numeric(df.get(f"PSE.{s}"), errors="coerce")
        rpm = pd.to_numeric(df.get(f"AvgRPM.{s}"), errors="coerce")
        cols[s] = pse * rpm
    return pd.DataFrame(cols, index=df.index)


def combine_score(P, mode):
    if mode == "3reads":
        return P["3READS"]
    if mode == "mean":
        return P.mean(axis=1, skipna=True)
    if mode == "max":
        return P.max(axis=1, skipna=True)
    if mode == "sum":
        return P.sum(axis=1, skipna=True, min_count=1)
    if mode == "best":
        base = P["3READS"].copy()
        alt = P[[c for c in P.columns if c != "3READS"]].mean(axis=1, skipna=True)
        return base.where(base.notna(), alt)
    raise ValueError(mode)


def load_longest_isoform(refgene_path):
    cols = ["bin", "name", "chrom", "strand", "txStart", "txEnd", "cdsStart", "cdsEnd",
            "exonCount", "exonStarts", "exonEnds", "score", "name2", "cdsStartStat",
            "cdsEndStat", "exonFrames"]
    rg = pd.read_csv(refgene_path, sep="\t", header=None, names=cols, comment="#",
                     compression="infer", dtype=str)

    def mrna_len(r):
        st = np.fromiter((int(x) for x in r.exonStarts.strip(",").split(",") if x), dtype=np.int64)
        en = np.fromiter((int(x) for x in r.exonEnds.strip(",").split(",") if x), dtype=np.int64)
        return int((en - st).sum()) if st.size else 0

    rg["mrna_len"] = rg.apply(mrna_len, axis=1)
    rg["nm"] = rg["name"].str.replace(r"\.\d+$", "", regex=True)
    idx = rg.groupby("name2")["mrna_len"].idxmax()
    return {g: nm for g, nm in zip(rg.loc[idx, "name2"], rg.loc[idx, "nm"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pas", required=True, help="hg38.PAS.main.tsv")
    ap.add_argument("--out", default="constitutive_cpa")
    ap.add_argument("--source", default="3reads", choices=["3reads", "mean", "max", "sum", "best"],
                    help="how to build PSExRPM across data sources (default 3reads = PolyA DB 3 native)")
    ap.add_argument("--region", default="3'UTR",
                    help="PAS_type_RefSeq_label to keep (default 3'UTR; 'any' to skip)")
    ap.add_argument("--use_tez", action="store_true",
                    help="also keep PAS_LRtype_label=='TEZ' (union with --region)")
    ap.add_argument("--biotype", default="protein_coding",
                    help="gene_biotype to keep (default protein_coding; 'any' to skip)")
    ap.add_argument("--keep_multimapped", action="store_true",
                    help="keep PAS assigned to >1 gene (default: drop, require PAS2gene==1)")
    ap.add_argument("--ucsc_refgene", default=None,
                    help="UCSC refGene table; restrict to the LONGEST isoform per gene (exact method)")
    ap.add_argument("--min_score", type=float, default=None,
                    help="optional: require final PSExRPM >= this")
    ap.add_argument("--top_percentile", type=float, default=None,
                    help="optional: keep only genes whose site is in the top X%% of PSExRPM")
    a = ap.parse_args()

    df = pd.read_csv(a.pas, sep="\t", dtype=str)
    n0 = len(df)

    df[["chrom", "strand", "pos"]] = df["PAS_ID"].apply(parse_pas_id)
    df = df[df["pos"].notna()].copy()
    df["pos"] = df["pos"].astype(int)

    if a.biotype != "any" and "gene_biotype" in df.columns:
        df = df[df["gene_biotype"] == a.biotype]
    if not a.keep_multimapped and "PAS2gene" in df.columns:
        df = df[pd.to_numeric(df["PAS2gene"], errors="coerce") == 1]
    n_after_basic = len(df)

    if a.region == "any":
        keep = pd.Series(True, index=df.index)
    else:
        keep = (df["PAS_type_RefSeq_label"] == a.region)
    if a.use_tez and "PAS_LRtype_label" in df.columns:
        keep = keep | (df["PAS_LRtype_label"] == "TEZ")
    df = df[keep]

    Psrc = score_per_source(df)
    df["PSExRPM"] = combine_score(Psrc, a.source)
    for s in SOURCES:
        df[f"PSExRPM.{s}"] = Psrc[s]
    n_drop = int(df["PSExRPM"].isna().sum())
    df = df[df["PSExRPM"].notna()].copy()

    if a.ucsc_refgene:
        longest = load_longest_isoform(a.ucsc_refgene)
        df["nm"] = df["transcript_id"].str.replace(r"\.\d+$", "", regex=True)
        df = df[df.apply(lambda r: longest.get(r["GeneSymbol"]) == r["nm"], axis=1)]
        iso_note = "restricted to the longest-mRNA RefSeq isoform per gene (refGene)"
    else:
        iso_note = ("NOTE: no --ucsc_refgene supplied; taking the gene's dominant 3'UTR site "
                    "(highest PSExRPM) as the constitutive site (longest-isoform step approximated)")

    df = df.sort_values("PSExRPM", ascending=False)
    top = df.groupby("GeneSymbol", as_index=False).first()

    if a.top_percentile is not None and len(top):
        thr = np.percentile(top["PSExRPM"], 100 - a.top_percentile)
        top = top[top["PSExRPM"] >= thr]
    if a.min_score is not None:
        top = top[top["PSExRPM"] >= a.min_score]

    top = top.sort_values(["chrom", "pos"]).reset_index(drop=True)

    # outputs
    tsv_cols = (["PAS_ID", "GeneSymbol", "chrom", "strand", "pos", "PSExRPM"]
                + [f"PSExRPM.{s}" for s in SOURCES]
                + [f"PSE.{s}" for s in SOURCES] + [f"AvgRPM.{s}" for s in SOURCES]
                + ["PAS_hexamer", "PolyaStrength", "PolyaStrength_percentile",
                   "PAS_type_RefSeq_label", "PAS_LRtype_label", "transcript_id", "TEZ_span",
                   "seq_us75", "pos_us75", "seq_ds75", "pos_ds75"])
    tsv_cols = [c for c in tsv_cols if c in top.columns]
    top[tsv_cols].to_csv(f"{a.out}.tsv", sep="\t", index=False)

    bed = top[["chrom", "pos", "GeneSymbol", "PSExRPM", "strand"]].copy()
    bed["start"] = bed["pos"] - 1
    bed["end"] = bed["pos"]
    bed["name"] = "canonical_" + bed["GeneSymbol"].astype(str)
    bed["score"] = bed["PSExRPM"].round(3)
    bed[["chrom", "start", "end", "name", "score", "strand"]].to_csv(
        f"{a.out}.bed", sep="\t", index=False, header=False)

    print(f"[select] input PAS                : {n0:,}", file=sys.stderr)
    print(f"[select] after biotype/unique     : {n_after_basic:,}", file=sys.stderr)
    print(f"[select] after region ({a.region}{'+TEZ' if a.use_tez else ''}) : {len(df)+n_drop:,} "
          f"(- {n_drop:,} NA {a.source} score = {len(df):,} rankable)", file=sys.stderr)
    print(f"[select] {iso_note}", file=sys.stderr)
    print(f"[select] constitutive sites (1/gene): {len(top):,} genes", file=sys.stderr)
    if len(top):
        q = top["PSExRPM"].quantile
        print(f"[select] PSExRPM  median={top['PSExRPM'].median():.1f}  "
              f"IQR=[{q(.25):.1f}, {q(.75):.1f}]  max={top['PSExRPM'].max():.1f}", file=sys.stderr)
    print(f"[select] wrote {a.out}.tsv and {a.out}.bed", file=sys.stderr)


if __name__ == "__main__":
    main()