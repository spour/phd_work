#!/usr/bin/env python3
# featurize the terminal-exon dataset windows with splice features and write
import argparse, os, sys
import numpy as np, pandas as pd
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k): return x

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

def parse_header(h):
    h = h[1:].split("|")
    loc = h[1].split(":")
    return h[0], loc[0], int(loc[1]), loc[2], h[2]

def read_fasta(path):
    recs = []; tag = chrom = center = strand = sp = None; seq = []
    for line in open(path):
        if line.startswith(">"):
            if tag is not None:
                recs.append((tag, chrom, center, strand, sp, "".join(seq)))
            tag, chrom, center, strand, sp = parse_header(line.rstrip("\n")); seq = []
        else:
            seq.append(line.strip())
    if tag is not None:
        recs.append((tag, chrom, center, strand, sp, "".join(seq)))
    return recs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--te_dir", required=True)
    ap.add_argument("--scripts_dir", required=True)
    ap.add_argument("--reference_fasta", required=True)
    ap.add_argument("--u1_lookup", required=True)
    ap.add_argument("--me5_lookup", required=True)
    ap.add_argument("--upstream", type=int, default=3000)
    ap.add_argument("--downstream", type=int, default=1000)
    ap.add_argument("--n_bins", type=int, default=8)
    ap.add_argument("--duplex_down_nt", type=int, default=300, help="downstream window for U1 duplex")
    ap.add_argument("--no_rnahybrid", action="store_true", help="skip the 2 ViennaRNA duplex features")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    sys.path.insert(0, a.scripts_dir)

    from extract_splicing_features import (GenomeFasta, load_u1_lookup, load_me5_lookup,
                                           init_maxentscan, extract_one_site)
    from compute_gene_baselines import derive_axes
    try:
        import splice_extra_features as XF; HAVE_XF = True
    except Exception as e:
        print(f"[warn] splice_extra_features not importable ({e}); axes only", file=sys.stderr)
        HAVE_XF = False
    use_rna = HAVE_RNA and not a.no_rnahybrid
    print(f"[features] axes=8  splice_extra={'4' if HAVE_XF else '0'}  "
          f"ViennaRNA_duplex={'2' if use_rna else '0 (ViennaRNA not importable)' if not HAVE_RNA else '0 (--no_rnahybrid)'}",
          file=sys.stderr)

    fa = GenomeFasta(a.reference_fasta)
    u1 = load_u1_lookup(a.u1_lookup); me5 = load_me5_lookup(a.me5_lookup)
    _, score3, _, m3 = init_maxentscan()
    W = a.upstream + a.downstream

    def featurize(chrom, center, strand):
        if strand == "+": w0 = max(0, center - a.upstream); w1 = center + a.downstream
        else:             w0 = max(0, center - a.downstream); w1 = center + a.upstream
        seq = fa.fetch(chrom, w0, w1, strand=strand)
        if seq is None or len(seq) < W - 100:
            return None
        feat = extract_one_site(seq, a.upstream, a.downstream, a.n_bins, me5, score3, m3, u1)
        axes = derive_axes(feat, seq, a.upstream)
        row = {}
        for ax in AXES:
            v = axes.get(ax, None) if hasattr(axes, "get") else None
            row[ax] = float(v) if v is not None else np.nan
        if HAVE_XF:
            try:
                acc = XF._find_acceptor(seq, a.upstream, score3, m3)
                row["bp_strength"]         = XF.branchpoint_score(seq, a.upstream, acc_pos=acc)
                row["u2af65_ppt"]          = XF.u2af65_ppt(seq, a.upstream, acc_pos=acc)
                row["u2af35_ag"]           = XF.u2af35_ag(seq, a.upstream, acc_pos=acc)
                row["u1_suppression_down"] = XF.u1_suppression_down(seq, a.upstream, a.downstream)
            except Exception:
                pass
        if use_rna:
            dn = seq[a.upstream:a.upstream + a.duplex_down_nt]
            up = seq[max(0, a.upstream - 50):a.upstream - 15]
            row["u1_duplex_down"] = duplex_mfe(dn, U1_5P)
            row["u2_duplex_bp"]   = duplex_mfe(up, U2_BP)
        return row

    for cls in ("pos", "neg"):
        for sp in ("train", "val", "test"):
            fpath = os.path.join(a.te_dir, f"{cls}_{sp}.fa")
            if not os.path.exists(fpath):
                continue
            recs = read_fasta(fpath); rows = {}
            for tag, chrom, center, strand, _sp, _seq in tqdm(recs, desc=f"{cls}_{sp}"):
                r = featurize(chrom, center, strand)
                if r is not None:
                    rows[f"{tag}_{chrom}_{center}_{strand}"] = r
            df = pd.DataFrame.from_dict(rows, orient="index").fillna(0.0)
            df.to_csv(os.path.join(a.out_dir, f"{cls}_{sp}.tsv"), sep="\t")
            print(f"[{cls}_{sp}] {len(df):,} rows, {df.shape[1]} features -> {cls}_{sp}.tsv",
                  file=sys.stderr)
    print(f"[done] feature TSVs in {a.out_dir}", file=sys.stderr)

if __name__ == "__main__":
    main()
