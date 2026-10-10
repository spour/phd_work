#!/usr/bin/env python3
# per-site signal dump for the figure 3 raster / heatmap
import argparse, sys
import numpy as np, pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scripts_dir", default=None)
    ap.add_argument("--shortlist", required=True)
    ap.add_argument("--reference_fasta", required=True)
    ap.add_argument("--u1_lookup", required=True)
    ap.add_argument("--me5_lookup", required=True)
    ap.add_argument("--max_nt", type=int, default=2000)
    ap.add_argument("--dn_nt", type=int, default=500)
    ap.add_argument("--n_sites", type=int, default=100, help="sites/class, or genes if --paired")
    ap.add_argument("--paired", action="store_true", help="same genes in both classes; shared site_id")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--acc_floor", type=float, default=0.0)
    ap.add_argument("--don_floor", type=float, default=0.0)
    ap.add_argument("--u1_keep_lo", type=float, default=-4.0)
    ap.add_argument("--u1_keep_hi", type=float, default=2.0)
    ap.add_argument("--out", default="fig3_raster_sites.tsv")
    a = ap.parse_args()
    if a.scripts_dir:
        sys.path.insert(0, a.scripts_dir)
    from extract_splicing_features import GenomeFasta, init_maxentscan, load_u1_lookup

    o = init_maxentscan(); score5, score3 = o[0], o[1]
    fa = GenomeFasta(a.reference_fasta)
    u1 = load_u1_lookup(a.u1_lookup)
    if len(u1) == 0:
        sys.exit("[err] U1 lookup is empty")
    U1LEN = len(next(iter(u1)))
    MAP = np.full(256, -1, np.int64)
    for b, code in zip(b"ACGT", range(4)):
        MAP[b] = code
    dense = np.full(4 ** U1LEN, np.nan, np.float32)
    powv = (4 ** np.arange(U1LEN - 1, -1, -1)).astype(np.int64)
    for kmer, val in u1.items():
        idx = 0
        for ch in kmer.encode():
            idx = idx * 4 + int(MAP[ch])
        dense[idx] = val

    def u1_full(seq):
        arr = MAP[np.frombuffer(seq.encode("ascii", "replace"), np.uint8)]
        n = len(arr)
        if n < U1LEN:
            return np.empty(0)
        codes = np.zeros(n - U1LEN + 1, np.int64); bad = np.zeros(n - U1LEN + 1, bool)
        for k in range(U1LEN):
            c = arr[k:k + (n - U1LEN + 1)]; codes += c * powv[k]; bad |= (c < 0)
        return np.where(bad, np.nan, dense[np.clip(codes, 0, len(dense) - 1)])

    df = pd.read_parquet(a.shortlist)

    # choose sites
    if a.paired:
        canon = df[df.is_canonical]
        comp = df[(~df.is_canonical) & (df.n_internal > 0)]
        genes = sorted(set(canon.gene) & set(comp.gene))
        rngg = np.random.default_rng(a.seed)
        genes = list(np.array(genes)[rngg.permutation(len(genes))][:a.n_sites])
        sel = []
        for gi, g in enumerate(genes):
            cr = canon[canon.gene == g]
            pr = comp[comp.gene == g]
            if len(cr) == 0 or len(pr) == 0:
                continue
            cr = cr.iloc[0]
            pr = (pr.sort_values("n_internal", ascending=False).iloc[0]
                  if "n_internal" in pr.columns else pr.iloc[0])
            sel.append({**cr.to_dict(), "cls": "Canonical", "sid": gi})
            sel.append({**pr.to_dict(), "cls": "Competitor", "sid": gi})
        samp = pd.DataFrame(sel)
    else:
        df = df[df.n_internal > 0].copy()
        df["cls"] = np.where(df.is_canonical, "Canonical", "Competitor")
        parts = [gg.sample(min(a.n_sites, len(gg)), random_state=a.seed).assign(cls=cls)
                 for cls, gg in df.groupby("cls")]
        samp = pd.concat(parts, ignore_index=True)
        samp["sid"] = samp.groupby("cls").cumcount()
    print(f"[data] {samp.cls.value_counts().to_dict()}  paired={a.paired}  "
          f"window [-{a.max_nt}, +{a.dn_nt}]", file=sys.stderr)

    rows = []; n_bad = 0
    for rec in samp.to_dict("records"):
        cls = rec["cls"]; c = int(rec["center"]); strand = rec["strand"]; chrom = str(rec["chrom"])
        sid = int(rec["sid"]); gene = rec.get("gene")
        if strand == "+":
            w0, w1 = max(0, c - a.max_nt), c + a.dn_nt
        else:
            w0, w1 = max(0, c - a.dn_nt), c + a.max_nt
        try:
            seq = fa.fetch(chrom, w0, w1, strand=strand)
        except Exception:
            seq = None
        if not seq or len(seq) < 30:
            n_bad += 1
            continue
        seq = seq.upper(); L = len(seq); site = L - a.dn_nt
        arr = np.frombuffer(seq.encode("ascii", "replace"), np.uint8)

        ags = np.where((arr[:-1] == 65) & (arr[1:] == 71))[0] + 2
        for i in ags:
            p = int(i)
            if p - 20 < 0 or p + 3 > L:
                continue
            try:
                v = score3(seq[p - 20:p + 3])
                v = float(np.nanmax(np.asarray(v, float))) if not np.isscalar(v) else float(v)
            except Exception:
                continue
            if np.isfinite(v) and v >= a.acc_floor:
                rows.append((cls, sid, gene, chrom, c, strand, "acceptor", float(p - site), v))

        gts = np.where((arr[:-1] == 71) & (arr[1:] == 84))[0]
        for i in gts:
            p = int(i)
            if p - 3 < 0 or p + 6 > L:
                continue
            try:
                v = score5(seq[p - 3:p + 6])
                v = float(np.nanmax(np.asarray(v, float))) if not np.isscalar(v) else float(v)
            except Exception:
                continue
            if np.isfinite(v) and v >= a.don_floor:
                rows.append((cls, sid, gene, chrom, c, strand, "donor", float(p - site), v))

        u1v = u1_full(seq)
        if len(u1v):
            starts = np.arange(len(u1v))
            keep = np.isfinite(u1v) & ((u1v <= a.u1_keep_lo) | (u1v >= a.u1_keep_hi))
            for p, v in zip(starts[keep].tolist(), u1v[keep].tolist()):
                rows.append((cls, sid, gene, chrom, c, strand, "u1", float(int(p) - site), float(v)))

    out = pd.DataFrame(rows, columns=["cls", "site_id", "gene", "chrom", "center",
                                      "strand", "signal", "pos", "score"])
    out.to_csv(a.out, sep="\t", index=False)
    print(f"[out] {a.out}: {len(out):,} rows; {out.groupby('cls').site_id.nunique().to_dict()} sites; "
          f"{n_bad} unfetchable", file=sys.stderr)


if __name__ == "__main__":
    main()