#!/usr/bin/env python3
# is the gc difference a local trough, a slope, or neither?
import argparse, os, sys, json, math
from collections import defaultdict

import numpy as np
import pandas as pd

RC = str.maketrans("ACGTN", "TGCAN")

try:
    from maxentpy import maxent
    from maxentpy.maxent import load_matrix5, load_matrix3
    _M5, _M3 = load_matrix5(), load_matrix3()
    HAVE_MAXENT = True
except Exception:
    HAVE_MAXENT = False


class Genome:
    def __init__(self, path):
        fai = path + ".fai"
        if not os.path.exists(fai):
            sys.exit("need " + fai + "; run `samtools faidx " + path + "`")
        self.idx = {}
        for line in open(fai):
            f = line.split("\t")
            self.idx[f[0]] = (int(f[1]), int(f[2]), int(f[3]), int(f[4]))
        self.fh = open(path, "rb")

    def fetch(self, chrom, start, end):
        if chrom not in self.idx or end <= start:
            return ""
        ln, off, lb, lw = self.idx[chrom]
        start, end = max(0, start), min(end, ln)
        if end <= start:
            return ""
        def seek(p):
            return off + p // lb * lw + p % lb
        self.fh.seek(seek(start))
        return self.fh.read(seek(end) - seek(start)).decode().replace("\n", "").upper()


def oriented(gen, chrom, pos, strand, up, dn):
    if strand == "-":
        s = gen.fetch(chrom, int(pos) - dn, int(pos) + up)
        return s.translate(RC)[::-1] if len(s) >= up + dn else ""
    s = gen.fetch(chrom, int(pos) - up, int(pos) + dn)
    return s if len(s) >= up + dn else ""


def gc_track(seq, win):
    arr = np.frombuffer(seq.encode(), dtype=np.uint8)
    v = ((arr == ord("G")) | (arr == ord("C"))).astype(float)
    ok = np.isin(arr, [ord(b) for b in "ACGT"]).astype(float)
    k = np.ones(win)
    num = np.convolve(v, k, mode="same")
    den = np.convolve(ok, k, mode="same")
    return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def skew_track(seq, win, pair=("G", "C")):
    arr = np.frombuffer(seq.encode(), dtype=np.uint8)
    a = (arr == ord(pair[0])).astype(float)
    b = (arr == ord(pair[1])).astype(float)
    k = np.ones(win)
    na = np.convolve(a, k, mode="same")
    nb = np.convolve(b, k, mode="same")
    tot = na + nb
    return np.where(tot > 0, (na - nb) / np.maximum(tot, 1), np.nan)


def splice_counts(seq, centre, lo, hi):
    i0 = max(0, centre + lo)
    i1 = min(len(seq), centre + hi)
    n5 = n3 = 0
    b5 = b3 = -99.0
    for i in range(i0, max(i0, i1 - 9)):
        if seq[i + 3:i + 5] == "GT":
            w = seq[i:i + 9]
            if "N" in w:
                continue
            n5 += 1
            if HAVE_MAXENT:
                try:
                    b5 = max(b5, float(maxent.score5(w, matrix=_M5)))
                except Exception:
                    pass
    for i in range(i0, max(i0, i1 - 23)):
        if seq[i + 18:i + 20] == "AG":
            w = seq[i:i + 23]
            if "N" in w:
                continue
            n3 += 1
            if HAVE_MAXENT:
                try:
                    b3 = max(b3, float(maxent.score3(w, matrix=_M3)))
                except Exception:
                    pass
    return (n5, b5 if b5 > -99 else np.nan,
            n3, b3 if b3 > -99 else np.nan)


def summarize(vals, label=None):
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if not len(v):
        return dict(n=0)
    sd = v.std(ddof=1)
    se = sd / np.sqrt(len(v))
    return dict(n=len(v), mean=float(v.mean()),
                lo=float(v.mean() - 1.96 * se), hi=float(v.mean() + 1.96 * se),
                median=float(np.median(v)), sd=float(sd),
                frac_positive=float((v > 0).mean()),
                dz=float(v.mean() / sd) if sd > 0 else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peaks", required=True)
    ap.add_argument("--genome", required=True)
    ap.add_argument("--up", type=int, default=3000)
    ap.add_argument("--dn", type=int, default=3000)
    ap.add_argument("--gc_win", type=int, default=201)
    ap.add_argument("--bin", type=int, default=50)
    ap.add_argument("--max_ccpa", type=int, default=8)
    ap.add_argument("--splice_span", type=int, default=2500,
                    help="half-width for the splice-site counts")
    ap.add_argument("--out", default="grad")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    W = lambda n, df: df.to_csv(os.path.join(a.out, n), sep="\t", index=False)
    UP, DN, L = a.up, a.dn, a.up + a.dn
    if UP != DN:
        sys.exit("this script needs a symmetric window to decompose the "
                 "profile into symmetric and antisymmetric parts")

    print("[cfg]  splice scoring: " +
          ("MaxEnt" if HAVE_MAXENT else "counts only, maxentpy absent"),
          file=sys.stderr)

    pk = pd.read_parquet(a.peaks)
    if "in_analysis_set" in pk.columns:
        pk = pk[pk.in_analysis_set.astype(bool) | pk.is_canonical.astype(bool)]
    pos = "center" if "center" in pk.columns else "pos"
    d = pd.DataFrame({"gene": pk.gene.astype(str), "chrom": pk.chrom.astype(str),
                      "pos": pk[pos].astype(int), "strand": pk.strand.astype(str),
                      "is_real": pk.is_canonical.astype(bool)})
    if a.max_ccpa > 0:
        keep = []
        for g, sub in d.groupby("gene", sort=False):
            r = sub[sub.is_real]; c = sub[~sub.is_real]
            if len(c) > a.max_ccpa:
                c = c.sample(a.max_ccpa, random_state=0)
            keep.append(pd.concat([r, c]))
        d = pd.concat(keep, ignore_index=True)
    print(f"[set]  {int(d.is_real.sum()):,} real, "
          f"{int((~d.is_real).sum()):,} false positives, "
          f"{d.gene.nunique():,} genes", file=sys.stderr)

    gen = Genome(a.genome)
    xs = np.arange(-UP, DN)

    tracks = {True: [], False: []}
    per_gene_delta = []
    rows = []
    n_bad = 0
    rng = np.random.default_rng(0)

    for gi, (g, sub) in enumerate(d.groupby("gene", sort=False)):
        if gi and gi % 250 == 0:
            print(f"[run]  {gi:,} genes...", file=sys.stderr)
        gp = {True: [], False: []}
        for r in sub.itertuples():
            seq = oriented(gen, r.chrom, r.pos, r.strand, UP, DN)
            if len(seq) != L:
                n_bad += 1
                continue
            cls = bool(r.is_real)
            gcv = gc_track(seq, a.gc_win)
            gp[cls].append(gcv)
            if len(tracks[cls]) < 4000:
                tracks[cls].append(gcv)

            # per-site geometry, not assuming a shared shape
            c = UP
            inner = gcv[c - 1000:c + 1000]
            flank = np.nanmean(np.r_[gcv[c - 3000:c - 2000],
                                     gcv[c + 2000:c + 3000]])
            j = int(np.nanargmin(inner)) if np.isfinite(inner).any() else -1
            rec = dict(gene=g, is_real=cls,
                       gc_up_1k=float(np.nanmean(gcv[c - 1000:c])),
                       gc_dn_1k=float(np.nanmean(gcv[c:c + 1000])),
                       gc_up_250=float(np.nanmean(gcv[c - 250:c])),
                       gc_dn_250=float(np.nanmean(gcv[c:c + 250])),
                       gc_flank=float(flank),
                       gc_min_pos=float(j - 1000) if j >= 0 else np.nan,
                       gc_min_val=float(np.nanmin(inner))
                       if np.isfinite(inner).any() else np.nan)
            rec["gc_gradient_1k"] = rec["gc_up_1k"] - rec["gc_dn_1k"]
            rec["gc_gradient_250"] = rec["gc_up_250"] - rec["gc_dn_250"]
            rec["gc_trough_depth"] = rec["gc_flank"] - rec["gc_min_val"]

            sk = skew_track(seq, a.gc_win, ("G", "C"))
            ak = skew_track(seq, a.gc_win, ("A", "T"))
            rec["gcskew_up_1k"] = float(np.nanmean(sk[c - 1000:c]))
            rec["gcskew_dn_1k"] = float(np.nanmean(sk[c:c + 1000]))
            rec["gcskew_gradient"] = rec["gcskew_up_1k"] - rec["gcskew_dn_1k"]
            rec["atskew_up_1k"] = float(np.nanmean(ak[c - 1000:c]))
            rec["atskew_dn_1k"] = float(np.nanmean(ak[c:c + 1000]))
            rec["atskew_gradient"] = rec["atskew_up_1k"] - rec["atskew_dn_1k"]

            S = a.splice_span
            n5u, b5u, n3u, b3u = splice_counts(seq, c, -S, 0)
            n5d, b5d, n3d, b3d = splice_counts(seq, c, 0, S)
            rec.update({"n5ss_up": n5u, "n5ss_dn": n5d, "n5ss_total": n5u + n5d,
                        "n3ss_up": n3u, "n3ss_dn": n3d, "n3ss_total": n3u + n3d,
                        "best5ss_up": b5u, "best5ss_dn": b5d,
                        "best3ss_up": b3u, "best3ss_dn": b3d,
                        "n5ss_gradient": n5u - n5d,
                        "n3ss_gradient": n3u - n3d})
            rows.append(rec)
            gp[cls].append(gcv) if False else None

        if len(gp[True]) == 1 and len(gp[False]) >= 1:
            per_gene_delta.append(gp[True][0] - np.nanmean(gp[False], axis=0))

    if n_bad:
        print(f"[seq]  {n_bad:,} sites dropped at contig edges", file=sys.stderr)
    PER = pd.DataFrame(rows)
    W("per_site.tsv", PER)
    print(f"[seq]  {len(PER):,} sites; {len(per_gene_delta):,} genes paired",
          file=sys.stderr)

    # profile shape
    def binned(v, w=a.bin):
        n = (len(v) // w) * w
        return np.nanmean(v[:n].reshape(-1, w), axis=1)
    xb = binned(xs.astype(float))

    D = np.vstack(per_gene_delta)
    mu = np.nanmean(D, axis=0)
    se = np.nanstd(D, axis=0, ddof=1) / np.sqrt(D.shape[0])

    rev = mu[::-1]
    sym = (mu + rev) / 2.0
    anti = (mu - rev) / 2.0
    W("profile_paired_gc.tsv", pd.DataFrame(dict(
        pos=xb.astype(int), delta=np.round(binned(mu), 6),
        lo=np.round(binned(mu - 1.96 * se), 6),
        hi=np.round(binned(mu + 1.96 * se), 6),
        symmetric=np.round(binned(sym), 6),
        antisymmetric=np.round(binned(anti), 6), n_genes=D.shape[0])))

    vs, va = float(np.nanvar(sym)), float(np.nanvar(anti))
    print(f"\n[shape] paired GC profile, -{UP}..+{DN}:", file=sys.stderr)
    print(f"        symmetric (trough) variance   {vs:.3e}", file=sys.stderr)
    print(f"        antisymmetric (slope) variance {va:.3e}", file=sys.stderr)
    print(f"        -> {100*vs/(vs+va):.1f}% of the structure is a symmetric "
          f"trough, {100*va/(vs+va):.1f}% is an upstream/downstream gradient",
          file=sys.stderr)
    j = int(np.nanargmin(mu))
    print(f"        minimum {mu[j]:+.4f} at {xs[j]:+d} nt; at -1500 "
          f"{mu[UP-1500]:+.4f}, at +1500 {mu[UP+1500]:+.4f}", file=sys.stderr)

    pooled = []
    for cls in (True, False):
        M = np.vstack(tracks[cls])
        m = np.nanmean(M, axis=0)
        for x_, y_ in zip(xb, binned(m)):
            pooled.append(dict(cls="real 3' end" if cls else "false positive",
                               pos=int(x_), gc=round(float(y_), 6),
                               n=M.shape[0]))
    W("profile_pooled_gc.tsv", pd.DataFrame(pooled))

    # per-site geometry
    geo = []
    for c in ("gc_min_pos", "gc_trough_depth", "gc_gradient_1k",
              "gc_gradient_250", "gcskew_gradient", "atskew_gradient"):
        for cls, lab in ((True, "real 3' end"), (False, "false positive")):
            v = PER.loc[PER.is_real == cls, c].to_numpy(float)
            v = v[np.isfinite(v)]
            if not len(v):
                continue
            geo.append(dict(feature=c, cls=lab, n=len(v),
                            mean=round(float(v.mean()), 5),
                            median=round(float(np.median(v)), 5),
                            sd=round(float(v.std(ddof=1)), 5),
                            q25=round(float(np.percentile(v, 25)), 5),
                            q75=round(float(np.percentile(v, 75)), 5)))
    W("per_site_geometry.tsv", pd.DataFrame(geo))

    mp = PER.loc[PER.is_real, "gc_min_pos"].dropna()
    print(f"\n[geom]  per-site GC minimum position, real ends: median "
          f"{np.median(mp):+.0f} nt, IQR {np.percentile(mp,25):+.0f} to "
          f"{np.percentile(mp,75):+.0f}", file=sys.stderr)
    print(f"        the average profile is only meaningful if these cluster; "
          f"an IQR spanning >1 kb means the trough is a construct of "
          f"averaging", file=sys.stderr)

    # within vs between gene
    feats = [c for c in PER.columns if c not in ("gene", "is_real")]
    out = []
    for c in feats:
        r = PER.loc[PER.is_real, c].to_numpy(float)
        k = PER.loc[~PER.is_real, c].to_numpy(float)
        r = r[np.isfinite(r)]; k = k[np.isfinite(k)]
        if len(r) < 50 or len(k) < 50:
            continue
        sd = np.sqrt((r.var(ddof=1) + k.var(ddof=1)) / 2)
        pooled_d = (r.mean() - k.mean()) / sd if sd > 0 else np.nan

        vals = []
        for g, sub in PER.groupby("gene", sort=False):
            rv = sub.loc[sub.is_real, c]
            cv = sub.loc[~sub.is_real, c]
            if len(rv) != 1 or not len(cv):
                continue
            a_ = pd.to_numeric(rv.iloc[0], errors="coerce")
            b_ = pd.to_numeric(cv, errors="coerce").mean()
            if np.isfinite(a_) and np.isfinite(b_):
                vals.append(float(a_ - b_))
        s = summarize(vals)
        if not s.get("n"):
            continue
        out.append(dict(feature=c,
                        pooled_real=round(float(r.mean()), 5),
                        pooled_fp=round(float(k.mean()), 5),
                        pooled_diff=round(float(r.mean() - k.mean()), 5),
                        pooled_d=round(float(pooled_d), 4),
                        paired_diff=round(s["mean"], 5),
                        paired_lo=round(s["lo"], 5),
                        paired_hi=round(s["hi"], 5),
                        paired_dz=round(s["dz"], 4),
                        frac_positive=round(s["frac_positive"], 4),
                        n_genes=s["n"],
                        sign_agrees=bool(np.sign(r.mean() - k.mean()) ==
                                         np.sign(s["mean"]))))
    CMP = pd.DataFrame(out).sort_values("paired_dz", key=abs, ascending=False)
    W("within_vs_between.tsv", CMP)

    print("\n[cmp]   within-gene (paired) vs between-gene (pooled)",
          file=sys.stderr)
    print(f"        {'feature':<22} {'pooled d':>9} {'paired dz':>10} "
          f"{'agree':>6}", file=sys.stderr)
    for r_ in CMP.head(22).itertuples():
        print(f"        {r_.feature:<22} {r_.pooled_d:>+9.3f} "
              f"{r_.paired_dz:>+10.3f} {'yes' if r_.sign_agrees else 'NO':>6}",
              file=sys.stderr)
    dis = CMP[~CMP.sign_agrees]
    if len(dis):
        print(f"\n        {len(dis)} feature(s) change sign between the two. "
              f"Those are the ones where a pooled comparison would give the "
              f"opposite answer:", file=sys.stderr)
        for r_ in dis.head(8).itertuples():
            print(f"          {r_.feature:<22} pooled {r_.pooled_diff:+.4f}, "
                  f"paired {r_.paired_diff:+.4f}", file=sys.stderr)

    with open(os.path.join(a.out, "counts.json"), "w") as fh:
        json.dump(dict(n_real=int(PER.is_real.sum()),
                       n_fp=int((~PER.is_real).sum()),
                       n_genes=int(PER.gene.nunique()),
                       n_genes_paired=int(D.shape[0]),
                       up=UP, dn=DN, gc_win=a.gc_win,
                       splice_span=a.splice_span, maxent=HAVE_MAXENT), fh,
                  indent=2)
    print(f"\n[out]  {a.out}/", file=sys.stderr)


if __name__ == "__main__":
    main()