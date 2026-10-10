#!/usr/bin/env python3
# two focused questions
import argparse, os, sys, json, gzip
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

RC = str.maketrans("ACGTN", "TGCAN")
PAS = ("AATAAA", "ATTAAA")
NOT_TE = {"Simple_repeat", "Low_complexity", "Satellite", "rRNA", "scRNA",
          "snRNA", "srpRNA", "tRNA", "RNA", "Unknown", "Other"}
FOCUS = ["Alu", "L1", "L2", "MIR", "ERV1", "ERVL", "ERVL-MaLR", "SVA",
         "hAT-Charlie", "TcMar-Tigger"]


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


def load_rmsk(path, chroms):
    op = gzip.open if path.endswith(".gz") else open
    by = defaultdict(list)
    n = 0
    with op(path, "rt") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 13 or f[5] not in chroms:
                continue
            by[f[5]].append((int(f[6]), int(f[7]), f[9], f[11], f[12], f[10]))
            n += 1
    out = {}
    for c, v in by.items():
        v.sort(key=lambda x: x[0])
        out[c] = dict(rows=v, start=np.array([x[0] for x in v]),
                      end=np.array([x[1] for x in v]))
    print(f"[rmsk] {n:,} intervals", file=sys.stderr)
    return out


def element_at(rm, chrom, pos):
    e = rm.get(chrom)
    if e is None or not len(e["start"]):
        return None
    j = int(np.searchsorted(e["start"], pos, side="right") - 1)
    if j < 0 or pos > e["rows"][j][1]:
        return None
    return j


def el_info(rm, chrom, j, pos, tstrand):
    s0, s1, estr, ecls, efam, ename = rm[chrom]["rows"][j]
    ln = max(s1 - s0, 1)
    frac = (pos - s0) / ln
    if estr == "-":
        frac = 1.0 - frac
    return dict(fam=efam, cls=ecls, name=ename, estrand=estr,
                sense=float(estr == tstrand),
                is_te=0 if ecls in NOT_TE else 1,
                start=s0, end=s1, length=ln, frac=float(frac))


def prev_element(rm, chrom, pos, tstrand, window=300):
    e = rm.get(chrom)
    if e is None or not len(e["start"]):
        return None, np.nan
    if tstrand != "-":
        j = int(np.searchsorted(e["end"], pos, side="left") - 1)
        while j >= 0 and pos - e["rows"][j][1] <= window:
            if e["rows"][j][1] < pos:
                return j, float(pos - e["rows"][j][1])
            j -= 1
        return None, np.nan
    j = int(np.searchsorted(e["start"], pos, side="right"))
    while j < len(e["rows"]) and e["rows"][j][0] - pos <= window:
        if e["rows"][j][0] > pos:
            return j, float(e["rows"][j][0] - pos)
        j += 1
    return None, np.nan


# part 1
def productivity_table(path, out_dir):
    S = pd.read_csv(path, sep="\t")
    W = lambda n, df: df.to_csv(os.path.join(out_dir, n), sep="\t", index=False)

    non = S[S.family == "non-TE"]
    non_oe = float(non.hex_obs_over_exp.iloc[0]) if len(non) else np.nan
    non_arch = float(non.arch_per_kb.iloc[0]) if len(non) else np.nan

    te = S[S.family != "non-TE"]
    sense = te[te.orientation == "sense"].set_index("family")
    anti = te[te.orientation == "antisense"].set_index("family")
    fams = [f for f in sense.index if f in anti.index]

    rows = []
    for f in fams:
        s, a = sense.loc[f], anti.loc[f]
        if float(s.kb) < 100:
            continue
        rows.append(dict(
            family=f,
            sense_Mb=round(float(s.kb) / 1000, 2),
            antisense_Mb=round(float(a.kb) / 1000, 2),
            sense_GC=round(float(s.gc), 3),
            sense_hex_per_kb=round(float(s.hex_per_kb), 3),
            sense_OE=round(float(s.hex_obs_over_exp), 3),
            antisense_OE=round(float(a.hex_obs_over_exp), 3),
            OE_over_nonTE=round(float(s.hex_obs_over_exp) / non_oe, 3)
            if np.isfinite(non_oe) and non_oe > 0 else np.nan,
            sense_arch_per_kb=round(float(s.arch_per_kb), 3),
            antisense_arch_per_kb=round(float(a.arch_per_kb), 3),
            arch_asymmetry=round(float(s.arch_per_kb) /
                                 max(float(a.arch_per_kb), 1e-9), 2),
            hex_asymmetry=round(float(s.hex_per_kb) /
                                max(float(a.hex_per_kb), 1e-9), 2),
            sense_architectures=int(round(float(s.arch_per_kb) * float(s.kb))),
            fp_per_arch=round(float(s.fp_per_architecture), 4)))
    R = pd.DataFrame(rows)

    te["total_arch"] = te.arch_per_kb * te.kb
    grand = float(te.total_arch.sum()) + non_arch * float(non.kb.iloc[0])
    R["pct_of_all_architectures"] = (100 * R.sense_architectures /
                                     grand).round(2)
    R = R.sort_values("sense_OE", ascending=False)
    W("productivity_sense.tsv", R)

    print("\n" + "=" * 78, file=sys.stderr)
    print("PART 1  Productivity per family, sense orientation only",
          file=sys.stderr)
    print("=" * 78, file=sys.stderr)
    print(f"       non-TE intronic reference: O/E {non_oe:.3f}, "
          f"{non_arch:.3f} architectures/kb", file=sys.stderr)
    print(f"\n       {'family':<15}{'Mb':>7}{'GC':>7}{'O/E':>7}"
          f"{'vs non-TE':>11}{'asym':>8}{'arch/kb':>9}{'% of all':>10}",
          file=sys.stderr)
    for r in R.itertuples():
        print(f"       {r.family:<15}{r.sense_Mb:>7.1f}{r.sense_GC:>7.3f}"
              f"{r.sense_OE:>7.2f}{r.OE_over_nonTE:>11.2f}"
              f"{r.arch_asymmetry:>8.1f}{r.sense_arch_per_kb:>9.3f}"
              f"{r.pct_of_all_architectures:>9.2f}%", file=sys.stderr)

    print("\n       the two axes, read together:", file=sys.stderr)
    hi_oe = R[R.OE_over_nonTE > 1.3]
    for r in hi_oe.itertuples():
        if r.arch_asymmetry > 1.8:
            kind = "structural: a positioned A-rich feature on one strand"
        else:
            kind = "compositional: productive, but symmetrically so"
        print(f"         {r.family:<14} O/E {r.sense_OE:.2f}, asymmetry "
              f"{r.arch_asymmetry:.1f}x  -- {kind}", file=sys.stderr)
    lo = R[(R.OE_over_nonTE <= 1.3) & (R.pct_of_all_architectures > 2)]
    for r in lo.itertuples():
        print(f"         {r.family:<14} O/E {r.sense_OE:.2f} but "
              f"{r.pct_of_all_architectures:.1f}% of all architectures "
              f"-- abundance, not productivity", file=sys.stderr)
    return R


# part 2
def geometry(peaks, rm, gen, out_dir, hex_up=60, cluster_window=300):
    W = lambda n, df: df.to_csv(os.path.join(out_dir, n), sep="\t", index=False)

    pk = pd.read_parquet(peaks)
    if "in_analysis_set" in pk.columns:
        pk = pk[pk.in_analysis_set.astype(bool) | pk.is_canonical.astype(bool)]
    POS = "center" if "center" in pk.columns else "pos"
    d = pd.DataFrame({"chrom": pk.chrom.astype(str),
                      "pos": pk[POS].astype(int),
                      "strand": pk.strand.astype(str),
                      "is_real": pk.is_canonical.astype(bool)})
    print(f"\n[geom] {len(d):,} sites", file=sys.stderr)

    rows = []
    for i, r in enumerate(d.itertuples()):
        if i and i % 10000 == 0:
            print(f"       {i:,}/{len(d):,}", file=sys.stderr)
        seq = oriented(gen, r.chrom, r.pos, r.strand, hex_up, 10)
        if len(seq) < hex_up + 10:
            continue
        up = seq[:hex_up]
        best = -1
        motif = None
        for h in PAS:
            k = up.rfind(h)
            if k > best:
                best, motif = k, h
        if best < 0:
            continue

        if r.strand != "-":
            hlo = r.pos - hex_up + best
            hhi = hlo + 5
        else:
            hhi = r.pos + hex_up - best
            hlo = hhi - 5
        hmid = (hlo + hhi) // 2

        js = element_at(rm, r.chrom, r.pos)
        jh = element_at(rm, r.chrom, hmid)

        rec = dict(chrom=r.chrom, pos=r.pos, strand=r.strand,
                   is_real=bool(r.is_real), motif=motif,
                   hex_offset=best - hex_up)

        if js is not None:
            si = el_info(rm, r.chrom, js, r.pos, r.strand)
            rec.update(site_fam=si["fam"], site_is_te=si["is_te"],
                       site_sense=si["sense"], site_frac=si["frac"],
                       site_el_len=si["length"])
        else:
            rec.update(site_fam="none", site_is_te=0, site_sense=np.nan,
                       site_frac=np.nan, site_el_len=np.nan)

        if jh is not None:
            hi_ = el_info(rm, r.chrom, jh, hmid, r.strand)
            rec.update(hex_fam=hi_["fam"], hex_is_te=hi_["is_te"],
                       hex_sense=hi_["sense"], hex_frac=hi_["frac"],
                       hex_el_len=hi_["length"])
        else:
            rec.update(hex_fam="none", hex_is_te=0, hex_sense=np.nan,
                       hex_frac=np.nan, hex_el_len=np.nan)

        rec["same_element"] = bool(js is not None and jh is not None and
                                   js == jh)
        rec["hexamer_outside_site_element"] = bool(
            js is not None and (jh is None or jh != js))

        jp, dist = prev_element(rm, r.chrom, r.pos, r.strand, cluster_window)
        if jp is not None:
            pi = el_info(rm, r.chrom, jp, r.pos, r.strand)
            rec.update(prev_fam=pi["fam"], prev_sense=pi["sense"],
                       dist_to_prev_end=dist)
        else:
            rec.update(prev_fam="none", prev_sense=np.nan,
                       dist_to_prev_end=np.nan)
        rows.append(rec)

    G = pd.DataFrame(rows)
    W("site_vs_hexamer_element.tsv", G)

    print("\n" + "=" * 78, file=sys.stderr)
    print("PART 2  Does the site's element supply its hexamer?", file=sys.stderr)
    print("=" * 78, file=sys.stderr)

    te = G[G.site_is_te == 1]
    print(f"       {len(te):,} sites lie in transposable element sequence",
          file=sys.stderr)
    print(f"       hexamer in the SAME element: "
          f"{100*te.same_element.mean():.1f}%", file=sys.stderr)
    print(f"       hexamer in a DIFFERENT element or outside any: "
          f"{100*te.hexamer_outside_site_element.mean():.1f}%",
          file=sys.stderr)

    print(f"\n       {'family':<14}{'n':>7}{'site frac':>11}{'hex frac':>10}"
          f"{'same el':>9}{'hex elsewhere':>15}", file=sys.stderr)
    frows = []
    for f in FOCUS:
        s = te[te.site_fam == f]
        if len(s) < 30:
            continue
        frows.append(dict(family=f, n=len(s),
                          median_site_frac=round(float(s.site_frac.median()), 3),
                          median_hex_frac=round(float(s.hex_frac.median()), 3)
                          if s.hex_frac.notna().any() else np.nan,
                          frac_same_element=round(float(s.same_element.mean()), 3),
                          frac_hex_elsewhere=round(
                              float(s.hexamer_outside_site_element.mean()), 3)))
        r = frows[-1]
        print(f"       {f:<14}{r['n']:>7,}{r['median_site_frac']:>11.3f}"
              f"{r['median_hex_frac']:>10.3f}"
              f"{100*r['frac_same_element']:>8.1f}%"
              f"{100*r['frac_hex_elsewhere']:>14.1f}%", file=sys.stderr)
    W("by_family_geometry.tsv", pd.DataFrame(frows))

    print("\n[test] sites at a fractional position below 0.10 in their element",
          file=sys.stderr)
    low = te[te.site_frac < 0.10]
    print(f"       {len(low):,} such sites "
          f"({100*len(low)/max(len(te),1):.1f}% of TE-overlapping sites)",
          file=sys.stderr)
    if len(low):
        print(f"       hexamer in a different element: "
              f"{100*low.hexamer_outside_site_element.mean():.1f}%",
              file=sys.stderr)
        hx = low[low.hex_is_te == 1]
        if len(hx):
            print(f"       of those whose hexamer IS in an element, its "
                  f"median fractional\n       position within that element: "
                  f"{hx.hex_frac.median():.3f}", file=sys.stderr)
            print(f"       hexamer's element family: " +
                  ", ".join(f"{k} {v:,}" for k, v in
                            hx.hex_fam.value_counts().head(5).items()),
                  file=sys.stderr)
        pv = low[low.dist_to_prev_end.notna()]
        if len(pv):
            print(f"       an element ends within {cluster_window} nt upstream "
                  f"for {100*len(pv)/len(low):.1f}% of them,\n"
                  f"       at a median distance of {pv.dist_to_prev_end.median():.0f} nt",
                  file=sys.stderr)
            print(f"       that upstream element's family: " +
                  ", ".join(f"{k} {v:,}" for k, v in
                            pv.prev_fam.value_counts().head(5).items()),
                  file=sys.stderr)
        W("low_fraction_sites.tsv", low)

    alu = te[(te.site_fam == "Alu")]
    if len(alu) > 50:
        print("\n[alu]  Alu, where the consensus mapping and the site-level "
              "assignment\n       disagreed", file=sys.stderr)
        print(f"       {len(alu):,} sites assigned to an Alu", file=sys.stderr)
        print(f"       median fractional position of the SITE: "
              f"{alu.site_frac.median():.3f}", file=sys.stderr)
        ah = alu[alu.hex_is_te == 1]
        if len(ah):
            print(f"       median fractional position of the HEXAMER within "
                  f"its own element: {ah.hex_frac.median():.3f}",
                  file=sys.stderr)
        print(f"       hexamer in the same Alu: "
              f"{100*alu.same_element.mean():.1f}%", file=sys.stderr)
        aa = alu[alu.hex_fam == "Alu"]
        print(f"       hexamer in an Alu, whether the same one or not: "
              f"{100*len(aa)/len(alu):.1f}%", file=sys.stderr)
        if len(ah):
            h, e = np.histogram(ah.hex_frac.dropna(), bins=20, range=(0, 1))
            W("alu_hexamer_fraction.tsv",
              pd.DataFrame(dict(bin_lo=e[:-1], bin_hi=e[1:], n=h)))
            top = np.argmax(h)
            print(f"       the hexamer fraction peaks in bin "
                  f"{e[top]:.2f}-{e[top+1]:.2f} ({h[top]:,} of {h.sum():,})",
                  file=sys.stderr)

    print("\n" + "-" * 78, file=sys.stderr)
    if len(low) and low.hexamer_outside_site_element.mean() > 0.5:
        print("       the discrepancy is explained: most low-fraction sites "
              "draw their hexamer\n       from a DIFFERENT element, so "
              "assigning a site to whichever element covers\n       it "
              "misattributes the geometry. these sites lie just downstream of "
              "the\n       preceding element's 3' end, which is the geometry a "
              "functional site has.", file=sys.stderr)
    else:
        print("       the discrepancy is NOT explained by element clustering: "
              "most low-fraction\n       sites carry their hexamer inside the "
              "same element, so the 5'-end\n       concentration is real and "
              "needs another explanation.", file=sys.stderr)
    print("-" * 78, file=sys.stderr)
    return G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--te_source", required=True,
                    help="tesrc/te_source.tsv from te_source.py")
    ap.add_argument("--peaks", required=True)
    ap.add_argument("--rmsk", required=True)
    ap.add_argument("--genome", required=True)
    ap.add_argument("--hex_up", type=int, default=60)
    ap.add_argument("--cluster_window", type=int, default=300)
    ap.add_argument("--out", default="geom")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    R = productivity_table(a.te_source, a.out)

    pk = pd.read_parquet(a.peaks)
    chroms = set(pk.chrom.astype(str).unique())
    rm = load_rmsk(a.rmsk, chroms)
    gen = Genome(a.genome)
    G = geometry(a.peaks, rm, gen, a.out, a.hex_up, a.cluster_window)

    with open(os.path.join(a.out, "counts.json"), "w") as fh:
        json.dump(dict(n_families=len(R), n_sites=len(G),
                       n_in_te=int((G.site_is_te == 1).sum()),
                       hex_up=a.hex_up), fh, indent=2)
    print(f"\n[out]  {a.out}/", file=sys.stderr)


if __name__ == "__main__":
    main()