#!/usr/bin/env python3
# does the element's own consensus carry a poly(a) signal?
import argparse, os, sys, json, gzip, re
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

RC = str.maketrans("ACGTN-", "TGCAN-")
PAS = ("AATAAA", "ATTAAA")
NOT_TE = {"Simple_repeat", "Low_complexity", "Satellite", "rRNA", "scRNA",
          "snRNA", "srpRNA", "tRNA", "RNA", "Unknown", "Other"}


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
            if f[11] in NOT_TE:
                continue
            by[f[5]].append(dict(start=int(f[6]), end=int(f[7]), strand=f[9],
                                 name=f[10], cls=f[11], fam=f[12],
                                 div=float(f[2]) / 10.0))
            n += 1
    out = {}
    for c, v in by.items():
        v.sort(key=lambda x: x["start"])
        out[c] = dict(rows=v, start=np.array([x["start"] for x in v]))
    print(f"[rmsk] {n:,} transposable-element intervals", file=sys.stderr)
    return out


def element_at(rm, chrom, pos):
    e = rm.get(chrom)
    if e is None or not len(e["start"]):
        return None
    j = int(np.searchsorted(e["start"], pos, side="right") - 1)
    if j < 0 or pos > e["rows"][j]["end"]:
        return None
    return j


HDR = re.compile(
    r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\S+)\s+(\d+)\s+(\d+)\s+"
    r"\((\d+)\)\s+(C\s+)?(\S+?)#(\S+)\s+")


def parse_align(path, targets):
    out = {}
    cur, gseq, cseq = None, [], []
    nseen = nkept = 0

    def flush():
        nonlocal cur, gseq, cseq
        if cur is not None and gseq and cseq:
            out[cur["key"]] = dict(meta=cur, genomic="".join(gseq),
                                   consensus="".join(cseq))
        cur, gseq, cseq = None, [], []

    with open(path) as fh:
        for line in fh:
            if line[:1].isdigit() or (line[:1] == " " and line[1:2].isdigit()):
                m = HDR.match(line)
                if m:
                    flush()
                    nseen += 1
                    if nseen % 1000000 == 0:
                        print(f"[align] {nseen:,} headers, {nkept:,} kept",
                              file=sys.stderr)
                    key = (m.group(5), int(m.group(6)), int(m.group(7)),
                           m.group(10))
                    if key in targets:
                        nkept += 1
                        cur = dict(key=key, chrom=m.group(5),
                                   gstart=int(m.group(6)), gend=int(m.group(7)),
                                   reverse=m.group(9) is not None,
                                   name=m.group(10), clsfam=m.group(11),
                                   div=float(m.group(2)))
                    continue
            if cur is None:
                continue
            s = line.rstrip("\n")
            if not s.strip():
                continue
            parts = s.split()
            if len(parts) < 4:
                continue
            body = parts[2]
            if not re.fullmatch(r"[ACGTNacgtn.\-]+", body):
                continue
            if parts[0] == cur["chrom"]:
                gseq.append(body.upper())
            else:
                cseq.append(body.upper())
    flush()
    print(f"[align] {nseen:,} headers scanned, {len(out):,} copies recovered",
          file=sys.stderr)
    return out


def walk(block):
    g, c = block["genomic"], block["consensus"]
    n = min(len(g), len(c))
    gp = block["meta"]["gstart"]
    ci = 0
    for i in range(n):
        gb, cb = g[i], c[i]
        if gb != "-":
            yield gp, ci, gb, cb
            gp += 1
        if cb != "-":
            ci += 1


def hamming_to_pas(six):
    if len(six) != 6 or "." in six or "N" in six:
        return np.nan
    return min(sum(1 for a, b in zip(six, h) if a != b) for h in PAS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peaks", required=True)
    ap.add_argument("--rmsk", required=True)
    ap.add_argument("--align", required=True)
    ap.add_argument("--genome", required=True)
    ap.add_argument("--pad", type=int, default=5000)
    ap.add_argument("--hex_up", type=int, default=60)
    ap.add_argument("--min_copies", type=int, default=60,
                    help="subfamilies with fewer aligned copies are skipped; "
                         "a consensus from a handful of copies is noise")
    ap.add_argument("--min_cov", type=int, default=20,
                    help="consensus positions covered by fewer copies are left "
                         "undefined rather than called from thin data")
    ap.add_argument("--max_subfam", type=int, default=60,
                    help="cap on subfamilies carried into the alignment pass")
    ap.add_argument("--max_copies_per_subfam", type=int, default=1500)
    ap.add_argument("--out", default="subfam")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    W = lambda n, df: df.to_csv(os.path.join(a.out, n), sep="\t", index=False)
    rng = np.random.default_rng(0)

    # sites and their hexamers
    pk = pd.read_parquet(a.peaks)
    if "in_analysis_set" in pk.columns:
        pk = pk[pk.in_analysis_set.astype(bool) | pk.is_canonical.astype(bool)]
    POS = "center" if "center" in pk.columns else "pos"
    d = pd.DataFrame({"gene": pk.gene.astype(str), "chrom": pk.chrom.astype(str),
                      "pos": pk[POS].astype(int), "strand": pk.strand.astype(str),
                      "is_real": pk.is_canonical.astype(bool)})
    span = (d.groupby(["gene", "chrom", "strand"], as_index=False)
              .agg(lo=("pos", "min"), hi=("pos", "max")))
    span["lo"] = (span.lo - a.pad).clip(lower=0)
    span["hi"] = span.hi + a.pad
    print(f"[set]  {len(d):,} sites, {len(span):,} genes", file=sys.stderr)

    rm = load_rmsk(a.rmsk, set(d.chrom.unique()))
    gen = Genome(a.genome)

    hits = []
    for r in d.itertuples():
        seq = oriented(gen, r.chrom, r.pos, r.strand, a.hex_up, 10)
        if len(seq) < a.hex_up + 10:
            continue
        up = seq[:a.hex_up]
        best, motif = -1, None
        for h in PAS:
            k = up.rfind(h)
            if k > best:
                best, motif = k, h
        if best < 0:
            continue
        if r.strand != "-":
            hlo = r.pos - a.hex_up + best
        else:
            hlo = r.pos + a.hex_up - best - 6
        gpos = list(range(hlo, hlo + 6))
        j = element_at(rm, r.chrom, int(np.mean(gpos)))
        if j is None:
            continue
        el = rm[r.chrom]["rows"][j]
        hits.append(dict(chrom=r.chrom, pos=r.pos, strand=r.strand,
                         is_real=bool(r.is_real), motif=motif,
                         gpos_lo=min(gpos), gpos_hi=max(gpos),
                         subfam=el["name"], fam=el["fam"], cls=el["cls"],
                         el_start=el["start"], el_end=el["end"],
                         el_strand=el["strand"], el_div=el["div"],
                         el_sense=float(el["strand"] == r.strand)))
    H = pd.DataFrame(hits)
    print(f"[hex]  {len(H):,} sites whose hexamer lies in an element",
          file=sys.stderr)
    if not len(H):
        sys.exit("no hexamers in elements; nothing to do")
    print("       top subfamilies by hexamer count:", file=sys.stderr)
    for k, v in H.subfam.value_counts().head(12).items():
        print(f"         {k:<16} {v:>6,}", file=sys.stderr)

    # pick subfamilies and gather their copies
    want_sub = [s for s, n in H.subfam.value_counts().items()
                if n >= 10][:a.max_subfam]
    print(f"\n[sub]  {len(want_sub)} subfamilies carried forward",
          file=sys.stderr)

    copies = defaultdict(list)
    for r in span.itertuples():
        e = rm.get(r.chrom)
        if not e:
            continue
        i0 = int(np.searchsorted(e["start"], r.lo, side="left"))
        i1 = int(np.searchsorted(e["start"], r.hi, side="right"))
        for k in range(i0, i1):
            el = e["rows"][k]
            if el["name"] not in want_sub:
                continue
            if el["end"] <= r.lo or el["start"] >= r.hi:
                continue
            copies[el["name"]].append(dict(chrom=r.chrom, start=el["start"],
                                           end=el["end"], name=el["name"],
                                           strand=el["strand"],
                                           div=el["div"],
                                           tstrand=r.strand))
    targets = set()
    keep = {}
    for s in want_sub:
        v = copies.get(s, [])
        if len(v) > a.max_copies_per_subfam:
            idx = rng.choice(len(v), a.max_copies_per_subfam, replace=False)
            v = [v[i] for i in idx]
        keep[s] = v
        for c in v:
            targets.add((c["chrom"], c["start"] + 1, c["end"], c["name"]))
    print(f"[sub]  {len(targets):,} element copies to align", file=sys.stderr)

    blocks = parse_align(a.align, targets)

    # empirical consensus per subfamily
    basecnt = defaultdict(lambda: defaultdict(Counter))
    covered = defaultdict(Counter)
    naligned = Counter()
    g2c_cache = {}
    for s in want_sub:
        for c in keep.get(s, []):
            key = (c["chrom"], c["start"] + 1, c["end"], c["name"])
            b = blocks.get(key)
            if b is None:
                continue
            naligned[s] += 1
            m = {}
            for gp, ci, gb, cb in walk(b):
                basecnt[s][ci][gb] += 1
                covered[s][ci] += 1
                m[gp] = ci
            g2c_cache[key] = m

    cons = {}
    crows = []
    for s in want_sub:
        if naligned[s] < a.min_copies:
            continue
        positions = sorted(basecnt[s])
        if not positions:
            continue
        seq = []
        for ci in range(max(positions) + 1):
            cnt = basecnt[s].get(ci)
            tot = sum(cnt[b] for b in "ACGT") if cnt else 0
            if tot < a.min_cov:
                seq.append(".")
            else:
                seq.append(max("ACGT", key=lambda b: cnt[b]))
        cons[s] = "".join(seq)
        crows.append(dict(subfamily=s, n_copies=naligned[s],
                          consensus_length=len(cons[s]),
                          defined_positions=sum(1 for x in cons[s] if x != ".")))
    print(f"\n[cons] consensus built for {len(cons)} subfamilies",
          file=sys.stderr)
    W("consensus_coverage.tsv", pd.DataFrame(crows))
    with open(os.path.join(a.out, "consensus.fa"), "w") as fh:
        for s, seq in cons.items():
            fh.write(">" + s + " n=" + str(naligned[s]) + "\n")
            for i in range(0, len(seq), 60):
                fh.write(seq[i:i + 60] + "\n")

    # scan each consensus
    scan = []
    hitrows = []
    for s, seq in cons.items():
        exact, one, two = [], [], []
        for i in range(len(seq) - 5):
            h = hamming_to_pas(seq[i:i + 6])
            if not np.isfinite(h):
                continue
            if h == 0:
                exact.append(i)
            elif h == 1:
                one.append(i)
            elif h == 2:
                two.append(i)
        fam = H[H.subfam == s].fam.iloc[0] if (H.subfam == s).any() else "?"
        scan.append(dict(subfamily=s, family=fam, n_copies=naligned[s],
                         cons_len=len(seq),
                         n_exact=len(exact), n_one_sub=len(one),
                         n_two_sub=len(two),
                         exact_positions=";".join(map(str, exact[:8])),
                         one_sub_positions=";".join(map(str, one[:12])),
                         one_sub_per_kb=round(1000 * len(one) /
                                              max(len(seq), 1), 2)))
        for i in exact:
            hitrows.append(dict(subfamily=s, family=fam, cons_pos=i,
                                kind="exact", consensus=seq[i:i + 6],
                                context=seq[max(0, i - 4):i + 10]))
        for i in one:
            hitrows.append(dict(subfamily=s, family=fam, cons_pos=i,
                                kind="one substitution", consensus=seq[i:i + 6],
                                context=seq[max(0, i - 4):i + 10]))
    SC = pd.DataFrame(scan).sort_values(["n_exact", "n_one_sub"],
                                        ascending=False)
    W("subfamily_scan.tsv", SC)
    W("consensus_hits.tsv", pd.DataFrame(hitrows))

    print("\n[scan] subfamily consensus sequences carrying a poly(A) signal",
          file=sys.stderr)
    ex = SC[SC.n_exact > 0]
    if len(ex):
        print(f"       {len(ex)} subfamilies carry AATAAA or ATTAAA outright:",
              file=sys.stderr)
        print(f"       {'subfamily':<16}{'family':<14}{'copies':>8}"
              f"{'len':>7}{'exact':>7}{'positions':>18}", file=sys.stderr)
        for r in ex.head(20).itertuples():
            print(f"       {r.subfamily:<16}{r.family:<14}{r.n_copies:>8,}"
                  f"{r.cons_len:>7,}{r.n_exact:>7}"
                  f"{r.exact_positions[:16]:>18}", file=sys.stderr)
    else:
        print("       none", file=sys.stderr)

    print("\n       subfamilies with the most one-substitution positions, "
          "per kb of consensus:", file=sys.stderr)
    for r in SC.nlargest(15, "one_sub_per_kb").itertuples():
        print(f"       {r.subfamily:<16}{r.family:<14}{r.n_copies:>8,}"
              f"{r.cons_len:>7,}  {r.n_one_sub:>4} positions, "
              f"{r.one_sub_per_kb:>6.1f}/kb", file=sys.stderr)

    land = []
    for r in H.itertuples():
        s = r.subfam
        if s not in cons:
            continue
        key = (r.chrom, r.el_start + 1, r.el_end, s)
        m = g2c_cache.get(key)
        if not m:
            continue
        idx = [m.get(p) for p in range(r.gpos_lo, r.gpos_hi + 1)]
        idx = [x for x in idx if x is not None]
        if len(idx) < 6:
            continue
        ci = min(idx)
        seq = cons[s]
        if ci + 6 > len(seq):
            continue
        csix = seq[ci:ci + 6]
        if r.el_sense == 0:
            csix = csix.translate(RC)[::-1]
        land.append(dict(subfamily=s, family=r.fam, is_real=r.is_real,
                         sense=r.el_sense, cons_pos=ci,
                         cons_hexamer=csix, motif=r.motif,
                         subs_from_pas=hamming_to_pas(csix),
                         frac_into_consensus=round(ci / max(len(seq), 1), 3),
                         el_div=r.el_div))
    L = pd.DataFrame(land)
    W("hexamer_landing.tsv", L)

    if len(L):
        print(f"\n[land] {len(L):,} observed hexamers mapped onto their "
              f"subfamily consensus", file=sys.stderr)
        vc = L.subs_from_pas.value_counts().sort_index()
        tot = int(vc.sum())
        print("       what the consensus says at the position the hexamer "
              "occupies:", file=sys.stderr)
        for k, v in vc.items():
            if not np.isfinite(k):
                continue
            lab = ("already a poly(A) signal" if k == 0 else
                   f"{int(k)} substitution" + ("s" if k > 1 else ""))
            print(f"         {lab:<28} {v:>6,}  ({100*v/tot:5.1f}%)",
                  file=sys.stderr)
        near = float((L.subs_from_pas <= 1).mean())
        print(f"       {100*near:.1f}% sit where the consensus is already a "
              f"signal or one substitution\n       away, against "
              f"{100*(L.subs_from_pas >= 3).mean():.1f}% three or more",
              file=sys.stderr)

        fl = (L.groupby("family")
                .agg(n=("subs_from_pas", "size"),
                     frac_exact=("subs_from_pas", lambda s: float((s == 0).mean())),
                     frac_one=("subs_from_pas", lambda s: float((s == 1).mean())),
                     median_subs=("subs_from_pas", "median"),
                     median_frac_into=("frac_into_consensus", "median"))
                .reset_index().sort_values("n", ascending=False))
        W("landing_by_family.tsv", fl)
        print("\n       by family:", file=sys.stderr)
        print(f"       {'family':<14}{'n':>7}{'exact':>9}{'one sub':>10}"
              f"{'median':>9}{'position':>10}", file=sys.stderr)
        for r in fl.head(10).itertuples():
            print(f"       {r.family:<14}{r.n:>7,}{100*r.frac_exact:>8.1f}%"
                  f"{100*r.frac_one:>9.1f}%{r.median_subs:>9.1f}"
                  f"{r.median_frac_into:>10.2f}", file=sys.stderr)

        sl = (L.groupby(["subfamily", "family"])
                .agg(n=("subs_from_pas", "size"),
                     frac_exact=("subs_from_pas", lambda s: float((s == 0).mean())),
                     frac_one_or_less=("subs_from_pas",
                                       lambda s: float((s <= 1).mean())),
                     median_frac_into=("frac_into_consensus", "median"))
                .reset_index())
        sl = sl[sl.n >= 20].sort_values("frac_one_or_less", ascending=False)
        W("landing_by_subfamily.tsv", sl)
        print("\n       subfamilies whose observed hexamers most often sit at "
              "a consensus\n       position already at or one step from a "
              "signal:", file=sys.stderr)
        for r in sl.head(15).itertuples():
            print(f"         {r.subfamily:<16}{r.family:<14}{r.n:>6,}  "
                  f"exact {100*r.frac_exact:>5.1f}%  "
                  f"within one {100*r.frac_one_or_less:>5.1f}%  "
                  f"at {r.median_frac_into:.2f} into the consensus",
                  file=sys.stderr)

    with open(os.path.join(a.out, "counts.json"), "w") as fh:
        json.dump(dict(n_sites=len(d), n_hex_in_element=len(H),
                       n_subfamilies=len(cons), n_copies_aligned=int(sum(naligned.values())),
                       n_landed=len(L)), fh, indent=2)
    print(f"\n[out]  {a.out}/", file=sys.stderr)
    print("\nRead [scan] for whether any subfamily consensus carries a signal "
          "natively, and\n[land] for whether the observed hexamers sit where "
          "the consensus was already\none step away. The second is the "
          "mutational-conversion mechanism; the first is\nnative inheritance. "
          "They are not mutually exclusive and different subfamilies\nmay show "
          "different ones.", file=sys.stderr)


if __name__ == "__main__":
    main()