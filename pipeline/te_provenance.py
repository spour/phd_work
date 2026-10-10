#!/usr/bin/env python3
# did the transposable element bring the poly(a) signal?
import argparse, os, sys, json, re, gzip
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

RC = str.maketrans("ACGTN-", "TGCAN-")
PAS = ["AATAAA", "ATTAAA"]
FOCUS = ["ERVL", "ERV1", "ERVL-MaLR", "ERVK", "L1", "L2", "Alu", "MIR",
         "hAT-Charlie", "TcMar-Tigger", "CR1", "hAT-Tip100"]


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


# rmsk
def load_rmsk(path, chroms):
    op = gzip.open if path.endswith(".gz") else open
    by = defaultdict(list)
    n = 0
    with op(path, "rt") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 13 or f[5] not in chroms:
                continue
            by[f[5]].append(dict(start=int(f[6]), end=int(f[7]), strand=f[9],
                                 name=f[10], cls=f[11], fam=f[12],
                                 divergence=float(f[2]) / 10.0))
            n += 1
    for c in by:
        by[c].sort(key=lambda x: x["start"])
    print(f"[rmsk] {n:,} intervals", file=sys.stderr)
    return by


def element_at(by, starts, chrom, pos):
    e = by.get(chrom)
    if e is None:
        return None
    j = np.searchsorted(starts[chrom], pos, side="right") - 1
    if j < 0 or pos > e[j]["end"]:
        return None
    return e[j]


# align
HDR = re.compile(
    r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\S+)\s+(\d+)\s+(\d+)\s+"
    r"\((\d+)\)\s+(C\s+)?(\S+?)#(\S+)\s+(\(?\d+\)?)\s+(\(?\d+\)?)\s+(\(?\d+\)?)")


def parse_align(path, targets):
    out = {}
    cur = None
    gseq, cseq = [], []
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
                    chrom, gs, ge = m.group(5), int(m.group(6)), int(m.group(7))
                    key = (chrom, gs, ge, m.group(10))
                    if key in targets:
                        nkept += 1
                        cur = dict(key=key, chrom=chrom, gstart=gs, gend=ge,
                                   reverse=m.group(9) is not None,
                                   name=m.group(10), clsfam=m.group(11),
                                   divergence=float(m.group(2)))
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
    print(f"[align] {nseen:,} headers scanned, {len(out):,} target blocks "
          f"recovered", file=sys.stderr)
    return out


def map_to_consensus(block, gpos_list):
    g, c = block["genomic"], block["consensus"]
    n = min(len(g), len(c))
    gp = block["meta"]["gstart"]
    ci = 0
    want = set(gpos_list)
    out = {}
    for i in range(n):
        gb, cb = g[i], c[i]
        if gb != "-":
            if gp in want:
                out[gp] = (ci, cb)
            gp += 1
        if cb != "-":
            ci += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peaks", required=True)
    ap.add_argument("--rmsk", required=True)
    ap.add_argument("--align", required=True)
    ap.add_argument("--genome", required=True)
    ap.add_argument("--hex_up", type=int, default=60)
    ap.add_argument("--out", default="te")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    W = lambda n, df: df.to_csv(os.path.join(a.out, n), sep="\t", index=False)

    pk = pd.read_parquet(a.peaks)
    if "in_analysis_set" in pk.columns:
        pk = pk[pk.in_analysis_set.astype(bool) | pk.is_canonical.astype(bool)]
    POS = "center" if "center" in pk.columns else "pos"
    d = pd.DataFrame({"gene": pk.gene.astype(str), "chrom": pk.chrom.astype(str),
                      "pos": pk[POS].astype(int), "strand": pk.strand.astype(str),
                      "is_real": pk.is_canonical.astype(bool)})
    print(f"[set]  {int(d.is_real.sum()):,} real, "
          f"{int((~d.is_real).sum()):,} false positives", file=sys.stderr)

    by = load_rmsk(a.rmsk, set(d.chrom.unique()))
    starts = {c: np.array([x["start"] for x in v]) for c, v in by.items()}
    gen = Genome(a.genome)

    # locate the hexamer and its element
    recs = []
    for r in d.itertuples():
        seq = oriented(gen, r.chrom, r.pos, r.strand, a.hex_up, 10)
        if len(seq) < a.hex_up + 10:
            continue
        up = seq[:a.hex_up]
        best = None
        for h in PAS:
            i = up.rfind(h)
            if i >= 0 and (best is None or i > best[0]):
                best = (i, h)
        if best is None:
            continue
        off, motif = best
        if r.strand == "-":
            gpos = [r.pos + (a.hex_up - off) - 1 - k for k in range(6)]
        else:
            gpos = [r.pos - a.hex_up + off + k for k in range(6)]
        el = element_at(by, starts, r.chrom, int(np.mean(gpos)))
        recs.append(dict(gene=r.gene, chrom=r.chrom, pos=r.pos,
                         strand=r.strand, is_real=bool(r.is_real),
                         motif=motif, hex_offset=off - a.hex_up,
                         gpos_lo=min(gpos), gpos_hi=max(gpos),
                         in_element=el is not None,
                         el_name=el["name"] if el else "none",
                         el_fam=el["fam"] if el else "none",
                         el_start=el["start"] if el else -1,
                         el_end=el["end"] if el else -1,
                         el_strand=el["strand"] if el else ".",
                         el_div=el["divergence"] if el else np.nan,
                         el_sense=float(el["strand"] == r.strand) if el else np.nan))
    H = pd.DataFrame(recs)
    W("hexamers.tsv", H)
    print(f"[hex]  {len(H):,} sites with a canonical hexamer; "
          f"{int(H.in_element.sum()):,} lie in a repeat", file=sys.stderr)

    # test 2: availability
    span = (d.groupby(["gene", "chrom"], as_index=False)
              .agg(lo=("pos", "min"), hi=("pos", "max")))
    span["lo"] -= 5000
    span["hi"] += 5000
    avail = Counter()
    for r in span.itertuples():
        e = by.get(r.chrom)
        if not e:
            continue
        i0 = np.searchsorted(starts[r.chrom], r.lo, side="left")
        i1 = np.searchsorted(starts[r.chrom], r.hi, side="right")
        for k in range(i0, i1):
            avail[e[k]["fam"]] += 1
    used_sense = Counter()
    for r in H[H.in_element & ~H.is_real].itertuples():
        used_sense[(r.el_fam, "sense" if r.el_sense == 1 else "antisense")] += 1
    av = []
    for fam in FOCUS:
        tot = avail[fam]
        for o in ("sense", "antisense"):
            u = used_sense[(fam, o)]
            av.append(dict(family=fam, orientation=o, n_used=u,
                           n_copies_available=tot,
                           per_copy_rate=round(u / max(tot, 1), 5)))
    AV = pd.DataFrame(av).sort_values("per_copy_rate", ascending=False)
    W("availability.tsv", AV)
    print("\n[avail] false positives per available element copy", file=sys.stderr)
    print(AV.head(12).to_string(index=False), file=sys.stderr)

    tgt = H[H.in_element & H.el_fam.isin(FOCUS)]
    targets = set((r.chrom, r.el_start + 1, r.el_end, r.el_name)
                  for r in tgt.itertuples())
    print(f"\n[align] scanning for {len(targets):,} target elements",
          file=sys.stderr)
    blocks = parse_align(a.align, targets)

    prov = []
    for r in tgt.itertuples():
        b = blocks.get((r.chrom, r.el_start + 1, r.el_end, r.el_name))
        if b is None:
            continue
        m = map_to_consensus(b, list(range(r.gpos_lo, r.gpos_hi + 1)))
        if len(m) < 6:
            continue
        ks = sorted(m)
        idxs = [m[p][0] for p in ks]
        bases = "".join(m[p][1] for p in ks)

        cons = bases if r.el_sense == 1 else bases.translate(RC)[::-1]

        prov.append(dict(gene=r.gene, chrom=r.chrom, pos=r.pos,
                         is_real=r.is_real, family=r.el_fam, name=r.el_name,
                         sense=r.el_sense, divergence=r.el_div, motif=r.motif,
                         cons_start=min(idxs), cons_end=max(idxs),
                         cons_bases=cons,
                         consensus_has_pas=float(cons in PAS),
                         n_matching=sum(1 for x, y in zip(cons, r.motif)
                                        if x == y)))
    P = pd.DataFrame(prov)
    W("provenance.tsv", P)
    if not len(P):
        print("[prov] no alignments recovered; check the header regex",
              file=sys.stderr)
        return

    print(f"\n[prov] {len(P):,} hexamers mapped to consensus coordinates",
          file=sys.stderr)
    print("       Alu sense is the internal control: its A-rich regions align "
          "regardless,\n       so a low base-agreement there means the mapping "
          "is broken, not that the\n       elements lack signals",
          file=sys.stderr)
    for fam in FOCUS:
        s = P[P.family == fam]
        if len(s) < 20:
            continue
        for lab, m in (("sense", s.sense == 1), ("antisense", s.sense == 0)):
            t = s[m]
            if len(t) < 10:
                continue
            print(f"       {fam:<14} {lab:<10} n {len(t):>5,}   "
                  f"consensus carries the hexamer "
                  f"{100*t.consensus_has_pas.mean():5.1f}%   "
                  f"mean bases matching {t.n_matching.mean():.2f}/6",
                  file=sys.stderr)

    pr = (P.groupby(["family", "sense"])
            .agg(n=("consensus_has_pas", "size"),
                 frac_consensus=("consensus_has_pas", "mean"),
                 mean_match=("n_matching", "mean"),
                 median_divergence=("divergence", "median")).reset_index())
    W("provenance_summary.tsv", pr)

    cp = []
    for fam in FOCUS:
        s = P[(P.family == fam) & (P.sense == 1)]
        if len(s) < 20:
            continue
        h, e = np.histogram(s.cons_start, bins=40)
        for x, y in zip(e[:-1], h):
            cp.append(dict(family=fam, cons_pos=float(x), n=int(y),
                           total=len(s)))
    W("consensus_positions.tsv", pd.DataFrame(cp))

    dv = []
    for fam in FOCUS:
        s = P[(P.family == fam) & (P.sense == 1)]
        if len(s) < 40 or s["divergence"].nunique() < 4:
            continue
        s = s.assign(q=pd.qcut(s["divergence"].rank(method="first"), 4,
                               labels=False))
        for q, t in s.groupby("q"):
            dv.append(dict(family=fam, div_quartile=int(q) + 1, n=len(t),
                           median_divergence=round(float(t["divergence"].median()), 2),
                           frac_consensus=round(float(t.consensus_has_pas.mean()), 4),
                           mean_match=round(float(t.n_matching.mean()), 3)))
    DV = pd.DataFrame(dv)
    W("divergence.tsv", DV)
    if len(DV):
        print("\n[div]  provenance by element age, sense only", file=sys.stderr)
        print(DV.to_string(index=False), file=sys.stderr)

    ex = P[P.family == "Alu"].head(8)[["motif", "cons_bases", "sense",
                                       "n_matching", "cons_start"]]
    if len(ex):
        print("\n[eg]   Alu examples, hexamer against consensus:",
              file=sys.stderr)
        print(ex.to_string(index=False), file=sys.stderr)

    with open(os.path.join(a.out, "counts.json"), "w") as fh:
        json.dump(dict(n_sites=len(d), n_with_hexamer=len(H),
                       n_in_element=int(H.in_element.sum()),
                       n_targets=len(targets), n_blocks=len(blocks),
                       n_mapped=len(P)), fh, indent=2)
    print(f"\n[out]  {a.out}/", file=sys.stderr)
    print("\nRead provenance_summary.tsv. If sense-oriented ERV false positives "
          "carry the\nhexamer in their consensus at a much higher rate than "
          "antisense ones, the\nelement brought the signal. If both are near "
          "zero while Alu sense shows high\nbase agreement, the mapping is "
          "sound and the hexamers are post-insertion.", file=sys.stderr)


if __name__ == "__main__":
    main()