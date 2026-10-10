#!/usr/bin/env python3
# how specific retrotransposon families seed cryptic poly(a) architecture
import argparse, os, sys, json, gzip, re
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

RC = str.maketrans("ACGTN-", "TGCAN-")
PAS = ["AATAAA", "ATTAAA"]
DSE = ["TTTT", "TGTG"]

SEEDERS = ["SVA", "Alu", "ERV1", "ERVK", "ERVL", "MIR", "L2", "ERVL-MaLR", "L1"]

NOMINAL = {"Alu": 300, "SVA": 2000, "L1": 6000, "L2": 3500, "MIR": 260,
           "ERVL-MaLR": 500, "ERV1": 1000, "ERVL": 700, "ERVK": 1000}


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


def load_rmsk(path, chroms):
    op = gzip.open if path.endswith(".gz") else open
    by = defaultdict(list)
    n = 0
    with op(path, "rt") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 13 or f[5] not in chroms:
                continue
            by[f[5]].append((int(f[6]), int(f[7]), f[9], f[10], f[11], f[12],
                             float(f[2]) / 10.0))
            n += 1
    out = {}
    for c, v in by.items():
        v.sort(key=lambda x: x[0])
        out[c] = v
    print(f"[rmsk] {n:,} intervals", file=sys.stderr)
    return out


HDR = re.compile(
    r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\S+)\s+(\d+)\s+(\d+)\s+"
    r"\((\d+)\)\s+(C\s+)?(\S+?)#(\S+)\s+")


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
                    key = (m.group(5), int(m.group(6)), int(m.group(7)),
                           m.group(10))
                    if key in targets:
                        nkept += 1
                        cur = dict(key=key, chrom=m.group(5),
                                   gstart=int(m.group(6)),
                                   gend=int(m.group(7)),
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
    print(f"[align] {nseen:,} headers scanned, {len(out):,} blocks recovered",
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peaks", required=True)
    ap.add_argument("--rmsk", required=True)
    ap.add_argument("--align", required=True)
    ap.add_argument("--genome", required=True)
    ap.add_argument("--pad", type=int, default=5000)
    ap.add_argument("--dse_lo", type=int, default=15)
    ap.add_argument("--dse_hi", type=int, default=70)
    ap.add_argument("--max_copies", type=int, default=6000,
                    help="copies sampled per family for the consensus and the "
                         "positional density; the alignment pass is the cost")
    ap.add_argument("--out", default="seed")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    W = lambda n, df: df.to_csv(os.path.join(a.out, n), sep="\t", index=False)
    rng = np.random.default_rng(0)

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
    fp_pos = defaultdict(set)
    for r in d[~d.is_real].itertuples():
        fp_pos[r.chrom].add(r.pos)
    print(f"[set]  {len(span):,} genes, "
          f"{(span.hi-span.lo).sum()/1e6:,.0f} Mb", file=sys.stderr)

    rm = load_rmsk(a.rmsk, set(span.chrom.unique()))
    gen = Genome(a.genome)

    # collect element copies inside the gene spans
    copies = defaultdict(list)
    for r in span.itertuples():
        e = rm.get(r.chrom)
        if not e:
            continue
        st = np.array([x[0] for x in e])
        i0 = np.searchsorted(st, r.lo, side="left")
        i1 = np.searchsorted(st, r.hi, side="right")
        for k in range(i0, i1):
            s0, s1, estr, ename, ecls, efam, ediv = e[k]
            if efam not in SEEDERS:
                continue
            if s1 <= r.lo or s0 >= r.hi:
                continue
            sense = "sense" if estr == r.strand else "antisense"
            copies[efam].append(dict(chrom=r.chrom, start=s0, end=s1,
                                     name=ename, strand=estr, div=ediv,
                                     sense=sense, tstrand=r.strand))
    for f in SEEDERS:
        print(f"[copy] {f:<12} {len(copies[f]):,} copies in the gene spans",
              file=sys.stderr)

    targets = set()
    keep = defaultdict(list)
    for f in SEEDERS:
        v = copies[f]
        if len(v) > a.max_copies:
            idx = rng.choice(len(v), a.max_copies, replace=False)
            v = [v[i] for i in idx]
        keep[f] = v
        for c in v:
            targets.add((c["chrom"], c["start"] + 1, c["end"], c["name"]))
    print(f"[align] {len(targets):,} target elements", file=sys.stderr)
    blocks = parse_align(a.align, targets)

    # walk each copy
    basecnt = {f: defaultdict(Counter) for f in SEEDERS}
    hexpos = {f: {"sense": Counter(), "antisense": Counter()} for f in SEEDERS}
    covpos = {f: {"sense": Counter(), "antisense": Counter()} for f in SEEDERS}
    percopy = []

    for f in SEEDERS:
        got = 0
        for c in keep[f]:
            b = blocks.get((c["chrom"], c["start"] + 1, c["end"], c["name"]))
            if b is None:
                continue
            got += 1
            seq = gen.fetch(c["chrom"], c["start"], c["end"])
            if len(seq) != c["end"] - c["start"] or "N" in seq[:50]:
                continue
            tseq = seq if c["tstrand"] != "-" else seq.translate(RC)[::-1]

            g2c = {}
            for gp, ci, gb, cb in walk(b):
                g2c[gp] = ci
                basecnt[f][ci][gb] += 1

            for ci in set(g2c.values()):
                covpos[f][c["sense"]][ci] += 1

            n = len(tseq)
            nhex = narch = nfp = 0
            for h in PAS:
                i = tseq.find(h)
                while i != -1:
                    nhex += 1
                    gp = (c["start"] + i) if c["tstrand"] != "-" \
                         else (c["end"] - i - 6)
                    ci = g2c.get(gp)
                    if ci is not None:
                        hexpos[f][c["sense"]][ci] += 1
                    w0, w1 = i + 6 + a.dse_lo, min(n, i + 6 + a.dse_hi)
                    if w1 > w0 and any(p in tseq[w0:w1] for p in DSE):
                        narch += 1
                        for dd in range(10, 31):
                            off = i + 6 + dd
                            gq = (c["start"] + off) if c["tstrand"] != "-" \
                                 else (c["end"] - off - 1)
                            if gq in fp_pos[c["chrom"]]:
                                nfp += 1
                                break
                    i = tseq.find(h, i + 1)
            percopy.append(dict(family=f, sense=c["sense"], div=c["div"],
                                length=c["end"] - c["start"],
                                n_hex=nhex, n_arch=narch, n_fp=nfp))
        print(f"[walk] {f:<12} {got:,} copies aligned", file=sys.stderr)

    PC = pd.DataFrame(percopy)
    W("per_copy.tsv", PC)

    # empirical consensus
    cons_rows = []
    for f in SEEDERS:
        if not basecnt[f]:
            continue
        mx = max(basecnt[f])
        lim = min(mx, NOMINAL.get(f, 2000))
        for ci in range(lim + 1):
            cnt = basecnt[f].get(ci)
            if not cnt:
                continue
            tot = sum(cnt[b] for b in "ACGT")
            if tot < 20:
                continue
            best = max("ACGT", key=lambda b: cnt[b])
            cons_rows.append(dict(family=f, cons_pos=ci, base=best,
                                  n=tot,
                                  frac=round(cnt[best] / tot, 4),
                                  fA=round(cnt["A"] / tot, 4),
                                  fC=round(cnt["C"] / tot, 4),
                                  fG=round(cnt["G"] / tot, 4),
                                  fT=round(cnt["T"] / tot, 4)))
    CS = pd.DataFrame(cons_rows)
    W("consensus.tsv", CS)
    print(f"\n[cons] empirical consensus built for "
          f"{CS.family.nunique()} families", file=sys.stderr)

    # positional hexamer density
    dens = []
    for f in SEEDERS:
        for o in ("sense", "antisense"):
            cov = covpos[f][o]
            hx = hexpos[f][o]
            if not cov:
                continue
            lim = min(max(cov), NOMINAL.get(f, 2000))
            for ci in range(lim + 1):
                c_ = cov.get(ci, 0)
                if c_ < 20:
                    continue
                dens.append(dict(family=f, orientation=o, cons_pos=ci,
                                 n_copies=c_, n_hex=hx.get(ci, 0),
                                 per_1000=round(1000 * hx.get(ci, 0) / c_, 3)))
    DN = pd.DataFrame(dens)
    W("hexamer_density.tsv", DN)

    print("\n[peak] where hexamers arise in each family, sense orientation",
          file=sys.stderr)
    pk_rows = []
    for f in SEEDERS:
        s = DN[(DN.family == f) & (DN.orientation == "sense")]
        if len(s) < 50:
            continue
        s = s.sort_values("per_1000", ascending=False)
        cs = CS[CS.family == f].set_index("cons_pos")
        seen = []
        for r in s.itertuples():
            if any(abs(r.cons_pos - p) < 30 for p in seen):
                continue
            seen.append(r.cons_pos)
            lo, hi = r.cons_pos - 3, r.cons_pos + 9
            ctx = "".join(cs.base.get(i, ".") for i in range(lo, hi))
            core = "".join(cs.base.get(i, ".") for i in range(r.cons_pos,
                                                              r.cons_pos + 6))
            dist = min(sum(1 for x, y in zip(core, h) if x != y) for h in PAS) \
                if len(core) == 6 and "." not in core else np.nan
            pk_rows.append(dict(family=f, cons_pos=r.cons_pos,
                                per_1000=r.per_1000, n_copies=r.n_copies,
                                consensus_context=ctx, consensus_hexamer=core,
                                substitutions_from_PAS=dist))
            if len(seen) >= 3:
                break
        for p in pk_rows[-len(seen):]:
            print(f"       {f:<12} pos {p['cons_pos']:>5}   "
                  f"{p['per_1000']:>6.1f} hexamers per 1000 copies   "
                  f"consensus {p['consensus_context']}   "
                  f"{p['substitutions_from_PAS']:.0f} substitution(s) from a PAS"
                  if np.isfinite(p['substitutions_from_PAS']) else
                  f"       {f:<12} pos {p['cons_pos']:>5}   "
                  f"{p['per_1000']:>6.1f} per 1000   "
                  f"consensus {p['consensus_context']}", file=sys.stderr)
    W("density_peaks.tsv", pd.DataFrame(pk_rows))

    age = []
    for f in SEEDERS:
        s = PC[(PC.family == f) & (PC.sense == "sense") & (PC.length > 100)]
        if len(s) < 200 or s["div"].nunique() < 4:
            continue
        s = s.assign(q=pd.qcut(s["div"].rank(method="first"), 4, labels=False))
        for q, t in s.groupby("q"):
            kb = t.length.sum() / 1000.0
            age.append(dict(family=f, quartile=int(q) + 1, n_copies=len(t),
                            median_div=round(float(t["div"].median()), 2),
                            kb=round(kb, 1),
                            hex_per_kb=round(t.n_hex.sum() / kb, 3),
                            arch_per_kb=round(t.n_arch.sum() / kb, 3),
                            fp_per_arch=round(t.n_fp.sum() /
                                              max(t.n_arch.sum(), 1), 4)))
    AG = pd.DataFrame(age)
    W("age.tsv", AG)
    if len(AG):
        print("\n[age]  architecture density by element divergence, sense only",
              file=sys.stderr)
        for f in AG.family.unique():
            s = AG[AG.family == f]
            tr = "declines with age" if s.arch_per_kb.iloc[0] > \
                 s.arch_per_kb.iloc[-1] * 1.15 else "flat"
            print(f"       {f:<12} " +
                  "  ".join(f"{r.median_div:>5.1f}%:{r.arch_per_kb:>6.3f}"
                            for r in s.itertuples()) + f"   {tr}",
                  file=sys.stderr)

    summ = []
    for f in SEEDERS:
        for o in ("sense", "antisense"):
            s = PC[(PC.family == f) & (PC.sense == o)]
            if not len(s):
                continue
            kb = s.length.sum() / 1000.0
            summ.append(dict(family=f, orientation=o, n_copies=len(s),
                             kb=round(kb, 1),
                             hex_per_kb=round(s.n_hex.sum() / kb, 3),
                             arch_per_kb=round(s.n_arch.sum() / kb, 3),
                             fp_per_arch=round(s.n_fp.sum() /
                                               max(s.n_arch.sum(), 1), 4),
                             median_div=round(float(s["div"].median()), 2)))
    SM = pd.DataFrame(summ)
    W("family_summary.tsv", SM)
    print("\n[fam]  per family, from the aligned copies", file=sys.stderr)
    print(SM.to_string(index=False), file=sys.stderr)

    with open(os.path.join(a.out, "counts.json"), "w") as fh:
        json.dump(dict(families=SEEDERS, n_targets=len(targets),
                       n_blocks=len(blocks), n_copies=int(len(PC)),
                       max_copies=a.max_copies), fh, indent=2)
    print(f"\n[out]  {a.out}/", file=sys.stderr)


if __name__ == "__main__":
    main()