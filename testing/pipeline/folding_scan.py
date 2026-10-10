#!/usr/bin/env python3
# are the false positives folded shut?
import argparse, os, sys, json, math, random
from collections import defaultdict

import numpy as np
import pandas as pd

try:
    import RNA
except ImportError:
    sys.exit("ViennaRNA not importable; pip install ViennaRNA")

try:
    from scipy import stats as sps
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

RC = str.maketrans("ACGTN", "TGCAN")
PAS_ALL = ["AATAAA", "ATTAAA", "TATAAA", "AGTAAA", "AATATA", "AATACA",
           "CATAAA", "GATAAA", "AATGAA", "TTTAAA", "ACTAAA", "AATAGA"]
PAS_CANON = ["AATAAA", "ATTAAA"]


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


def dinuc_shuffle(s, rng):
    if len(s) < 4:
        return s
    edges = defaultdict(list)
    for i in range(len(s) - 1):
        edges[s[i]].append(s[i + 1])
    last = s[-1]
    for _ in range(20):
        lastedge = {}
        ok = True
        for v in edges:
            if v == last:
                continue
            if not edges[v]:
                ok = False
                break
            lastedge[v] = rng.choice(edges[v])
        if not ok:
            return s
        good = True
        for v in lastedge:
            seen, cur = set(), v
            while cur != last:
                if cur in seen or cur not in lastedge:
                    good = False
                    break
                seen.add(cur)
                cur = lastedge[cur]
            if not good:
                break
        if good:
            break
    else:
        return s
    rem = {v: list(e) for v, e in edges.items()}
    for v, e in lastedge.items():
        rem[v].remove(e)
    for v in rem:
        rng.shuffle(rem[v])
    for v, e in lastedge.items():
        rem[v].append(e)
    out = [s[0]]
    cur = s[0]
    for _ in range(len(s) - 1):
        if not rem.get(cur):
            break
        nxt = rem[cur].pop(0)
        out.append(nxt)
        cur = nxt
    return "".join(out) if len(out) == len(s) else s


def fold_stats(rna, md=None):
    fc = RNA.fold_compound(rna, md) if md is not None else RNA.fold_compound(rna)
    ss, mfe = fc.mfe()
    fc.exp_params_rescale(mfe)
    _, efe = fc.pf()
    bpp = fc.bpp()
    n = len(rna)
    paired = np.zeros(n + 1)
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            p = bpp[i][j]
            if p:
                paired[i] += p
                paired[j] += p
    unpaired = 1.0 - np.clip(paired[1:], 0, 1)
    return dict(mfe=float(mfe), efe=float(efe), ss=ss), unpaired


def gc_of(s):
    n = sum(s.count(b) for b in "ACGU" + "T")
    g = s.count("G") + s.count("C")
    return g / n if n else np.nan


def find_pas(seq, centre, up=60, dn=0, motifs=PAS_ALL):
    lo, hi = centre - up, centre + dn
    best = None
    for m in motifs:
        i = seq.find(m, max(0, lo), min(len(seq), hi))
        while i != -1:
            off = i - centre
            if best is None or abs(off + 21) < abs(best[0] + 21):
                best = (off, m, i)
            i = seq.find(m, i + 1, min(len(seq), hi))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peaks", required=True)
    ap.add_argument("--genome", required=True)
    ap.add_argument("--win", type=int, default=150,
                    help="half-width of the folded window. Folding is O(n^3), "
                         "so 150 gives a 300 nt window and tractable runtime; "
                         "CPSF footprints are far smaller than this")
    ap.add_argument("--local_win", type=int, default=40,
                    help="half-width for the local fold around the hexamer")
    ap.add_argument("--n_shuffle", type=int, default=3,
                    help="dinucleotide shuffles per site. Each costs a full "
                         "fold, so this dominates the runtime")
    ap.add_argument("--max_ccpa", type=int, default=4,
                    help="false positives per gene. Lower than elsewhere "
                         "because folding is the expensive step")
    ap.add_argument("--temperature", type=float, default=37.0)
    ap.add_argument("--nperm", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="fold")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    W = lambda n, df: df.to_csv(os.path.join(a.out, n), sep="\t", index=False)
    rng = np.random.default_rng(a.seed)
    pyrng = random.Random(a.seed)

    md = RNA.md()
    md.temperature = a.temperature
    print(f"[cfg]  ViennaRNA {RNA.__version__ if hasattr(RNA,'__version__') else ''}"
          f", {a.temperature} C, window +/-{a.win} nt, "
          f"{a.n_shuffle} dinucleotide shuffles per site", file=sys.stderr)

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
    W_, LW = a.win, a.local_win
    rows = []
    nbad = nohex = 0

    for i, r in enumerate(d.itertuples()):
        if i and i % 1000 == 0:
            print(f"[run]  {i:,}/{len(d):,} sites...", file=sys.stderr)
        dna = oriented(gen, r.chrom, r.pos, r.strand, W_, W_)
        if len(dna) != 2 * W_ or "N" in dna:
            nbad += 1
            continue
        rna = dna.replace("T", "U")
        c = W_

        rec = dict(gene=r.gene, chrom=r.chrom, pos=r.pos, is_real=bool(r.is_real))
        rec["gc_window"] = gc_of(dna)

        # global folding of the window
        st, unp = fold_stats(rna, md)
        n = len(rna)
        rec["mfe_per_nt"] = st["mfe"] / n
        rec["efe_per_nt"] = st["efe"] / n
        rec["mean_unpaired"] = float(np.mean(unp))
        rec["unpaired_cleave"] = float(np.mean(unp[max(0, c - 10):c + 10]))
        rec["unpaired_dse"] = float(np.mean(unp[c + 10:c + 60]))

        # the hexamer specifically
        hit = find_pas(dna, c, up=60, dn=0)
        if hit is None:
            nohex += 1
            rec["has_pas"] = 0.0
            for k in ("pas_unpaired", "pas_unpaired_local", "pas_offset",
                      "pas_is_canon", "pas_unpaired_shuf",
                      "pas_unpaired_delta", "pas_open_energy"):
                rec[k] = np.nan
        else:
            off, motif, idx = hit
            rec["has_pas"] = 1.0
            rec["pas_offset"] = float(off)
            rec["pas_is_canon"] = float(motif in PAS_CANON)
            rec["pas_unpaired"] = float(np.mean(unp[idx:idx + 6]))

            l0 = max(0, idx + 3 - LW)
            l1 = min(len(rna), idx + 3 + LW)
            sub = rna[l0:l1]
            if len(sub) > 20:
                st2, unp2 = fold_stats(sub, md)
                j = idx - l0
                rec["pas_unpaired_local"] = float(np.mean(unp2[j:j + 6]))
                fc = RNA.fold_compound(sub, md)
                fc.mfe(); _, efe_free = fc.pf()
                fc2 = RNA.fold_compound(sub, md)
                for p_ in range(j + 1, j + 7):
                    if 1 <= p_ <= len(sub):
                        fc2.hc_add_up(p_)
                fc2.mfe(); _, efe_open = fc2.pf()
                rec["pas_open_energy"] = float(efe_open - efe_free)
            else:
                rec["pas_unpaired_local"] = np.nan
                rec["pas_open_energy"] = np.nan

            # dinucleotide-shuffled control
            vals = []
            for _ in range(a.n_shuffle):
                sh = dinuc_shuffle(rna, pyrng)
                k = sh.find(motif.replace("T", "U"))
                if k < 0:
                    continue
                _, unps = fold_stats(sh, md)
                vals.append(float(np.mean(unps[k:k + 6])))
            if vals:
                rec["pas_unpaired_shuf"] = float(np.mean(vals))
                rec["pas_unpaired_delta"] = (rec["pas_unpaired"] -
                                             rec["pas_unpaired_shuf"])
            else:
                rec["pas_unpaired_shuf"] = np.nan
                rec["pas_unpaired_delta"] = np.nan
        rows.append(rec)

    if nbad:
        print(f"[seq]  {nbad:,} sites dropped: short window or ambiguous bases",
              file=sys.stderr)
    P = pd.DataFrame(rows)
    W("per_site.tsv", P)
    print(f"[seq]  {len(P):,} sites folded; {nohex:,} had no hexamer upstream",
          file=sys.stderr)
    print(f"[pas]  hexamer present: real "
          f"{100*P.loc[P.is_real,'has_pas'].mean():.1f}%, false positive "
          f"{100*P.loc[~P.is_real,'has_pas'].mean():.1f}%", file=sys.stderr)

    FEATS = ["mfe_per_nt", "efe_per_nt", "mean_unpaired", "unpaired_cleave",
             "unpaired_dse", "pas_unpaired", "pas_unpaired_local",
             "pas_open_energy", "pas_unpaired_shuf", "pas_unpaired_delta",
             "gc_window"]

    # paired within gene
    R = P[P.is_real].drop_duplicates("gene").set_index("gene")
    F = P[~P.is_real].groupby("gene")[FEATS].mean()
    common = R.index.intersection(F.index)
    DIFF = (R.loc[common, FEATS].astype(float) - F.loc[common, FEATS].astype(float))
    W("paired_differences.tsv", DIFF.reset_index())

    def dz_of(v):
        v = np.asarray(v, float); v = v[np.isfinite(v)]
        if len(v) < 20:
            return np.nan, len(v), np.nan
        sd = v.std(ddof=1)
        dz = v.mean() / sd if sd > 0 else np.nan
        p = np.nan
        if HAVE_SCIPY and sd > 0:
            try:
                _, p = sps.wilcoxon(v, zero_method="zsplit")
            except Exception:
                pass
        return float(dz), len(v), float(p)

    res = []
    for f in FEATS:
        dz, n, p = dz_of(DIFF[f].to_numpy())
        rl = P.loc[P.is_real, f].astype(float)
        fp = P.loc[~P.is_real, f].astype(float)
        res.append(dict(feature=f, n_genes=n,
                        paired_diff=round(float(DIFF[f].mean()), 5),
                        paired_dz=round(dz, 4) if np.isfinite(dz) else np.nan,
                        p_wilcoxon=p,
                        real_mean=round(float(rl.mean()), 5),
                        fp_mean=round(float(fp.mean()), 5)))
    S = pd.DataFrame(res).sort_values("paired_dz", key=abs, ascending=False)
    W("headline.tsv", S)
    print("\n[main] paired within gene, real minus its own false positives",
          file=sys.stderr)
    print(f"       {'feature':<22}{'real':>10}{'fp':>10}{'dz':>9}{'p':>12}",
          file=sys.stderr)
    for r_ in S.itertuples():
        print(f"       {r_.feature:<22}{r_.real_mean:>10.4f}{r_.fp_mean:>10.4f}"
              f"{r_.paired_dz:>+9.3f}{r_.p_wilcoxon:>12.2e}", file=sys.stderr)

    print("\n[gc]   how much of each folding measure is GC content?",
          file=sys.stderr)
    gcd = DIFF["gc_window"].to_numpy(float)
    dec = []
    for f in FEATS:
        if f == "gc_window":
            continue
        v = DIFF[f].to_numpy(float)
        m = np.isfinite(v) & np.isfinite(gcd)
        if m.sum() < 100:
            continue
        r = float(np.corrcoef(v[m], gcd[m])[0, 1])
        bt = np.polyfit(gcd[m], v[m], 1)
        resid = v[m] - np.polyval(bt, gcd[m])
        dz_raw, _, _ = dz_of(v[m])
        dz_res, _, p_res = dz_of(resid)
        dec.append(dict(feature=f, r_with_gc=round(r, 4),
                        dz_raw=round(dz_raw, 4),
                        dz_given_gc=round(dz_res, 4),
                        retained=round(abs(dz_res) / abs(dz_raw), 3)
                        if np.isfinite(dz_raw) and abs(dz_raw) > 1e-9 else np.nan,
                        p_given_gc=p_res))
        print(f"       {f:<22} r(GC) {r:+.3f}   dz {dz_raw:+.3f} -> "
              f"{dz_res:+.3f} given GC   ({100*dec[-1]['retained']:.0f}% "
              f"retained)", file=sys.stderr)
    W("gc_decomposition.tsv", pd.DataFrame(dec))

    print("\n[shuf] hexamer accessibility against dinucleotide-shuffled "
          "controls", file=sys.stderr)
    print("       the shuffle preserves composition and dinucleotide "
          "frequency exactly, so\n       a difference here is structure that "
          "composition does not force",
          file=sys.stderr)
    for lab, m in (("real 3' end", P.is_real), ("false positive", ~P.is_real)):
        s = P[m & (P.has_pas == 1)]
        o = s.pas_unpaired.astype(float)
        h = s.pas_unpaired_shuf.astype(float)
        dd = (o - h).dropna()
        if not len(dd):
            continue
        sd = dd.std(ddof=1)
        print(f"       {lab:<16} observed {o.mean():.4f}, shuffled "
              f"{h.mean():.4f}, delta {dd.mean():+.4f} "
              f"(dz {dd.mean()/sd:+.3f}, n {len(dd):,})", file=sys.stderr)
    dz, n, p = dz_of(DIFF["pas_unpaired_delta"].to_numpy())
    print(f"       paired within gene, real minus fp, of that delta: "
          f"dz {dz:+.3f}, p {p:.2e}, n {n:,}", file=sys.stderr)
    print("       this is the number that tests the folding hypothesis "
          "independently of GC", file=sys.stderr)

    key = ["pas_unpaired", "pas_unpaired_local", "pas_open_energy",
           "pas_unpaired_delta", "mfe_per_nt"]
    key = [k for k in key if k in P.columns]
    gl = {g: sub[key].to_numpy(float)
          for g, sub in P.groupby("gene", sort=False) if len(sub) >= 2}
    obs = {k: dz_of(DIFF[k].to_numpy())[0] for k in key}
    nulls = {k: [] for k in key}
    print(f"\n[perm] {a.nperm} within-gene label permutations", file=sys.stderr)
    for it in range(a.nperm):
        acc = []
        for g, M in gl.items():
            j = rng.integers(0, len(M))
            acc.append(M[j] - np.nanmean(np.delete(M, j, axis=0), axis=0))
        A = np.array(acc, float)
        sd = np.nanstd(A, axis=0, ddof=1)
        dzv = np.nanmean(A, axis=0) / np.where(sd > 0, sd, np.nan)
        for j2, k in enumerate(key):
            nulls[k].append(dzv[j2])
    prow = []
    for k in key:
        nl = np.array(nulls[k], float); nl = nl[np.isfinite(nl)]
        if not len(nl) or not np.isfinite(obs[k]):
            continue
        pp = (1 + int((np.abs(nl) >= abs(obs[k])).sum())) / (len(nl) + 1)
        prow.append(dict(feature=k, observed_dz=round(obs[k], 4),
                         null_sd=round(float(nl.std(ddof=1)), 4), p_perm=pp))
        print(f"       {k:<22} observed {obs[k]:+.3f}, null sd "
              f"{nl.std(ddof=1):.3f}, p {pp:.3g}", file=sys.stderr)
    W("permutation.tsv", pd.DataFrame(prow))

    W("plot_values.tsv", P[["gene", "is_real", "has_pas"] + FEATS])

    with open(os.path.join(a.out, "counts.json"), "w") as fh:
        json.dump(dict(n_sites=len(P), n_genes=int(P.gene.nunique()),
                       n_real=int(P.is_real.sum()),
                       n_fp=int((~P.is_real).sum()),
                       win=W_, local_win=LW, n_shuffle=a.n_shuffle,
                       temperature=a.temperature), fh, indent=2)
    print(f"\n[out]  {a.out}/", file=sys.stderr)
    print("\nRead [gc] then [shuf]. Global folding energy will track GC and is "
          "not a\nseparate finding. The hypothesis stands or falls on whether "
          "hexamer\naccessibility survives residualisation on GC, and on "
          "whether the real sites\nexpose their hexamer more than their own "
          "dinucleotide shuffles do.", file=sys.stderr)


if __name__ == "__main__":
    main()