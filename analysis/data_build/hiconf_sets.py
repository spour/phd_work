#!/usr/bin/env python3
# one high-confidence set per experiment, then all-by-all

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict

import numpy as np
import pandas as pd

HEX = ("AATAAA", "ATTAAA")
RC = str.maketrans("ACGTN", "TGCAN")


# util
def find_first(*c):
    for x in c:
        if x and os.path.exists(x):
            return x
    return None


def load_config(path):
    import yaml
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    d = cfg.setdefault("defaults", {})
    d.setdefault("tolerance", 50)
    d.setdefault("min_reads", 2)
    d.setdefault("enrich_ratio", 3.0)
    d.setdefault("merge_within", 25)
    cfg["experiments"] = [e for e in cfg.get("experiments", [])
                          if e.get("enabled", True)]
    for e in cfg["experiments"]:
        if str(e.get("species", "human")).lower() != "human":
            sys.exit(e["name"] + ": non-human, disable it")
    return cfg


def liftover(df, chain, exe, label):
    if not exe or not chain:
        sys.exit(label + ": liftOver/chain missing")
    tmp = tempfile.mkdtemp(prefix="lo_")
    src, dst, unm = (os.path.join(tmp, x) for x in ("i", "o", "u"))
    df = df.reset_index(drop=True)
    with open(src, "w") as fh:
        for i, (c, p, s) in enumerate(zip(df.chrom, df.pos, df.strand)):
            fh.write(f"{c}\t{int(p)-1}\t{int(p)}\t{i}\t0\t"
                     f"{s if s in '+-' else '+'}\n")
    subprocess.run([exe, "-minMatch=0.95", src, chain, dst, unm],
                   capture_output=True, text=True)
    o = pd.read_csv(dst, sep="\t", header=None,
                    names=["chrom", "s", "pos", "idx", "sc", "strand"])
    shutil.rmtree(tmp, ignore_errors=True)
    o = o.loc[~o.idx.duplicated(keep=False)]
    k = df.loc[o.idx.to_numpy()].copy()
    k["chrom"] = o.chrom.to_numpy()
    k["pos"] = o.pos.to_numpy()
    print("     lifted " + f"{len(k):,}" + "/" + f"{len(df):,}")
    return k.reset_index(drop=True)


def collapse(df, within):
    if within <= 0 or not len(df):
        return df.reset_index(drop=True)
    out = []
    for (c, s), g in df.groupby(["chrom", "strand"], sort=False):
        g = g.sort_values("pos", kind="mergesort")
        pos = g.pos.to_numpy()
        val = np.nan_to_num(g.value.to_numpy(float))
        brk = np.concatenate([[0], np.cumsum(np.diff(pos) > within)])
        for _, ix in pd.Series(np.arange(len(pos))).groupby(brk):
            i = ix.to_numpy()
            b = i[np.argmax(val[i])]
            out.append((c, int(pos[b]), s, float(val[b])))
    return pd.DataFrame(out, columns=["chrom", "pos", "strand", "value"])


# enriched-block calls

STRAND_PATS = [(".fwd.", ".rev."), ("_Fwd.", "_Rev."), ("_fwd.", "_rev."),
               ("_forward", "_reverse"), ("_F.", "_R."), ("_plus", "_minus"),
               ("_pos.", "_neg."), (".plus.", ".minus."), ("Plus", "Minus")]


def strand_mate(path):
    base = os.path.basename(path)
    for fwd, rev in STRAND_PATS:
        if fwd in base:
            cand = path.replace(fwd, rev)
            if os.path.exists(cand):
                return cand, fwd + " / " + rev
    return None, None


def call_enriched(treat_path, bg_path, p, label):
    import pyBigWig

    min_reads = float(p.get("min_reads", 2))
    ratio = float(p.get("enrich_ratio", 3.0))
    merge = int(p.get("merge_within", 25))
    direction = str(p.get("direction", "up")).lower()
    if direction not in ("up", "down"):
        sys.exit(label + ": direction must be up or down")
    if direction == "down":
        treat_path, bg_path = bg_path, treat_path
    if p.get("value_is_rpm"):
        lib = float(p.get("library_size", 1e8))
        thr = min_reads * 1e6 / lib
        eps = 0.5 * 1e6 / lib
    else:
        thr, eps = min_reads, 0.5
    print("     direction=" + direction + "; blocks called from " +
          os.path.basename(treat_path))
    print("     floor " + f"{thr:.4g}" + " (>= " + f"{min_reads:.0f}" +
          " reads), " + ("gain" if direction == "up" else "loss") + " >= " +
          f"{ratio:g}" + "x vs " + os.path.basename(bg_path))

    def blocks(fp, strand):
        bw = pyBigWig.open(fp)
        rows = []
        for chrom in bw.chroms():
            try:
                ivs = bw.intervals(chrom)
            except RuntimeError:
                continue
            if not ivs:
                continue
            keep = [(s, e, abs(v)) for s, e, v in ivs if abs(v) >= thr * 0.999]
            if not keep:
                continue
            be, bv, bp = keep[0][1], keep[0][2], keep[0][0]
            for s, e, v in keep[1:]:
                if s - be <= merge:
                    be = max(be, e)
                    if v > bv:
                        bv, bp = v, s
                else:
                    rows.append((chrom, bp, strand, bv))
                    be, bv, bp = e, v, s
            rows.append((chrom, bp, strand, bv))
        bw.close()
        return rows

    rev, pat = strand_mate(treat_path)
    if rev:
        print("     stranded pair (" + pat + ")")
        rows = blocks(treat_path, "+") + blocks(rev, "-")
    else:
        print("     no strand mate found; treating as UNSTRANDED. If this file "
              "should have one, check the naming against STRAND_PATS.")
        rows = blocks(treat_path, ".")
    t = pd.DataFrame(rows, columns=["chrom", "pos", "strand", "value"])
    n_pre = len(t)
    if not n_pre:
        return t

    bg_rev, _ = strand_mate(bg_path)
    h = {"+": pyBigWig.open(bg_path)}
    h["-"] = pyBigWig.open(bg_rev) if bg_rev else h["+"]
    if bool(rev) != bool(bg_rev):
        print("     WARNING: treatment and background differ in strandedness; "
              "the ratio will mix strands")
    h["."] = h["+"]
    half = merge
    bgv = np.zeros(len(t))
    for i, (c, pp_, s) in enumerate(zip(t.chrom, t.pos, t.strand)):
        bw = h.get(s if s in "+-" else ".")
        ch = bw.chroms()
        if c not in ch:
            continue
        a, b = max(0, int(pp_) - half), min(int(ch[c]), int(pp_) + half)
        if b <= a:
            continue
        try:
            v = bw.values(c, a, b, numpy=True)
        except RuntimeError:
            continue
        v = np.abs(v[np.isfinite(v)])
        bgv[i] = float(v.max()) if v.size else 0.0
    for x in set(h.values()):
        x.close()

    t["bg"] = bgv
    t["enrich"] = (t.value + eps) / (bgv + eps)
    require_bg = bool(p.get("require_other_detected", False))
    sel = t.enrich >= ratio
    if require_bg:
        sel = sel & (t.bg > 0)
    out = t[sel].reset_index(drop=True)
    n_auto = int(((t.enrich >= ratio) & (t.bg <= 0)).sum())
    print("     " + f"{n_pre:,}" + " blocks over floor -> " +
          f"{len(out):,}" + " kept (" +
          f"{100*len(out)/max(n_pre,1):.1f}" + "%)")
    print("     of the " + f"{int((t.enrich >= ratio).sum()):,}" +
          " passing the ratio, " + f"{n_auto:,}" + " (" +
          f"{100*n_auto/max(int((t.enrich >= ratio).sum()),1):.1f}" +
          "%) have ZERO signal in the other condition and pass on the "
          "pseudocount alone" +
          ("  [excluded: require_other_detected]" if require_bg
           else "  [included; set require_other_detected: true to drop them]"))
    return out


def parse_table(path, p):
    df = pd.read_csv(path, sep="\t", low_memory=False,
                     compression="gzip" if path.endswith(".gz") else None)
    st = p.get("strand_col")
    strand = (df[st].astype(str) if st and st in df.columns
              else pd.Series(".", index=df.index))
    s_ = pd.to_numeric(df[p["start_col"]], errors="coerce")
    ec = p.get("end_col")
    anchor = p.get("anchor", "three_prime")
    if ec and ec in df.columns:
        e_ = pd.to_numeric(df[ec], errors="coerce")
        w = np.nanmedian(e_ - s_)
        plus = (strand != "-").to_numpy()
        if anchor == "three_prime":
            pos = np.where(plus, e_, s_)
        elif anchor == "five_prime":
            pos = np.where(plus, s_, e_)
        elif anchor == "midpoint":
            pos = ((s_ + e_) // 2).to_numpy()
        elif anchor == "end":
            pos = e_.to_numpy()
        else:
            pos = s_.to_numpy()
        print("     median width " + f"{w:.0f}" + " nt; anchor=" + anchor
              + (" (3' end: end on +, start on -)"
                 if anchor == "three_prime" else ""))
    else:
        pos = s_.to_numpy()
        print("     no end column; using start")
    out = pd.DataFrame({"chrom": df[p["chrom_col"]].astype(str),
                        "pos": pos, "strand": strand, "value": np.nan})
    return out.dropna(subset=["chrom", "pos"]).assign(
        pos=lambda d: d.pos.astype(int)).reset_index(drop=True)


# set algebra
def nearest(a, b, ig):
    out = np.full(len(a), np.inf)
    idx = defaultdict(list)
    for c, s, p in zip(b.chrom, b.strand, b.pos):
        idx[(c, "." if ig else s)].append(p)
    idx = {k: np.sort(np.asarray(v)) for k, v in idx.items()}
    for k, sub in a.groupby(["chrom"] if ig else ["chrom", "strand"],
                            sort=False):
        if not isinstance(k, tuple):
            k = (k,)
        arr = idx.get((k[0], ".") if ig else (k[0], k[1]))
        if arr is None or not arr.size:
            continue
        p = sub.pos.to_numpy()
        i = np.searchsorted(arr, p)
        l = np.where(i > 0, np.abs(p - arr[np.clip(i-1, 0, arr.size-1)]), np.inf)
        r = np.where(i < arr.size, np.abs(arr[np.clip(i, 0, arr.size-1)] - p), np.inf)
        out[sub.index.to_numpy()] = np.minimum(l, r)
    return out


def pct_hexamer(sites, gen, unstranded, up=60, cap=20000):
    if not len(sites):
        return np.nan, 0
    d = (sites.sample(cap, random_state=0) if len(sites) > cap
         else sites).reset_index(drop=True)
    hit = tot = 0
    for c, p, s in zip(d.chrom, d.pos, d.strand):
        if unstranded or s not in "+-":
            sq = [gen.fetch(c, int(p)-up, int(p)),
                  gen.fetch(c, int(p), int(p)+up).translate(RC)[::-1]]
        elif s == "-":
            sq = [gen.fetch(c, int(p), int(p)+up).translate(RC)[::-1]]
        else:
            sq = [gen.fetch(c, int(p)-up, int(p))]
        if not any(len(x) >= up for x in sq):
            continue
        tot += 1
        if any(h in x for x in sq for h in HEX):
            hit += 1
    return (100*hit/tot if tot else np.nan), tot


class Genome:
    def __init__(self, path):
        fai = path + ".fai"
        if not os.path.exists(fai):
            sys.exit("need " + fai)
        self.idx = {}
        for line in open(fai):
            f = line.split("\t")
            self.idx[f[0]] = (int(f[1]), int(f[2]), int(f[3]), int(f[4]))
        self.fh = open(path, "rb")

    def fetch(self, c, a, b):
        if c not in self.idx or b <= a:
            return ""
        ln, off, lb, lw = self.idx[c]
        a, b = max(0, a), min(b, ln)
        if b <= a:
            return ""
        f = lambda p: off + p//lb*lw + p % lb
        self.fh.seek(f(a))
        return self.fh.read(f(b)-f(a)).decode().replace("\n", "").upper()


# figure
def make_figure(J, C, hexdf, sizes, labels, tol, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    CM = 1 / 2.54
    plt.rcParams.update({
        "font.size": 7, "axes.linewidth": .6, "pdf.fonttype": 42,
        "font.family": "sans-serif", "savefig.dpi": 600})

    names = list(J.index)
    n = len(names)
    M = J.to_numpy(float).copy()
    np.fill_diagonal(M, np.nan)
    vmax = np.nanmax(M) if np.isfinite(M).any() else 1.0
    vmax = max(vmax, 1e-6)

    cell = 0.85
    fig_w = (cell * n + 5.6) * CM
    fig_h = (cell * n + 4.2) * CM
    fig = plt.figure(figsize=(fig_w, fig_h))
    left = 0.30
    hm_w = cell * n * CM / fig_w
    ax = fig.add_axes([left, 0.26, hm_w, 0.62])
    axh = fig.add_axes([left + hm_w + 0.055, 0.26, 0.20, 0.62])
    cax = fig.add_axes([left + hm_w + 0.005, 0.26, 0.014, 0.62])

    im = ax.imshow(M, cmap="YlGnBu", vmin=0, vmax=vmax, aspect="auto")
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "\u2014", ha="center", va="center",
                        fontsize=6, color="#AAAAAA")
                continue
            v = M[i, j]
            if not np.isfinite(v):
                continue
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=5.6,
                    color="white" if v > vmax * .6 else "#222222")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([labels[x].replace("\n", " ") + "\n" + f"{sizes[x]:,}"
                        for x in names], rotation=40, ha="right", fontsize=6)
    ax.set_yticklabels([labels[x].replace("\n", " ") for x in names],
                       fontsize=6)
    ax.set_ylim(n - 0.5, -0.5)
    ax.tick_params(length=2, width=.6)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=5.5, length=2)
    cb.outline.set_linewidth(.5)

    h = hexdf.set_index("name").loc[names]
    y = np.arange(n)
    axh.barh(y - .19, h.pct_obs, .36, color="#B2182B", lw=0, label="sites")
    axh.barh(y + .19, h.pct_rand, .36, color="#CFCFCF", lw=0, label="+5 kb")
    for yi, (o, r) in zip(y, zip(h.pct_obs, h.pct_rand)):
        if np.isfinite(o):
            axh.text(min(max(o, r) + 3, 86), yi, f"{o:.0f}%", va="center",
                     fontsize=5.6, color="#333333")
    axh.set_ylim(n - 0.5, -0.5)
    axh.set_yticks(y)
    axh.set_yticklabels([])
    axh.set_xlim(0, 100)
    axh.set_xticks([0, 50, 100])
    axh.tick_params(length=2, width=.6, labelsize=5.5)
    axh.set_xlabel("% AATAAA/ATTAAA", fontsize=6, labelpad=2)
    axh.legend(fontsize=5.2, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, -0.10), ncol=2, handlelength=.9,
               borderpad=.1, columnspacing=1.0)
    axh.spines[["top", "right", "left"]].set_visible(False)

    ax.set_title("Jaccard between high-confidence\nsite sets (within "
                 + str(tol) + " nt)", fontsize=7.5, loc="left", pad=4)
    axh.set_title("canonical\nPAS", fontsize=7, loc="center", pad=4)
    fig.text(0.01, 0.015,
             "One row per experiment. Perturbations are blocks enriched over "
             "that experiment's own control at matched\ncoordinates, so the "
             "control is a background, not a row. Jaccard is capped by the "
             "size ratio (sizes under the x\nlabels); containment |AnB|/|A| "
             "and the ceiling are in jaccard_matrix.tsv.",
             fontsize=5.5, color="#555555", ha="left", va="bottom")
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, "fig_jaccard_hexamer." + e),
                    bbox_inches="tight", pad_inches=.04)
    plt.close(fig)
    print("\nwrote fig_jaccard_hexamer.pdf/.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="hiconf.yaml")
    ap.add_argument("--peaks", default=None,
                    help="optional: adds model FP and canonical as rows")
    ap.add_argument("--genome", required=True)
    ap.add_argument("--chain", default=None)
    ap.add_argument("--liftover-bin", default=None)
    ap.add_argument("--restricted-shuffles", type=int, default=50,
                    help="within-gene shuffles for the panel d chance baseline")
    ap.add_argument("--gene-pad", type=int, default=5000,
                    help="pad gene spans by this much on each side")
    ap.add_argument("--outdir", default="hiconf_out")
    ap.add_argument("--genes", default=None,
                    help="file of gene names, one per line. Every site set is "
                         "restricted to sites falling inside those genes' "
                         "spans. Restricting only the peak-derived rows would "
                         "compare a filtered model_fp against unfiltered "
                         "external sets, and the Jaccard would fall for "
                         "reasons unrelated to the biology.")
    a = ap.parse_args()
    os.makedirs(a.outdir + "/sites", exist_ok=True)
    cfg = load_config(a.config)
    D = cfg["defaults"]
    tol = int(D["tolerance"])
    chain = find_first(a.chain, "external/chain/hg19ToHg38.over.chain.gz")
    lo = find_first(a.liftover_bin, "external/bin/liftOver",
                    shutil.which("liftOver"))

    sets, labels, unstr = {}, {}, {}
    for e in cfg["experiments"]:
        nm = e["name"]
        print("\n== " + nm + "  " + str(e.get("label", "")))
        p = dict(D)
        p.update(e.get("parse", {}))
        if p.get("format") == "bigwig":
            bg = e.get("background")
            if not bg:
                sys.exit(nm + ": bigwig experiment needs `background:`")
            df = call_enriched(e["path"], bg, p, nm)
        else:
            paths = [e["path"]] + list(e.get("merge_with", []))
            frames = []
            for pth in paths:
                d1 = parse_table(pth, p)
                print("     " + os.path.basename(pth) + ": "
                      + f"{len(d1):,}" + " sites")
                frames.append(d1)
            df = pd.concat(frames, ignore_index=True)
            if len(paths) > 1:
                print("     pooled " + str(len(paths)) + " files -> "
                      + f"{len(df):,}" + " before collapse")
        if not len(df):
            print("     empty, skipped")
            continue
        if e.get("build") == "hg19":
            df = liftover(df, chain, lo, nm)
        df = collapse(df, int(p.get("merge_within", 25)))
        df["chrom"] = df.chrom.where(df.chrom.str.startswith("chr"),
                                     "chr" + df.chrom)
        df.to_csv(a.outdir + "/sites/" + nm + ".tsv", sep="\t", index=False)
        sets[nm] = df
        labels[nm] = e.get("label", nm)
        unstr[nm] = not bool(e.get("stranded", True))
        print("     -> " + f"{len(df):,}" + " high-confidence sites")

    gene_spans = None
    if a.peaks:
        pk = (pd.read_parquet(a.peaks) if a.peaks.endswith(".parquet")
              else pd.read_csv(a.peaks, sep="\t"))
        if "in_analysis_set" in pk.columns:
            pk = pk[pk.in_analysis_set.astype(bool)]
        pos = "center" if "center" in pk.columns else "pos"
        for key, mask, lab in (("model_fp", ~pk.is_canonical.astype(bool),
                                "model false positives"),
                               ("canonical", pk.is_canonical.astype(bool),
                                "canonical CPA")):
            d = pk[mask]
            sets[key] = pd.DataFrame({"chrom": d.chrom.astype(str),
                                      "pos": d[pos].astype(int),
                                      "strand": d.strand.astype(str),
                                      "value": np.nan}).reset_index(drop=True)
            labels[key] = lab
            unstr[key] = False
            print("\n== " + key + ": " + f"{len(sets[key]):,}" + " sites")

        pad = int(a.gene_pad)
        gs = (pk.assign(_p=pk[pos].astype(int))
                .groupby(["gene", "chrom", "strand"], as_index=False)
                .agg(gstart=("_p", "min"), gend=("_p", "max")))
        gs["gstart"] = (gs.gstart - pad).clip(lower=0)
        gs["gend"] = gs.gend + pad
        gene_spans = gs
        w = gs.gend - gs.gstart
        print("\ngene spans from peak extents, padded " + f"{pad:,}" + " nt: "
              + f"{len(gs):,}" + " genes, median width "
              + f"{np.median(w):,.0f}" + " nt, total "
              + f"{w.sum()/1e6:,.0f}" + " Mb ("
              + f"{100*w.sum()/3.1e9:.1f}" + "% of the genome)")
        print("  This is a LOWER bound on gene bodies: peak extents do not "
              "reach the TSS or TES.")

    if a.genes:
        if gene_spans is None:
            sys.exit("--genes needs --peaks, which is where gene spans "
                     "come from")
        with open(a.genes) as fh:
            keep_g = {ln.strip() for ln in fh if ln.strip()}
        gs2 = gene_spans[gene_spans.gene.isin(keep_g)]
        print("\ngene filter: " + f"{gs2.gene.nunique():,}" + "/" +
              f"{gene_spans.gene.nunique():,}" + " genes  [" +
              os.path.basename(a.genes) + "]")
        iv2 = {}
        for c, g in gs2.groupby("chrom"):
            g = g.sort_values("gstart")
            iv2[c] = (g.gstart.to_numpy(), g.gend.to_numpy())

        def inside(d):
            keep = np.zeros(len(d), bool)
            for c, sub in d.groupby("chrom", sort=False):
                if c not in iv2:
                    continue
                st, en = iv2[c]
                p_ = sub.pos.to_numpy()
                j = np.searchsorted(st, p_, side="right") - 1
                ok = (j >= 0) & (p_ <= en[np.clip(j, 0, len(en) - 1)])
                keep[sub.index.to_numpy()] = ok
            return keep

        for k in list(sets):
            d = sets[k].reset_index(drop=True)
            m = inside(d)
            sets[k] = d[m].reset_index(drop=True)
            print("  " + labels[k].ljust(30) + f"{len(d):>8,}" + " -> " +
                  f"{int(m.sum()):>7,}" + "  (" + f"{100*m.mean():5.1f}" + "%)")
            if m.sum() == 0:
                print("     WARNING: empty after filtering")
        gene_spans = gs2

    names = list(sets)
    sizes = {k: len(sets[k]) for k in names}
    print("\nall-by-all at " + str(tol) + " nt")
    rows, dist_rows = [], []
    J = pd.DataFrame(np.nan, index=names, columns=names)
    C = pd.DataFrame(np.nan, index=names, columns=names)
    for x in names:
        for y in names:
            if x == y:
                J.loc[x, y] = 1.0
                C.loc[x, y] = 1.0
                continue
            ig = unstr[x] or unstr[y]
            dxy = nearest(sets[x].reset_index(drop=True), sets[y], ig)
            dyx = nearest(sets[y].reset_index(drop=True), sets[x], ig)
            fin = dxy[np.isfinite(dxy)]
            if fin.size:
                keep = (fin if fin.size <= 20000 else
                        np.random.default_rng(0).choice(fin, 20000,
                                                        replace=False))
                dist_rows.append(pd.DataFrame({
                    "set_a": x, "set_b": y, "dist": keep}))
            a_in_b = int((dxy <= tol).sum())
            b_in_a = int((dyx <= tol).sum())
            inter = (a_in_b + b_in_a) / 2.0
            union = len(sets[x]) + len(sets[y]) - inter
            J.loc[x, y] = inter / union if union else np.nan
            C.loc[x, y] = a_in_b / len(sets[x])
            rows.append(dict(set_a=x, set_b=y, n_a=len(sets[x]),
                             n_b=len(sets[y]), n_a_in_b=a_in_b,
                             n_b_in_a=b_in_a, jaccard=J.loc[x, y],
                             containment_a_in_b=C.loc[x, y],
                             max_possible_jaccard=min(len(sets[x]), len(sets[y]))
                             / max(len(sets[x]), len(sets[y]))))
    pd.DataFrame(rows).to_csv(a.outdir + "/jaccard_matrix.tsv", sep="\t",
                              index=False)
    if dist_rows:
        dd = pd.concat(dist_rows, ignore_index=True)
        dd.to_csv(a.outdir + "/distances.tsv", sep="\t", index=False)
        print("  distances.tsv: " + f"{len(dd):,}" + " nearest-neighbour "
              "distances (subsampled to 20k per ordered pair)")
    print(J.round(3).to_string())

    if gene_spans is not None:
        print("\ncontainment restricted to the " + f"{gene_spans.gene.nunique():,}"
              + " analysis genes")
        iv = {}
        for c, g in gene_spans.groupby("chrom"):
            g = g.sort_values("gstart")
            iv[c] = (g.gstart.to_numpy(), g.gend.to_numpy())

        def in_genes(d):
            keep = np.zeros(len(d), bool)
            for c, sub in d.groupby("chrom", sort=False):
                if c not in iv:
                    continue
                st, en = iv[c]
                p = sub.pos.to_numpy()
                j = np.searchsorted(st, p, side="right") - 1
                ok = (j >= 0) & (p <= en[np.clip(j, 0, len(en) - 1)])
                keep[sub.index.to_numpy()] = ok
            return keep

        rsets, rrows = {}, []
        for k in names:
            d = sets[k].reset_index(drop=True)
            m = in_genes(d)
            rsets[k] = d[m].reset_index(drop=True)
            print("  " + labels[k].replace("\n", " ").ljust(30)
                  + f"{len(sets[k]):>8,}" + " -> " + f"{int(m.sum()):>7,}"
                  + "  (" + f"{100*m.mean():5.1f}" + "% in analysis genes)")
        rng2 = np.random.default_rng(0)

        def host_span(d):
            gs_ = np.full(len(d), np.nan)
            ge_ = np.full(len(d), np.nan)
            for c, sub in d.groupby("chrom", sort=False):
                if c not in iv:
                    continue
                st, en = iv[c]
                p_ = sub.pos.to_numpy()
                j = np.searchsorted(st, p_, side="right") - 1
                ok = (j >= 0) & (p_ <= en[np.clip(j, 0, len(en) - 1)])
                idx = sub.index.to_numpy()
                gs_[idx[ok]] = st[j[ok]]
                ge_[idx[ok]] = en[j[ok]]
            return gs_, ge_

        host_cache = {}

        def shuffle_in_own_gene(k):
            d = rsets[k]
            if k not in host_cache:
                host_cache[k] = host_span(d)
            gs_, ge_ = host_cache[k]
            out = d.copy()
            ok = np.isfinite(gs_) & np.isfinite(ge_) & (ge_ > gs_)
            newp = out.pos.to_numpy().copy()
            if ok.any():
                newp[ok] = rng2.integers(gs_[ok].astype(np.int64),
                                         ge_[ok].astype(np.int64))
            out["pos"] = newp
            return out.reset_index(drop=True)

        NSH = int(a.restricted_shuffles)
        for x in names:
            for y in names:
                if x == y or not len(rsets[x]) or not len(rsets[y]):
                    continue
                ig = unstr[x] or unstr[y]
                dd = nearest(rsets[x], rsets[y], ig)
                k_ = int((dd <= tol).sum())
                nulls = np.empty(NSH)
                for i in range(NSH):
                    nulls[i] = (nearest(shuffle_in_own_gene(x), rsets[y],
                                        ig) <= tol).sum()
                exp = float(nulls.mean())
                obs_f = k_ / len(rsets[x])
                exp_f = exp / len(rsets[x])
                rrows.append(dict(set_a=x, set_b=y, n_a=len(rsets[x]),
                                  n_b=len(rsets[y]), n_a_in_b=k_,
                                  containment_a_in_b=obs_f,
                                  expected_containment=exp_f,
                                  fold_over_chance=obs_f / exp_f
                                  if exp_f > 0 else np.inf,
                                  p_shuffle=(1 + int((nulls >= k_).sum())) / (NSH + 1),
                                  median_dist=float(np.median(dd[np.isfinite(dd)]))
                                  if np.isfinite(dd).any() else np.nan))
        if rrows:
            RR = pd.DataFrame(rrows)
            RR.to_csv(a.outdir + "/restricted.tsv", sep="\t", index=False)
            print("\n  observed vs within-gene chance, top pairs")
            top = RR.nlargest(8, "fold_over_chance")
            for _, r in top.iterrows():
                print("    " + (r.set_a + " in " + r.set_b).ljust(30)
                      + f"{100*r.containment_a_in_b:6.2f}%" + " vs chance "
                      + f"{100*r.expected_containment:5.2f}%" + "  = "
                      + f"{r.fold_over_chance:5.2f}" + "x  p="
                      + f"{r.p_shuffle:.3g}")
            print("  -> restricted.tsv")

    print("\nhexamer")
    gen = Genome(a.genome)
    hx = []
    for k in names:
        o, n_ = pct_hexamer(sets[k], gen, unstr[k])
        d2 = sets[k].copy()
        d2["pos"] = d2.pos + 5000
        r_, _ = pct_hexamer(d2, gen, unstr[k])
        hx.append(dict(name=k, label=labels[k], n=len(sets[k]), n_scored=n_,
                       pct_obs=o, pct_rand=r_))
        print("  " + labels[k].ljust(28) + f"{o:5.1f}" + "%  vs displaced " +
              f"{r_:5.1f}" + "%")
    hexdf = pd.DataFrame(hx)
    hexdf.to_csv(a.outdir + "/hexamer.tsv", sep="\t", index=False)

    make_figure(J, C, hexdf, sizes, labels, tol, a.outdir)
    print("\nwrote to " + a.outdir + "/")


if __name__ == "__main__":
    main()