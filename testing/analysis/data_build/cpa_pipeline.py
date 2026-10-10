#!/usr/bin/env python3
# one pipeline for external cpa site datasets

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict

import numpy as np
import pandas as pd

HEXAMERS = ["AATAAA", "ATTAAA", "TATAAA", "AGTAAA", "AATACA", "CATAAA",
            "GATAAA", "AATATA", "AATAGA", "AAAAAG", "ACTAAA", "AAGAAA"]
STRONG = ("AATAAA", "ATTAAA")


# config
def load_config(path):
    try:
        import yaml
    except ImportError:
        sys.exit("pyyaml needed:  pip install pyyaml")
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    cfg.setdefault("defaults", {})
    d = cfg["defaults"]
    d.setdefault("window", 500)
    d.setdefault("tolerances", [50, 100, 250, 500])
    d.setdefault("shuffle_rounds", 200)
    d.setdefault("shuffle_offset", [10000, 100000])
    d.setdefault("quant_half", 50)

    cfg["datasets"] = [x for x in cfg.get("datasets", [])
                       if x.get("enabled", True)]

    for ds in cfg["datasets"]:
        nm = ds.get("name", "<unnamed>")
        sp = ds.get("species")
        if sp is None:
            sys.exit(nm + ": no `species` field. Declare it explicitly; "
                     "build alone does not tell you the organism.")
        if str(sp).lower() != "human":
            sys.exit(nm + ": species=" + str(sp) + ". Only human datasets can "
                     "be overlapped against hg38 peaks. Set enabled: false.")
        src = ds.get("source", {})
        if src.get("type") != "local":
            sys.exit(nm + ": source.type must be `local`; geo_supp and url are "
                     "documented but NOT implemented. Download the file first.")
        if "path" not in src:
            sys.exit(nm + ": source.path missing.")
        if "build" not in ds:
            sys.exit(nm + ": build missing.")
        if "parse" not in ds or "format" not in ds["parse"]:
            sys.exit(nm + ": parse.format missing.")
        if ds["parse"]["format"] not in PARSERS:
            sys.exit(nm + ": unknown parse.format " + str(ds["parse"]["format"]))
        ds.setdefault("direction", "none")
        if ds["direction"] not in ("up", "down", "none"):
            sys.exit(nm + ": direction must be up, down or none.")
        ctl = ds.get("control")
        if ctl and ctl not in [x.get("name") for x in cfg["datasets"]]:
            sys.exit(nm + ": control `" + str(ctl) + "` is not an enabled "
                     "dataset. Pairing would silently vanish.")
    return cfg


def expand_datasets(cfg, args):
    extra = []
    for ds in cfg["datasets"]:
        for gname in (ds.get("parse", {}).get("groups") or {}):
            name = ds["name"] + "__" + gname
            pth = os.path.join(args.outdir, "sites", name + ".tsv")
            if not os.path.exists(pth):
                continue
            d2 = dict(ds)
            d2["name"] = name
            d2["label"] = ds.get("label", ds["name"]) + " [" + gname + "]"
            d2["derived_from"] = ds["name"]
            d2.pop("control", None)
            d2.pop("merge_with", None)
            extra.append(d2)
    cfg["datasets"] = cfg["datasets"] + extra
    return cfg


def find_first(*cands):
    for c in cands:
        if c and os.path.exists(c):
            return c
    return None


# parsers
def parse_table(path, p):
    df = pd.read_csv(path, sep="\t", low_memory=False,
                     compression="gzip" if path.endswith(".gz") else None)
    cc, sc = p["chrom_col"], p["start_col"]
    st = p.get("strand_col")
    ec = p.get("end_col")
    pos = pd.to_numeric(df[sc], errors="coerce")
    width = np.nan
    if ec and ec in df.columns:
        end = pd.to_numeric(df[ec], errors="coerce")
        w = end - pos
        if np.nanmedian(w) > 1:
            width = float(np.nanmedian(w))
            pos = (pos + end) // 2
    out = pd.DataFrame({
        "chrom": df[cc].astype(str),
        "pos": pos,
        "strand": df[st].astype(str) if st and st in df.columns else ".",
        "value": np.nan,
    })

    groups = p.get("groups") or {}
    gmin = float(p.get("group_min", 0))
    all_cols = []
    for gname, pats in groups.items():
        cols = [c for c in df.columns
                if any(str(pat).lower() in str(c).lower() for pat in pats)]
        if not cols:
            print("     WARNING: group " + gname + " matched 0 columns; "
                  "check the header against parse.groups")
            out["grp_" + gname] = False
            continue
        print("     group " + gname + ": " + str(len(cols)) + " columns")
        vals = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        nmin = int(p.get("group_min_samples", 1))
        out["grp_" + gname] = ((vals > gmin).sum(axis=1) >= nmin).to_numpy()
        print("       flagged " + str(int(out["grp_" + gname].sum())) +
              " sites (TPM > " + str(gmin) + " in >= " + str(nmin) +
              " of " + str(len(cols)) + " samples)")
        all_cols = all_cols + cols
    if all_cols:
        v = df[all_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        out["value"] = v.max(axis=1).to_numpy()

    out = out.dropna(subset=["chrom", "pos"]).reset_index(drop=True)
    out["pos"] = out["pos"].astype(int)
    out.attrs["src_width"] = width
    return out


def parse_bed(path, p):
    df = pd.read_csv(path, sep="\t", header=None, comment="#", low_memory=False,
                     compression="gzip" if path.endswith(".gz") else None)
    df = df.rename(columns={0: "chrom", 1: "start", 2: "end"})
    strand = df[5].astype(str) if df.shape[1] > 5 else "."
    val = pd.to_numeric(df[4], errors="coerce") if df.shape[1] > 4 else np.nan
    out = pd.DataFrame({
        "chrom": df["chrom"].astype(str),
        "pos": ((df["start"] + df["end"]) // 2).astype(int),
        "strand": strand, "value": val,
    })
    out.attrs["src_width"] = float(np.nanmedian(df["end"] - df["start"]))
    return out


def parse_bigwig(path, p):
    try:
        import pyBigWig
    except ImportError:
        sys.exit("pyBigWig needed for bigwig datasets")

    min_reads = float(p.get("min_reads", 2))
    if p.get("value_is_rpm"):
        lib = float(p.get("library_size", 1e8))
        thr = min_reads * 1e6 / lib
    else:
        thr = min_reads
    merge = int(p.get("merge_within", 25))

    def call(fp, strand):
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
            b_end, bv, bp = keep[0][1], keep[0][2], keep[0][0]
            for s, e, v in keep[1:]:
                if s - b_end <= merge:
                    b_end = max(b_end, e)
                    if v > bv:
                        bv, bp = v, s
                else:
                    rows.append((chrom, bp, strand, bv))
                    b_end, bv, bp = e, v, s
            rows.append((chrom, bp, strand, bv))
        bw.close()
        return rows

    rev = path.replace(".fwd.bw", ".rev.bw")
    if path.endswith(".fwd.bw") and os.path.exists(rev):
        rows = call(path, "+") + call(rev, "-")
    else:
        if path.endswith(".fwd.bw"):
            print("     WARNING: " + os.path.basename(path) + " looks stranded "
                  "but no .rev.bw beside it; calling as unstranded")
        rows = call(path, ".")
    out = pd.DataFrame(rows, columns=["chrom", "pos", "strand", "value"])
    out.attrs["threshold_rpm"] = thr
    out.attrs["min_reads"] = min_reads
    out.attrs["src_width"] = 1.0
    return out


PARSERS = {"table": parse_table, "bed": parse_bed, "bigwig": parse_bigwig}


# liftover
def liftover(df, chain, exe, label):
    if exe is None or chain is None:
        sys.exit(label + ": liftOver or chain not found")
    tmp = tempfile.mkdtemp(prefix="lo_")
    src, dst, unm = (os.path.join(tmp, x) for x in ("i.bed", "o.bed", "u.bed"))
    df = df.reset_index(drop=True)
    with open(src, "w") as fh:
        for i, (c, p, s) in enumerate(zip(df["chrom"], df["pos"], df["strand"])):
            fh.write(f"{c}\t{int(p)-1}\t{int(p)}\t{i}\t0\t"
                     f"{s if s in '+-' else '+'}\n")
    subprocess.run([exe, "-minMatch=0.95", src, chain, dst, unm],
                   capture_output=True, text=True)
    if not os.path.exists(dst):
        shutil.rmtree(tmp, ignore_errors=True)
        sys.exit(label + ": liftOver produced nothing")
    o = pd.read_csv(dst, sep="\t", header=None,
                    names=["chrom", "s", "pos", "idx", "sc", "strand"])
    shutil.rmtree(tmp, ignore_errors=True)
    dup = o["idx"].duplicated(keep=False)
    n_multi = int(o.loc[dup, "idx"].nunique())
    o = o.loc[~dup]
    kept = df.loc[o["idx"].to_numpy()].copy()
    kept["chrom"] = o["chrom"].to_numpy()
    kept["pos"] = o["pos"].to_numpy()
    return kept.reset_index(drop=True), len(df) - len(kept), n_multi


def collapse(df, within):
    if len(df) == 0:
        return df.reset_index(drop=True)
    gcols = [c for c in df.columns if c.startswith("grp_")]
    if within <= 0:
        return df.reset_index(drop=True)
    if not np.isfinite(df["value"].to_numpy(float)).any():
        print("     WARNING: all values NaN; collapse ties are broken by "
              "position (leftmost kept), not by abundance")
    out = []
    for (c, s), g in df.groupby(["chrom", "strand"], sort=False):
        g = g.sort_values("pos", kind="mergesort")
        pos = g["pos"].to_numpy()
        val = g["value"].to_numpy(float)
        val = np.where(np.isfinite(val), val, 0.0)
        gmat = g[gcols].to_numpy(bool) if gcols else None
        brk = np.concatenate([[0], np.cumsum(np.diff(pos) > within)])
        for _, idx in pd.Series(np.arange(len(pos))).groupby(brk):
            i = idx.to_numpy()
            best = i[np.argmax(val[i])]
            row = [c, int(pos[best]), s, float(val[best]), len(i)]
            if gcols:
                row = row + [bool(x) for x in gmat[i].any(axis=0)]
            out.append(tuple(row))
    cols = ["chrom", "pos", "strand", "value", "n_merged"] + gcols
    return pd.DataFrame(out, columns=cols).reset_index(drop=True)


# prep
def stage_prep(cfg, args):
    os.makedirs(f"{args.outdir}/sites", exist_ok=True)
    chain = find_first(args.chain, "external/chain/hg19ToHg38.over.chain.gz",
                       "chain/hg19ToHg38.over.chain.gz")
    lo = find_first(args.liftover_bin, "external/bin/liftOver", "bin/liftOver",
                    shutil.which("liftOver"))
    win = int(cfg["defaults"]["window"])
    half = win // 2
    acct = []

    for ds in cfg["datasets"]:
        name = ds["name"]
        print("\n== " + name + "  (" + str(ds.get("label", "")) + ")")
        paths = [ds["source"]["path"]] + [m["path"] for m in ds.get("merge_with", [])]
        frames = []
        for pth in paths:
            if not os.path.exists(pth):
                sys.exit(name + ": missing " + pth)
            fn = PARSERS[ds["parse"]["format"]]
            d = fn(pth, ds["parse"])
            msg = "   " + os.path.basename(pth) + ": " + f"{len(d):,}" + " sites"
            if np.isfinite(d.attrs.get("src_width", np.nan)):
                msg = msg + ", median source width " + \
                      f"{d.attrs['src_width']:.0f}" + " nt"
            print(msg)
            if "threshold_rpm" in d.attrs:
                print("     threshold " + f"{d.attrs['threshold_rpm']:.4g}" +
                      " (>= " + f"{d.attrs['min_reads']:.0f}" + " reads)")
            frames.append(d)
        raw = sum(len(f) for f in frames)
        df = pd.concat(frames, ignore_index=True)
        for c in [x for x in df.columns if x.startswith("grp_")]:
            df[c] = df[c].fillna(False).astype(bool)

        lost_lift = multi = 0
        if ds["build"] == "hg19":
            df, lost_lift, multi = liftover(df, chain, lo, name)
            print("   lifted: " + f"{len(df):,}" + " kept, " + str(lost_lift) +
                  " lost, " + str(multi) + " multi")
        n_pre = len(df)

        mw = int(ds["parse"].get("merge_within", 0))
        df = collapse(df, mw)
        merged = n_pre - len(df)
        if merged:
            print("   merged within " + str(mw) + " nt: " +
                  f"{merged:,}" + " collapsed")

        df["chrom"] = df["chrom"].where(df["chrom"].str.startswith("chr"),
                                        "chr" + df["chrom"])
        df["start"] = (df["pos"] - half).clip(lower=0)
        df["end"] = df["pos"] + half
        df["dataset"] = name
        df["stranded"] = bool(ds.get("stranded", True))
        out = f"{args.outdir}/sites/{name}.tsv"
        df.to_csv(out, sep="\t", index=False)
        print("   -> " + f"{len(df):,}" + " sites, expanded to " + str(win) +
              " nt  [" + out + "]")

        subs = {}
        for gname in (ds["parse"].get("groups") or {}):
            gc = "grp_" + gname
            if gc not in df.columns:
                continue
            sub = df[df[gc].astype(bool)].copy()
            sub["dataset"] = name + "__" + gname
            sp = f"{args.outdir}/sites/{name}__{gname}.tsv"
            sub.to_csv(sp, sep="\t", index=False)
            subs[gname] = len(sub)
            print("   subset " + gname + ": " + f"{len(sub):,}" +
                  " sites  [" + sp + "]")
        if len(subs) > 1:
            tot = sum(subs.values())
            print("   subset total " + f"{tot:,}" + " vs union " +
                  f"{len(df):,}" + "  (excess = sites used in more than one "
                  "condition)")

        acct.append(dict(dataset=name, label=ds.get("label", name),
                         kind=ds.get("kind"), lab=ds.get("lab"),
                         species=ds.get("species"), build=ds["build"],
                         stranded=ds.get("stranded", True),
                         circular=bool(ds.get("circular", False)),
                         direction=ds.get("direction", "none"),
                         n_raw=raw, n_lost_liftover=lost_lift,
                         n_multimapped=multi, n_merged=merged,
                         n_final=len(df), window=win,
                         subsets=";".join(k + "=" + str(v)
                                          for k, v in subs.items())))

    a = pd.DataFrame(acct)
    a.to_csv(f"{args.outdir}/accounting_external.tsv", sep="\t", index=False)
    print("\n" + a[["dataset", "n_raw", "n_lost_liftover", "n_merged",
                    "n_final", "subsets"]].to_string(index=False))
    return a


def load_peaks(path, genes_file=None):
    df = (pd.read_parquet(path) if path.endswith(".parquet")
          else pd.read_csv(path, sep="\t", low_memory=False))
    if "in_analysis_set" in df.columns:
        n0 = len(df)
        df = df[df["in_analysis_set"].astype(bool)]
        print("peaks: " + f"{len(df):,}" + " in analysis set (of " +
              f"{n0:,}" + ")")
    else:
        print("WARNING: no in_analysis_set column; is this the stale parquet?")

    if genes_file:
        with open(genes_file) as fh:
            keep = {ln.strip() for ln in fh if ln.strip()}
        n0, g0 = len(df), df["gene"].nunique()
        df = df[df["gene"].isin(keep)]
        print("gene filter: " + f"{df['gene'].nunique():,}" + "/" +
              f"{g0:,}" + " genes, " + f"{len(df):,}" + "/" + f"{n0:,}" +
              " peaks kept  [" + os.path.basename(genes_file) + "]")
        if df["gene"].nunique() < len(keep) * 0.9:
            print("  WARNING: many listed genes are absent from the peak set; "
                  "check the gene naming matches")
    pos = "center" if "center" in df.columns else "pos"
    q = pd.DataFrame({"chrom": df["chrom"].astype(str),
                      "pos": df[pos].astype(int),
                      "strand": df["strand"].astype(str),
                      "gene": df.get("gene", "NA"),
                      "is_canonical": df["is_canonical"].astype(bool)})
    for c in ("discriminant", "spl_upstream_max_me3", "gc_contrast"):
        if c in df.columns:
            q[c] = df[c].to_numpy()
    return q.reset_index(drop=True)


def nearest(query, target, ignore_strand=False):
    out = np.full(len(query), np.inf)
    idx = defaultdict(list)
    for c, s, p in zip(target["chrom"], target["strand"], target["pos"]):
        idx[(c, "." if ignore_strand else s)].append(p)
    idx = {k: np.sort(np.asarray(v)) for k, v in idx.items()}
    keys = ["chrom"] if ignore_strand else ["chrom", "strand"]
    n_grp = n_hit = 0
    for k, sub in query.groupby(keys, sort=False):
        if not isinstance(k, tuple):
            k = (k,)
        kk = (k[0], ".") if ignore_strand else (k[0], k[1])
        arr = idx.get(kk)
        n_grp += 1
        if arr is None or arr.size == 0:
            continue
        n_hit += 1
        p = sub["pos"].to_numpy()
        i = np.searchsorted(arr, p)
        lft = np.where(i > 0, np.abs(p - arr[np.clip(i - 1, 0, arr.size - 1)]), np.inf)
        rgt = np.where(i < arr.size, np.abs(arr[np.clip(i, 0, arr.size - 1)] - p), np.inf)
        out[sub.index.to_numpy()] = np.minimum(lft, rgt)
    if n_grp and n_hit == 0:
        print("     WARNING: no query group matched any target key -- "
              "chrom/strand convention mismatch, not an absence of overlap")
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def stage_overlap(cfg, args, q):
    rng = np.random.default_rng(0)
    d = cfg["defaults"]
    tols = sorted(set(list(d["tolerances"]) + [int(d["window"])]))
    lo_off, hi_off = d["shuffle_offset"]
    nsh = int(d["shuffle_rounds"])
    comp = q[~q["is_canonical"]].reset_index(drop=True)
    canon = q[q["is_canonical"]].reset_index(drop=True)
    print("\nfalse positives " + f"{len(comp):,}" + ", canonical " +
          f"{len(canon):,}" + ", genes " + f"{q['gene'].nunique():,}")

    by_name = {x["name"]: x for x in cfg["datasets"]}
    rows, hitcols = [], {}
    for ds in cfg["datasets"]:
        name = ds["name"]
        ext = pd.read_csv(f"{args.outdir}/sites/{name}.tsv", sep="\t")
        ig = not bool(ds.get("stranded", True))
        dc = nearest(comp, ext, ig)
        dk = nearest(canon, ext, ig)
        hitcols[name] = dc
        for t in tols:
            obs = int((dc <= t).sum())
            null = np.empty(nsh)
            for i in range(nsh):
                e2 = ext.copy()
                e2["pos"] = e2["pos"] + int(rng.integers(lo_off, hi_off)) * \
                    int(rng.choice([-1, 1]))
                null[i] = (nearest(comp, e2, ig) <= t).sum()
            exp = float(null.mean())
            lo, hi = wilson(obs, len(comp))
            rows.append(dict(
                dataset=name, label=ds.get("label", name),
                kind=ds.get("kind"), circular=bool(ds.get("circular", False)),
                derived_from=ds.get("derived_from", ""),
                tolerance=t, n_external=len(ext),
                n_fp=len(comp), n_fp_hit=obs,
                pct_fp_hit=100 * obs / len(comp),
                ci_lo=100 * lo, ci_hi=100 * hi,
                n_canon=len(canon), n_canon_hit=int((dk <= t).sum()),
                pct_canon_hit=100 * (dk <= t).mean(),
                expected_displaced=exp,
                fold=obs / exp if exp > 0 else np.inf,
                p_displaced=(1 + int((null >= obs).sum())) / (nsh + 1)))
        print("  " + name + ": " + f"{int((dc <= tols[0]).sum()):,}" +
              " hits at " + str(tols[0]) + " nt, " +
              f"{int((dc <= tols[-1]).sum()):,}" + " at " + str(tols[-1]) + " nt")

    r = pd.DataFrame(rows)

    pair = []
    for ds in cfg["datasets"]:
        ctl = ds.get("control")
        if not ctl or ctl not in by_name:
            continue
        for t in tols:
            a = r[(r.dataset == ds["name"]) & (r.tolerance == t)].iloc[0]
            b = r[(r.dataset == ctl) & (r.tolerance == t)].iloc[0]
            direction = ds.get("direction", "none")
            if not np.isfinite(a.fold) or not np.isfinite(b.fold) or b.fold == 0:
                ratio, ok = np.nan, None
            elif direction == "up":
                ratio = a.fold / b.fold
                ok = bool(ratio > 1)
            elif direction == "down":
                ratio = a.fold / b.fold
                ok = bool(ratio < 1)
            else:
                ratio, ok = a.fold / b.fold, None
            pair.append(dict(perturbation=ds["name"], control=ctl, tolerance=t,
                             direction=direction, matches_direction=ok,
                             fold_pert=a.fold, fold_ctrl=b.fold,
                             fold_ratio=ratio,
                             pct_pert=a.pct_fp_hit, pct_ctrl=b.pct_fp_hit,
                             n_pert_hit=a.n_fp_hit, n_ctrl_hit=b.n_fp_hit))

    r.to_csv(f"{args.outdir}/overlap.tsv", sep="\t", index=False)
    if pair:
        p = pd.DataFrame(pair)
        p.to_csv(f"{args.outdir}/overlap_paired.tsv", sep="\t", index=False)
        print("\npaired perturbation vs its own control")
        print(p.to_string(index=False))
        if (p["direction"] == "down").any():
            print("\nNOTE: presence/absence overlap detects GAIN of sites only. "
                  "A `down` perturbation is under-powered here because a site "
                  "stays called until it falls below min_reads. Read quant.tsv.")

    hits = comp[["chrom", "pos", "strand", "gene"]].copy()
    for n_, dd in hitcols.items():
        hits["dist_" + n_] = dd
        for t in tols:
            hits["hit_" + n_ + "_" + str(t)] = dd <= t
    hits.to_csv(f"{args.outdir}/fp_hits.tsv", sep="\t", index=False)
    return r


class Genome:

    def __init__(self, path):
        self.path = path
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
        raw = self.fh.read(seek(end) - seek(start))
        return raw.decode().replace("\n", "").replace("\r", "").upper()


RC = str.maketrans("ACGTN", "TGCAN")


def motif_table(sites, gen, up=60, dn=40, label=""):
    recs = []
    for c, p, s in zip(sites["chrom"], sites["pos"], sites["strand"]):
        plus = s != "-"
        if plus:
            seq = gen.fetch(c, p - up, p + dn)
        else:
            seq = gen.fetch(c, p - dn, p + up).translate(RC)[::-1]
        if len(seq) < up + dn:
            continue
        usq, dsq = seq[:up], seq[up:]
        best, bpos = None, None
        for h in HEXAMERS:
            i = usq.rfind(h)
            if i >= 0 and (bpos is None or i > bpos):
                best, bpos = h, i
        arun = 0
        run = 0
        for ch in dsq[:20]:
            run = run + 1 if ch == "A" else 0
            arun = max(arun, run)
        d20 = dsq[:20]
        d1040 = dsq[10:40]
        recs.append(dict(
            hexamer=best,
            hex_strong=best in STRONG if best else False,
            hex_offset=(bpos - up) if bpos is not None else np.nan,
            down_A_frac=d20.count("A") / max(len(d20), 1),
            down_A_run=arun,
            down_U_frac=d1040.count("T") / max(len(d1040), 1),
            down_GU_frac=(d1040.count("T") + d1040.count("G")) / max(len(d1040), 1),
        ))
    df = pd.DataFrame(recs)
    if len(df) == 0:
        return df, {}
    hx = df["hex_offset"].dropna()
    mode = np.nan
    conc = np.nan
    if len(hx):
        h, edges = np.histogram(hx, bins=np.arange(-up, 1, 5))
        mode = float(edges[np.argmax(h)] + 2.5)
        conc = float(((hx >= mode - 5) & (hx <= mode + 5)).mean())
    summ = dict(
        group=label, n=len(df),
        pct_any_hexamer=100 * df["hexamer"].notna().mean(),
        pct_strong_hexamer=100 * df["hex_strong"].mean(),
        hex_offset_median=float(hx.median()) if len(hx) else np.nan,
        hex_offset_mode=mode, hex_concentration_pm5=conc,
        pct_in_minus40_minus10=100 * float(((hx >= -40) & (hx <= -10)).mean())
        if len(hx) else np.nan,
        down_A_frac_median=float(df["down_A_frac"].median()),
        pct_A_run_ge6=100 * float((df["down_A_run"] >= 6).mean()),
        down_U_frac_median=float(df["down_U_frac"].median()),
        pct_internal_priming=100 * float(
            ((df["down_A_run"] >= 6) & (df["hexamer"].isna())).mean()),
    )
    return df, summ


def stage_motif(cfg, args, q):
    if not args.genome:
        print("no --genome, skipping motif stage")
        return None
    gen = Genome(args.genome)
    comp = q[~q["is_canonical"]].reset_index(drop=True)
    canon = q[q["is_canonical"]].reset_index(drop=True)
    hits = pd.read_csv(f"{args.outdir}/fp_hits.tsv", sep="\t")
    tol = int(cfg["defaults"]["tolerances"][0])

    groups = {"canonical (positive control)": canon}
    for ds in cfg["datasets"]:
        col = "hit_" + ds["name"] + "_" + str(tol)
        if col in hits.columns:
            m = hits[col].astype(bool).to_numpy()
            if m.sum() >= 10:
                groups["FP hit by " + ds["name"]] = comp[m].reset_index(drop=True)
    anyhit = np.zeros(len(comp), bool)
    for ds in cfg["datasets"]:
        col = "hit_" + ds["name"] + "_" + str(tol)
        if col in hits.columns:
            anyhit |= hits[col].astype(bool).to_numpy()
    groups["FP not hit (background)"] = comp[~anyhit].reset_index(drop=True)

    shifted = comp[~anyhit].copy().reset_index(drop=True)
    shifted["pos"] = shifted["pos"] + 5000
    groups["FP not hit, +5 kb (null)"] = shifted

    for ds in cfg["datasets"]:
        e = pd.read_csv(f"{args.outdir}/sites/{ds['name']}.tsv", sep="\t")
        if len(e) > 20000:
            e = e.sample(20000, random_state=0).reset_index(drop=True)
        if not bool(ds.get("stranded", True)):
            e = e.copy()
            e["strand"] = "+"
        groups[ds["name"] + " sites"] = e

    summ, per = [], []
    for lab, g in groups.items():
        df, s = motif_table(g, gen, label=lab)
        if s:
            summ.append(s)
            df["group"] = lab
            per.append(df)
            print("  " + lab.ljust(34) + " n=" + f"{s['n']:>6,}" +
                  "  hexamer " + f"{s['pct_any_hexamer']:>5.1f}" + "%" +
                  "  mode " + str(s["hex_offset_mode"]) +
                  "  conc " + f"{s['hex_concentration_pm5']:.2f}" +
                  "  A-run>=6 " + f"{s['pct_A_run_ge6']:>5.1f}" + "%")
    S = pd.DataFrame(summ)
    S.to_csv(f"{args.outdir}/motif_summary.tsv", sep="\t", index=False)
    pd.concat(per, ignore_index=True).to_csv(
        f"{args.outdir}/motif_per_site.tsv", sep="\t", index=False)
    print("\nRead the canonical row first: if its hexamer mode is not near -21 "
          "with high concentration, the coordinate or strand convention is "
          "wrong and no other row means anything.")
    print("Known gap: unstranded datasets are scored as plus-strand, so their "
          "hexamer offsets are meaningful only for the true-plus subset.")
    return S


def cliffs_delta(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return np.nan
    bs = np.sort(b)
    gt = int(np.searchsorted(bs, a, side="left").sum())
    ge = int(np.searchsorted(bs, a, side="right").sum())
    lt = a.size * b.size - ge
    return float((gt - lt) / (a.size * b.size))


def bw_signal(path, sites, half):
    try:
        import pyBigWig
    except ImportError:
        sys.exit("pyBigWig needed for the quant stage")

    rev_path = path.replace(".fwd.bw", ".rev.bw")
    stranded = path.endswith(".fwd.bw") and os.path.exists(rev_path)
    handles = {"+": pyBigWig.open(path)}
    handles["-"] = pyBigWig.open(rev_path) if stranded else handles["+"]
    handles["."] = handles["+"]
    if stranded:
        print("     stranded pair: " + os.path.basename(path) + " / " +
              os.path.basename(rev_path))

    out = np.full(len(sites), np.nan)
    miss = set()
    strands = (sites["strand"].to_numpy() if "strand" in sites.columns
               else np.full(len(sites), "."))
    for i, (c, p, s) in enumerate(zip(sites["chrom"].to_numpy(),
                                      sites["pos"].to_numpy(), strands)):
        bw = handles.get(s if s in ("+", "-") else ".")
        chroms = bw.chroms()
        if c not in chroms:
            miss.add(c)
            continue
        st = max(0, int(p) - half)
        en = min(int(chroms[c]), int(p) + half)
        if en <= st:
            continue
        try:
            v = bw.values(c, st, en, numpy=True)
        except RuntimeError:
            continue
        v = np.abs(v[np.isfinite(v)])
        out[i] = float(v.max()) if v.size else 0.0

    for h in set(handles.values()):
        h.close()
    if miss:
        show = ", ".join(sorted(miss)[:5])
        if len(miss) > 5:
            show = show + ", ..."
        print("     WARNING: " + str(len(miss)) + " contig(s) absent from " +
              os.path.basename(path) + ": " + show +
              "  (post-liftOver scaffolds are expected; a missing autosome "
              "is not)")
    return out


def stage_quant(cfg, args, q):
    half = int(cfg["defaults"]["quant_half"])
    by_name = {x["name"]: x for x in cfg["datasets"]}
    chain = find_first(args.back_chain,
                       "external/chain/hg38ToHg19.over.chain.gz",
                       "chain/hg38ToHg19.over.chain.gz")
    lo = find_first(args.liftover_bin, "external/bin/liftOver", "bin/liftOver",
                    shutil.which("liftOver"))
    rows, per = [], []
    cache = {}

    for ds in cfg["datasets"]:
        ctl = ds.get("control")
        if not ctl or ctl not in by_name:
            continue
        if ds["parse"]["format"] != "bigwig":
            print("  quant: skipping " + ds["name"] + " (not bigwig)")
            continue
        print("\n== quant " + ds["name"] + " vs " + ctl)
        key = ds["build"]
        if key in cache:
            qq = cache[key]
        else:
            qq = q.copy()
            if key == "hg19":
                if chain is None or lo is None:
                    sys.exit("quant: hg19 dataset needs the hg38ToHg19 chain "
                             "and liftOver; pass --back-chain")
                qq, lost, multi = liftover(qq, chain, lo, ds["name"] + " back-lift")
                print("   back-lifted peaks: " + f"{len(qq):,}" + " kept, " +
                      str(lost) + " lost, " + str(multi) + " multi")
            cache[key] = qq

        pv = bw_signal(ds["source"]["path"], qq, half)
        cv = bw_signal(by_name[ctl]["source"]["path"], qq, half)

        eps = ds["parse"].get("quant_eps", cfg["defaults"].get("quant_eps"))
        if eps is None:
            pp = ds["parse"]
            if pp.get("value_is_rpm"):
                eps = 0.5 * 1e6 / float(pp.get("library_size", 1e8))
            else:
                eps = 0.5
        eps = float(eps)
        print("   pseudocount " + f"{eps:.4g}")

        lr = np.log2((pv + eps) / (cv + eps))
        fp = ~qq["is_canonical"].to_numpy(bool)

        det_p = np.nan_to_num(pv) > 0
        det_c = np.nan_to_num(cv) > 0
        inf = det_p | det_c
        fp_i = fp & inf
        cn_i = (~fp) & inf

        p_only = int((fp & det_p & ~det_c).sum())
        c_only = int((fp & det_c & ~det_p).sum())
        try:
            from scipy.stats import binomtest
            p_disc = (binomtest(p_only, p_only + c_only, 0.5).pvalue
                      if (p_only + c_only) else np.nan)
        except ImportError:
            p_disc = np.nan

        med_fp = float(np.nanmedian(lr[fp_i])) if fp_i.sum() else np.nan
        med_cn = float(np.nanmedian(lr[cn_i])) if cn_i.sum() else np.nan
        d = cliffs_delta(lr[fp_i], lr[cn_i])
        d_all = cliffs_delta(lr[fp], lr[~fp])

        direction = ds.get("direction", "none")
        if direction in ("up", "down") and (p_only + c_only) > 0:
            frac = p_only / (p_only + c_only)
            ok = bool(frac > 0.5) if direction == "up" else bool(frac < 0.5)
        else:
            ok = None

        rows.append(dict(perturbation=ds["name"], control=ctl,
                         direction=direction, matches_direction=ok,
                         half=half, eps=eps, build=ds["build"],
                         n_fp=int(fp.sum()), n_canon=int((~fp).sum()),
                         n_fp_informative=int(fp_i.sum()),
                         n_canon_informative=int(cn_i.sum()),
                         frac_fp_zero_both=float(1 - fp_i.sum() / max(fp.sum(), 1)),
                         n_fp_pert_only=p_only, n_fp_ctrl_only=c_only,
                         p_discordant=p_disc,
                         med_lr_fp_inf=med_fp, med_lr_canon_inf=med_cn,
                         delta_fp_vs_canon_inf=d,
                         delta_fp_vs_canon_all=d_all))
        t = qq[["chrom", "pos", "strand", "gene", "is_canonical"]].copy()
        t["dataset"] = ds["name"]
        t["sig_pert"] = pv
        t["sig_ctrl"] = cv
        t["log2_ratio"] = lr
        t["informative"] = inf
        per.append(t)
        print("   informative FP " + f"{int(fp_i.sum()):,}" + "/" +
              f"{int(fp.sum()):,}" + "  (" +
              f"{100 * fp_i.sum() / max(fp.sum(), 1):.1f}" + "%)")
        print("   discordant FP: " + str(p_only) + " pert-only vs " +
              str(c_only) + " ctrl-only, p=" + f"{p_disc:.3g}" +
              ("  [matches " + direction + "]" if ok
               else ("  [AGAINST " + direction + "]" if ok is False else "")))
        print("   median log2 (informative): FP " + f"{med_fp:.3f}" +
              ", canonical " + f"{med_cn:.3f}" + ", delta " + f"{d:.3f}"
              + "   <- normalisation-sensitive, do not lead with this")

    if rows:
        pd.DataFrame(rows).to_csv(f"{args.outdir}/quant.tsv", sep="\t",
                                  index=False)
        pd.concat(per, ignore_index=True).to_csv(
            f"{args.outdir}/quant_per_site.tsv", sep="\t", index=False)
        print("\nRead n_fp_informative first: sites with zero coverage in both "
              "conditions are ties, not evidence. Then the discordant counts, "
              "which use only sites where something changed and are immune to "
              "library composition. The medians assume RPM is comparable "
              "between conditions, which it is not when a perturbation shifts "
              "a large fraction of reads.")
    else:
        print("quant: no bigwig perturbation/control pairs, nothing to do")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "overlap", "motif", "quant",
                                      "report", "all"])
    ap.add_argument("--config", default="datasets.yaml")
    ap.add_argument("--peaks", required=False)
    ap.add_argument("--genome", default=None)
    ap.add_argument("--chain", default=None,
                    help="hg19ToHg38 chain, for prep")
    ap.add_argument("--back-chain", default=None,
                    help="hg38ToHg19 chain, for quant")
    ap.add_argument("--liftover-bin", default=None)
    ap.add_argument("--outdir", default="cpa_out")
    ap.add_argument("--genes", default=None,
                    help="file of gene names, one per line. Restricts the peak "
                         "set to those genes. Used for the four-way coverage "
                         "intersection: a gene absent from a dataset's coverage "
                         "cannot produce an overlap there, so including it "
                         "conflates 'not used' with 'not assayed'.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    os.makedirs(args.outdir, exist_ok=True)

    if args.stage in ("prep", "all"):
        stage_prep(cfg, args)
    if args.stage != "prep":
        cfg = expand_datasets(cfg, args)

    with open(f"{args.outdir}/config_used.json", "w") as fh:
        json.dump(cfg, fh, indent=2, default=str)

    q = None
    if args.stage in ("overlap", "motif", "quant", "report", "all"):
        if not args.peaks:
            sys.exit("--peaks required for this stage")
        q = load_peaks(args.peaks, args.genes)
    if args.stage in ("overlap", "all"):
        stage_overlap(cfg, args, q)
    if args.stage in ("motif", "all"):
        stage_motif(cfg, args, q)
    if args.stage in ("quant", "all"):
        stage_quant(cfg, args, q)
    if args.stage in ("report", "all"):
        f = f"{args.outdir}/overlap.tsv"
        if os.path.exists(f):
            r = pd.read_csv(f, sep="\t")
            w = int(cfg["defaults"]["window"])
            print("\n=== overlap at the configured window, " + str(w) + " nt ===")
            print(r[r.tolerance == w][
                ["dataset", "kind", "circular", "n_external", "n_fp_hit",
                 "pct_fp_hit", "ci_lo", "ci_hi", "fold", "p_displaced"]
            ].to_string(index=False))
            print("\ncircular=True: this dataset built the competitor filter, "
                  "so its overlap with the filtered set is tautological.")
            print("Rows with derived_from set are condition subsets of another "
                  "dataset, not independent observations.")
        f2 = f"{args.outdir}/quant.tsv"
        if os.path.exists(f2):
            print("\n=== quant: perturbation vs control at peak coordinates ===")
            print(pd.read_csv(f2, sep="\t").to_string(index=False))
    print("\nwrote to " + args.outdir + "/")


if __name__ == "__main__":
    main()