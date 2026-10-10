#!/usr/bin/env python3
# confirm the scale, units, and direction of the maxentscan and rnahybrid
import argparse, sys, re
import numpy as np

CONSENSUS_5SS = "CAGGTAAGT"
DECOY_5SS     = "CAGGTGACG"
CONSENSUS_3SS = ("T" * 17) + "CAGGTG"
DOC_3SS       = "CTCTACTACTATCTATCTAGATC"
DONOR_MOTIF   = re.compile(r"G[TU][AG]AG[TU]")


def qline(name, vals):
    x = np.asarray(list(vals), float); x = x[np.isfinite(x)]
    q = np.quantile(x, [0, .01, .05, .25, .5, .75, .95, .99, 1.0])
    print(f"  {name}: n={len(x):,}  min={q[0]:.2f} q01={q[1]:.2f} q05={q[2]:.2f} "
          f"q25={q[3]:.2f} med={q[4]:.2f} q75={q[5]:.2f} q95={q[6]:.2f} "
          f"q99={q[7]:.2f} max={q[8]:.2f}")
    return x, q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scripts_dir", default=None)
    ap.add_argument("--u1_lookup", required=True)
    ap.add_argument("--me5_lookup", required=True)
    ap.add_argument("--sample", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.scripts_dir:
        sys.path.insert(0, a.scripts_dir)
    from extract_splicing_features import init_maxentscan, load_u1_lookup, load_me5_lookup

    o = init_maxentscan(); score5, score3 = o[0], o[1]

    def s5(seq):
        seq = seq.upper().replace("U", "T")
        v = score5(seq)
        return float(np.nanmax(np.asarray(v, float))) if not np.isscalar(v) else float(v)

    def s3(seq):
        seq = seq.upper().replace("U", "T")
        v = score3(seq)
        return float(np.nanmax(np.asarray(v, float))) if not np.isscalar(v) else float(v)

    print("=" * 72)
    print("1. MaxEntScan anchors (Yeo & Burge 2004; >0 = site, higher = stronger)")
    print("=" * 72)
    print(f"  score5 consensus  {CONSENSUS_5SS:24} = {s5(CONSENSUS_5SS):7.3f}   expect near 5'ss max ~10.8-11.8")
    print(f"  score5 decoy      {DECOY_5SS:24} = {s5(DECOY_5SS):7.3f}   expect low / negative")
    print(f"  score3 documented {DOC_3SS:24} = {s3(DOC_3SS):7.3f}   expect 6.71 (Perl MaxEntScan)")
    print(f"  score3 polyY+AG   {CONSENSUS_3SS:24} = {s3(CONSENSUS_3SS):7.3f}   expect high")

    me5 = load_me5_lookup(a.me5_lookup)
    print("\n" + "=" * 72)
    print("2. me5 lookup (expected: MaxEntScan 5'ss bits per 9-mer, higher = stronger)")
    print("=" * 72)
    mklen = len(next(iter(me5)))
    _, _ = qline(f"me5 values (keylen={mklen})", me5.values())
    if mklen == 9:
        rng = np.random.default_rng(a.seed)
        keys = list(me5.keys())
        idx = rng.choice(len(keys), min(a.sample, len(keys)), replace=False)
        diff = []
        for i in idx:
            k = keys[i]
            try:
                diff.append(me5[k] - s5(k))
            except Exception:
                pass
        diff = np.asarray(diff, float)
        mx = np.nanmax(np.abs(diff)) if diff.size else np.nan
        print(f"  me5[k] - score5(k) over {diff.size:,} k-mers: max|diff|={mx:.4g}  mean|diff|={np.nanmean(np.abs(diff)):.4g}")
        print("  -> me5 IS MaxEntScan 5'ss bits" if mx < 1e-3
              else "  -> me5 differs from raw MaxEntScan (transform or different table); inspect")
    else:
        print("  (me5 keys are not 9-mers; skipping score5 agreement check)")

    u1 = load_u1_lookup(a.u1_lookup)
    print("\n" + "=" * 72)
    print("3. U1 (RNAhybrid) lookup: scale and units")
    print("=" * 72)
    u1len = len(next(iter(u1)))
    uv, uq = qline(f"U1 values (keylen={u1len})", u1.values())
    pct_pos = float((uv > 0).mean() * 100)
    print(f"  {pct_pos:.1f}% of U1 values are > 0")
    if pct_pos > 0.5:
        print("  -> positive values exist, so this is NOT raw RNAhybrid MFE (MFE <= 0).")
        print("     It is a transformed hybridization score. The -8.1 kcal/mol literature")
        print("     threshold does not apply to these numbers.")
    else:
        print("  -> all <= 0, consistent with raw RNAhybrid MFE (kcal/mol); -8.1 applies.")

    print("\n" + "=" * 72)
    print("4. U1 direction (which end is a real 5'ss?)")
    print("=" * 72)
    items = sorted(u1.items(), key=lambda kv: kv[1])
    lo_k = [k for k, _ in items[:15]]
    hi_k = [k for k, _ in items[-15:]]

    def motif_frac(ks):
        return float(np.mean([1 if DONOR_MOTIF.search(k.upper().replace("U", "T")) else 0 for k in ks]))

    fhi, flo = motif_frac(hi_k), motif_frac(lo_k)
    print(f"  5'ss-motif fraction  high-value end = {fhi:.2f}   low-value end = {flo:.2f}")
    print(f"  highest-U1 k-mers: {hi_k[-6:]}")
    print(f"  lowest-U1  k-mers: {lo_k[:6]}")

    r = np.nan
    if u1len == 9:
        rng = np.random.default_rng(a.seed)
        keys = list(u1.keys())
        idx = rng.choice(len(keys), min(a.sample, len(keys)), replace=False)
        uu, mm = [], []
        for i in idx:
            k = keys[i]
            try:
                m = s5(k)
                if np.isfinite(m):
                    uu.append(u1[k]); mm.append(m)
            except Exception:
                pass
        if len(uu) > 100:
            r = float(np.corrcoef(uu, mm)[0, 1])
            print(f"  corr(U1 value, MaxEntScan 5'ss) over {len(uu):,} 9-mers = {r:+.3f}")

    high_is_strong = (r > 0) if np.isfinite(r) else (fhi > flo)

    print("\n" + "-" * 72)
    if high_is_strong:
        thr = float(np.quantile(uv, 0.90))
        print("  VERDICT: U1 higher = stronger. Strong binders are the POSITIVE tail.")
        print("  make_fig3_raster.R:   U1_STRONG <- \"high\"")
        print(f"                        U1_THR    <- {thr:.1f}   (90th pct; raise toward {uq[8]:.1f} to declutter)")
    else:
        thr = float(np.quantile(uv, 0.10))
        print("  VERDICT: U1 lower = stronger (MFE-like). Strong binders are the NEGATIVE tail.")
        print("  make_fig3_raster.R:   U1_STRONG <- \"low\"")
        print(f"                        U1_THR    <- {thr:.1f}   (10th pct; lower toward {uq[0]:.1f} to declutter)")
    print("-" * 72)


if __name__ == "__main__":
    main()