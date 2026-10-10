#!/usr/bin/env python3
# do the u1 and maxentscan lookups show expected trends?
import argparse, sys, itertools as it
import numpy as np

U1_5P   = "ACUUACCUG"
CONS5   = "CAGGTAAGT"
CONS3   = "T" * 17 + "CAGGTG"


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    xr = np.argsort(np.argsort(x)); yr = np.argsort(np.argsort(y))
    return float(np.corrcoef(xr, yr)[0, 1])


def get_score3(scripts_dir):
    try:
        from maxentpy.maxent import load_matrix3, score3
        m3 = load_matrix3()
        return lambda s: float(score3(s.upper().replace("U", "T"), matrix=m3))
    except Exception:
        if scripts_dir:
            sys.path.insert(0, scripts_dir)
        from extract_splicing_features import init_maxentscan
        o = init_maxentscan(); s3 = o[1]
        def f(s):
            v = s3(s.upper().replace("U", "T"))
            return float(np.nanmax(np.asarray(v, float))) if not np.isscalar(v) else float(v)
        return f


def build_me5():
    from maxentpy.maxent import load_matrix5, score5
    m5 = load_matrix5()
    kmers = ["".join(p) for p in it.product("ACGT", repeat=9)]
    vals = np.fromiter((score5(k, matrix=m5) for k in kmers), float, len(kmers))
    return dict(zip(kmers, vals))


def build_u1():
    import RNA
    dna = ["".join(p) for p in it.product("ACGT", repeat=9)]
    rna = [k.replace("T", "U") for k in dna]
    vals = np.fromiter((RNA.duplexfold(U1_5P, r).energy for r in rna), float, len(rna))
    return dict(zip(dna, vals))


def mutants(seq, k, n, rng, alphabet="ACGT"):
    out = []
    L = len(seq)
    for _ in range(n):
        pos = rng.choice(L, k, replace=False)
        s = list(seq)
        for p in pos:
            s[p] = rng.choice([b for b in alphabet if b != seq[p]])
        out.append("".join(s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--u1_npy", default=None, help="U1hyb_lookup.npy (dict); build if omitted")
    ap.add_argument("--me5_npy", default=None, help="me5_lookup.npy (dict); build if omitted")
    ap.add_argument("--scripts_dir", default=None, help="for score3 via extract_splicing_features if maxentpy absent")
    ap.add_argument("--out", default="lookup_trends.png")
    ap.add_argument("--plot", action="store_true",
                    help="also write the diagnostic figure (needs a matplotlib built for your NumPy)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    print("loading / building lookups ...", file=sys.stderr)
    u1 = np.load(a.u1_npy, allow_pickle=True).item() if a.u1_npy else build_u1()
    me5 = np.load(a.me5_npy, allow_pickle=True).item() if a.me5_npy else build_me5()
    score3 = get_score3(a.scripts_dir)

    keys = np.array(sorted(set(u1) & set(me5)))
    U = np.array([u1[k] for k in keys], float)
    M = np.array([me5[k] for k in keys], float)
    is_gt = np.array([k[3:5] == "GT" for k in keys])

    print("\n" + "=" * 70)
    print("U1 (ViennaRNA duplexfold): lower = stronger")
    print("=" * 70)
    kmin = keys[np.argmin(U)]
    rank_cons = int((U < u1[CONS5]).sum())
    print(f"  global min energy 9-mer   = {kmin}  ({u1[kmin]:.2f} kcal/mol)")
    print(f"  consensus {CONS5}       = {u1[CONS5]:.2f} kcal/mol  (rank {rank_cons+1} of {len(keys):,}; 1 = strongest)")
    print(f"  energy range              = [{U.min():.2f}, {U.max():.2f}]   ({(U>0).mean()*100:.1f}% > 0)")

    print("\n" + "=" * 70)
    print("MaxEntScan 5'ss (bits): higher = stronger")
    print("=" * 70)
    kmax = keys[np.argmax(M)]
    rank_cons5 = int((M > me5[CONS5]).sum())
    print(f"  global max 9-mer          = {kmax}  ({me5[kmax]:.2f} bits)")
    print(f"  consensus {CONS5}       = {me5[CONS5]:.2f} bits  (rank {rank_cons5+1} of {len(keys):,})")
    print(f"  score range               = [{M.min():.2f}, {M.max():.2f}]")

    print("\n" + "=" * 70)
    print("Cross-signal coupling: U1 energy vs MaxEntScan 5'ss (expect NEGATIVE)")
    print("=" * 70)
    print(f"  Spearman over all 9-mers        = {spearman(U, M):+.3f}")
    print(f"  Spearman over GT-donor 9-mers   = {spearman(U[is_gt], M[is_gt]):+.3f}  (n={is_gt.sum():,})")

    print("\n" + "=" * 70)
    print("Mutation ladder from consensus (0..6 substitutions, 400 variants each)")
    print("=" * 70)
    ks = list(range(7)); mu_u1 = []; mu_m5 = []
    for k in ks:
        vs = [CONS5] if k == 0 else mutants(CONS5, k, 400, rng)
        mu_u1.append(np.mean([u1[v] for v in vs]))
        mu_m5.append(np.mean([me5[v] for v in vs]))
    for k, eu, em in zip(ks, mu_u1, mu_m5):
        print(f"  {k} mismatch:  U1 energy={eu:6.2f}   MaxEnt5={em:6.2f}")
    print("  -> U1 energy should rise (weaker) and MaxEnt5 should fall as mismatches increase")

    pos_drop = []
    for i in range(9):
        alts = [CONS5[:i] + b + CONS5[i+1:] for b in "ACGT" if b != CONS5[i]]
        pos_drop.append(me5[CONS5] - np.mean([me5[v] for v in alts]))
    print("\n  MaxEnt5 mean score drop per mutated position (1..9):",
          " ".join(f"{d:.1f}" for d in pos_drop), "  (GT at pos 4-5 should dominate)")

    print("\n" + "=" * 70)
    print("MaxEntScan 3'ss (23-mer): mutational series")
    print("=" * 70)
    kill_ag = CONS3[:17] + "C" + "C" + CONS3[19:]
    weak_tract = "A" * 17 + CONS3[17:]
    rand23 = "".join(rng.choice(list("ACGT"), 23))
    series = [("strong consensus", CONS3), ("AG killed", kill_ag),
              ("tract degraded", weak_tract), ("random", rand23)]
    s3vals = [(lab, score3(s)) for lab, s in series]
    for lab, v in s3vals:
        print(f"  {lab:18} = {v:6.2f} bits")

    # figure (opt-in
    if not a.plot:
        print("\n[sanity] numeric checks complete (no figure). Add --plot to also write one.")
        return
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"\n[plot skipped] matplotlib import failed ({type(e).__name__}: NumPy 1.x/2.x ABI "
              f"in the module stack). The numeric checks above are the sanity result.")
        return
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    hb = ax[0,0].hexbin(M, U, gridsize=60, bins="log", cmap="Greys", mincnt=1)
    ax[0,0].scatter([me5[CONS5]], [u1[CONS5]], color="#D55E00", s=70, zorder=5, label="consensus")
    ax[0,0].set_xlabel("MaxEntScan 5'ss (bits)"); ax[0,0].set_ylabel("U1 duplex energy (kcal/mol)")
    ax[0,0].set_title(f"(a) U1 vs MaxEnt5   Spearman={spearman(U,M):+.2f}  (all 9-mers)", fontsize=11)
    ax[0,0].legend(loc="upper right", fontsize=9); fig.colorbar(hb, ax=ax[0,0], label="log10 count")

    axl = ax[0,1]; axr = axl.twinx()
    axl.plot(ks, mu_u1, "-o", color="#1D9E75", label="U1 energy"); axl.set_ylabel("U1 energy (kcal/mol)", color="#1D9E75")
    axr.plot(ks, mu_m5, "-s", color="#0072B2", label="MaxEnt5"); axr.set_ylabel("MaxEnt5 (bits)", color="#0072B2")
    axl.set_xlabel("substitutions from consensus"); axl.set_title("(b) mutation ladder (both degrade together)", fontsize=11)
    axl.tick_params(axis="y", colors="#1D9E75"); axr.tick_params(axis="y", colors="#0072B2")

    cols = ["#D55E00" if i in (3,4) else "#0072B2" for i in range(9)]
    ax[1,0].bar(range(1,10), pos_drop, color=cols)
    ax[1,0].set_xlabel("5'ss position (exon 1-3 | intron 4-9)"); ax[1,0].set_ylabel("mean MaxEnt5 drop when mutated")
    ax[1,0].set_title("(c) MaxEnt5 per-position sensitivity (GT = orange)", fontsize=11)
    ax[1,0].set_xticks(range(1,10))

    labs = [l for l,_ in s3vals]; vals = [v for _,v in s3vals]
    ax[1,1].bar(range(len(labs)), vals, color=["#0072B2","#D55E00","#E69F00","#8A8F98"])
    ax[1,1].axhline(0, color="k", lw=.6); ax[1,1].set_xticks(range(len(labs)))
    ax[1,1].set_xticklabels(labs, rotation=20, ha="right", fontsize=9)
    ax[1,1].set_ylabel("MaxEntScan 3'ss (bits)"); ax[1,1].set_title("(d) 3'ss mutational series", fontsize=11)

    src = "your .npy lookups" if (a.u1_npy and a.me5_npy) else "built from ViennaRNA + maxentpy"
    fig.suptitle(f"Lookup sanity: expected trends for U1, 5'ss, 3'ss  ({src})", fontweight="bold", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(a.out, dpi=200)
    print(f"\n[out] {a.out}")


if __name__ == "__main__":
    main()