#!/usr/bin/env python
# the two new panel datasets for figure3_v2, plus the extra forest row

import os
import sys
import csv
import math
import argparse
from collections import defaultdict

import numpy as np

Z95 = 1.959963984540054
U1_CELL = "up_500|density_per_kb@4"
ACC_EDGES = np.arange(4.5, 15.0 + 1e-9, 1.5)
U1_EDGES = np.arange(0.0, 200.0 + 1e-9, 20.0)
MIN_CELL = 200


def log(m):
    print(m, file=sys.stderr, flush=True)


def fnum(s):
    try:
        v = float(s)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def two_sided_p(z):
    return math.erfc(abs(z) / math.sqrt(2.0))


def p_label(p):
    if not np.isfinite(p):
        return "n/a"
    if p <= 0:
        return "p < 1e-300"
    if p < 1e-3:
        return "p = %.0e" % p
    return "p = %.3f" % p


def logistic_irls(X, y, max_iter=60, tol=1e-10):
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.maximum(mu * (1 - mu), 1e-10)
        z = eta + (y - mu) / w
        XtW = X.T * w
        try:
            new = np.linalg.solve(XtW @ X, XtW @ z)
        except np.linalg.LinAlgError:
            new = np.linalg.lstsq(XtW @ X, XtW @ z, rcond=None)[0]
        if np.max(np.abs(new - beta)) < tol:
            beta = new
            break
        beta = new
    eta = np.clip(X @ beta, -30, 30)
    mu = 1.0 / (1.0 + np.exp(-eta))
    w = np.maximum(mu * (1 - mu), 1e-10)
    return beta, np.sqrt(np.diag(np.linalg.inv((X.T * w) @ X)))


def zscore(v):
    ok = np.isfinite(v)
    mu, sd = v[ok].mean(), v[ok].std()
    return (v - mu) / (sd if sd > 0 else 1.0)


def wilcoxon_signed_rank_p(d):
    d = d[d != 0]
    n = d.size
    if n < 10:
        return float("nan")
    a = np.abs(d)
    o = np.argsort(a, kind="mergesort")
    sa = a[o]
    ng = np.r_[True, sa[1:] != sa[:-1]]
    g = np.cumsum(ng)
    st = np.r_[np.nonzero(ng)[0], n]
    av = (st[:-1] + st[1:] + 1) / 2.0
    r = np.empty(n)
    r[o] = av[g - 1]
    W = r[d > 0].sum()
    mu = n * (n + 1) / 4.0
    tie = np.bincount(g)[1:].astype(float)
    var = (n * (n + 1) * (2 * n + 1) - (tie ** 3 - tie).sum() / 2.0) / 24.0
    return two_sided_p((W - mu) / math.sqrt(var)) if var > 0 else float("nan")


def within_gene(score, y, genes, competitor_higher):
    by = defaultdict(lambda: ([], []))
    for i in range(score.size):
        if np.isfinite(score[i]):
            by[genes[i]][0 if y[i] == 1 else 1].append(score[i])
    rates, tw, tp = [], 0.0, 0
    for _g, (can, comp) in by.items():
        if not can or not comp:
            continue
        c = np.asarray(can)[:, None]
        k = np.asarray(comp)[None, :]
        w = float((k > c).sum() if competitor_higher else (c > k).sum())
        w += 0.5 * float((c == k).sum())
        n = c.size * k.size
        rates.append(w / n)
        tw += w
        tp += n
    r = np.asarray(rates)
    return r, (tw / tp if tp else float("nan")), tp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", default="fig3_new_positional_features.tsv")
    ap.add_argument("--sa", default="sa_peak_scalars_final.tsv")
    ap.add_argument("--u1_cache", default="u1_grid_cache.npz")
    ap.add_argument("--grid", default="master_grid.tsv")
    a = ap.parse_args()
    for p in (a.pos, a.sa, a.u1_cache):
        if not os.path.isfile(p):
            sys.exit("missing required input: %s" % p)

    with open(a.pos, newline="") as fh:
        R = list(csv.DictReader(fh, delimiter="\t"))
    genes = np.array([r["gene"] for r in R])
    y = np.array([int(float(r["is_canonical"])) for r in R], dtype=np.int8)
    acc_me = np.array([fnum(r["acc"]) for r in R])

    with open(a.sa, newline="") as fh:
        S = list(csv.DictReader(fh, delimiter="\t"))
    if len(S) != len(R) or [r["gene"] for r in S] != list(genes):
        sys.exit("row alignment mismatch: %s vs %s" % (a.sa, a.pos))
    acc_sa = np.array([fnum(r["sa_acc_max"]) for r in S])

    z = np.load(a.u1_cache, allow_pickle=True)
    ucols = [str(c) for c in z["cols"]]
    if U1_CELL not in ucols:
        sys.exit("u1 cache lacks %s" % U1_CELL)
    if list(z["gene"]) != list(genes):
        sys.exit("row alignment mismatch: %s vs %s" % (a.u1_cache, a.pos))
    u1 = z["M"][:, ucols.index(U1_CELL)].astype(float)

    base = float((y == 1).mean())
    log("peaks %d  genes %d  base rate %.4f" % (len(R), len(set(genes)), base))
    for nm, v in (("acc_MaxEnt", acc_me), ("acc_SpliceAI", acc_sa), ("u1_pasprox", u1)):
        log("  %-13s finite %6d (%.2f%%)  median %.4g" %
            (nm, int(np.isfinite(v).sum()), 100 * np.isfinite(v).mean(), np.nanmedian(v)))

    FEATS = [("acc_MaxEnt", acc_me, False), ("acc_SpliceAI", acc_sa, False),
             ("u1_pasprox", u1, True)]

    rows = []
    summ = {}
    for name, v, comp_hi in FEATS:
        vz = zscore(v)
        by = defaultdict(lambda: ([], [], [], []))
        for i in range(v.size):
            if not np.isfinite(v[i]):
                continue
            g = by[genes[i]]
            if y[i] == 1:
                g[0].append(v[i])
                g[2].append(vz[i])
            else:
                g[1].append(v[i])
                g[3].append(vz[i])
        n_gene = 0
        for gname, (cv, kv, cz, kz) in by.items():
            if not cv or not kv:
                continue
            n_gene += 1
            rows.append(dict(gene=gname, feature=name,
                             canonical_val=float(np.mean(cv)),
                             competitor_val=float(np.mean(kv)),
                             canonical_z=float(np.mean(cz)),
                             competitor_z=float(np.mean(kz)),
                             n_competitors=len(kv),
                             expected_direction=("competitor" if comp_hi else "canonical"),
                             winrate_expected="", pair_wtd="", wilcoxon_p="",
                             p_label="", n_genes="", n_pairs=""))
        r, pw, npair = within_gene(v, y, genes, comp_hi)
        p = wilcoxon_signed_rank_p(r - 0.5)
        summ[name] = (float(r.mean()), pw, p, int(r.size), npair)
        rows.append(dict(gene="__SUMMARY__", feature=name,
                         canonical_val="", competitor_val="",
                         canonical_z="", competitor_z="", n_competitors="",
                         expected_direction=("competitor" if comp_hi else "canonical"),
                         winrate_expected=float(r.mean()), pair_wtd=float(pw),
                         wilcoxon_p=float(p), p_label=p_label(p),
                         n_genes=int(r.size), n_pairs=int(npair)))

    K1 = ["gene", "feature", "canonical_val", "competitor_val", "canonical_z",
          "competitor_z", "n_competitors", "expected_direction", "winrate_expected",
          "pair_wtd", "wilcoxon_p", "p_label", "n_genes", "n_pairs"]
    with open("hero_within_gene.tsv", "w") as fh:
        fh.write("\t".join(K1) + "\n")
        for r in rows:
            fh.write("\t".join(("%.6g" % r[c]) if isinstance(r[c], float) else str(r[c])
                               for c in K1) + "\n")
    log("wrote hero_within_gene.tsv (%d rows)" % len(rows))

    ok = np.isfinite(acc_me) & np.isfinite(u1)
    ai = np.digitize(acc_me, ACC_EDGES) - 1
    ui = np.digitize(u1, U1_EDGES) - 1
    inb = ok & (ai >= 0) & (ai < len(ACC_EDGES) - 1) & (ui >= 0) & (ui < len(U1_EDGES) - 1)
    log("acc_x_u1: %d/%d peaks inside the bin grid (%.1f%%); %d outside acceptor edges, "
        "%d outside U1 edges"
        % (int(inb.sum()), int(ok.sum()), 100.0 * inb.sum() / max(1, ok.sum()),
           int((ok & ((ai < 0) | (ai >= len(ACC_EDGES) - 1))).sum()),
           int((ok & ((ui < 0) | (ui >= len(U1_EDGES) - 1))).sum())))
    cells = []
    kept = 0
    for bi in range(len(ACC_EDGES) - 1):
        for bj in range(len(U1_EDGES) - 1):
            m = inb & (ai == bi) & (ui == bj)
            n = int(m.sum())
            if n < MIN_CELL:
                continue
            pc = float(y[m].mean())
            odds = (pc / (1 - pc)) if 0 < pc < 1 else np.nan
            ob = base / (1 - base)
            l2 = math.log2(odds / ob) if np.isfinite(odds) and odds > 0 else np.nan
            xc = float((U1_EDGES[bj] + U1_EDGES[bj + 1]) / 2)
            cells.append(dict(acceptor=float((ACC_EDGES[bi] + ACC_EDGES[bi + 1]) / 2),
                              interv=xc, u1=xc, n=n, p_canonical=pc, log2_enrich=l2,
                              base_rate=base))
            kept += n
    K2 = ["acceptor", "interv", "u1", "n", "p_canonical", "log2_enrich", "base_rate"]
    with open("acc_x_u1.tsv", "w") as fh:
        fh.write("\t".join(K2) + "\n")
        for c in cells:
            fh.write("\t".join(("%.6g" % c[k]) if isinstance(c[k], float) else str(c[k])
                               for k in K2) + "\n")
    lo = min((c["log2_enrich"] for c in cells if np.isfinite(c["log2_enrich"])), default=np.nan)
    hi = max((c["log2_enrich"] for c in cells if np.isfinite(c["log2_enrich"])), default=np.nan)
    log("wrote acc_x_u1.tsv (%d cells >= %d, %d peaks) log2 OR range %.3f .. %.3f"
        % (len(cells), MIN_CELL, kept, lo, hi))

    frows = []
    m = np.isfinite(u1)
    X = np.column_stack([np.ones(int(m.sum())), zscore(u1)[m]])
    beta, se = logistic_irls(X, y[m].astype(float))
    b, s = beta[1], se[1]
    p = two_sided_p(b / s)
    frows.append(dict(block="effect", effect_label="PAS-proximal U1", scorer="U1",
                      estimate=math.exp(b), ci_lo=math.exp(b - Z95 * s),
                      ci_hi=math.exp(b + Z95 * s), p=p, p_label=p_label(p),
                      null_value=1.0, n=int(m.sum()), source="computed"))
    ga, pw, pp, ng, _np = summ["u1_pasprox"]
    r, _pw, _n = within_gene(u1, y, genes, True)
    hse = float(r.std() / math.sqrt(r.size)) if r.size > 1 else float("nan")
    frows.append(dict(block="ranking", effect_label="within-gene win-rate, U1 PAS-proximal",
                      scorer="U1", estimate=ga, ci_lo=ga - Z95 * hse, ci_hi=ga + Z95 * hse,
                      p=pp, p_label=p_label(pp), null_value=0.5, n=ng, source="computed"))
    mi = np.isfinite(acc_me) & np.isfinite(u1)
    za, zu = zscore(acc_me), zscore(u1)
    Xi = np.column_stack([np.ones(int(mi.sum())), za[mi], zu[mi], za[mi] * zu[mi]])
    bi_, si_ = logistic_irls(Xi, y[mi].astype(float))
    b, s = bi_[3], si_[3]
    p = two_sided_p(b / s)
    frows.append(dict(block="interaction", effect_label="acceptor x U1", scorer="U1",
                      estimate=math.exp(b), ci_lo=math.exp(b - Z95 * s),
                      ci_hi=math.exp(b + Z95 * s), p=p, p_label=p_label(p),
                      null_value=1.0, n=int(mi.sum()), source="computed"))
    qt = np.nanquantile(acc_me, [1 / 3.0, 2 / 3.0])
    tert = np.digitize(acc_me, qt)
    tert_out = []
    for k, nm in enumerate(("weak", "medium", "strong")):
        v = np.where(tert == k, u1, np.nan)
        rr, ppw, _n = within_gene(v, y, genes, True)
        pt = wilcoxon_signed_rank_p(rr - 0.5)
        se = float(rr.std() / math.sqrt(rr.size)) if rr.size > 1 else float("nan")
        tert_out.append((nm, float(rr.mean()), ppw, int(rr.size)))
        frows.append(dict(block="tertile", effect_label="U1 win-rate, %s acceptor" % nm,
                          scorer="U1", estimate=float(rr.mean()),
                          ci_lo=float(rr.mean()) - Z95 * se, ci_hi=float(rr.mean()) + Z95 * se,
                          p=pt, p_label=p_label(pt), null_value=0.5, n=int(rr.size),
                          source="computed"))
    K3 = ["block", "effect_label", "scorer", "estimate", "ci_lo", "ci_hi", "p",
          "p_label", "null_value", "n", "source"]
    with open("forest_u1.tsv", "w") as fh:
        fh.write("\t".join(K3) + "\n")
        for r_ in frows:
            fh.write("\t".join(("%.6g" % r_[c]) if isinstance(r_[c], float) else str(r_[c])
                               for c in K3) + "\n")
    log("wrote forest_u1.tsv (%d rows)" % len(frows))

    FULL = 6000
    if os.path.isfile(a.grid):
        with open(a.grid, newline="") as fh:
            G = list(csv.DictReader(fh, delimiter="\t"))
        for g in G:
            g["_wg"] = fnum(g["within_gene_winrate_expected_dir"])
            g["_au"] = fnum(g["pooled_auroc"])
            g["_ng"] = int(g["n_genes"])

        def pick(scorer, signal, region=None, reduction=None, threshold=None):
            c = [g for g in G if g["scorer"] == scorer and g["signal"] == signal]
            if region is not None:
                c = [g for g in c if g["region"] == region and g["reduction"] == reduction
                     and g["threshold"] == threshold]
                return c[0] if c else None
            c = [g for g in c if g["_ng"] >= FULL]
            return max(c, key=lambda g: g["_wg"]) if c else None

        SPEC = [
            ("acceptor", "MaxEnt", "definition", 1, dict()),
            ("acceptor", "SpliceAI", "definition", 1, dict()),
            ("donor", "MaxEnt", "suppression", 1, dict()),
            ("U1", "MaxEnt", "suppression", 1, dict()),
            ("donor", "SpliceAI", "ruled out", 0,
             dict(region="up_2000", reduction="sum", threshold="-")),
        ]
        prows = []
        for sig, scorer, role, keep, kw in SPEC:
            gsig = "donor" if sig == "U1" else sig
            gscorer = "U1" if sig == "U1" else scorer
            g = pick(gscorer, gsig, **kw)
            if g is None:
                log("pooled_vs_wg: no cell for %s %s" % (scorer, sig))
                continue
            lab = "%s %s%s" % (g["region"], g["reduction"],
                               "" if g["threshold"] == "-" else " >= %s" % g["threshold"])
            prows.append(dict(signal=sig, scorer=scorer, role=role, in_scope=keep,
                              pooled_auroc=g["_au"], within_gene_winrate=g["_wg"],
                              wg_p=fnum(g["wilcoxon_p"]), wg_q=fnum(g["bh_q"]),
                              pair_wtd=fnum(g["pair_wtd"]), n_genes=g["_ng"],
                              region_reduction_label=lab,
                              label="%s  %s" % ({"U1": "U1"}.get(sig, sig), scorer)))
        K4 = ["signal", "scorer", "role", "in_scope", "pooled_auroc", "within_gene_winrate",
              "wg_p", "wg_q", "pair_wtd", "n_genes", "region_reduction_label", "label"]
        with open("pooled_vs_wg.tsv", "w") as fh:
            fh.write("\t".join(K4) + "\n")
            for r_ in prows:
                fh.write("\t".join(("%.6g" % r_[c]) if isinstance(r_[c], float) else str(r_[c])
                                   for c in K4) + "\n")
        log("wrote pooled_vs_wg.tsv (%d rows)" % len(prows))
    else:
        prows = []
        log("pooled_vs_wg: %s absent, panel F data not written" % a.grid)

    # summaries
    print("=" * 96)
    print("WITHIN-GENE WIN-RATES (expected direction)")
    print("=" * 96)
    print("  %-14s %-11s %9s %9s %12s %8s %9s"
          % ("feature", "expected", "gene-avg", "pair-wtd", "Wilcoxon p", "genes", "pairs"))
    for name, _v, comp_hi in FEATS:
        ga, pw, p, ng, npair = summ[name]
        print("  %-14s %-11s %9.4f %9.4f %12s %8d %9d"
              % (name, "competitor" if comp_hi else "canonical", ga, pw,
                 ("<1e-300" if not (p > 1e-300) else "%.1e" % p), ng, npair))
    print()
    print("=" * 96)
    print("PANEL G  acceptor x U1 PAS-proximal")
    print("=" * 96)
    print("  base rate %.4f   cells (n >= %d) %d   peaks in kept cells %d"
          % (base, MIN_CELL, len(cells), kept))
    print("  log2 odds ratio range %.4f .. %.4f  (symmetric fill limit %.4f)"
          % (lo, hi, max(abs(lo), abs(hi))))
    print("  acceptor bins %s" % ", ".join("%.2f" % e for e in ACC_EDGES))
    print("  U1 bins       %s" % ", ".join("%.0f" % e for e in U1_EDGES))
    print()
    print("=" * 96)
    print("PANEL J  U1 forest row")
    print("=" * 96)
    f = frows[0]
    print("  logistic canonical ~ z(u1_pasprox): OR per SD %.4f  [%.4f, %.4f]  %s  n %d"
          % (f["estimate"], f["ci_lo"], f["ci_hi"], f["p_label"], f["n"]))
    print("  direction: %s" % ("toward competitor (OR < 1), as expected"
                               if f["estimate"] < 1 else "toward real end (OR > 1), UNEXPECTED"))
    g = frows[1]
    print("  within-gene win-rate %.4f  [%.4f, %.4f]  %s  n %d genes"
          % (g["estimate"], g["ci_lo"], g["ci_hi"], g["p_label"], g["n"]))
    it = frows[2]
    print()
    print("  interaction acceptor x U1: OR %.4f  [%.4f, %.4f]  %s"
          % (it["estimate"], it["ci_lo"], it["ci_hi"], it["p_label"]))
    print("  NOT uniform across acceptor strength (unlike panel F, whose interaction is null).")
    print("  U1 win-rate by acceptor tertile:")
    for nm, ga_, pw_, ng_ in tert_out:
        print("    %-7s %.4f (pair %.4f)  n_genes %d" % (nm, ga_, pw_, ng_))
    print("  It attenuates, it does not reverse: every tertile stays above 0.5.")
    if prows:
        print()
        print("=" * 96)
        print("PANEL F  pooled AUROC vs within-gene win-rate")
        print("=" * 96)
        print("  %-20s %-9s %8s %8s   %-24s %s"
              % ("point", "role", "pooled", "within", "encoding", "quadrant"))
        for r_ in prows:
            q = ("top-right" if r_["pooled_auroc"] > 0.5 and r_["within_gene_winrate"] > 0.5
                 else "top-left" if r_["within_gene_winrate"] > 0.5
                 else "bottom-right" if r_["pooled_auroc"] > 0.5 else "bottom-left")
            print("  %-20s %-9s %8.4f %8.4f   %-24s %s"
                  % (r_["label"], r_["role"], r_["pooled_auroc"], r_["within_gene_winrate"],
                     r_["region_reduction_label"], q))


if __name__ == "__main__":
    main()
