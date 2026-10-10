#!/usr/bin/env python3
# use of unannotated competitors relative to random intronic positions of the same genes
import os, numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); Z = np.load(os.path.join(ROOT, "data/usage_sites.npz"), allow_pickle=True)
files = list(Z["files"]); S = Z["signal"]; cls = Z["cls"]; ana = Z["analysis"]; gene = Z["gene"]
genes = sorted(set(gene)); gix = {g: k for k, g in enumerate(genes)}; kg = np.array([gix[g] for g in gene]); NG = len(genes); RNG = np.random.default_rng(0)
sel = lambda *pats: [i for i, f in enumerate(files) if any(p in f for p in pats)]
EXP = [("HeLa (Oh 2017)", "U1 AMO", sel("cAMO_15nmol"), sel("U1AMO_15nmol")), ("HeLa (Feng 2025)", "U1 AMO", sel("_Control-a", "_Control-b"), sel("_U1-a", "_U1-b")),
       ("HeLa (Feng 2024)", "U1 AMO", sel("HeLa-ctr-"), sel("HeLa-U1-")), ("HEK293 (Feng 2025)", "U1 AMO", sel("_293-ctr-"), sel("_293-U1-")),
       ("HEK293T (Soles 2026)", "U1 AMO", sel("_c25_"), sel("_u1_")), ("SW480 (Feng 2025)", "U1 AMO", sel("SW480-NC-"), sel("SW480-U1-")),
       ("HeLa (Kamieniarz-Gdula 2019)", "PCF11 knockdown", sel("siLUC"), sel("siPCF11"))]
comp = (cls == "competitor") & ana; ctl = cls == "intronic control"
out = open(os.path.join(ROOT, "data/usage_ratio.tsv"), "w"); out.write("experiment\ttreatment\tcondition\tcompetitor_fraction\tcontrol_fraction\tratio\tlo\thi\tP_change\n")
for lab, trt, fc, ft in EXP:
    assert fc and ft, lab
    ok = np.all(np.isfinite(S[fc + ft]), 0); H = {"control": (np.nan_to_num(S[fc]) > 0).any(0), trt: (np.nan_to_num(S[ft]) > 0).any(0)}
    G = {}
    for cond, h in H.items():
        for nm, m in (("c", comp & ok), ("d", ctl & ok)):
            G[(cond, nm, "s")] = np.bincount(kg[m], weights=h[m].astype(float), minlength=NG); G[(cond, nm, "n")] = np.bincount(kg[m], minlength=NG).astype(float)
    fr = lambda cond, nm, k: G[(cond, nm, "s")][k].sum() / G[(cond, nm, "n")][k].sum(); ratio = lambda cond, k: fr(cond, "c", k) / fr(cond, "d", k)
    k0 = np.arange(NG); B = [RNG.integers(0, NG, NG) for _ in range(2000)]
    ch = np.array([np.log2(ratio(trt, k) / ratio("control", k)) for k in B]); ch = ch[np.isfinite(ch)]; p = max(2 * min((ch <= 0).mean(), (ch >= 0).mean()), 1 / 2000)
    for cond in ("control", trt):
        b = np.array([ratio(cond, k) for k in B]); lo, hi = np.percentile(b[np.isfinite(b)], [2.5, 97.5])
        out.write(f"{lab}\t{trt}\t{'control' if cond == 'control' else 'treated'}\t{fr(cond,'c',k0):.5f}\t{fr(cond,'d',k0):.5f}\t{ratio(cond,k0):.3f}\t{lo:.3f}\t{hi:.3f}\t{p:.3g}\n")
        print(f"{lab:30s} {cond:16s} comp {fr(cond,'c',k0):.4f} ctl {fr(cond,'d',k0):.4f} ratio {ratio(cond,k0):.2f} [{lo:.2f},{hi:.2f}]  P change {p:.3g}")
out.close()
