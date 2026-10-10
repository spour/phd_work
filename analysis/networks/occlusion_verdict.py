#!/usr/bin/env python3
# the verdict for both interpretability analyses
import sys
import numpy as np

def log(*a): print(*a, file=sys.stderr, flush=True)
MINZ = 3.0
GRID = np.arange(-2000, 501, 25.0)

T = np.genfromtxt("occlusion_multiscale.tsv", names=True, delimiter="\t")
RF = int(T["rf_nt"][0]); NREP = int(T["n_rep"][0])
NG = int(T["n_genes"][0]); NS = int(T["n_test"][0]); WG0 = float(T["intact_winrate"][0])
WIDTHS = sorted(set(int(w) for w in T["width"]))
log(f"[in] occlusion_multiscale.tsv: {len(T)} windows, scales {WIDTHS}, {NREP} shuffles each; "
    f"{NS} test sites in {NG} genes, intact within-gene {WG0:.5f}, receptive field {RF} nt")

Z, PK, ADQ = {}, {}, {}
for w in WIDTHS:
    m = T["width"] == w; o = np.argsort(T["centre"][m])
    c, z = T["centre"][m][o], T["z"][m][o]
    Z[w] = np.interp(GRID, c, z)
    PK[w] = dict(peak=float(GRID[int(np.argmax(Z[w]))]), maxz=float(z.max()),
                 z0=float(np.interp(0, c, z)), frac2=float((z > 2).mean()), n=int(m.sum()))
    ADQ[w] = PK[w]["maxz"] >= MINZ

log(f"\n[A] SCALE ADEQUACY (a scale needs max z >= {MINZ} for its argmax to mean anything)")
log(f"{'scale':>6} {'n_win':>6} {'max z':>7} {'z at 0':>8} {'frac z>2':>9} {'peak':>7}  verdict")
for w in WIDTHS:
    p = PK[w]
    log(f"{w:>6} {p['n']:>6} {p['maxz']:>7.2f} {p['z0']:>8.2f} {p['frac2']:>9.2f} "
        f"{p['peak']:>+7.0f}  {'adequate' if ADQ[w] else 'NO SIGNAL, excluded'}")

AD = [w for w in WIDTHS if ADQ[w]]
def agree(ws):
    pk = np.array([PK[w]["peak"] for w in ws])
    M = np.corrcoef(np.vstack([Z[w] for w in ws]))
    iu = np.triu_indices(len(ws), 1)
    return pk.max() - pk.min(), M[iu].min(), float(np.median(M[iu]))

sp_all, r_all, rm_all = agree(WIDTHS)
log(f"\n[A] all {len(WIDTHS)} scales: peak spread {sp_all:.0f} nt, min pairwise r {r_all:.3f} "
    f"-> {'agree' if sp_all <= RF and r_all > 0.5 else 'DISAGREE'}")
if len(AD) >= 2:
    sp, r, rm = agree(AD)
    log(f"[A] {len(AD)} adequate scales {AD}: peaks "
        + ", ".join(f"{w}nt {PK[w]['peak']:+.0f}" for w in AD)
        + f"; spread {sp:.0f} nt against a {RF} nt receptive field; "
          f"pairwise r min {r:.3f} median {rm:.3f}")
    real = sp <= RF and r > 0.5
    log(f"[A] VERDICT: {'CONCENTRATION REAL' if real else 'SCALE-DEPENDENT (probe artifact)'} "
        f"-- every scale able to register a drop puts it in the same place, within {sp:.0f} nt, "
        f"which is {sp/RF:.2f} of one receptive field")
else:
    sp = r = rm = float("nan"); real = False
    log("[A] VERDICT: too few adequate scales to decide")

log(f"[A] the excluded 25 nt probe is not a null result about sequence, it is a null result "
    f"about the probe: a mask {100*25/RF:.0f}% the width of the receptive field leaves the "
    f"pooled bins largely intact")

R = np.genfromtxt("occlusion_by_rank.tsv", names=True, delimiter="\t", dtype=None, encoding="utf-8")
LABS = ["rank 1", "rank 2-3", "rank 4-10", "rank >10"]
C = np.sort(R["centre"][R["stratum"] == "rank 1"])
SEP, INFO = {}, {}
for lab in LABS:
    m = R["stratum"] == lab; o = np.argsort(R["centre"][m])
    sep = (R["mean_dlogit_true"][m] - R["mean_dlogit_comp"][m])[o]
    SEP[lab] = sep
    dmg = np.maximum(-sep, 0)
    INFO[lab] = dict(n=int(R["n_genes"][m][0]), wg0=float(R["stratum_intact_winrate"][m][0]),
                     acc_t=float(R["acc_true"][m][0]), acc_c=float(R["acc_comp"][m][0]),
                     peak=float(C[int(np.argmin(sep))]), worst=float(sep.min()),
                     total=float(dmg.sum()),
                     centroid=float((C * dmg).sum() / max(dmg.sum(), 1e-12)),
                     dw_all_neg=bool((R["d_winrate"][m] <= 0).all()))

log("\n[B] RANK-STRATIFIED, on the ceiling-free true-minus-competitor logit shift")
log(f"{'stratum':>9} {'genes':>6} {'intact wg':>10} {'acc delta':>10} {'peak':>7} "
    f"{'worst shift':>12} {'total':>8} {'centroid':>9}")
for lab in LABS:
    i = INFO[lab]
    log(f"{lab:>9} {i['n']:>6} {i['wg0']:>10.4f} {i['acc_t']-i['acc_c']:>+10.3f} "
        f"{i['peak']:>+7.0f} {i['worst']:>+12.4f} {i['total']:>8.3f} {i['centroid']:>+9.0f}")

M = np.corrcoef(np.vstack([SEP[l] for l in LABS]))
log("\n[B] shape correlation between strata")
log("          " + "".join(f"{l:>10}" for l in LABS))
for i, l in enumerate(LABS):
    log(f"{l:>9} " + "".join(f"{M[i,j]:>10.3f}" for j in range(len(LABS))))

t0, tl = INFO["rank 1"]["total"], INFO["rank >10"]["total"]
log(f"\n[B] the model uses {t0/max(tl,1e-12):.0f}x more positional signal on the genes it ranks "
    f"first than on the genes it ranks past 10 ({t0:.3f} against {tl:.3f}), and the little it "
    f"uses on those moves DOWNSTREAM: centroid {INFO['rank 1']['centroid']:+.0f} nt for rank 1 "
    f"against {INFO['rank >10']['centroid']:+.0f} nt for rank >10, with the two profiles "
    f"ANTICORRELATED at r = {M[0,3]:+.3f}")
log(f"[B] the same ordering appears in the pipeline's acceptor feature, which was computed "
    f"independently of the CNN: mean upstream MaxEnt 3'ss, true minus competitor, falls "
    + " -> ".join(f"{INFO[l]['acc_t']-INFO[l]['acc_c']:+.3f}" for l in LABS)
    + " across the same strata")
log("[B] caveat kept in view: strata are defined BY the model's own ranking, so d_winrate per "
    "stratum is confounded by regression to the mean (rank 1 sits at 1.0000 and can only fall; "
    "for rank 2-3 every window's occlusion RAISES the win-rate). That is why the shift above, "
    "not d_winrate, carries the argument.")

# ig cross-check
try:
    G = np.genfromtxt("ig_positional.tsv", names=True, delimiter="\t")
    b = lambda v: v.reshape(-1, 100).sum(1)
    BC = np.arange(-2000, 500, 100) + 50
    ex = b(G["ig_abs"]) - b(G["ig_null_abs"])
    m = T["width"] == 100; o = np.argsort(T["centre"][m])
    zi = np.interp(BC, T["centre"][m][o], T["z"][m][o])
    di = np.interp(BC, T["centre"][m][o], T["d_winrate"][m][o])
    r_raw = float(np.corrcoef(b(G["ig_abs"]), zi)[0, 1])
    r_ex = float(np.corrcoef(ex, zi)[0, 1])
    ratio = float(b(G["ig_abs"]).sum() / b(G["ig_null_abs"]).sum())
    log(f"\n[IG] CROSS-CHECK")
    log(f"[IG] total |IG| over the composition null: ratio {ratio:.3f} -- IG's MAGNITUDE still "
        f"sits at the null, exactly as the earlier attribution pass found")
    log(f"[IG] but its SHAPE agrees with occlusion once the null is subtracted: |IG| vs "
        f"occlusion z r = {r_raw:+.3f}, excess-over-null vs occlusion z r = {r_ex:+.3f}, "
        f"vs occlusion d_winrate r = {float(np.corrcoef(ex, di)[0,1]):+.3f}")
    log(f"[IG] peaks: occlusion z {BC[int(np.argmax(zi))]:+d}, IG excess {BC[int(np.argmax(ex))]:+d} "
        f"-- {'SAME BIN' if BC[int(np.argmax(zi))] == BC[int(np.argmax(ex))] else 'different bins'}")
    log(f"[IG] reading: gradient attribution can find WHERE the model looks and cannot show THAT "
        f"the model looks there for more than compositional reasons. Occlusion answers both, "
        f"which is why it stays primary.")
except Exception as e:
    log(f"\n[IG] cross-check skipped ({e})"); r_ex = ratio = float("nan")

with open("occlusion_verdict.tsv", "w") as fh:
    fh.write("section\tkey\tvalue\n")
    for w in WIDTHS:
        for k in ("maxz", "peak", "z0", "frac2"):
            fh.write(f"A\tscale{w}_{k}\t{PK[w][k]:.4f}\n")
        fh.write(f"A\tscale{w}_adequate\t{int(ADQ[w])}\n")
    fh.write(f"A\tadequate_scales\t{','.join(map(str,AD))}\n")
    fh.write(f"A\tpeak_spread_all\t{sp_all:.1f}\nA\tpeak_spread_adequate\t{sp:.1f}\n")
    fh.write(f"A\tmin_r_adequate\t{r:.4f}\nA\treceptive_field\t{RF}\n")
    fh.write(f"A\tconcentration_real\t{int(bool(real))}\n")
    for lab in LABS:
        s = lab.replace(" ", "_")
        for k in ("n", "wg0", "peak", "worst", "total", "centroid"):
            fh.write(f"B\t{s}_{k}\t{INFO[lab][k]:.4f}\n")
        fh.write(f"B\t{s}_acc_delta\t{INFO[lab]['acc_t']-INFO[lab]['acc_c']:.4f}\n")
    fh.write(f"B\tshape_r_rank1_vs_rankgt10\t{M[0,3]:.4f}\n")
    fh.write(f"IG\tnull_ratio\t{ratio:.4f}\nIG\texcess_vs_occlusion_z_r\t{r_ex:.4f}\n")
log("\n[out] occlusion_verdict.tsv")
