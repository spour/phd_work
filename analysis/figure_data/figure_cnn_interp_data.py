#!/usr/bin/env python3
# panel inputs for the cnn interpretability figure
import sys, re
import numpy as np

def log(*a): print(*a, file=sys.stderr, flush=True)

V = {f"{r['section']}:{r['key']}": r["value"] for r in
     np.genfromtxt("occlusion_verdict.tsv", names=True, delimiter="\t", dtype=None,
                   encoding="utf-8")}
RF = int(float(V["A:receptive_field"]))
ADQ = [w for w in (25, 50, 100, 200, 400) if float(V[f"A:scale{w}_adequate"]) == 1]
log(f"[A] receptive field {RF} nt; adequate scales (max z >= 3): {ADQ}; "
    f"excluded: {[w for w in (25,50,100,200,400) if w not in ADQ]}")

T = np.genfromtxt("occlusion_multiscale.tsv", names=True, delimiter="\t")
assert int(T["rf_nt"][0]) == RF
with open("figCI_occlusion.tsv", "w") as fh:
    fh.write("width\tcentre\tz\td_winrate\tadequate\tmax_z\tpeak\trf_nt\tn_genes\tn_peaks\n")
    for w in (25, 50, 100, 200, 400):
        m = T["width"] == w; o = np.argsort(T["centre"][m])
        c, z, d = T["centre"][m][o], T["z"][m][o], T["d_winrate"][m][o]
        for i in range(len(c)):
            fh.write(f"{w}\t{c[i]:.0f}\t{z[i]:.5f}\t{d[i]:.6f}\t{int(w in ADQ)}\t"
                     f"{float(V[f'A:scale{w}_maxz']):.4f}\t{float(V[f'A:scale{w}_peak']):.0f}\t"
                     f"{RF}\t{int(T['n_genes'][0])}\t{int(T['n_test'][0])}\n")
        log(f"  {w:>3} nt: max z {float(V[f'A:scale{w}_maxz']):5.2f}  peak "
            f"{float(V[f'A:scale{w}_peak']):+6.0f}  "
            f"{'ADEQUATE' if w in ADQ else 'excluded, below receptive-field resolution'}")
log(f"[A] the {len(ADQ)} adequate scales peak within {float(V['A:peak_spread_adequate']):.0f} nt "
    f"of each other, {float(V['A:peak_spread_adequate'])/RF:.2f} of one receptive field; "
    f"all five would read {float(V['A:peak_spread_all']):.0f} nt")
log("[out] figCI_occlusion.tsv")

R = np.genfromtxt("occlusion_by_rank.tsv", names=True, delimiter="\t", dtype=None,
                  encoding="utf-8")
LABS = ["rank 1", "rank 2-3", "rank 4-10", "rank >10"]
with open("figCI_rank.tsv", "w") as fh:
    fh.write("stratum\tcentre\tshift\tn_genes\ttotal_signal\tcentroid\tacc_delta\tn_genes_total\n")
    tot = {}
    for lab in LABS:
        m = R["stratum"] == lab; o = np.argsort(R["centre"][m])
        c = R["centre"][m][o]
        s = (R["mean_dlogit_true"][m] - R["mean_dlogit_comp"][m])[o]
        k = lab.replace(" ", "_")
        tot[lab] = float(V[f"B:{k}_total"])
        for i in range(len(c)):
            fh.write(f"{lab}\t{c[i]:.0f}\t{s[i]:.6f}\t{int(R['n_genes'][m][0])}\t"
                     f"{tot[lab]:.4f}\t{float(V[f'B:{k}_centroid']):.1f}\t"
                     f"{float(V[f'B:{k}_acc_delta']):.4f}\t{int(R['n_genes_total'][0])}\n")
        log(f"  {lab:>9} n={int(R['n_genes'][m][0]):5d}  total signal {tot[lab]:6.3f}  "
            f"centroid {float(V[f'B:{k}_centroid']):+7.0f} nt  "
            f"acceptor delta {float(V[f'B:{k}_acc_delta']):+.3f}")
log(f"[B] rank 1 carries {tot['rank 1']/tot['rank >10']:.0f}x the positional signal of rank >10 "
    f"({tot['rank 1']:.2f} against {tot['rank >10']:.2f}); the two profiles correlate "
    f"r = {float(V['B:shape_r_rank1_vs_rankgt10']):+.3f}")
log("[out] figCI_rank.tsv")

G = np.genfromtxt("gia_donor_test.tsv", names=True, delimiter="\t", dtype=None, encoding="utf-8")
SHOW = [("acceptor_pyAG", "3' splice site (pyrimidine tract + AG)", "motif", ""),
        ("donor_AGGTAAGT", "5' splice site (AGGTAAGT)", "motif", "donor_perm_ctrl_TAGAGTAG"),
        ("donor_perm_ctrl_TAGAGTAG", "5' splice site, scrambled", "control", ""),
        ("PAS_AATAAA", "Polyadenylation signal (AATAAA)", "motif", "PAS_perm_ctrl_AAAATA"),
        ("PAS_perm_ctrl_AAAATA", "Polyadenylation signal, scrambled", "control", "")]
BGL = {"dinuc_shuffled": "Dinucleotide-shuffled background",
       "real_competitor": "Real competitor background"}
with open("figCI_gia.tsv", "w") as fh:
    fh.write("background\tbg_label\tmotif\tlabel\tkind\tdelta_logit\tse\tposition\tn_bg\t"
             "has_scramble\n")
    for bg in BGL:
        log(f"[C] {BGL[bg]}")
        for mid, lab, kind, ctrl in SHOW:
            s = G[(G["background"] == bg) & (G["motif"] == mid)]
            i = int(np.argmax(s["delta_logit"]))
            fh.write(f"{bg}\t{BGL[bg]}\t{mid}\t{lab}\t{kind}\t{s['delta_logit'][i]:.5f}\t"
                     f"{s['se'][i]:.5f}\t{int(s['position'][i])}\t{int(s['n'][i])}\t"
                     f"{int(kind=='motif' and mid!='acceptor_pyAG')}\n")
            log(f"    {lab:>42}  delta {s['delta_logit'][i]:+.4f} +/- {s['se'][i]:.4f} "
                f"at {int(s['position'][i]):+5d}")
log("[C] the GIA carries permutation controls for the donor and the PAS but not for the "
    "acceptor, so the acceptor has no paired scramble bar")
log("[out] figCI_gia.tsv")

F = np.genfromtxt("filter_matches_reference.tsv", names=True, delimiter="\t", dtype=None,
                  encoding="utf-8")
W = 19
ic = F["info_content_bits"] / W
log(f"[D] 32 filters, {ic.min():.3f} to {ic.max():.3f} bits per column against a maximum of 2.0 "
    f"(median {np.median(ic):.3f}); no filter is a crisp motif")
matched = np.array([str(x) != "none" for x in F["best_match"]])
log(f"[D] {int(matched.sum())}/32 match a reference motif, {int((matched & (F['r']>=0.8)).sum())} "
    f"at r >= 0.8; usage spans {int(F['n_sites'].min())} to {int(F['n_sites'].max())} sites")
with open("figCI_filters.tsv", "w") as fh:
    fh.write("filter\tic_bits\tic_per_col\tn_sites\tbest_match\tr\tstrand\tconsensus\twidth\n")
    for i in range(len(F)):
        fh.write(f"{F['filter'][i]}\t{F['info_content_bits'][i]:.4f}\t{ic[i]:.5f}\t"
                 f"{int(F['n_sites'][i])}\t{F['best_match'][i]}\t{F['r'][i]:.4f}\t"
                 f"{F['strand'][i]}\t{F['consensus'][i]}\t{W}\n")
LOGO = ["filter27", "filter12"]
for f in LOGO:
    i = int(np.flatnonzero(F["filter"] == f)[0])
    log(f"[D] logo {f}: IC {ic[i]:.3f} bits/col, {int(F['n_sites'][i])} sites, "
        f"best match {F['best_match'][i]} r {F['r'][i]:.3f}, consensus {F['consensus'][i]}")
log("[out] figCI_filters.tsv")

txt = open("filter_pwms_reference.meme").read()
blocks = re.findall(r"MOTIF (\S+)\nletter-probability matrix:[^\n]*\n((?:\s*[\d.]+\s+[\d.]+\s+"
                    r"[\d.]+\s+[\d.]+\n)+)", txt)
PW = {n: np.array([[float(x) for x in ln.split()] for ln in b.strip().split("\n")])
      for n, b in blocks}
assert len(PW) == 32 and all(v.shape == (W, 4) for v in PW.values())
NTS = "ACGT"
for i in range(len(F)):
    f = str(F["filter"][i]); cons = str(F["consensus"][i])
    der = "".join(NTS[j] for j in PW[f].argmax(1))
    assert der == cons, f"{f}: MEME argmax {der} does not match table consensus {cons}"
log(f"[D] all {len(PW)} MEME matrices reproduce their table consensus exactly")
with open("figCI_logos.tsv", "w") as fh:
    fh.write("filter\tposition\tbase\tprob\theight\tic_col\n")
    for f in LOGO:
        M = PW[f]
        for p in range(W):
            q = np.clip(M[p], 1e-9, 1)
            icc = 2.0 + float((q * np.log2(q)).sum())
            for b, base in enumerate("ACGT"):
                fh.write(f"{f}\t{p+1}\t{base}\t{M[p,b]:.6f}\t{M[p,b]*icc:.6f}\t{icc:.6f}\n")
log("[out] figCI_logos.tsv")

with open("figCI_notes.tsv", "w") as fh:
    fh.write("key\tvalue\n")
    for k in ("A:receptive_field", "A:peak_spread_adequate", "A:peak_spread_all",
              "A:min_r_adequate", "B:shape_r_rank1_vs_rankgt10", "IG:null_ratio",
              "IG:excess_vs_occlusion_z_r"):
        fh.write(f"{k}\t{V[k]}\n")
    fh.write(f"A:adequate_scales\t{','.join(map(str,ADQ))}\n")
    fh.write(f"D:n_filters\t{len(F)}\nD:ic_min\t{ic.min():.4f}\nD:ic_max\t{ic.max():.4f}\n")
    fh.write(f"D:n_matched\t{int(matched.sum())}\n")
log(f"[IG] cross-check: magnitude ratio to the composition null "
    f"{float(V['IG:null_ratio']):.3f}, shape correlation with the occlusion z profile "
    f"r = {float(V['IG:excess_vs_occlusion_z_r']):+.3f}")
log("[out] figCI_notes.tsv")
