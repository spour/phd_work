#!/usr/bin/env python3
# inputs for the two-panel cnn interpretability figure and its two supplementary panels
import sys, re
import numpy as np

def log(*a): print(*a, file=sys.stderr, flush=True)
W, NF = 19, 32

PM = np.genfromtxt("cnn_posmap_occlusion.tsv", names=True, delimiter="\t", dtype=None,
                   encoding="utf-8")
WG0 = float(PM["intact_winrate"][0]); NTE = int(PM["n_test"][0]); NGE = int(PM["n_genes"][0])
assert abs(WG0 - 0.821101) < 1e-5, f"posmap intact win-rate {WG0}, not the reported model"
assert len(PM) == NF * 25
log(f"[A] cnn_posmap_occlusion.tsv: {NF} filters x 25 bands, {NTE} test peaks in {NGE} genes, "
    f"intact within-gene {WG0:.5f} (reported model)")

FM = np.genfromtxt("filter_matches_reference.tsv", names=True, delimiter="\t", dtype=None,
                   encoding="utf-8")
assert len(FM) == NF
txt = open("filter_pwms_reference.meme").read()
blocks = re.findall(r"MOTIF (\S+)\nletter-probability matrix:[^\n]*\n((?:\s*[\d.]+\s+[\d.]+\s+"
                    r"[\d.]+\s+[\d.]+\n)+)", txt)
PW = {n: np.array([[float(x) for x in ln.split()] for ln in b.strip().split("\n")])
      for n, b in blocks}
assert len(PW) == NF and all(v.shape == (W, 4) for v in PW.values())
NTS = "ACGT"
for i in range(NF):
    f = str(FM["filter"][i]); cons = str(FM["consensus"][i])
    der = "".join(NTS[j] for j in PW[f].argmax(1))
    assert der == cons, f"{f}: MEME argmax {der} != table consensus {cons}"
log(f"[A] all {NF} MEME matrices reproduce their table consensus exactly")

tot = {f: float(np.abs(PM["contrib"][PM["filter"] == f]).sum()) for f in np.unique(PM["filter"])}
ORD = sorted(tot, key=lambda f: -tot[f])
RANK = {f: i + 1 for i, f in enumerate(ORD)}
IC = {str(FM["filter"][i]): float(FM["info_content_bits"][i]) / W for i in range(NF)}
MT = {str(FM["filter"][i]): str(FM["best_match"][i]) for i in range(NF)}
RR = {str(FM["filter"][i]): float(FM["r"][i]) for i in range(NF)}
NS = {str(FM["filter"][i]): int(FM["n_sites"][i]) for i in range(NF)}
CO = {str(FM["filter"][i]): str(FM["consensus"][i]) for i in range(NF)}

with open("cnnmain_posmap.tsv", "w") as fh:
    fh.write("filter\trow_order\tbin\tbin_start\tbin_end\tcentre\tcontrib\tabs_total\t"
             "ic_per_col\tbest_match\tr\tn_sites\tconsensus\tn_test\tn_genes\tintact_winrate\n")
    for r in PM:
        f = str(r["filter"])
        fh.write(f"{f}\t{RANK[f]}\t{int(r['bin'])}\t{int(r['bin_start'])}\t{int(r['bin_end'])}\t"
                 f"{(r['bin_start']+r['bin_end'])/2:.0f}\t{r['contrib']:.6f}\t{tot[f]:.6f}\t"
                 f"{IC[f]:.4f}\t{MT[f]}\t{RR[f]:.3f}\t{NS[f]}\t{CO[f]}\t{NTE}\t{NGE}\t"
                 f"{WG0:.6f}\n")
with open("cnnmain_logos.tsv", "w") as fh:
    fh.write("filter\trow_order\tposition\tbase\tprob\tconsensus\n")
    for f in ORD:
        for p in range(W):
            for b, base in enumerate(NTS):
                fh.write(f"{f}\t{RANK[f]}\t{p+1}\t{base}\t{PW[f][p,b]:.6f}\t{CO[f]}\n")
log("[out] cnnmain_posmap.tsv cnnmain_logos.tsv")

PPT = max(((f, RR[f]) for f in ORD if MT[f] == "acceptor_3ss_pyAG"), key=lambda t: t[1])[0]
log(f"[A] the one clean motif: {PPT}  consensus {CO[PPT]}  matches acceptor_3ss_pyAG at "
    f"r = {RR[PPT]:.3f}, {IC[PPT]:.3f} bits/column, {NS[PPT]:,} activating sites, "
    f"row {RANK[PPT]} of {NF} by summed |contribution|")
nm = sum(1 for f in ORD if MT[f] != "none")
at = {f: float(PW[f][:, [0, 3]].sum() / W) for f in ORD}
log(f"[A] {nm}/{NF} filters match a reference motif; A+T share of the PWM mass "
    f"{np.mean([at[f] for f in ORD if MT[f]!='none']):.3f} for those against "
    f"{np.mean([at[f] for f in ORD if MT[f]=='none']):.3f} for the rest")
log(f"[A] top 5 by summed |contribution|: " + ", ".join(
    f"{f} ({MT[f] if MT[f]!='none' else 'no match'}, {tot[f]:.2f})" for f in ORD[:5]))

G = np.genfromtxt("figCI_gia.tsv", names=True, delimiter="\t", dtype=None, encoding="utf-8")
with open("cnnmain_gia.tsv", "w") as fh:
    fh.write("background\tbg_label\tmotif\tlabel\tkind\tdelta_logit\tse\tposition\tn_bg\n")
    for r in G:
        fh.write(f"{r['background']}\t{r['bg_label']}\t{r['motif']}\t{r['label']}\t{r['kind']}\t"
                 f"{r['delta_logit']:.5f}\t{r['se']:.5f}\t{int(r['position'])}\t{int(r['n_bg'])}\n")
log("\n[B] implantation, at each motif's most favourable position")
for bg in ("dinuc_shuffled", "real_competitor"):
    for r in G[G["background"] == bg]:
        log(f"    {bg:16s} {str(r['label']):>42}  {r['delta_logit']:+.4f} +/- {r['se']:.4f}")
log("[out] cnnmain_gia.tsv")

V = {f"{r['section']}:{r['key']}": r["value"] for r in
     np.genfromtxt("occlusion_verdict.tsv", names=True, delimiter="\t", dtype=None,
                   encoding="utf-8")}
with open("cnnmain_notes.tsv", "w") as fh:
    fh.write("key\tvalue\n")
    for k in ("A:receptive_field", "A:adequate_scales", "A:peak_spread_adequate",
              "B:shape_r_rank1_vs_rankgt10", "B:rank_1_total", "B:rank_>10_total",
              "IG:null_ratio", "IG:excess_vs_occlusion_z_r"):
        fh.write(f"{k}\t{V[k]}\n")
    fh.write(f"ppt_filter\t{PPT}\nppt_r\t{RR[PPT]:.3f}\nppt_ic\t{IC[PPT]:.3f}\n")
    fh.write(f"ppt_sites\t{NS[PPT]}\nppt_consensus\t{CO[PPT]}\n")
    fh.write(f"n_filters\t{NF}\nn_matched\t{nm}\nn_test\t{NTE}\nn_genes\t{NGE}\n")
    fh.write(f"at_matched\t{np.mean([at[f] for f in ORD if MT[f]!='none']):.4f}\n")
    fh.write(f"at_unmatched\t{np.mean([at[f] for f in ORD if MT[f]=='none']):.4f}\n")
log("[out] cnnmain_notes.tsv")
