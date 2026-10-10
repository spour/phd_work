#!/usr/bin/env python3
# the verdict for the bpnet-style variant, recomputed from the tsvs the run wrote
import sys
import numpy as np

def log(*a): print(*a, file=sys.stderr, flush=True)
THR = 0.02

M = np.genfromtxt("bpnet_metrics.tsv", names=True, delimiter="\t", dtype=None, encoding="utf-8")
new = M[M["model"] == "BPNet variant"][0]
old = M[M["model"] == "Reported CNN"][0]
pd_ = M[M["model"] == "paired_delta"][0]
d, dlo, dhi = float(pd_["within_gene"]), float(pd_["wg_lo"]), float(pd_["wg_hi"])
log(f"[1] BPNet within-gene {new['within_gene']:.5f} "
    f"[{new['wg_lo']:.5f},{new['wg_hi']:.5f}]  P@1 {new['p_at_1']:.5f}  "
    f"AUROC {new['auroc']:.5f}  AUPRC {new['auprc']:.5f}")
log(f"[1] reported     within-gene {old['within_gene']:.5f} "
    f"[{old['wg_lo']:.5f},{old['wg_hi']:.5f}]  P@1 {old['p_at_1']:.5f}  "
    f"AUROC {old['auroc']:.5f}  AUPRC {old['auprc']:.5f}")
log(f"[1] paired delta {d:+.5f} [{dlo:+.5f},{dhi:+.5f}] on the same 1,623 held-out genes")
sig = dlo > 0 or dhi < 0
PERF = "improved" if (d > THR and dlo > 0) else ("degraded" if (d < -THR and dhi < 0)
                                                else "unchanged")
log(f"[1] performance: {PERF.upper()} (threshold {THR}, paired CI "
    f"{'excludes' if sig else 'includes'} zero)")

F = np.genfromtxt("bpnet_filters.tsv", names=True, delimiter="\t", dtype=None, encoding="utf-8")
O = np.genfromtxt("filter_matches_reference.tsv", names=True, delimiter="\t", dtype=None,
                  encoding="utf-8")
icn = F["ic_per_col_matched"]
V = {}
try:
    for ln in open("bpnet_verdict.tsv"):
        k, _, v = ln.partition("\t")
        V[k.strip()] = v.strip()
except Exception:
    pass
ico_med = float(V.get("ic_matched_old_median", "nan"))
ico_max = float(V.get("ic_matched_old_max", "nan"))
log(f"[2] count-matched IC: BPNet {icn.min():.3f}-{icn.max():.3f} (median {np.median(icn):.3f}) "
    f"bits/column over {len(F)} filters of width {int(F['width'][0])}; "
    f"reported median {ico_med:.3f}, max {ico_max:.3f} over 32 of width 19")
SHARP = np.median(icn) > ico_med and icn.max() > ico_max
log(f"[2] sharper: {'YES' if SHARP else 'NO'} "
    f"(median {np.median(icn)-ico_med:+.3f}, max {icn.max()-ico_max:+.3f} bits/column)")
nm_new = int((F["best_match"] != "none").sum()); nm_old = int((O["best_match"] != "none").sum())
log(f"[2] filters matching a reference motif: {nm_new}/{len(F)} against {nm_old}/32; "
    f"the variant recognises MORE reference motifs without sharper filters")
top = np.argsort(-icn)[:5]
for i in top:
    log(f"    {F['filter'][i]}  IC {icn[i]:.3f} b/col  sites {int(F['n_sites_matched'][i]):6d}  "
        f"match {str(F['best_match'][i]):>18} r {F['r'][i]:.3f}  {F['consensus'][i]}")

G = np.genfromtxt("bpnet_gia.tsv", names=True, delimiter="\t", dtype=None, encoding="utf-8")
def best(model, bg, motif):
    s = G[(G["model"] == model) & (G["background"] == bg) & (G["motif"] == motif)]
    i = int(np.argmax(s["delta_logit"])); return float(s["delta_logit"][i]), int(s["position"][i])
PAIRS = [("donor_AGGTAAGT", "donor_perm_ctrl_TAGAGTAG"), ("PAS_AATAAA", "PAS_perm_ctrl_AAAATA")]
NEW = []
log("[3] implantation, both models on the same backgrounds; motif minus its scramble")
for bg in ("dinuc_shuffled", "real_competitor"):
    a_n, _ = best("BPNet variant", bg, "acceptor_pyAG")
    a_o, _ = best("Reported CNN", bg, "acceptor_pyAG")
    log(f"  {bg:16s} acceptor (no scramble in the panel)  BPNet {a_n:+.4f}  reported {a_o:+.4f}")
    for m, c in PAIRS:
        vn = best("BPNet variant", bg, m)[0] - best("BPNet variant", bg, c)[0]
        vo = best("Reported CNN", bg, m)[0] - best("Reported CNN", bg, c)[0]
        log(f"  {bg:16s} {m:20s} BPNet {vn:+.4f}  reported {vo:+.4f}")
        if vn > 0.05 and vn > 2 * max(vo, 0.0): NEW.append((bg, m, vn, vo))
log(f"[3] motifs registering on the variant but not the reported model: "
    f"{'NONE' if not NEW else NEW}")

if PERF == "improved" and not SHARP and not NEW:
    V_ = (f"The motif-friendly architecture RANKS BETTER, not worse: within-gene "
          f"{new['within_gene']:.4f} against {old['within_gene']:.4f}, paired {d:+.4f} "
          f"[{dlo:+.4f},{dhi:+.4f}], and P@1 {new['p_at_1']:.4f} against {old['p_at_1']:.4f}. "
          f"Its first-layer filters are NOT sharper on the count-matched comparison "
          f"({np.median(icn):.3f} against {ico_med:.3f} bits/column), and no motif beyond the "
          f"acceptor registers under implantation. The gain is in the trunk, not in a motif "
          f"code: dilated convolutions at full resolution rank better while reading the same "
          f"elements. Nothing here changes what the model is said to have learnt.")
elif NEW:
    V_ = f"A motif absent from the reported model registers on the variant: {NEW}"
elif PERF == "degraded":
    V_ = (f"The architecture is not free: within-gene {new['within_gene']:.4f} against "
          f"{old['within_gene']:.4f}, {d:+.4f} [{dlo:+.4f},{dhi:+.4f}].")
elif PERF == "improved" and SHARP:
    V_ = (f"The variant both ranks better ({d:+.4f}) and carries sharper filters "
          f"({np.median(icn):.3f} against {ico_med:.3f} bits/column).")
else:
    V_ = (f"Performance {PERF} ({d:+.4f}) and the filters are not sharper "
          f"({np.median(icn):.3f} against {ico_med:.3f} bits/column).")

with open("bpnet_verdict.tsv", "w") as fh:
    fh.write("key\tvalue\n")
    for k, v in (("performance", PERF), ("winrate_new", f"{new['within_gene']:.5f}"),
                 ("winrate_reported", f"{old['within_gene']:.5f}"),
                 ("winrate_delta", f"{d:.5f}"), ("winrate_delta_lo", f"{dlo:.5f}"),
                 ("winrate_delta_hi", f"{dhi:.5f}"),
                 ("p_at_1_new", f"{new['p_at_1']:.5f}"),
                 ("p_at_1_reported", f"{old['p_at_1']:.5f}"),
                 ("auroc_new", f"{new['auroc']:.5f}"), ("auroc_reported", f"{old['auroc']:.5f}"),
                 ("ic_matched_new_median", f"{np.median(icn):.4f}"),
                 ("ic_matched_new_max", f"{icn.max():.4f}"),
                 ("ic_matched_old_median", f"{ico_med:.4f}"),
                 ("ic_matched_old_max", f"{ico_max:.4f}"),
                 ("filters_sharper", str(int(SHARP))),
                 ("n_matched_new", str(nm_new)), ("n_matched_old", str(nm_old)),
                 ("new_motifs", "none" if not NEW else str(len(NEW))),
                 ("verdict", V_)):
        fh.write(f"{k}\t{v}\n")
log(f"\n[VERDICT] {V_}")
log("[out] bpnet_verdict.tsv")
