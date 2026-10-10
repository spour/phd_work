#!/usr/bin/env python3
# one function that computes the raw splice feature row for a candidate
import numpy as np
AXES=["upstream_mean_u1","upstream_max_me3","term_3ss","term_ppt",
      "local_me5","pas_prox_u1","dn5ss_has_strong","interv_me5_max","dist_interv_me5"]
NEW =["bp_strength","u2af65_ppt","u2af35_ag","u1_suppression_up","u1_suppression_down",
      "dist_to_max_3ss"]
REL =["term_3ss_abs","comp_up_3ss","comp_dn_3ss","rel_3ss_up","rel_3ss_dn",
      "dist_nearest_5ss","dist_up_5ss","dist_dn_5ss",
      "n_5ss_500","n_5ss_1k","sum_5ss","mean_top3_5ss"]
SPLICE_COLS=[f"spl_{a}" for a in AXES]+[f"spl_{n}" for n in NEW]+[f"spl_{r}" for r in REL]
def compute_splice_row(seq, upstream_nt, downstream_nt, n_bins,
                       me5, score3, m3, u1, derive_axes, XF):
    from extract_splicing_features import extract_one_site
    row={c: np.nan for c in SPLICE_COLS}
    if seq is None or len(seq) < upstream_nt+downstream_nt-100:
        return row
    try:
        feat=extract_one_site(seq, upstream_nt, downstream_nt, n_bins, me5, score3, m3, u1)
        ax=derive_axes(feat, seq, upstream_nt)
        for a in AXES:
            v=ax.get(a, np.nan); row[f"spl_{a}"]=float(v) if v is not None and np.isfinite(v) else np.nan
    except Exception:
        pass
    try:
        acc=XF._find_acceptor(seq, upstream_nt, score3, m3)
        row["spl_bp_strength"]=XF.branchpoint_score(seq, upstream_nt, acc_pos=acc)
        row["spl_u2af65_ppt"]=XF.u2af65_ppt(seq, upstream_nt, acc_pos=acc)
        row["spl_u2af35_ag"]=XF.u2af35_ag(seq, upstream_nt, acc_pos=acc)
        row["spl_u1_suppression_up"]=XF.u1_suppression_up(seq, upstream_nt, downstream_nt)
        row["spl_u1_suppression_down"]=XF.u1_suppression_down(seq, upstream_nt, downstream_nt)
    except Exception:
        pass
    try:
        if hasattr(XF,"acceptor_competition"):
            ac=XF.acceptor_competition(seq, upstream_nt, downstream_nt, score3, m3)
            for k,v in ac.items(): row[f"spl_{k}"]=float(v) if v is not None and np.isfinite(v) else np.nan
            dg=XF.donor_geometry(seq, upstream_nt, downstream_nt)
            for k,v in dg.items(): row[f"spl_{k}"]=float(v) if v is not None and np.isfinite(v) else np.nan
            dc=XF.donor_counts(seq, upstream_nt, downstream_nt)
            for k,v in dc.items(): row[f"spl_{k}"]=float(v) if v is not None and np.isfinite(v) else np.nan
            ad=XF.acceptor_distance(seq, upstream_nt, downstream_nt, score3, m3)
            for k,v in ad.items(): row[f"spl_{k}"]=float(v) if v is not None and np.isfinite(v) else np.nan
    except Exception:
        pass
    return row