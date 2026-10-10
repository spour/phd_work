#!/usr/bin/env python3
# surgically add two features to the splice feature set: spl_dist_to_max_3ss : distance
import re, sys, shutil, os

XF="splice_extra_features.py"
SC="splice_common.py"

FUNC = '''

# ============================================================================
# acceptor_distance: position/distance of the STRONGEST upstream 3'ss.
# Added to test whether HOW FAR the defining acceptor sits (not just its strength)
# carries reranking signal, and to exploit a wider upstream featurization window.
# ============================================================================
def _call_score3_ad(score3, win, m3):
    try:
        s=score3(win)
    except TypeError:
        try: s=score3(win, m3)
        except Exception: return None
    except Exception: return None
    if s is None: return None
    if np.isscalar(s): return float(s)
    try:
        a=np.asarray(s,float); return float(np.nanmax(a)) if a.size else None
    except Exception: return None

def acceptor_distance(seq, upstream_nt, downstream_nt, score3=None, m3=None, search_nt=None):
    """Distance/position of the strongest upstream acceptor relative to cleavage.
    Cleavage is at index `upstream_nt` in seq. Returns dict with
    dist_to_max_3ss and pos_of_max_3ss (positive = upstream; NaN if none)."""
    out=dict(dist_to_max_3ss=np.nan, pos_of_max_3ss=np.nan)
    if seq is None or len(seq) < upstream_nt+3: return out
    cand=upstream_nt
    span = upstream_nt if (search_nt is None) else min(int(search_nt), upstream_nt)
    lo=max(0, cand-span); hi=cand
    if hi-lo < 23: return out
    region=seq[lo:hi].upper()
    ra=np.frombuffer(region.encode("ascii","replace"),dtype=np.uint8)
    ag=np.where((ra[:-1]==65)&(ra[1:]==71))[0]+2
    best=-1e9; best_p=None
    for i in ag:
        p=lo+int(i); s0=p-20
        if s0<0 or p+3>len(seq): continue
        v=_call_score3_ad(score3, seq[s0:p+3].upper(), m3) if score3 is not None else None
        if v is None: continue
        if v>best: best=v; best_p=p
    if best_p is None: return out
    dist=cand-best_p
    out["pos_of_max_3ss"]=float(dist); out["dist_to_max_3ss"]=float(dist)
    return out
'''

CALL_TEMPLATE = (
'{ind}ad=XF.acceptor_distance(seq, upstream_nt, downstream_nt, score3, m3)\n'
'{ind}for k,v in ad.items(): row[f"spl_{{k}}"]=float(v) if v is not None and np.isfinite(v) else np.nan\n'
)

def patch_xf():
    src=open(XF).read()
    if "def acceptor_distance(" in src:
        print(f"[{XF}] acceptor_distance already present, skipping append"); return
    shutil.copy(XF, XF+".bak")
    open(XF,"a").write(FUNC)
    print(f"[{XF}] appended acceptor_distance() (backup {XF}.bak)")

def patch_sc():
    src=open(SC).read()
    changed=False
    # 1) add the two names to the new list so splice_cols includes them
    if "dist_to_max_3ss" not in src:
        m=re.search(r'NEW\s*=\s*\[(.*?)\]', src, re.S)
        if not m: sys.exit(f"[{SC}] could not find NEW=[...] to extend")
        inner=m.group(1).rstrip()
        if not inner.rstrip().endswith(","): inner=inner+","
        newlist='NEW =['+inner+'\n      "dist_to_max_3ss","pos_of_max_3ss"]'
        src=src[:m.start()]+newlist+src[m.end():]
        changed=True
        print(f"[{SC}] added dist_to_max_3ss, pos_of_max_3ss to NEW (SPLICE_COLS)")
    else:
        print(f"[{SC}] NEW already contains the columns, skipping")
    # 2) wire the call into compute_splice_row, right after the donor_counts block
    if "acceptor_distance(seq" not in src:
        anchor='dc=XF.donor_counts(seq, upstream_nt, downstream_nt)'
        if anchor not in src:
            sys.exit(f"[{SC}] could not find donor_counts anchor to insert after")
        m2=re.search(r'(?P<ind>[ \t]*)dc=XF\.donor_counts\([^\n]*\n[ \t]*for k,v in dc\.items\(\):[^\n]*\n', src)
        if not m2: sys.exit(f"[{SC}] could not locate donor_counts for-loop to insert after")
        ind=m2.group("ind")
        call_block=CALL_TEMPLATE.format(ind=ind)
        src=src[:m2.end()]+call_block+src[m2.end():]
        changed=True
        print(f"[{SC}] inserted acceptor_distance call into compute_splice_row")
    else:
        print(f"[{SC}] acceptor_distance call already wired")
    if changed:
        shutil.copy(SC, SC+".bak2") if not os.path.exists(SC+".bak") else shutil.copy(SC, SC+".bak")
        open(SC,"w").write(src)
        print(f"[{SC}] written (backup made)")

if __name__=="__main__":
    for f in (XF, SC):
        if not os.path.exists(f): sys.exit(f"[err] {f} not found; run from the scripts/ dir")
    patch_xf(); patch_sc()
    import ast
    ast.parse(open(XF).read()); ast.parse(open(SC).read())
    print("\n[ok] both files parse. New SPLICE_COLS should now include spl_dist_to_max_3ss, spl_pos_of_max_3ss.")
    print("Next: re-featurize at -2000/+500, then rerun score_rerank.")
