#!/usr/bin/env python3
# add competitor-field-distribution features
import re, sys, shutil, os

XF="splice_extra_features.py"
SC="splice_common.py"
SR="score_rerank.py"

FUNC = '''

# ============================================================================
# acceptor_field: SHAPE of the upstream acceptor-score distribution (dominance),
# orthogonal to the absolute max. True terminal exon = one dominant acceptor;
# cryptic intronic site = flat field of comparable weak acceptors.
# ============================================================================
def _call_score3_af(score3, win, m3):
    try: s=score3(win)
    except TypeError:
        try: s=score3(win, m3)
        except Exception: return None
    except Exception: return None
    if s is None: return None
    if np.isscalar(s): return float(s)
    try:
        a=np.asarray(s,float); return float(np.nanmax(a)) if a.size else None
    except Exception: return None

def _all_acceptor_scores_af(seq, lo, hi, score3, m3):
    n=len(seq); lo=max(0,lo); hi=min(n,hi)
    if hi-lo<23: return np.array([])
    region=seq[lo:hi].upper()
    ra=np.frombuffer(region.encode("ascii","replace"),dtype=np.uint8)
    ag=np.where((ra[:-1]==65)&(ra[1:]==71))[0]+2
    out=[]
    for i in ag:
        p=lo+int(i); s0=p-20
        if s0<0 or p+3>n: continue
        v=_call_score3_af(score3, seq[s0:p+3].upper(), m3)
        if v is not None: out.append(v)
    return np.array(out,float)

def acceptor_field(seq, upstream_nt, downstream_nt, score3=None, m3=None, min_pos=4):
    """Shape (dominance) features of the upstream acceptor field. Scale-free /
    ratio-based so they don't re-encode the absolute max. NaN if too few acceptors."""
    out=dict(field_z_top=np.nan, field_gap_ratio=np.nan, field_top_frac=np.nan,
             field_n_strong=np.nan, field_entropy=np.nan)
    if seq is None or len(seq) < upstream_nt: return out
    s=_all_acceptor_scores_af(seq, max(0,upstream_nt-upstream_nt), upstream_nt, score3, m3)
    if s.size < min_pos: return out
    smax=float(np.max(s)); mu=float(np.mean(s)); sd=float(np.std(s))
    if sd>1e-9: out["field_z_top"]=(smax-mu)/sd
    srt=np.sort(s)[::-1]
    if smax>1e-9 and srt.size>=2: out["field_gap_ratio"]=float(srt[1]/smax)
    pos=s[s>0]
    if pos.size>=1 and pos.sum()>1e-9:
        out["field_top_frac"]=float(pos.max()/pos.sum())
        p=pos/pos.sum(); ent=-np.sum(p*np.log(p+1e-12))
        out["field_entropy"]=float(ent/np.log(len(p))) if len(p)>1 else 0.0
    if smax>1e-9: out["field_n_strong"]=float(np.sum(s>=0.5*smax))
    return out
'''

FIELD_NAMES=["field_z_top","field_gap_ratio","field_top_frac","field_n_strong","field_entropy"]

CALL_TEMPLATE = (
'{ind}af=XF.acceptor_field(seq, upstream_nt, downstream_nt, score3, m3)\n'
'{ind}for k,v in af.items(): row[f"spl_{{k}}"]=float(v) if v is not None and np.isfinite(v) else np.nan\n'
)

DIRP_ADD = {
    "spl_field_z_top":+1, "spl_field_top_frac":+1,
    "spl_field_gap_ratio":-1, "spl_field_n_strong":-1, "spl_field_entropy":-1,
}

def patch_xf():
    src=open(XF).read()
    if "def acceptor_field(" in src:
        print(f"[{XF}] acceptor_field already present, skipping"); return
    shutil.copy(XF, XF+".bak_field")
    open(XF,"a").write(FUNC)
    print(f"[{XF}] appended acceptor_field() (backup {XF}.bak_field)")

def patch_sc():
    src=open(SC).read(); changed=False
    if "field_z_top" not in src:
        m=re.search(r'NEW\s*=\s*\[(.*?)\]', src, re.S)
        if not m: sys.exit(f"[{SC}] could not find NEW=[...]")
        inner=m.group(1).rstrip()
        if not inner.rstrip().endswith(","): inner=inner+","
        addn=",".join(f'"{n}"' for n in FIELD_NAMES)
        newlist='NEW =['+inner+'\n      '+addn+']'
        src=src[:m.start()]+newlist+src[m.end():]; changed=True
        print(f"[{SC}] added {FIELD_NAMES} to NEW")
    else:
        print(f"[{SC}] field names already in NEW, skipping")
    if "acceptor_field(seq" not in src:
        m2=re.search(r'(?P<ind>[ \t]*)dc=XF\.donor_counts\([^\n]*\n[ \t]*for k,v in dc\.items\(\):[^\n]*\n', src)
        if not m2:
            m2=re.search(r'(?P<ind>[ \t]*)ad=XF\.acceptor_distance\([^\n]*\n[ \t]*for k,v in ad\.items\(\):[^\n]*\n', src)
        if not m2: sys.exit(f"[{SC}] could not find an anchor (donor_counts/acceptor_distance) to insert after")
        ind=m2.group("ind")
        src=src[:m2.end()]+CALL_TEMPLATE.format(ind=ind)+src[m2.end():]; changed=True
        print(f"[{SC}] inserted acceptor_field call into compute_splice_row")
    else:
        print(f"[{SC}] acceptor_field call already wired")
    if changed:
        shutil.copy(SC, SC+".bak_field"); open(SC,"w").write(src)
        print(f"[{SC}] written (backup {SC}.bak_field)")

def patch_sr():
    if not os.path.exists(SR):
        print(f"[{SR}] not in this dir; skipping DIRP edit (features still work as unconstrained, sign 0)")
        return
    src=open(SR).read()
    if "spl_field_z_top" in src:
        print(f"[{SR}] DIRP already has field priors, skipping"); return
    m=re.search(r'(DIRP\s*=\s*\{.*?)(\})', src, re.S)
    if not m:
        print(f"[{SR}] could not find DIRP dict; features will run unconstrained (sign 0)"); return
    adds="".join(f'\n      "{k}":{v:+d},' for k,v in DIRP_ADD.items())
    new=m.group(1).rstrip()
    if not new.rstrip().endswith(","): new=new+","
    new=new+adds+"\n      "+m.group(2)
    src=src[:m.start()]+new+src[m.end():]
    shutil.copy(SR, SR+".bak_field"); open(SR,"w").write(src)
    print(f"[{SR}] added field monotone priors to DIRP (backup {SR}.bak_field)")

if __name__=="__main__":
    for f in (XF,SC):
        if not os.path.exists(f): sys.exit(f"[err] {f} not found; run from the scripts/ dir")
    patch_xf(); patch_sc(); patch_sr()
    import ast
    ast.parse(open(XF).read()); ast.parse(open(SC).read())
    if os.path.exists(SR): ast.parse(open(SR).read())
    print("\n[ok] files parse. SPLICE_COLS now includes 5 field features.")
    print("Next: re-featurize at -2000/+500 (same as before), then rerun score_rerank.")