#!/usr/bin/env python3
# reduce the feature set to exactly "25 original + dist_to_max_3ss" by removing
import re, os, sys, shutil

XF="splice_extra_features.py"
SC="splice_common.py"

def patch_sc():
    if not os.path.exists(SC): sys.exit(f"[err] {SC} not found; run from scripts/")
    src=open(SC).read()
    if '"pos_of_max_3ss"' not in src:
        print(f"[{SC}] pos_of_max_3ss already absent from NEW, skipping"); return
    shutil.copy(SC, SC+".bak_pos")
    src=re.sub(r',?\s*"pos_of_max_3ss"', '', src)
    open(SC,"w").write(src)
    print(f"[{SC}] removed pos_of_max_3ss from NEW (backup {SC}.bak_pos)")

def patch_xf():
    if not os.path.exists(XF): sys.exit(f"[err] {XF} not found; run from scripts/")
    src=open(XF).read()
    if "pos_of_max_3ss" not in src:
        print(f"[{XF}] acceptor_distance already pos-free, skipping"); return
    shutil.copy(XF, XF+".bak_pos")
    src=src.replace('out=dict(dist_to_max_3ss=np.nan, pos_of_max_3ss=np.nan)',
                    'out=dict(dist_to_max_3ss=np.nan)')
    src=re.sub(
        r'(?P<ind>[ \t]*)out\["pos_of_max_3ss"\]\s*=\s*float\(dist\)\s*;\s*out\["dist_to_max_3ss"\]\s*=\s*float\(dist\)',
        r'\g<ind>out["dist_to_max_3ss"]=float(dist)', src)
    src=re.sub(
        r'(?P<ind>[ \t]*)out\["dist_to_max_3ss"\]\s*=\s*float\(dist\)\s*;\s*out\["pos_of_max_3ss"\]\s*=\s*float\(dist\)',
        r'\g<ind>out["dist_to_max_3ss"]=float(dist)', src)
    src=re.sub(r'[ \t]*out\["pos_of_max_3ss"\]\s*=\s*float\(dist\)\s*;?\s*\n', '', src)
    open(XF,"w").write(src)
    print(f"[{XF}] acceptor_distance now returns only dist_to_max_3ss (backup {XF}.bak_pos)")

if __name__=="__main__":
    patch_xf(); patch_sc()
    import ast
    ast.parse(open(XF).read()); ast.parse(open(SC).read())
    try:
        sys.path.insert(0,"."); import importlib, splice_common; importlib.reload(splice_common)
        cols=splice_common.SPLICE_COLS
        print(f"\n[ok] SPLICE_COLS now {len(cols)} cols.")
        print(f"     dist_to_max_3ss present: {'spl_dist_to_max_3ss' in cols}")
        print(f"     pos_of_max_3ss present:  {'spl_pos_of_max_3ss' in cols}  (should be False)")
        if len(cols)==26: print("     -> exactly 25 original + dist_to_max_3ss. Correct.")
        else: print(f"     -> NOTE: {len(cols)} cols (expected 26); check no other extras remain.")
    except Exception as e:
        print(f"\n[ok] files parse. (could not import splice_common: {e})")