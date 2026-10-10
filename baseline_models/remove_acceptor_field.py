#!/usr/bin/env python3
# cleanly remove the competitor-field-distribution features (acceptor_field)
import re, os, sys, shutil

XF="splice_extra_features.py"
SC="splice_common.py"
SR="score_rerank.py"
FIELD_NAMES=["field_z_top","field_gap_ratio","field_top_frac","field_n_strong","field_entropy"]

def restore_or_edit(path, edit_fn):
    bak=path+".bak_field"
    if os.path.exists(bak):
        shutil.copy(path, path+".prefieldremove")
        shutil.copy(bak, path)
        print(f"[{path}] restored from {bak} (exact revert; current saved to {path}.prefieldremove)")
        return
    if not os.path.exists(path):
        print(f"[{path}] not present, skipping"); return
    src=open(path).read(); new=edit_fn(src)
    if new!=src:
        shutil.copy(path, path+".prefieldremove"); open(path,"w").write(new)
        print(f"[{path}] surgically removed field features (no backup found; saved {path}.prefieldremove)")
    else:
        print(f"[{path}] nothing to remove")

def strip_xf(src):
    src=re.sub(r'\n# =+\n# acceptor_field:.*?(?=\n\ndef |\Z)', '\n', src, flags=re.S)
    src=re.sub(r'\ndef _call_score3_af\(.*?\n(?=\ndef |\Z)', '\n', src, flags=re.S)
    src=re.sub(r'\ndef _all_acceptor_scores_af\(.*?\n(?=\ndef |\Z)', '\n', src, flags=re.S)
    src=re.sub(r'\ndef acceptor_field\(.*?\n(?=\ndef |\Z)', '\n', src, flags=re.S)
    return src

def strip_sc(src):
    for n in FIELD_NAMES:
        src=re.sub(r',?\s*"'+n+r'"', '', src)
    src=re.sub(r'[ \t]*af=XF\.acceptor_field\([^\n]*\n[ \t]*for k,v in af\.items\(\):[^\n]*\n', '', src)
    return src

def strip_sr(src):
    for n in FIELD_NAMES:
        src=re.sub(r'\s*"spl_'+n+r'":[+-]?\d+,?', '', src)
    return src

if __name__=="__main__":
    restore_or_edit(XF, strip_xf)
    restore_or_edit(SC, strip_sc)
    restore_or_edit(SR, strip_sr)
    import ast
    for f in (XF,SC,SR):
        if os.path.exists(f): ast.parse(open(f).read())
    try:
        sys.path.insert(0,".")
        import importlib, splice_common
        importlib.reload(splice_common)
        has_field=[c for c in splice_common.SPLICE_COLS if "field_" in c]
        has_dist=[c for c in splice_common.SPLICE_COLS if "dist_to_max_3ss" in c or "pos_of_max_3ss" in c]
        print(f"\n[ok] SPLICE_COLS now {len(splice_common.SPLICE_COLS)} cols; "
              f"field features remaining: {has_field} (should be []); "
              f"acceptor_distance kept: {has_dist}")
    except Exception as e:
        print(f"\n[ok] files parse. (Could not import splice_common to count: {e})")
    print("Next: re-featurize at -2000/+500 WITHOUT field cols (or just reuse the "
          "pre-field w2000 parquets), then rerun score_rerank.")
