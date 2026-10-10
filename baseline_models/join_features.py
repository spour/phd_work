#!/usr/bin/env python3
# augmented tsv = your pwm+kmer featurization tsv + splice tsv
import argparse, sys
import numpy as np, pandas as pd

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pwm",required=True); ap.add_argument("--splice",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--force_positional",action="store_true",
                    help="bind by row order even if counts let an id-join run")
    a=ap.parse_args()
    P=pd.read_csv(a.pwm,sep="\t",index_col=0)
    S=pd.read_csv(a.splice,sep="\t",index_col=0)
    npw,nsp=len(P),len(S)

    if npw==nsp:
        Pv=P.reset_index(drop=True)
        Sv=S.reset_index(drop=True)
        dup=[c for c in Sv.columns if c in Pv.columns]
        if dup:
            print(f"[warn] dropping {len(dup)} splice cols already in pwm: {dup[:5]}...",file=sys.stderr)
            Sv=Sv.drop(columns=dup)
        J=pd.concat([Pv,Sv],axis=1)
        J.index=P.index
        mode="positional"
    else:
        if P.index.is_unique and S.index.is_unique:
            J=P.join(S,how="inner"); mode="id-join(unique)"
        else:
            sys.exit(f"[err] row counts differ ({npw} pwm vs {nsp} splice) AND ids are "
                     f"not unique on both sides (pwm_unique={P.index.is_unique}, "
                     f"splice_unique={S.index.is_unique}). Cannot align safely. "
                     f"Re-run splice_features_for_bed.py on the SAME bed used for the "
                     f"pwm featurization so row counts match for a positional bind.")
    if len(J)!=npw:
        sys.exit(f"[err] post-join rows {len(J)} != pwm rows {npw} (mode={mode})")
    spl=[c for c in S.columns if c in J.columns]
    nonnan=J[spl].notna().mean().mean() if spl else 0.0
    J.to_csv(a.out,sep="\t")
    print(f"[done|{mode}] {J.shape[0]:,} x {J.shape[1]} (pwm {P.shape[1]} + splice {len(spl)}) "
          f"splice non-NaN {nonnan:.3f} -> {a.out}",file=sys.stderr)

if __name__=="__main__": main()