#!/usr/bin/env python3
# compute, across all shortlist peaks (from the saved one-hot)
import argparse, sys
import numpy as np, pandas as pd

PAS_HEX=["AATAAA","ATTAAA","AGTAAA","TATAAA","CATAAA","GATAAA",
         "AATATA","AATACA","AATAGA","AAAAAG","ACTAAA","AAGAAA","AATGAA"]
HEXSET=set(PAS_HEX)
B={"A":0,"C":1,"G":2,"T":3}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train_npz",required=True); ap.add_argument("--train_meta",required=True)
    ap.add_argument("--test_npz",default=None); ap.add_argument("--test_meta",default=None)
    ap.add_argument("--upstream_nt",type=int,default=1000); ap.add_argument("--downstream_nt",type=int,default=500)
    ap.add_argument("--gc_win",type=int,default=21,help="sliding window for GC%")
    ap.add_argument("--out",default="posprof")
    a=ap.parse_args()

    X=np.load(a.train_npz)["X"].astype(np.float32); m=pd.read_parquet(a.train_meta).reset_index(drop=True)
    if a.test_npz:
        X=np.concatenate([X,np.load(a.test_npz)["X"].astype(np.float32)],0)
        m=pd.concat([m,pd.read_parquet(a.test_meta)],ignore_index=True)
    L=X.shape[1]; cleave=a.upstream_nt
    isc=m["is_canonical"].to_numpy().astype(bool)
    has_any=(X.sum(2)>0)
    cov_canon=has_any[isc].mean(0); cov_comp=has_any[~isc].mean(0)
    print(f"[data] {len(X):,} peaks, window {L}, cleavage at index {cleave}",file=sys.stderr)

    def freqs(mask):
        c=X[mask].mean(0)
        return c[:,0],c[:,1],c[:,2],c[:,3]
    Ac,Cc,Gc,Tc=freqs(isc); Ao,Co,Go,To=freqs(~isc)
    gc_canon=Cc+Gc; gc_comp=Co+Go
    eps=1e-6
    skew_canon=(Gc-Cc)/(Gc+Cc+eps); skew_comp=(Go-Co)/(Go+Co+eps)
    def smooth(v,w):
        k=np.ones(w)/w; return np.convolve(v,k,mode="same")
    gc_canon_s=smooth(gc_canon,a.gc_win); gc_comp_s=smooth(gc_comp,a.gc_win)
    skew_canon_s=smooth(skew_canon,a.gc_win); skew_comp_s=smooth(skew_comp,a.gc_win)

    seqidx=X.argmax(2)
    has=X.sum(2)>0
    def hex_track(mask):
        sub=seqidx[mask]; H=has[mask]; n=sub.shape[0]
        track=np.zeros(L)
        codes={}
        for hx in PAS_HEX:
            c=0
            for ch in hx: c=c*4+B[ch]
            codes[hx]=c
        codeset=set(codes.values())
        pow4=np.array([4**5,4**4,4**3,4**2,4,1])
        for st in range(L-6):
            window=sub[:,st:st+6]
            valid=H[:,st:st+6].all(1)
            code=(window*pow4).sum(1)
            isp=np.isin(code,list(codeset)) & valid
            track[st]=isp.mean()
        return track
    pas_canon=hex_track(isc); pas_comp=hex_track(~isc)

    rel=np.arange(L)-cleave
    out=pd.DataFrame({
        "rel_pos":rel,
        "gc_canonical":gc_canon_s,"gc_competitor":gc_comp_s,"gc_diff":gc_canon_s-gc_comp_s,
        "A_canonical":Ac,"U_canonical":Tc,"A_competitor":Ao,"U_competitor":To,
        "pas_canonical":pas_canon,"pas_competitor":pas_comp,"pas_diff":pas_canon-pas_comp,
        "gcskew_canonical":skew_canon_s,"gcskew_competitor":skew_comp_s,
        "gcskew_diff":skew_canon_s-skew_comp_s,
        "cov_canonical":cov_canon,"cov_competitor":cov_comp,
    })
    out.to_csv(f"{a.out}_profiles.tsv",sep="\t",index=False)

    def at(relpos,col):
        i=int(np.argmin(np.abs(rel-relpos))); return out[col].iloc[i]
    pas_peak_rel=int(rel[np.argmax(pas_canon)])
    rel_arr=out['rel_pos'].to_numpy(); gcd=np.abs(out['gc_diff'].to_numpy())
    cov_ok=(cov_canon>=0.8)&(cov_comp>=0.8)
    win=(rel_arr>=-500)&(rel_arr<=100)&cov_ok
    gcdiff_peak_rel=int(rel_arr[win][np.argmax(gcd[win])]) if win.any() else 0
    print(f"\n=== alignment summary ===",file=sys.stderr)
    print(f"  PAS-hexamer detector peaks at rel {pas_peak_rel} (expect ~ -21)",file=sys.stderr)
    print(f"  A% peaks at rel {int(rel[np.argmax(Ac)])}; U% (DSE) peaks downstream at "
          f"rel {int(rel[cleave:cleave+60].argmax()) if False else int(rel[np.argmax(Tc[cleave:cleave+80])+cleave])}",file=sys.stderr)
    print(f"  |GC canon-comp| max within [-500,+100] & covered at rel {gcdiff_peak_rel} "
          f"(CNN filters fired ~ -330)",file=sys.stderr)
    band=(rel_arr>=-400)&(rel_arr<=-250)
    print(f"  mean GC diff in [-400,-250] band: {out['gc_diff'].to_numpy()[band].mean():+.4f} "
          f"(canonical minus competitor)",file=sys.stderr)
    print(f"  coverage at -330: canon {cov_canon[int(np.argmin(np.abs(rel_arr+330)))]:.2f} "
          f"comp {cov_comp[int(np.argmin(np.abs(rel_arr+330)))]:.2f}; "
          f"at -900: canon {cov_canon[int(np.argmin(np.abs(rel_arr+900)))]:.2f} "
          f"(low coverage = edge noise)",file=sys.stderr)
    print(f"  GC diff at -330: {at(-330,'gc_diff'):+.3f}   at +20 (DSE): {at(20,'gc_diff'):+.3f}   "
          f"at -21 (PAS): {at(-21,'gc_diff'):+.3f}",file=sys.stderr)
    print(f"  GC-SKEW diff at -330: {at(-330,'gcskew_diff'):+.3f}  (positive => true ends more "
          f"G-skewed = R-loop-favorable sequence)",file=sys.stderr)
    print(f"\n[out] {a.out}_profiles.tsv  (columns: rel_pos, gc_canonical/competitor/diff, "
          f"A/U tracks, pas_canonical/competitor/diff)",file=sys.stderr)

if __name__=="__main__": main()