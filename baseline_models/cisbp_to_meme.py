#!/usr/bin/env python3
# convert a cisbp-rna bulk download into a single meme file with rbp names in the motif
import argparse, os, sys
import numpy as np

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cisbp_dir",required=True)
    ap.add_argument("--out",default="cisbp_rna.meme")
    ap.add_argument("--min_w",type=int,default=4,help="skip motifs shorter than this")
    a=ap.parse_args()
    info=os.path.join(a.cisbp_dir,"RBP_Information.txt")
    pdir=os.path.join(a.cisbp_dir,"pwms_all_motifs")
    m2name={}
    with open(info) as fh:
        hdr=fh.readline().rstrip("\n").split("\t")
        mi=hdr.index("Motif_ID"); ni=hdr.index("RBP_Name"); si=hdr.index("RBP_Species")
        for ln in fh:
            f=ln.rstrip("\n").split("\t")
            if len(f)<=max(mi,ni,si): continue
            mid=f[mi]; name=f[ni]; sp=f[si]
            if mid=="." or not mid: continue
            if mid not in m2name and name and name!=".":
                m2name[mid]=name
    print(f"[info] {len(m2name)} motif->RBP mappings",file=sys.stderr)

    n_written=0; n_skip=0
    with open(a.out,"w") as out:
        out.write("MEME version 4\n\nALPHABET= ACGT\n\n")
        out.write("strands: +\n\n")
        out.write("Background letter frequencies\nA 0.25 C 0.25 G 0.25 T 0.25\n\n")
        for fn in sorted(os.listdir(pdir)):
            if not fn.endswith(".txt"): continue
            mid=fn[:-4]
            name=m2name.get(mid,None)
            path=os.path.join(pdir,fn)
            rows=[]
            try:
                with open(path) as fh:
                    h=fh.readline()
                    for ln in fh:
                        p=ln.rstrip("\n").split("\t")
                        if len(p)<5: continue
                        rows.append([float(p[1]),float(p[2]),float(p[3]),float(p[4])])
            except Exception:
                n_skip+=1; continue
            if len(rows)<a.min_w: n_skip+=1; continue
            P=np.array(rows); P=P/P.sum(1,keepdims=True)
            label = f"{mid}_{name}" if name else mid
            out.write(f"MOTIF {label}\n")
            out.write(f"letter-probability matrix: alength= 4 w= {len(P)} nsites= 20\n")
            for r in P:
                out.write(" "+" ".join(f"{x:.6f}" for x in r)+"\n")
            out.write("\n")
            n_written+=1
    print(f"[out] {a.out}: {n_written} motifs written, {n_skip} skipped",file=sys.stderr)

if __name__=="__main__": main()
