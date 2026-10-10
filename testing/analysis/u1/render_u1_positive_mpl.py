#!/usr/bin/env python3
# render the task a calibration figure with matplotlib (cvmfs
import csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK="#1A1A1A"; CAN="#0072B2"; COMP="#E69F00"; NUL="#8A8F98"
OLD=4.0; PRIM=5.25

def readtsv(path, stop_on_comment=True):
    rows=[];
    with open(path) as fh:
        r=csv.reader(fh, delimiter="\t"); hdr=next(r)
        for p in r:
            if not p or p[0].startswith("#"):
                if stop_on_comment: break
                continue
            rows.append(p)
    return hdr, rows

# panel a: score distributions
hdr,D=readtsv("u1_positive_dist.tsv")
sc=[float(x[0]) for x in D]
don=[float(x[1]) for x in D]; intr=[float(x[2]) for x in D]; allk=[float(x[3]) for x in D]

# panel b: sensitivity vs enrichment
hdr,C=readtsv("u1_positive_calibration.tsv")
thr=[float(x[0]) for x in C]; sens=[float(x[1]) for x in C]; enr=[float(x[4]) for x in C]
pts=[(t,s,e) for t,s,e in zip(thr,sens,enr) if e>0 and s>0]

fig,(a1,a2)=plt.subplots(1,2,figsize=(13,5.2))

a1.step(sc,don,where="mid",color=CAN,lw=1.8,label="annotated donors")
a1.step(sc,intr,where="mid",color=COMP,lw=1.8,label="random intronic")
a1.step(sc,allk,where="mid",color=NUL,lw=1.5,label="all 9-mers")
for x,c,lab in [(OLD,NUL,"old 4.0"),(PRIM,INK,"primary 5.25")]:
    a1.axvline(x,ls="--",color=c,lw=0.8)
    a1.text(x,a1.get_ylim()[1]*0.98,lab,rotation=0,ha="left" if x==PRIM else "right",
            va="top",fontsize=9,color=c)
a1.set_xlabel("U1 score (-dG)"); a1.set_ylabel("density")
a1.set_title("Real 5'ss carry high U1 scores", fontweight="bold", color=INK, loc="left")
a1.legend(frameon=False, loc="upper right", fontsize=9)
for s in ("top","right"): a1.spines[s].set_visible(False)

xs=[p[1] for p in pts]; ys=[p[2] for p in pts]
a2.plot(xs,ys,color=CAN,lw=1.8,marker="o",ms=3)
for t,s,e in pts:
    if round(t,2) in (4.0,5.25,6.5,7.0,8.0):
        a2.plot(s,e,"o",color=INK,ms=6)
        a2.annotate(f"{t:.2f}",(s,e),textcoords="offset points",xytext=(6,2),fontsize=9,color=INK)
a2.set_yscale("log"); a2.invert_xaxis()
a2.set_xlabel("sensitivity (fraction of annotated donors kept)")
a2.set_ylabel("fold enrichment vs intronic (log)")
a2.set_title("Sensitivity and enrichment trade off with the cutoff", fontweight="bold", color=INK, loc="left")
for s in ("top","right"): a2.spines[s].set_visible(False)

plt.tight_layout()
plt.savefig("u1_positive_calibration.png", dpi=170, facecolor="white")
print("[out] u1_positive_calibration.png")
