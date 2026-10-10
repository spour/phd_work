#!/usr/bin/env python3
# multi-panel figure for the de-novo cnn that separates dominant ends from cpccpa cryptic
import argparse, sys, re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

COL={"U-rich":"#C65B3C","GA-rich/ESE":"#5B7C6F","C-rich":"#3C6E9E",
     "AU-rich":"#D89A4C","mixed":"#9AA0A6"}

def classify(cons):
    s=cons.upper(); n=len(s)
    A=s.count("A")/n; T=s.count("T")/n; G=s.count("G")/n; C=s.count("C")/n
    if T>=0.55: return "U-rich"
    if (A+T)>=0.7: return "AU-rich"
    if (G+C)>=0.6 and C>=G: return "C-rich"
    if (G+A)>=0.6 and re.search(r"(AAG|GAA|AGA){2,}",s): return "GA-rich/ESE"
    if (G+A)>=0.6: return "GA-rich/ESE"
    if (G+C)>=0.55: return "C-rich"
    return "mixed"

def parse_meme(path):
    pwms={}
    try:
        txt=open(path).read()
    except Exception:
        return pwms
    blocks=re.split(r"\nMOTIF ",txt)
    for b in blocks[1:]:
        name=b.split()[0]
        m=re.search(r"w=\s*(\d+)",b)
        if not m: continue
        rows=[]
        for ln in b.splitlines():
            nums=re.findall(r"[0-9]*\.?[0-9]+(?:e-?\d+)?",ln)
            vals=[float(x) for x in nums]
            if len(vals)==4 and 0.9<sum(vals)<1.1:
                rows.append(vals)
        if rows: pwms[name]=np.array(rows)
    return pwms

def letter(ax,ch,x,y,w,h,color):
    ax.text(x+w/2,y,ch,fontsize=80*h,ha="center",va="bottom",
            family="monospace",weight="bold",color=color)

def logo(ax,P,title):
    BASES=["A","C","G","T"]; bcol={"A":"#3DA35D","C":"#3C6E9E","G":"#E6A817","T":"#C65B3C"}
    Ln=len(P)
    for i in range(Ln):
        col=P[i]; ic=2+np.sum(col*np.log2(col+1e-9))
        order=np.argsort(col)
        ybase=0
        for j in order:
            h=col[j]*ic/2.0
            if h>0.02:
                ax.text(i+0.5,ybase,BASES[j],fontsize=1,alpha=0)
                ax.add_patch(Rectangle((i+0.1,ybase),0.8,h,color=bcol[BASES[j]],alpha=0.0))
                ax.text(i+0.5,ybase+h/2,BASES[j],ha="center",va="center",
                        family="monospace",weight="bold",color=bcol[BASES[j]],
                        fontsize=max(6,min(26,260*h/max(Ln,1)*0.6)))
                ybase+=h
    ax.set_xlim(0,Ln); ax.set_ylim(0,2); ax.set_yticks([0,1,2])
    ax.set_title(title,fontsize=8); ax.set_xticks([])
    for s in ["top","right"]: ax.spines[s].set_visible(False)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--summary",required=True,help="first motif set (e.g. dominant-marking/used)")
    ap.add_argument("--summary2",default=None,help="second motif set (e.g. cryptic-marking/suppressed)")
    ap.add_argument("--label1",default="marks DOMINANT"); ap.add_argument("--label2",default="marks CRYPTIC")
    ap.add_argument("--meme",default=None); ap.add_argument("--meme2",default=None)
    ap.add_argument("--auc_overall",type=float,required=True)
    ap.add_argument("--auc_internal",type=float,required=True)
    ap.add_argument("--title",default="De-novo CNN motifs")
    ap.add_argument("--n_logos",type=int,default=6)
    ap.add_argument("--out",default="cnn_motif_figure")
    a=ap.parse_args()
    S=pd.read_csv(a.summary,sep="\t"); S["class"]=S["consensus"].map(classify)
    S=S.sort_values("n",ascending=False).reset_index(drop=True)
    S2=None
    if a.summary2:
        S2=pd.read_csv(a.summary2,sep="\t"); S2["class"]=S2["consensus"].map(classify)
        S2=S2.sort_values("n",ascending=False).reset_index(drop=True)
    pwms=parse_meme(a.meme) if a.meme else {}

    if S2 is not None:
        fig=plt.figure(figsize=(14,9))
        gs=fig.add_gridspec(2,3,height_ratios=[1,1],hspace=0.42,wspace=0.35)
        axA=fig.add_subplot(gs[0,0])
        bars=axA.bar(["overall","within\nINTERNAL"],[a.auc_overall,a.auc_internal],color=["#444","#5B7C6F"],width=.6)
        axA.axhline(0.5,ls="--",c="grey",lw=1); axA.set_ylim(0.4,0.85)
        for b,v in zip(bars,[a.auc_overall,a.auc_internal]): axA.text(b.get_x()+b.get_width()/2,v+0.005,f"{v:.3f}",ha="center",fontsize=10,weight="bold")
        axA.set_ylabel("AUC dominant vs cryptic"); axA.set_title("A  Separation",fontsize=11,loc="left",weight="bold")
        for s in ["top","right"]: axA.spines[s].set_visible(False)
        def catalog(ax,SS,title):
            yp=np.arange(len(SS))[::-1]
            ax.barh(yp,SS["ic"],color=[COL[c] for c in SS["class"]])
            for yi,(_,r) in zip(yp,SS.iterrows()):
                ax.text(0.05,yi,r["consensus"],va="center",fontsize=7.5,family="monospace")
                ax.text(r["ic"]+0.05,yi,f"{r['ic']:.1f} n={int(r['n'])}",va="center",fontsize=6.5,color="grey")
            ax.set_yticks([]); ax.set_xlabel("IC (bits)"); ax.set_xlim(0,max(SS["ic"])*1.6)
            ax.set_title(title,fontsize=11,loc="left",weight="bold")
            for s in ["top","right"]: ax.spines[s].set_visible(False)
        catalog(fig.add_subplot(gs[0,1]),S,f"B  Motifs that {a.label1}")
        catalog(fig.add_subplot(gs[0,2]),S2,f"C  Motifs that {a.label2}")
        axD=fig.add_subplot(gs[1,:])
        axD.scatter(S["medpos"],np.full(len(S),1.0),c=[COL[c] for c in S["class"]],s=80,marker="o",edgecolor="k",linewidth=.4,zorder=3,label=a.label1)
        axD.scatter(S2["medpos"],np.full(len(S2),0.0),c=[COL[c] for c in S2["class"]],s=80,marker="s",edgecolor="k",linewidth=.4,zorder=3,label=a.label2)
        for _,r in S.iterrows(): axD.text(r["medpos"],1.08,r["consensus"][:6],fontsize=5.5,rotation=45,ha="left",color="grey")
        for _,r in S2.iterrows(): axD.text(r["medpos"],-0.18,r["consensus"][:6],fontsize=5.5,rotation=45,ha="left",color="grey")
        axD.axvline(0,c="k",lw=1); axD.axvline(-21,ls=":",c="grey")
        axD.set_yticks([0,1]); axD.set_yticklabels([a.label2,a.label1]); axD.set_ylim(-0.6,1.6)
        axD.set_xlabel("position relative to cleavage site (nt)")
        axD.set_title("D  Motif positions (circle = dominant-marking, square = cryptic-marking)",fontsize=11,loc="left",weight="bold")
        handles=[Rectangle((0,0),1,1,color=COL[k]) for k in COL]
        axD.legend(handles,list(COL.keys()),fontsize=7,loc="upper left",frameon=False,ncol=5)
        for s in ["top","right","left"]: axD.spines[s].set_visible(False)
        fig.suptitle(a.title,fontsize=14,weight="bold",y=0.98)
        fig.savefig(f"{a.out}.png",dpi=200,bbox_inches="tight"); fig.savefig(f"{a.out}.pdf",bbox_inches="tight")
        print(f"[out] {a.out}.png, {a.out}.pdf",file=sys.stderr)
        pd.concat([S.assign(set=a.label1),S2.assign(set=a.label2)]).to_csv(f"{a.out}_annotated.tsv",sep="\t",index=False)
        for nm,SS in [(a.label1,S),(a.label2,S2)]:
            print(f"\n{nm}:",file=sys.stderr)
            print(SS[["motif","n","ic","medpos","consensus","class"]].to_string(index=False),file=sys.stderr)
        return

    fig=plt.figure(figsize=(13,9))
    gs=fig.add_gridspec(2,3,height_ratios=[1,1.1],hspace=0.38,wspace=0.32)

    # a: auc bars
    axA=fig.add_subplot(gs[0,0])
    bars=axA.bar(["overall","within\nINTERNAL"],[a.auc_overall,a.auc_internal],
                 color=["#444","#5B7C6F"],width=.6)
    axA.axhline(0.5,ls="--",c="grey",lw=1); axA.set_ylim(0.4,0.85)
    for b,v in zip(bars,[a.auc_overall,a.auc_internal]):
        axA.text(b.get_x()+b.get_width()/2,v+0.005,f"{v:.3f}",ha="center",fontsize=10,weight="bold")
    axA.set_ylabel("AUC (dominant vs CPCCPA)")
    axA.set_title("A  Separation + compartment control",fontsize=11,loc="left",weight="bold")
    axA.annotate("drop = compartment\ncomponent",xy=(1,a.auc_internal),xytext=(0.55,0.62),
                 fontsize=8,color="grey",ha="center")
    for s in ["top","right"]: axA.spines[s].set_visible(False)

    # b: motif catalog (ic and n, colored by class)
    axB=fig.add_subplot(gs[0,1:])
    yp=np.arange(len(S))[::-1]
    axB.barh(yp,S["ic"],color=[COL[c] for c in S["class"]])
    for yi,(_,r) in zip(yp,S.iterrows()):
        axB.text(0.05,yi,r["consensus"],va="center",fontsize=8,family="monospace")
        axB.text(r["ic"]+0.05,yi,f"IC {r['ic']:.1f}  n={int(r['n'])}",va="center",fontsize=7,color="grey")
    axB.set_yticks([]); axB.set_xlabel("information content (bits)")
    axB.set_xlim(0,max(S["ic"])*1.5)
    axB.set_title("B  Learned motifs (IG seqlets), colored by composition class",fontsize=11,loc="left",weight="bold")
    handles=[Rectangle((0,0),1,1,color=COL[k]) for k in COL]
    axB.legend(handles,list(COL.keys()),fontsize=7,loc="lower right",frameon=False)
    for s in ["top","right"]: axB.spines[s].set_visible(False)

    # c: motif position relative to cleavage
    axC=fig.add_subplot(gs[1,0])
    axC.scatter(S["medpos"],np.arange(len(S)),c=[COL[c] for c in S["class"]],s=60,zorder=3)
    for i,(_,r) in enumerate(S.iterrows()):
        axC.text(r["medpos"]-8,i,r["consensus"][:6],va="center",ha="right",fontsize=6,color="grey")
    axC.axvline(0,c="k",lw=1); axC.axvline(-21,ls=":",c="grey",lw=1)
    axC.text(0,len(S)+0.3,"cleavage",fontsize=7,ha="center",color="k")
    axC.set_xlabel("position relative to cleavage site (nt)")
    axC.set_yticks([]); axC.set_xlim(min(S["medpos"])-60,40)
    axC.set_title("C  Motif positions (upstream)",fontsize=11,loc="left",weight="bold")
    for s in ["top","right","left"]: axC.spines[s].set_visible(False)

    # d: sequence logos for top motifs
    if pwms:
        sub=fig.add_gridspec(2,3,height_ratios=[1,1.1],hspace=0.38,wspace=0.32)[1,1:]
        gsd=sub.subgridspec(2,3,hspace=0.6,wspace=0.3)
        top=S.head(a.n_logos)
        for k,(_,r) in enumerate(top.iterrows()):
            ax=fig.add_subplot(gsd[k//3,k%3])
            key=None
            for nm in pwms:
                if re.search(rf"motif_{int(r['motif'])}\b",nm) or nm.endswith(f"_{int(r['motif'])}"):
                    key=nm; break
            if key is None and len(pwms)>k: key=list(pwms.keys())[k]
            if key: logo(ax,pwms[key],f"{r['consensus'][:8]} ({r['class']})")
            else: ax.axis("off")
        fig.text(0.66,0.46,"D  Sequence logos (top motifs)",fontsize=11,weight="bold")
    else:
        axD=fig.add_subplot(gs[1,1:]); axD.axis("off")
        axD.text(0.5,0.5,"D  (provide --meme for sequence logos)",ha="center",fontsize=10,color="grey")

    fig.suptitle(a.title,fontsize=14,weight="bold",y=0.98)
    fig.savefig(f"{a.out}.png",dpi=200,bbox_inches="tight")
    fig.savefig(f"{a.out}.pdf",bbox_inches="tight")
    print(f"[out] {a.out}.png, {a.out}.pdf",file=sys.stderr)
    S.to_csv(f"{a.out}_annotated.tsv",sep="\t",index=False)
    print(S[["motif","n","ic","medpos","consensus","class"]].to_string(index=False),file=sys.stderr)

if __name__=="__main__": main()