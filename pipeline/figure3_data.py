#!/usr/bin/env python3
# user's known-good base (acceptor auroc ~0.61)
import argparse, os, sys
import numpy as np, pandas as pd
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k): return x


_G = {}

IPROF_EDGES = np.arange(0, 600 + 30, 30)
IPROF_NB = len(IPROF_EDGES) - 1

MG_EDGES = np.arange(-500, 500 + 40, 40)
MG_NB = len(MG_EDGES) - 1
MG_SPAN = 500.0

UC_EDGES = np.arange(-500, 1000 + 50, 50)
UC_NB = len(UC_EDGES) - 1
UC_THRS = np.array([0.0, 2.0, 3.0, 4.0, 4.58, 5.0, 5.70, 6.0, 7.30, 8.0, 9.0])
DW_THRS = {"u1": np.array([0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]),
           "donor": np.array([3.0, 4.0, 5.0, 6.5, 8.0])}
DW_MECHS = ("u1", "donor")
MC_THRS = np.array([0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
UC_NT = len(UC_THRS)
UC_NF = 3 + UC_NT


def u1_winner_features(cube, edges=None, thrs=None):
    import numpy as np
    if edges is None: edges = UC_EDGES
    if thrs is None: thrs = UC_THRS
    edges = np.asarray(edges); thrs = np.asarray(thrs)
    b = lambda x: int(np.searchsorted(edges, x))
    ti = lambda t: 3 + int(np.argmin(np.abs(thrs - t)))
    def _dens(lo, hi, t):
        return cube[:, b(lo):b(hi), ti(t)].sum(1) / ((hi - lo) / 1000.0)
    with np.errstate(invalid="ignore"):
        mx = np.nanmax(cube[:, b(-450):b(50), 2], axis=1)
    return dict(u1_dens_win=_dens(-500, 250, 4.0),
                u1_max_disc=mx,
                u1_dens_strong=_dens(-500, 400, 8.0),
                u1_dens_3p=_dens(0, 250, 4.0))


def _u1_cube(rel_up, val_up, rel_dn, val_dn, thrs=None):
    import numpy as np
    if thrs is None: thrs = UC_THRS
    cube = np.zeros((UC_NB, UC_NF), np.float32)
    cube[:, 2] = np.nan
    parts = []
    if len(rel_up):
        m = np.isfinite(val_up) & (rel_up < 0) & (rel_up >= UC_EDGES[0])
        if m.any(): parts.append((rel_up[m], val_up[m]))
    if len(rel_dn):
        m = np.isfinite(val_dn) & (rel_dn >= 0) & (rel_dn < UC_EDGES[-1])
        if m.any(): parts.append((rel_dn[m], val_dn[m]))
    if not parts: return cube, 0.0, 0.0
    rel = np.concatenate([p[0] for p in parts]).astype(float)
    val = np.concatenate([p[1] for p in parts]).astype(float)
    bi = np.clip(np.searchsorted(UC_EDGES, rel, side="right") - 1, 0, UC_NB - 1)
    cube[:, 0] = np.bincount(bi, minlength=UC_NB)
    cube[:, 1] = np.bincount(bi, weights=val, minlength=UC_NB)
    mx = np.full(UC_NB, -np.inf)
    np.maximum.at(mx, bi, val)
    cube[:, 2] = np.where(np.isfinite(mx), mx, np.nan)
    for ti in range(UC_NT):
        cube[:, 3 + ti] = np.bincount(bi, weights=(val >= thrs[ti]).astype(float),
                                      minlength=UC_NB)
    w = 1.0 / (1.0 + np.abs(rel))
    up = rel < 0
    return cube, float(np.sum(val[up] * w[up])), float(np.sum(val[~up] * w[~up]))

def _init_worker(cfg):
    import numpy as np, sys
    sys.path.insert(0, cfg["scripts_dir"])
    from extract_splicing_features import GenomeFasta, init_maxentscan, load_u1_lookup
    o = init_maxentscan()
    fa = GenomeFasta(cfg["reference_fasta"])
    u1 = load_u1_lookup(cfg["u1_lookup"])
    U1LEN = len(next(iter(u1)))
    MAP = np.full(256, -1, np.int64)
    for b, code in zip(b"ACGT", range(4)): MAP[b] = code
    dense = np.full(4**U1LEN, np.nan, np.float32)
    powv = (4**np.arange(U1LEN-1, -1, -1)).astype(np.int64)
    for kmer, val in u1.items():
        idx = 0
        for ch in kmer.encode(): idx = idx*4 + int(MAP[ch])
        dense[idx] = -val
    wins = np.arange(cfg["dist_bin"], cfg["max_nt"] + cfg["dist_bin"], cfg["dist_bin"])
    SEDGES = {m: np.arange(lo, hi + cfg["score_bin"], cfg["score_bin"])
              for m,(lo,hi) in cfg["SRANGE"].items()}
    U1DN_EDGES = np.arange(0, cfg["u1_dn_span"] + cfg["u1_dn_bin"], cfg["u1_dn_bin"])
    _G.update(dict(fa=fa, s5=o[0], s3=o[1], u1dense=dense, MAP=MAP, powv=powv,
                   U1LEN=U1LEN, wins=wins, nW=len(wins), SEDGES=SEDGES,
                   NS={m: len(e)-1 for m,e in SEDGES.items()},
                   STRONG_THR={"acceptor":8.0,"donor":8.0,"u1":0.0},
                   MECHS=["acceptor","u1","donor"], U1DN_EDGES=U1DN_EDGES,
                   nU=len(U1DN_EDGES)-1, cfg=cfg,
                   U1G_BANDS=np.array([(0,50),(50,100),(100,150),(150,200),(200,300),
                                       (300,400),(400,500),(500,700),(700,1000),
                                       (0,100),(0,200),(0,300),(0,500),(100,300),(100,500)]),
                   U1G_THRS=np.array([0.0,2.0,4.0,6.0,8.0,10.0]),
                   ME5G_THRS=np.array([0.0,2.0,4.0,6.5,8.0,10.0])))
    _G["ANN"] = _load_annot_5ss(cfg["annot_5ss"]) if cfg.get("annot_5ss") else {}

def _pas_strength(seq_region):
    s = seq_region.upper()
    for h in ("AATAAA","ATTAAA"):
        if h in s: return 2.0
    for h in ("AGTAAA","TATAAA","CATAAA","GATAAA","AATATA","AATACA",
              "AATAGA","ACTAAA","AAGAAA","AATGAA"):
        if h in s: return 1.0
    return 0.0

def _u1_full_mp(seq):
    import numpy as np
    MAP=_G["MAP"]; U1LEN=_G["U1LEN"]; powv=_G["powv"]; dense=_G["u1dense"]
    arr = MAP[np.frombuffer(seq.encode("ascii","replace"), np.uint8)]; n=len(arr)
    if n < U1LEN: return np.empty(0)
    codes = np.zeros(n-U1LEN+1, np.int64); bad = np.zeros(n-U1LEN+1, bool)
    for k in range(U1LEN):
        c = arr[k:k+(n-U1LEN+1)]; codes += c*powv[k]; bad |= (c<0)
    return np.where(bad, np.nan, dense[np.clip(codes,0,len(dense)-1)])

def _scan_signals_mp(seq, site):
    import numpy as np
    s5=_G["s5"]; s3=_G["s3"]; MECHS=_G["MECHS"]
    L=len(seq); res={m:([],[]) for m in MECHS}
    arr=np.frombuffer(seq.encode("ascii","replace"),dtype=np.uint8)
    ags=np.where((arr[:-1]==65)&(arr[1:]==71))[0]+2
    for i in ags:
        p=int(i)
        if p-20<0 or p+3>L or p>site: continue
        try:
            v=s3(seq[p-20:p+3]); v=float(np.nanmax(np.asarray(v,float))) if not np.isscalar(v) else float(v)
        except Exception: continue
        if np.isfinite(v) and v>-99: res["acceptor"][0].append(site-p); res["acceptor"][1].append(v)
    gts=np.where((arr[:-1]==71)&(arr[1:]==84))[0]
    for i in gts:
        p=int(i)
        if p-3<0 or p+6>L or p>site: continue
        try:
            v=s5(seq[p-3:p+6]); v=float(np.nanmax(np.asarray(v,float))) if not np.isscalar(v) else float(v)
        except Exception: continue
        if np.isfinite(v) and v>-99: res["donor"][0].append(site-p); res["donor"][1].append(v)
    u1v=_u1_full_mp(seq); starts=np.arange(len(u1v)); keep=np.isfinite(u1v)&(starts<=site)
    res["u1"]=((site-starts[keep]).astype(float).tolist(), u1v[keep].tolist())
    return res

def _work_mp(rec):
    import numpy as np
    g=_G; a=g["cfg"]; fa=g["fa"]; MECHS=g["MECHS"]; wins=g["wins"]; nW=g["nW"]
    SEDGES=g["SEDGES"]; NS=g["NS"]; STRONG_THR=g["STRONG_THR"]; U1LEN=g["U1LEN"]
    U1DN_EDGES=g["U1DN_EDGES"]; nU=g["nU"]
    c=int(rec["center"]); strand=rec["strand"]; chrom=str(rec["chrom"]); cls=rec["cls"]
    if strand=="+": w0,w1=max(0,c-a["max_nt"]),c+a["pad"]
    else:           w0,w1=max(0,c-a["pad"]),c+a["max_nt"]
    try: seq=fa.fetch(chrom,w0,w1,strand=strand)
    except Exception: seq=None
    if not seq or len(seq)<30: return None
    seq=seq.upper(); L=len(seq); site=L-a["pad"]
    sig=_scan_signals_mp(seq,site)
    rows={}; dcm={}
    for m in MECHS:
        ds,ss=sig[m]
        if not ds: rows[m]=None; continue
        ds=np.asarray(ds,float); ss=np.asarray(ss,float)
        o=np.argsort(ds); ds=ds[o]; ss=ss[o]
        runmax=np.maximum.accumulate(ss); idx=np.searchsorted(ds,wins,side="right")-1
        row=np.full(nW,np.nan)
        for wi,k in enumerate(idx):
            if k>=0: row[wi]=runmax[k]
        strong=(ss>=STRONG_THR[m]).astype(np.int64); cum=np.cumsum(strong)
        dcount=np.zeros(nW)
        for wi,k in enumerate(idx): dcount[wi]=cum[k] if k>=0 else 0
        rows[m]=(row,dcount)
        if m in DW_MECHS:
            tl=DW_THRS[m]; gmat=np.zeros((len(tl),nW),np.float32)
            for ti_,tv in enumerate(tl):
                cw=np.cumsum((ss>=tv).astype(np.int64))
                for wi,k in enumerate(idx): gmat[ti_,wi]=cw[k] if k>=0 else 0
            dcm[m]=gmat
    ie=dict(acc=np.nan,dn_donor=np.nan,dn_dist=np.nan,dn_u1=np.nan,interv_donor=np.nan,
            interv_dist=np.nan,n_strong_interv=0,interv_sum5=np.nan,interv_density5=np.nan,
            interv_sum5_perkb=np.nan,interv_u1flank=np.nan,
            interv_u1_sum=np.nan,interv_u1_nstrong=0,interv_u1_density=np.nan,interv_u1_sum_perkb=np.nan,
            edef_best5=np.nan, edef_pas=np.nan, edef_compete=np.nan, edef_coupling=np.nan, nearest_dn=np.nan,
            lone5_best=np.nan, unit5_best=np.nan, interv_complete_frac=np.nan,
            interv_n_cand=0, interv_sum_cand=np.nan,
            dn5_best=np.nan, dn5_dist=np.nan, dn5_nearest_strong=np.nan,
            first5_from_anchor=np.nan, internal_exon_like=0,
            annot_dn5_dist=np.nan, annot_dn5_within=0,
            annot_first5_from_anchor=np.nan, annot_internal_exon_like=0,
            anchor_3ss_pos=np.nan, d_anchor_to_pa=np.nan, anchor_up_check=np.nan,
            first5_up_dist=np.nan, first5_up_present=0, first5_up_score=np.nan,
            first5_dn_dist=np.nan, first5_dn_present=0, first5_dn_score=np.nan,
            first5_pastpa_dist=np.nan, first5_pastpa_present=0, first5_pastpa_score=np.nan)
    anchor_up=np.nan
    iprof=np.zeros(IPROF_NB)
    mg5=np.zeros(MG_NB); mgus=np.zeros(MG_NB); mgun=np.zeros(MG_NB)
    ie["edef_pas"] = _pas_strength(seq[max(0,site-50):site+min(a["pad"],10)])
    ad,asc=sig["acceptor"]
    if ad:
        ad=np.asarray(ad,float); asc=np.asarray(asc,float)
        ai=int(np.argmax(asc)); anchor_up=ad[ai]; ie["acc"]=float(asc[ai])
        dd0,vv0=sig["donor"]
        if dd0:
            dd0=np.asarray(dd0,float); vv0=np.asarray(vv0,float); rel0=anchor_up-dd0
            iv=(rel0>0)&(rel0<anchor_up)
            if iv.any():
                vint=vv0[iv]; rint=rel0[iv]; bi=int(np.argmax(vint))
                sm=vint>=STRONG_THR["donor"]
                if sm.any():
                    bb=np.clip(np.searchsorted(IPROF_EDGES,rint[sm],side="right")-1,0,IPROF_NB-1)
                    np.add.at(iprof,bb,1.0)
                ada=np.asarray(sig["acceptor"][0],float); ddint=dd0[iv]
                if ada.size:
                    ada_s=np.sort(ada); loI=ddint-1500.0; hiI=ddint-70.0
                    cmask=(np.searchsorted(ada_s,hiI,side="right")-np.searchsorted(ada_s,loI,side="left"))>0
                else: cmask=np.zeros(ddint.shape,bool)
                if (~cmask).any(): ie["lone5_best"]=float(vint[~cmask].max())
                if cmask.any(): ie["unit5_best"]=float(vint[cmask].max())
                ie["interv_complete_frac"]=float(cmask.mean()) if cmask.size else np.nan
                ie["interv_donor"]=float(vint[bi]); ie["interv_dist"]=float(rint[bi])
                ie["n_strong_interv"]=int((vint>=STRONG_THR["donor"]).sum())
                ie["interv_sum5"]=float(np.sum(np.clip(vint,0,None)))
                cand=vint>=a["me5_cand"]
                ie["interv_n_cand"]=int(cand.sum())
                ie["interv_sum_cand"]=float(vint[cand].sum()) if cand.any() else 0.0
                exon_kb=max(anchor_up,1)/1000.0
                ie["interv_density5"]=float(ie["n_strong_interv"]/exon_kb)
                ie["interv_sum5_perkb"]=float(ie["interv_sum5"]/exon_kb)
                du1,vu1=sig["u1"]
                if du1:
                    du1=np.asarray(du1,float); vu1=np.asarray(vu1,float); relu=anchor_up-du1
                    near=np.abs(relu-rint[bi])<=50
                    if near.any(): ie["interv_u1flank"]=float(np.max(vu1[near]))
                    ivu=(relu>0)&(relu<anchor_up)
                    if ivu.any():
                        vintu=vu1[ivu]
                        ie["interv_u1_sum"]=float(np.sum(np.clip(vintu,0,None)))
                        ie["interv_u1_nstrong"]=int((vintu>=STRONG_THR["u1"]).sum())
                        ie["interv_u1_density"]=float(ie["interv_u1_nstrong"]/exon_kb)
                        ie["interv_u1_sum_perkb"]=float(ie["interv_u1_sum"]/exon_kb)
        for mm,key in (("donor","dn_donor"),("u1","dn_u1")):
            dd,vv=sig[mm]
            if not dd: continue
            dd=np.asarray(dd,float); vv=np.asarray(vv,float); rel=anchor_up-dd
            win=(rel>=a["dn_lo"])&(rel<=a["dn_hi"])
            if win.any(): ie[key]=float(np.max(vv[win]))
            if mm=="donor":
                pwin=(rel>=20)&(rel<=a["per_max"])
                if pwin.any(): ie["dn_dist"]=float(rel[pwin][int(np.argmax(vv[pwin]))])
                strong_dn = (rel > 0) & (vv >= a["nd_thr"])
                if strong_dn.any():
                    ie["nearest_dn"] = float(np.min(rel[strong_dn]))
        # exon-definition measures (window = edef_lo..edef_hi past the anchor)
        dd,vv=sig["donor"]
        if dd:
            dd=np.asarray(dd,float); vv=np.asarray(vv,float); rel=anchor_up-dd
            ew=(rel>=a["edef_lo"])&(rel<=a["edef_hi"])
            if ew.any():
                best5=float(np.max(vv[ew]))
                ie["edef_best5"]=best5
                ie["edef_compete"]=best5 - 3.0*ie["edef_pas"]
                ie["edef_coupling"]=best5 * ie["acc"]
    u1prof=None; u1v=np.empty(0)
    if strand=="+": d0,d1=c,c+a["u1_dn_span"]
    else:           d0,d1=max(0,c-a["u1_dn_span"]),c
    try: dn_seq=fa.fetch(chrom,d0,d1,strand=strand)
    except Exception: dn_seq=None
    if dn_seq: dn_seq=dn_seq.upper()
    if dn_seq and len(dn_seq)>=U1LEN:
        u1v=_u1_full_mp(dn_seq)
        if len(u1v):
            pos=np.arange(len(u1v)); keep=np.isfinite(u1v)&(pos<=a["u1_dn_span"])
            if keep.any():
                bi=np.clip(np.searchsorted(U1DN_EDGES,pos[keep],side="right")-1,0,nU-1)
                u1prof=(bi,u1v[keep])
    # search grid: density (sites/kb >= thr) per band past the anchor 3'ss
    u1grid=None; me5grid=None
    if np.isfinite(ie["acc"]):
        B=_G["U1G_BANDS"]
        def _grid(pos_vals, thrs):
            dd, vv = pos_vals
            if not dd: return None
            dd=np.asarray(dd,float); vv=np.asarray(vv,float); rel=anchor_up-dd
            g=np.full((len(B),len(thrs)), np.nan)
            for bi_ in range(len(B)):
                lo,hi=B[bi_]; m=(rel>=lo)&(rel<hi); kb=max(hi-lo,1)/1000.0
                if m.any():
                    seg=vv[m]
                    for ti in range(len(thrs)): g[bi_,ti]=float((seg>=thrs[ti]).sum())/kb
                else: g[bi_,:]=0.0
            return g
        u1grid  = _grid(sig["u1"],    _G["U1G_THRS"])
        me5grid = _grid(sig["donor"], _G["ME5G_THRS"])
    # downstream of the site: score 5'ss on dn_seq (position 0 = the site, + = downstream)
    dn5_pos=np.empty(0); dn5_sc=np.empty(0)
    if dn_seq and len(dn_seq)>=9:
        arrd=np.frombuffer(dn_seq.encode("ascii","replace"),np.uint8)
        gtd=np.where((arrd[:-1]==71)&(arrd[1:]==84))[0]
        pp=[]; ss5=[]
        for i in gtd:
            p=int(i)
            if p-3<0 or p+6>len(dn_seq): continue
            try:
                v=g["s5"](dn_seq[p-3:p+6]); v=float(np.nanmax(np.asarray(v,float))) if not np.isscalar(v) else float(v)
            except Exception: continue
            if np.isfinite(v) and v>-99: pp.append(p); ss5.append(v)
        dn5_pos=np.asarray(pp,float); dn5_sc=np.asarray(ss5,float)
    if dn5_sc.size:
        bi=int(np.argmax(dn5_sc)); ie["dn5_best"]=float(dn5_sc[bi]); ie["dn5_dist"]=float(dn5_pos[bi])
        sdn=dn5_sc>=a["nd_thr"]
        if sdn.any(): ie["dn5_nearest_strong"]=float(dn5_pos[sdn].min())
    cand_anchor=[]
    if np.isfinite(ie["nearest_dn"]): cand_anchor.append(ie["nearest_dn"])
    if dn5_sc.size and np.isfinite(anchor_up):
        sdn=dn5_sc>=a["nd_thr"]
        if sdn.any(): cand_anchor.append(float(anchor_up+dn5_pos[sdn].min()))
    if cand_anchor:
        ie["first5_from_anchor"]=float(min(cand_anchor))
        ie["internal_exon_like"]=1 if (50.0<=ie["first5_from_anchor"]<=300.0) else 0
    _positional_feats(ie, sig, anchor_up, dn5_pos, dn5_sc, c, strand, a["nd_thr"])
    ie.update(_annot_feats(g.get("ANN") or {}, chrom, c, strand, anchor_up))
    # metagene (site-centered): strong-5'ss density + u1 mean, both sides
    dU,sU=sig["donor"]
    if dU:
        dU=np.asarray(dU,float); sU=np.asarray(sU,float); st=(sU>=a["nd_thr"])&(dU<=MG_SPAN)
        if st.any():
            bb=np.clip(np.searchsorted(MG_EDGES,-dU[st],side="right")-1,0,MG_NB-1); np.add.at(mg5,bb,1.0)
    duU,vuU=sig["u1"]
    if duU:
        duU=np.asarray(duU,float); vuU=np.asarray(vuU,float); mk=duU<=MG_SPAN
        if mk.any():
            bb=np.clip(np.searchsorted(MG_EDGES,-duU[mk],side="right")-1,0,MG_NB-1)
            np.add.at(mgus,bb,vuU[mk]); np.add.at(mgun,bb,1.0)
    if dn5_sc.size:
        st=(dn5_sc>=a["nd_thr"])&(dn5_pos<=MG_SPAN)
        if st.any():
            bb=np.clip(np.searchsorted(MG_EDGES,dn5_pos[st],side="right")-1,0,MG_NB-1); np.add.at(mg5,bb,1.0)
    if dn_seq and len(u1v):
        posd=np.arange(len(u1v)); kd=np.isfinite(u1v)&(posd<=MG_SPAN)
        if kd.any():
            bb=np.clip(np.searchsorted(MG_EDGES,posd[kd].astype(float),side="right")-1,0,MG_NB-1)
            np.add.at(mgus,bb,u1v[kd]); np.add.at(mgun,bb,1.0)
    _du,_vu = sig["u1"]
    _ru = -np.asarray(_du,float) if _du else np.empty(0)
    _vu = np.asarray(_vu,float) if _du else np.empty(0)
    _rd = np.arange(len(u1v),dtype=float) if len(u1v) else np.empty(0)
    ucube,dwu,dwd = _u1_cube(_ru,_vu,_rd,u1v if len(u1v) else np.empty(0))
    ie["u1_dw_up"]=dwu; ie["u1_dw_dn"]=dwd
    _dd5,_vv5 = sig["donor"]
    _r5u = -np.asarray(_dd5,float) if _dd5 else np.empty(0)
    _v5u = np.asarray(_vv5,float) if _dd5 else np.empty(0)
    mcube,_,_ = _u1_cube(_r5u,_v5u,dn5_pos,dn5_sc,thrs=MC_THRS)
    ie["chrom"]=chrom; ie["center"]=int(c); ie["strand"]=strand
    return (cls,str(rec["gene"]),rows,ie,u1prof,u1grid,me5grid,iprof,mg5,mgus,mgun,ucube,mcube,dcm)


POSITIONAL_KEYS = (
    "anchor_3ss_pos", "d_anchor_to_pa", "anchor_up_check",
    "first5_up_dist", "first5_up_present", "first5_up_score",
    "first5_dn_dist", "first5_dn_present", "first5_dn_score",
    "first5_pastpa_dist", "first5_pastpa_present", "first5_pastpa_score",
)


def _positional_feats(ie, sig, anchor_up, dn5_pos, dn5_sc, c, strand, nd_thr):
    import numpy as np
    if not np.isfinite(anchor_up):
        return
    ie["anchor_3ss_pos"] = float(c - anchor_up) if strand == "+" else float(c + anchor_up)
    ie["d_anchor_to_pa"] = float(anchor_up)
    ie["anchor_up_check"] = float(anchor_up)
    dd, vv = sig["donor"]
    if dd:
        dd = np.asarray(dd, float); vv = np.asarray(vv, float)
        rel = anchor_up - dd
        strong = vv >= nd_thr
        up = strong & (rel < 0)
        if up.any():
            j = int(np.argmin(-rel[up]))
            ie["first5_up_dist"] = float((-rel[up])[j])
            ie["first5_up_score"] = float(vv[up][j])
            ie["first5_up_present"] = 1
        dn = strong & (rel > 0)
        if dn.any():
            j = int(np.argmin(rel[dn]))
            ie["first5_dn_dist"] = float(rel[dn][j])
            ie["first5_dn_score"] = float(vv[dn][j])
            ie["first5_dn_present"] = 1
    if dn5_sc is not None and getattr(dn5_sc, "size", 0):
        s = dn5_sc >= nd_thr
        if s.any():
            j = int(np.argmin(dn5_pos[s]))
            ie["first5_pastpa_dist"] = float(dn5_pos[s][j])
            ie["first5_pastpa_score"] = float(dn5_sc[s][j])
            ie["first5_pastpa_present"] = 1

def _load_annot_5ss(path):
    import numpy as np, pandas as pd, gzip, sys
    D = {}
    try:
        if any(path.endswith(x) for x in (".gtf",".gtf.gz",".gff",".gff3",".gff.gz",".gff3.gz")):
            op = gzip.open if path.endswith(".gz") else open
            tmp = {}
            with op(path, "rt") as fh:
                for line in fh:
                    if not line or line[0] == "#": continue
                    f = line.rstrip("\n").split("\t")
                    if len(f) < 8 or f[2] != "exon": continue
                    chrom, start, end, strand = f[0], int(f[3])-1, int(f[4]), f[6]
                    donor = end if strand == "+" else start
                    tmp.setdefault((chrom, strand), []).append(donor)
            for k, v in tmp.items(): D[k] = np.array(sorted(set(v)), dtype=np.int64)
        else:
            df = pd.read_csv(path, sep=None, engine="python")
            low = {c.lower(): c for c in df.columns}
            def pk(*names):
                for n in names:
                    if n in low: return low[n]
                return None
            cc = pk("chrom","chr","seqname","seqnames","#chrom"); sc = pk("strand")
            dc = pk("donor","5ss","five_prime_ss","pos","position")
            stc = pk("start","exon_start"); ec = pk("end","exon_end")
            if cc is None or sc is None:
                df = pd.read_csv(path, sep="\t", header=None)
                if df.shape[1] >= 6: cc, stc, ec, sc = 0, 1, 2, 5
                else: raise ValueError("could not identify chrom/strand columns")
            for (chrom, strand), sub in df.groupby([cc, sc]):
                if dc is not None:
                    pos = sub[dc].astype(np.int64).values
                else:
                    st = sub[stc].astype(np.int64).values; en = sub[ec].astype(np.int64).values
                    pos = en if str(strand) == "+" else st
                D[(str(chrom), str(strand))] = np.array(sorted(set(pos.tolist())), dtype=np.int64)
        nsite = sum(len(v) for v in D.values())
        print(f"[annot-5ss] loaded {nsite} donor sites, {len(D)} (chrom,strand) keys from {path}", file=sys.stderr)
        print(f"[annot-5ss] example keys: {list(D.keys())[:3]} (must match FASTA naming, e.g. 'chr1')", file=sys.stderr)
    except Exception as e:
        print(f"[annot-5ss] WARNING failed to load {path}: {e}; annotated test disabled", file=sys.stderr)
        return {}
    return D

def _annot_feats(ANN, chrom, c, strand, anchor_up):
    import numpy as np
    out = dict(annot_dn5_dist=np.nan, annot_dn5_within=0,
               annot_first5_from_anchor=np.nan, annot_internal_exon_like=0)
    arr = ANN.get((chrom, strand))
    if arr is None or not np.isfinite(anchor_up): return out
    lo = np.searchsorted(arr, c-2000); hi = np.searchsorted(arr, c+2000)
    near = arr[lo:hi]
    if near.size == 0: return out
    d = (near.astype(float)-c) if strand == "+" else (c-near.astype(float))
    dn = d[d > 0]
    if dn.size:
        out["annot_dn5_dist"] = float(dn.min())
        out["annot_dn5_within"] = 1 if (20.0 <= dn.min() <= 400.0) else 0
    fanc = d + anchor_up; fanc = fanc[fanc > 0]
    if fanc.size:
        out["annot_first5_from_anchor"] = float(fanc.min())
        out["annot_internal_exon_like"] = 1 if (50.0 <= fanc.min() <= 300.0) else 0
    return out


def auroc_auprc(pos, neg):
    s = np.concatenate([pos, neg]); y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s)); r[o] = np.arange(1, len(s)+1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    cs = np.cumsum(cnt); r = ((cs - cnt) + cs + 1)[inv] / 2.0
    auroc = (r[y==1].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))
    order = np.argsort(-s); yy = y[order]
    tp = np.cumsum(yy); fp = np.cumsum(1-yy)
    prec = tp / np.maximum(tp+fp, 1); rec = tp / max(tp[-1], 1)
    _trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auprc = float(_trap(prec, rec))
    return float(auroc), auprc, float(y.mean())

def bcshow_tables(dw, thrs, wins, y, acc_rows, u1_thr=4.0,
                  donor_search=(3.0, 4.0, 6.5, 8.0), nybin=14):
    import numpy as np, pandas as pd
    y = np.asarray(y).astype(int)
    wins = np.asarray(wins, float)

    def _au(score):
        pos = score[y == 1]; neg = score[y == 0]
        if len(pos) < 20 or len(neg) < 20: return np.nan
        return auroc_auprc(pos, neg)[0]

    picks = {}
    for mech in dw:
        T = np.asarray(thrs[mech], float)
        if mech == "u1":
            ti = int(np.argmin(np.abs(T - u1_thr)))
            picks[mech] = (ti, float(T[ti]), "grid-search winner")
            continue
        best = None
        for c in donor_search:
            ti = int(np.argmin(np.abs(T - c)))
            aus = [_au(dw[mech][:, ti, wi] / (wins[wi] / 1000.0)) for wi in range(len(wins))]
            disc = np.nanmax(np.abs(np.asarray(aus, float) - 0.5))
            print(f"    [bcshow] {mech} thr {T[ti]:>4.1f}: peak |AUROC-0.5| = {disc:.4f}", file=sys.stderr)
            if best is None or disc > best[0]: best = (disc, ti, float(T[ti]))
        picks[mech] = (best[1], best[2],
                       "max |AUROC-0.5| over " + ",".join(f"{c:g}" for c in donor_search))
    for m, (_ti, tv, why) in picks.items():
        print(f"    [bcshow] pick {m}: threshold {tv:g} ({why})", file=sys.stderr)

    rows = [dict(mech="acceptor", window=float(r["window"]), auroc_raw=float(r["auroc"]),
                 auroc=float(r["auroc"]), predicts="true end", thr=np.nan,
                 encoding="running-max peak strength")
            for r in acc_rows if r["mech"] == "acceptor"]
    for mech, (ti, tv, _why) in picks.items():
        for wi, w in enumerate(wins):
            raw = _au(dw[mech][:, ti, wi] / (w / 1000.0))
            if not np.isfinite(raw): continue
            rows.append(dict(mech=mech, window=float(w), auroc_raw=raw,
                             auroc=1.0 - raw if raw < 0.5 else raw,
                             predicts="competitor" if raw < 0.5 else "true end", thr=tv,
                             encoding=f"strong-site density/kb, threshold {tv:g}"))
    AU = pd.DataFrame(rows)

    hrows = []
    n1 = max(int((y == 1).sum()), 1); n0 = max(int((y == 0).sum()), 1)
    eps = 1.0 / max(n1, n0)
    for mech, (ti, tv, _why) in picks.items():
        dens = dw[mech][:, ti, :] / (wins[None, :] / 1000.0)
        hi = float(np.nanpercentile(dens, 99.0))
        edges = np.linspace(0.0, hi if hi > 0 else 1.0, nybin + 1)
        ctr = (edges[:-1] + edges[1:]) / 2.0
        for wi, w in enumerate(wins):
            bi = np.clip(np.searchsorted(edges, dens[:, wi], side="right") - 1, 0, nybin - 1)
            for j in range(nybin):
                m_ = bi == j
                n = int(m_.sum())
                if n == 0: continue
                pc = float((m_ & (y == 1)).sum()) / n1
                pk = float((m_ & (y == 0)).sum()) / n0
                hrows.append(dict(mech=mech, window=float(w), yval=float(ctr[j]),
                                  log2ratio=float(np.log2((pc + eps) / (pk + eps))),
                                  n=n, strong_thr=tv))
    return AU, pd.DataFrame(hrows), picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scripts_dir", default=None)
    ap.add_argument("--shortlist", required=True)
    ap.add_argument("--reference_fasta", required=True)
    ap.add_argument("--u1_lookup", required=True)
    ap.add_argument("--me5_lookup", required=True)
    ap.add_argument("--max_nt", type=int, default=2000)
    ap.add_argument("--pad", type=int, default=10)
    ap.add_argument("--dist_bin", type=int, default=50)
    ap.add_argument("--score_bin", type=float, default=0.5)
    ap.add_argument("--score_lo", type=float, default=-15.0)
    ap.add_argument("--score_hi", type=float, default=14.0)
    ap.add_argument("--max_per_class", type=int, default=8000)
    ap.add_argument("--imbalanced", action="store_true")
    ap.add_argument("--min_cell", type=int, default=30)
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--dn_lo", type=int, default=70, help="downstream-donor window start (nt past anchor)")
    ap.add_argument("--dn_hi", type=int, default=300, help="downstream-donor window end")
    ap.add_argument("--per_max", type=int, default=500, help="max anchor->downstream-donor distance for periodicity")
    ap.add_argument("--u1_dn_span", type=int, default=500, help="U1 profile span downstream of the peak's own site")
    ap.add_argument("--u1_dn_bin", type=int, default=25)
    ap.add_argument("--edef_lo", type=int, default=50, help="exon-definition window start past anchor (nt)")
    ap.add_argument("--edef_hi", type=int, default=300, help="exon-definition window end past anchor (nt)")
    ap.add_argument("--nd_thr", type=float, default=8.0,
                    help="MaxEntScan 5'ss threshold for a STRONG downstream donor "
                         "(literature: <6 weak, >8 strong; Wang/Yeo-Burge)")
    ap.add_argument("--me5_cand", type=float, default=3.0,
                    help="MaxEntScan 5'ss threshold for a CANDIDATE (plausible U1-binding) 5'ss; "
                         "sites below this are near-background noise (Yeo & Burge 2004, log-odds bits)")
    ap.add_argument("--annot_5ss", default=None,
                    help="optional: GENCODE GTF(.gz) or a table of ANNOTATED 5'ss (donor) sites "
                         "(chrom/strand + donor|pos or start/end). Enables the annotated internal-exon test.")
    ap.add_argument("--strong_acc", type=float, default=8.0,
                    help="MaxEntScan 3'ss threshold for a STRONG anchor acceptor (candidate real exon start), "
                         "used to condition the U1-relay conjunction test")
    ap.add_argument("--bcshow_u1_thr", type=float, default=4.0,
                    help="U1 cutoff for the between-class showcase (grid-search winner)")
    ap.add_argument("--bcshow_donor_search", default="3,4,6.5,8",
                    help="donor cutoffs searched; the one maximising |AUROC-0.5| wins")
    ap.add_argument("--bcshow_nybin", type=int, default=14,
                    help="density bins on the showcase heatmap y axis")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="fig3")
    a = ap.parse_args()
    if a.scripts_dir: sys.path.insert(0, a.scripts_dir)
    from extract_splicing_features import (GenomeFasta, init_maxentscan,
                                           load_u1_lookup, load_me5_lookup)
    import threading
    _tls = threading.local()
    def _scorers():
        if not hasattr(_tls, "s3"):
            o = init_maxentscan(); _tls.s5, _tls.s3 = o[0], o[1]
        return _tls.s5, _tls.s3
    _m = init_maxentscan(); score5 = _m[0]; score3 = _m[1]
    fa = GenomeFasta(a.reference_fasta)
    def _fa():
        if not hasattr(_tls, "fa"):
            _tls.fa = GenomeFasta(a.reference_fasta)
        return _tls.fa
    u1 = load_u1_lookup(a.u1_lookup); me5 = load_me5_lookup(a.me5_lookup)
    ANN_MAIN = _load_annot_5ss(a.annot_5ss) if (a.annot_5ss and a.n_jobs <= 1) else {}
    if len(u1) == 0: sys.exit("[err] U1 lookup is empty")
    U1LEN = len(next(iter(u1)))
    _MAP = np.full(256, -1, np.int64)
    for b,code in zip(b"ACGT", range(4)): _MAP[b] = code
    u1dense = np.full(4**U1LEN, np.nan, np.float32)
    _pow = (4**np.arange(U1LEN-1, -1, -1)).astype(np.int64)
    for kmer, val in u1.items():
        idx = 0
        for ch in kmer.encode(): idx = idx*4 + int(_MAP[ch])
        u1dense[idx] = -val

    def u1_full(seq):
        arr = _MAP[np.frombuffer(seq.encode("ascii","replace"), np.uint8)]
        n = len(arr)
        if n < U1LEN: return np.empty(0)
        codes = np.zeros(n-U1LEN+1, np.int64); bad = np.zeros(n-U1LEN+1, bool)
        for k in range(U1LEN):
            c = arr[k:k+(n-U1LEN+1)]; codes += c*_pow[k]; bad |= (c<0)
        return np.where(bad, np.nan, u1dense[np.clip(codes,0,len(u1dense)-1)])

    df = (pd.read_csv(a.shortlist, sep="\t") if a.shortlist.endswith((".tsv",".txt"))
          else pd.read_parquet(a.shortlist))
    df = df[df.n_internal > 0]
    df["cls"] = np.where(df.is_canonical, "Canonical", "Competitor")
    if a.imbalanced:
        parts = [gg.sample(min(len(gg), a.max_per_class), random_state=a.seed) for _, gg in df.groupby("cls")]
        df = pd.concat(parts, ignore_index=True)
        print(f"[data] {len(df):,} peaks (IMBALANCED base {df.is_canonical.mean():.3f}); upstream {a.max_nt}nt", file=sys.stderr)
    else:
        nt = min(a.max_per_class, int(df.cls.value_counts().min()))
        parts = [gg.sample(nt, random_state=a.seed) if len(gg) > nt else gg for _, gg in df.groupby("cls")]
        df = pd.concat(parts, ignore_index=True)
        print(f"[data] {len(df):,} peaks (balanced {nt:,}/class); upstream {a.max_nt}nt", file=sys.stderr)

    wins = np.arange(a.dist_bin, a.max_nt + a.dist_bin, a.dist_bin)
    MECHS = ["acceptor", "u1", "donor"]
    SRANGE = {"acceptor": (a.score_lo, a.score_hi), "donor": (a.score_lo, a.score_hi), "u1": (-5.0, 12.0)}
    SEDGES = {m: np.arange(lo, hi + a.score_bin, a.score_bin) for m,(lo,hi) in SRANGE.items()}
    SCTR = {m: (e[:-1]+e[1:])/2 for m,e in SEDGES.items()}; NS = {m: len(e)-1 for m,e in SEDGES.items()}
    STRONG_THR = {"acceptor": 8.0, "donor": 8.0, "u1": 0.0}; nW = len(wins)
    RM = {m: {"Canonical": [], "Competitor": []} for m in MECHS}
    DEN = {m: {"Canonical": [], "Competitor": []} for m in MECHS}
    GENE = {"Canonical": [], "Competitor": []}
    acc = {m: {"Canonical": np.zeros((nW, NS[m])), "Competitor": np.zeros((nW, NS[m]))} for m in MECHS}
    npk = {"Canonical": 0, "Competitor": 0}; n_bad = 0

    IE = {"Canonical": [], "Competitor": []}
    U1DN_EDGES = np.arange(0, a.u1_dn_span + a.u1_dn_bin, a.u1_dn_bin)
    nU = len(U1DN_EDGES) - 1
    u1dn_sum = {c: np.zeros(nU) for c in ("Canonical","Competitor")}
    U1G_BANDS = np.array([(0,50),(50,100),(100,150),(150,200),(200,300),
                          (300,400),(400,500),(500,700),(700,1000),
                          (0,100),(0,200),(0,300),(0,500),(100,300),(100,500)])
    U1G_THRS = np.array([0.0,2.0,4.0,6.0,8.0,10.0])
    ME5G_THRS = np.array([0.0,2.0,4.0,6.5,8.0,10.0])
    DWIN = {m: {c: [] for c in ("Canonical","Competitor")} for m in DW_MECHS}
    UCUBE = {c: [] for c in ("Canonical","Competitor")}
    MCUBE = {c: [] for c in ("Canonical","Competitor")}
    u1grid_vals = {c: [] for c in ("Canonical","Competitor")}
    me5grid_vals = {c: [] for c in ("Canonical","Competitor")}
    u1dn_cnt = {c: np.zeros(nU) for c in ("Canonical","Competitor")}
    IPROF = {c: np.zeros(IPROF_NB) for c in ("Canonical","Competitor")}
    MG5 = {c: np.zeros(MG_NB) for c in ("Canonical","Competitor")}
    MGUS = {c: np.zeros(MG_NB) for c in ("Canonical","Competitor")}
    MGUN = {c: np.zeros(MG_NB) for c in ("Canonical","Competitor")}

    def scan_signals(seq, site):
        score5, score3 = _scorers()
        L = len(seq); res = {m: ([], []) for m in MECHS}
        up_arr = np.frombuffer(seq.encode("ascii", "replace"), dtype=np.uint8)
        ags = np.where((up_arr[:-1] == 65) & (up_arr[1:] == 71))[0] + 2
        for i in ags:
            p = int(i)
            if p-20 < 0 or p+3 > L or p > site: continue
            try:
                v = score3(seq[p-20:p+3]); v = float(np.nanmax(np.asarray(v, float))) if not np.isscalar(v) else float(v)
            except Exception: continue
            if np.isfinite(v) and v > -99: res["acceptor"][0].append(site-p); res["acceptor"][1].append(v)
        gts = np.where((up_arr[:-1] == 71) & (up_arr[1:] == 84))[0]
        for i in gts:
            p = int(i)
            if p-3 < 0 or p+6 > L or p > site: continue
            try:
                v = score5(seq[p-3:p+6]); v = float(np.nanmax(np.asarray(v, float))) if not np.isscalar(v) else float(v)
            except Exception: continue
            if np.isfinite(v) and v > -99: res["donor"][0].append(site-p); res["donor"][1].append(v)
        u1v = u1_full(seq)
        starts = np.arange(len(u1v))
        keep = np.isfinite(u1v) & (starts <= site)
        res["u1"] = ((site - starts[keep]).astype(float).tolist(), u1v[keep].tolist())
        return res

    def _work(rec):
        c = int(rec["center"]); strand = rec["strand"]; chrom = str(rec["chrom"]); cls = rec["cls"]
        if strand == "+": w0, w1 = max(0, c - a.max_nt), c + a.pad
        else:             w0, w1 = max(0, c - a.pad), c + a.max_nt
        try: seq = _fa().fetch(chrom, w0, w1, strand=strand)
        except Exception: seq = None
        if not seq or len(seq) < 30: return None
        seq = seq.upper(); L = len(seq); site = L - a.pad
        up_seq = seq
        sig = scan_signals(up_seq, site)
        rows = {}; dcm = {}
        for m in MECHS:
            ds, ss = sig[m]
            if not ds: rows[m] = None; continue
            ds = np.asarray(ds, float); ss = np.asarray(ss, float)
            o = np.argsort(ds); ds = ds[o]; ss = ss[o]
            runmax = np.maximum.accumulate(ss)
            idx = np.searchsorted(ds, wins, side="right") - 1
            row = np.full(nW, np.nan)
            for wi, k in enumerate(idx):
                if k >= 0: row[wi] = runmax[k]
            strong = (ss >= STRONG_THR[m]).astype(np.int64); cum = np.cumsum(strong)
            dcount = np.zeros(nW)
            for wi, k in enumerate(idx):
                dcount[wi] = cum[k] if k >= 0 else 0
            rows[m] = (row, dcount)
            if m in DW_MECHS:
                tl = DW_THRS[m]; gmat = np.zeros((len(tl), nW), np.float32)
                for ti_, tv in enumerate(tl):
                    cw = np.cumsum((ss >= tv).astype(np.int64))
                    for wi, k in enumerate(idx): gmat[ti_, wi] = cw[k] if k >= 0 else 0
                dcm[m] = gmat
        ie = dict(acc=np.nan, dn_donor=np.nan, dn_dist=np.nan, dn_u1=np.nan,
                  interv_donor=np.nan, interv_dist=np.nan, n_strong_interv=0,
                  interv_sum5=np.nan, interv_density5=np.nan, interv_sum5_perkb=np.nan,
                  interv_u1flank=np.nan,
                  interv_u1_sum=np.nan, interv_u1_nstrong=0, interv_u1_density=np.nan,
                  interv_u1_sum_perkb=np.nan,
                  edef_best5=np.nan, edef_pas=np.nan, edef_compete=np.nan, edef_coupling=np.nan,
                  nearest_dn=np.nan,
                  lone5_best=np.nan, unit5_best=np.nan, interv_complete_frac=np.nan,
                  interv_n_cand=0, interv_sum_cand=np.nan,
                  dn5_best=np.nan, dn5_dist=np.nan, dn5_nearest_strong=np.nan,
                  first5_from_anchor=np.nan, internal_exon_like=0,
                  annot_dn5_dist=np.nan, annot_dn5_within=0,
                  annot_first5_from_anchor=np.nan, annot_internal_exon_like=0,
                  anchor_3ss_pos=np.nan, d_anchor_to_pa=np.nan, anchor_up_check=np.nan,
                  first5_up_dist=np.nan, first5_up_present=0, first5_up_score=np.nan,
                  first5_dn_dist=np.nan, first5_dn_present=0, first5_dn_score=np.nan,
                  first5_pastpa_dist=np.nan, first5_pastpa_present=0, first5_pastpa_score=np.nan)
        anchor_up=np.nan
        iprof=np.zeros(IPROF_NB)
        mg5=np.zeros(MG_NB); mgus=np.zeros(MG_NB); mgun=np.zeros(MG_NB)
        ie["edef_pas"] = _pas_strength(seq[max(0,site-50):site+min(a.pad,10)])
        ad, asc = sig["acceptor"]
        if ad:
            ad = np.asarray(ad, float); asc = np.asarray(asc, float)
            ai = int(np.argmax(asc)); anchor_up = ad[ai]; ie["acc"] = float(asc[ai])
            dd0, vv0 = sig["donor"]
            if dd0:
                dd0 = np.asarray(dd0, float); vv0 = np.asarray(vv0, float)
                rel0 = anchor_up - dd0
                iv = (rel0 > 0) & (rel0 < anchor_up)
                if iv.any():
                    vint = vv0[iv]; rint = rel0[iv]
                    bi = int(np.argmax(vint))
                    sm = vint >= STRONG_THR["donor"]
                    if sm.any():
                        bb = np.clip(np.searchsorted(IPROF_EDGES, rint[sm], side="right")-1, 0, IPROF_NB-1)
                        np.add.at(iprof, bb, 1.0)
                    ada = np.asarray(sig["acceptor"][0], float); ddint = dd0[iv]
                    if ada.size:
                        ada_s = np.sort(ada); loI = ddint-1500.0; hiI = ddint-70.0
                        cmask = (np.searchsorted(ada_s,hiI,side="right")-np.searchsorted(ada_s,loI,side="left")) > 0
                    else: cmask = np.zeros(ddint.shape, bool)
                    if (~cmask).any(): ie["lone5_best"] = float(vint[~cmask].max())
                    if cmask.any(): ie["unit5_best"] = float(vint[cmask].max())
                    ie["interv_complete_frac"] = float(cmask.mean()) if cmask.size else np.nan
                    ie["interv_donor"] = float(vint[bi])
                    ie["interv_dist"]  = float(rint[bi])
                    ie["n_strong_interv"] = int((vint >= STRONG_THR["donor"]).sum())
                    ie["interv_sum5"] = float(np.sum(np.clip(vint, 0, None)))
                    cand = vint >= a.me5_cand
                    ie["interv_n_cand"] = int(cand.sum())
                    ie["interv_sum_cand"] = float(vint[cand].sum()) if cand.any() else 0.0
                    exon_kb = max(anchor_up, 1) / 1000.0
                    ie["interv_density5"] = float(ie["n_strong_interv"] / exon_kb)
                    ie["interv_sum5_perkb"] = float(ie["interv_sum5"] / exon_kb)
                    du1, vu1 = sig["u1"]
                    if du1:
                        du1 = np.asarray(du1, float); vu1 = np.asarray(vu1, float)
                        relu = anchor_up - du1
                        near = np.abs(relu - rint[bi]) <= 50
                        if near.any(): ie["interv_u1flank"] = float(np.max(vu1[near]))
                        ivu = (relu > 0) & (relu < anchor_up)
                        if ivu.any():
                            vintu = vu1[ivu]
                            ie["interv_u1_sum"] = float(np.sum(np.clip(vintu, 0, None)))
                            ie["interv_u1_nstrong"] = int((vintu >= STRONG_THR["u1"]).sum())
                            ie["interv_u1_density"] = float(ie["interv_u1_nstrong"] / exon_kb)
                            ie["interv_u1_sum_perkb"] = float(ie["interv_u1_sum"] / exon_kb)
            for mm, key in (("donor","dn_donor"), ("u1","dn_u1")):
                dd, vv = sig[mm]
                if not dd: continue
                dd = np.asarray(dd, float); vv = np.asarray(vv, float)
                rel = anchor_up - dd
                win = (rel >= a.dn_lo) & (rel <= a.dn_hi)
                if win.any():
                    ie[key] = float(np.max(vv[win]))
                if mm == "donor":
                    pwin = (rel >= 20) & (rel <= a.per_max)
                    if pwin.any():
                        ie["dn_dist"] = float(rel[pwin][int(np.argmax(vv[pwin]))])
                    strong_dn = (rel > 0) & (vv >= a.nd_thr)
                    if strong_dn.any():
                        ie["nearest_dn"] = float(np.min(rel[strong_dn]))
            dd2, vv2 = sig["donor"]
            if dd2:
                dd2 = np.asarray(dd2, float); vv2 = np.asarray(vv2, float); rel2 = anchor_up - dd2
                ew = (rel2 >= a.edef_lo) & (rel2 <= a.edef_hi)
                if ew.any():
                    best5 = float(np.max(vv2[ew]))
                    ie["edef_best5"] = best5
                    ie["edef_compete"] = best5 - 3.0*ie["edef_pas"]
                    ie["edef_coupling"] = best5 * ie["acc"]
        u1prof = None; u1v = np.empty(0)
        if strand == "+": d0, d1 = c, c + a.u1_dn_span
        else:             d0, d1 = max(0, c - a.u1_dn_span), c
        try: dn_seq = _fa().fetch(chrom, d0, d1, strand=strand)
        except Exception: dn_seq = None
        if dn_seq: dn_seq = dn_seq.upper()
        if dn_seq and len(dn_seq) >= U1LEN:
            u1v = u1_full(dn_seq)
            if len(u1v):
                pos = np.arange(len(u1v))
                keep = np.isfinite(u1v) & (pos <= a.u1_dn_span)
                if keep.any():
                    bi = np.clip(np.searchsorted(U1DN_EDGES, pos[keep], side="right")-1, 0, nU-1)
                    u1prof = (bi, u1v[keep])
        u1grid = None; me5grid = None
        if np.isfinite(ie["acc"]):
            def _grid(pv, thrs):
                dd, vv = pv
                if not dd: return None
                dd = np.asarray(dd, float); vv = np.asarray(vv, float); rel = anchor_up - dd
                g = np.full((len(U1G_BANDS), len(thrs)), np.nan)
                for bi_ in range(len(U1G_BANDS)):
                    lo, hi = U1G_BANDS[bi_]; mm2 = (rel >= lo) & (rel < hi); kb = max(hi-lo,1)/1000.0
                    if mm2.any():
                        seg = vv[mm2]
                        for ti in range(len(thrs)): g[bi_,ti] = float((seg >= thrs[ti]).sum())/kb
                    else: g[bi_,:] = 0.0
                return g
            u1grid  = _grid(sig["u1"], U1G_THRS)
            me5grid = _grid(sig["donor"], ME5G_THRS)
        dn5_pos = np.empty(0); dn5_sc = np.empty(0)
        if dn_seq and len(dn_seq) >= 9:
            arrd = np.frombuffer(dn_seq.encode("ascii","replace"), np.uint8)
            gtd = np.where((arrd[:-1]==71) & (arrd[1:]==84))[0]
            pp=[]; ss5=[]
            for i in gtd:
                p=int(i)
                if p-3<0 or p+6>len(dn_seq): continue
                try:
                    v=score5(dn_seq[p-3:p+6]); v=float(np.nanmax(np.asarray(v,float))) if not np.isscalar(v) else float(v)
                except Exception: continue
                if np.isfinite(v) and v>-99: pp.append(p); ss5.append(v)
            dn5_pos=np.asarray(pp,float); dn5_sc=np.asarray(ss5,float)
        if dn5_sc.size:
            bi=int(np.argmax(dn5_sc)); ie["dn5_best"]=float(dn5_sc[bi]); ie["dn5_dist"]=float(dn5_pos[bi])
            sdn=dn5_sc>=a.nd_thr
            if sdn.any(): ie["dn5_nearest_strong"]=float(dn5_pos[sdn].min())
        cand_anchor=[]
        if np.isfinite(ie["nearest_dn"]): cand_anchor.append(ie["nearest_dn"])
        if dn5_sc.size and np.isfinite(anchor_up):
            sdn=dn5_sc>=a.nd_thr
            if sdn.any(): cand_anchor.append(float(anchor_up+dn5_pos[sdn].min()))
        if cand_anchor:
            ie["first5_from_anchor"]=float(min(cand_anchor))
            ie["internal_exon_like"]=1 if (50.0<=ie["first5_from_anchor"]<=300.0) else 0
        _positional_feats(ie, sig, anchor_up, dn5_pos, dn5_sc, c, strand, a.nd_thr)
        ie.update(_annot_feats(ANN_MAIN, chrom, c, strand, anchor_up))
        dU,sU=sig["donor"]
        if dU:
            dU=np.asarray(dU,float); sU=np.asarray(sU,float); st=(sU>=a.nd_thr)&(dU<=MG_SPAN)
            if st.any():
                bb=np.clip(np.searchsorted(MG_EDGES,-dU[st],side="right")-1,0,MG_NB-1); np.add.at(mg5,bb,1.0)
        duU,vuU=sig["u1"]
        if duU:
            duU=np.asarray(duU,float); vuU=np.asarray(vuU,float); mk=duU<=MG_SPAN
            if mk.any():
                bb=np.clip(np.searchsorted(MG_EDGES,-duU[mk],side="right")-1,0,MG_NB-1)
                np.add.at(mgus,bb,vuU[mk]); np.add.at(mgun,bb,1.0)
        if dn5_sc.size:
            st=(dn5_sc>=a.nd_thr)&(dn5_pos<=MG_SPAN)
            if st.any():
                bb=np.clip(np.searchsorted(MG_EDGES,dn5_pos[st],side="right")-1,0,MG_NB-1); np.add.at(mg5,bb,1.0)
        if dn_seq and len(u1v):
            posd=np.arange(len(u1v)); kd=np.isfinite(u1v)&(posd<=MG_SPAN)
            if kd.any():
                bb=np.clip(np.searchsorted(MG_EDGES,posd[kd].astype(float),side="right")-1,0,MG_NB-1)
                np.add.at(mgus,bb,u1v[kd]); np.add.at(mgun,bb,1.0)
        _du,_vu = sig["u1"]
        _ru = -np.asarray(_du,float) if _du else np.empty(0)
        _vu = np.asarray(_vu,float) if _du else np.empty(0)
        _rd = np.arange(len(u1v),dtype=float) if len(u1v) else np.empty(0)
        ucube,dwu,dwd = _u1_cube(_ru,_vu,_rd,u1v if len(u1v) else np.empty(0))
        ie["u1_dw_up"]=dwu; ie["u1_dw_dn"]=dwd
        _dd5,_vv5 = sig["donor"]
        _r5u = -np.asarray(_dd5,float) if _dd5 else np.empty(0)
        _v5u = np.asarray(_vv5,float) if _dd5 else np.empty(0)
        mcube,_,_ = _u1_cube(_r5u,_v5u,dn5_pos,dn5_sc,thrs=MC_THRS)
        ie["chrom"]=chrom; ie["center"]=int(c); ie["strand"]=strand
        return (cls, str(rec["gene"]), rows, ie, u1prof, u1grid, me5grid, iprof, mg5, mgus, mgun, ucube, mcube, dcm)

    recs = df.to_dict("records")
    def _consume(res):
        nonlocal n_bad
        if res is None: n_bad += 1; return
        cls, gene, rows, ie, u1prof, u1grid, me5grid, iprof, mg5, mgus, mgun, ucube, mcube, dcm = res
        UCUBE[cls].append(ucube); MCUBE[cls].append(mcube)
        for _m in DW_MECHS:
            DWIN[_m][cls].append(dcm.get(_m) if dcm.get(_m) is not None else np.zeros((len(DW_THRS[_m]),nW),np.float32))
        npk[cls] += 1; GENE[cls].append(gene)
        IPROF[cls] += iprof; MG5[cls] += mg5; MGUS[cls] += mgus; MGUN[cls] += mgun
        for m in MECHS:
            if rows[m] is None:
                RM[m][cls].append(np.full(nW, np.nan)); DEN[m][cls].append(np.zeros(nW)); continue
            row, dcount = rows[m]; RM[m][cls].append(row); DEN[m][cls].append(dcount)
            fin = np.isfinite(row)
            if fin.any():
                jj = np.clip(np.searchsorted(SEDGES[m], row[fin], side="right")-1, 0, NS[m]-1)
                for wi, j in zip(np.nonzero(fin)[0], jj): acc[m][cls][wi, int(j)] += 1.0
        ie["gene"] = gene; IE[cls].append(ie)
        if u1prof is not None:
            bi, vals = u1prof
            np.add.at(u1dn_sum[cls], bi, vals); np.add.at(u1dn_cnt[cls], bi, 1.0)
        if u1grid is not None:
            u1grid_vals[cls].append(u1grid)
        if me5grid is not None:
            me5grid_vals[cls].append(me5grid)

    if a.n_jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        cfg = dict(scripts_dir=a.scripts_dir or "../scripts",
                   reference_fasta=a.reference_fasta, u1_lookup=a.u1_lookup,
                   max_nt=a.max_nt, pad=a.pad, dist_bin=a.dist_bin, score_bin=a.score_bin,
                   dn_lo=a.dn_lo, dn_hi=a.dn_hi, per_max=a.per_max,
                   u1_dn_span=a.u1_dn_span, u1_dn_bin=a.u1_dn_bin, edef_lo=a.edef_lo, edef_hi=a.edef_hi, nd_thr=a.nd_thr, me5_cand=a.me5_cand, annot_5ss=a.annot_5ss, SRANGE=SRANGE)
        with ProcessPoolExecutor(max_workers=a.n_jobs, initializer=_init_worker,
                                 initargs=(cfg,)) as ex:
            for res in tqdm(ex.map(_work_mp, recs, chunksize=64),
                            total=len(recs), desc="scan"): _consume(res)
    else:
        for rec in tqdm(recs, total=len(recs), desc="scan"): _consume(_work(rec))

    ratio_all, auroc_all, paired_all = [], [], []
    for m in MECHS:
        sc = SCTR[m]; nS = NS[m]
        RMc = np.array(RM[m]["Canonical"], float); RMk = np.array(RM[m]["Competitor"], float)
        MC = acc[m]["Canonical"]/max(npk["Canonical"],1); MK = acc[m]["Competitor"]/max(npk["Competitor"],1)
        eps = 1.0/max(npk["Canonical"], npk["Competitor"], 1)
        ratio = np.log2((MC+eps)/(MK+eps)); tot = acc[m]["Canonical"] + acc[m]["Competitor"]
        for i in range(nW):
            for j in range(nS):
                if tot[i,j] < a.min_cell: continue
                ratio_all.append(dict(mech=m, window=int(wins[i]), maxscore=float(sc[j]),
                                      strong_thr=float(STRONG_THR[m]), log2ratio=float(ratio[i,j]),
                                      n=int(tot[i,j])))
        for wi, w in enumerate(wins):
            pc = RMc[:, wi]; kc = RMk[:, wi]; pc = pc[np.isfinite(pc)]; kc = kc[np.isfinite(kc)]
            if len(pc) < 20 or len(kc) < 20: continue
            au, ap_, base = auroc_auprc(pc, kc)
            auroc_all.append(dict(mech=m, window=int(w), auroc=au, auprc=ap_, base_rate=base))
        wlast = nW - 1; cg, kg = {}, {}
        use_dens = (m != "acceptor")
        if use_dens:
            Dc = np.array(DEN[m]["Canonical"], float)[:, wlast] / (a.max_nt/1000.0)
            Dk = np.array(DEN[m]["Competitor"], float)[:, wlast] / (a.max_nt/1000.0)
        for cls, store in (("Canonical", cg), ("Competitor", kg)):
            M = RMc if cls=="Canonical" else RMk
            src = (Dc if cls=="Canonical" else Dk) if use_dens else M[:, wlast]
            for gi, g in enumerate(GENE[cls]):
                v = src[gi] if use_dens else M[gi, wlast]
                if np.isfinite(v): store.setdefault(g, []).append(v)
        rows = []; gdiff = []
        for g in set(cg) & set(kg):
            cval = np.mean(cg[g]); comp = np.array(kg[g])
            rows.append((cval, comp.mean(), comp.max(), comp.min(), len(comp))); gdiff.append(cval - comp.mean())
        if rows:
            P = pd.DataFrame(rows, columns=["canon","cmean","cmax","cmin","k"])
            pd.DataFrame({"mech": m, "diff": gdiff}).to_csv(f"{a.out}_paired_diff_{m}.tsv", sep="\t", index=False)
            def _wil(k,nn,z=1.96):
                if nn==0: return (0.,0.,0.)
                ph=k/nn; d=1+z*z/nn; c=(ph+z*z/(2*nn))/d; h=z*((ph*(1-ph)/nn+z*z/(4*nn*nn))**0.5)/d
                return ph,max(0,c-h),min(1,c+h)
            for agg, hits, null in [("pair (mean competitor)", int((P.canon>P.cmean).sum()), 0.5),
                                    ("vs strongest competitor", int((P.canon>P.cmax).sum()), float((1.0/(P.k+1)).mean())),
                                    ("vs weakest competitor", int((P.canon>P.cmin).sum()), float((P.k/(P.k+1)).mean()))]:
                obs,lo,hi=_wil(hits,len(P))
                paired_all.append(dict(mech=m, agg=agg, observed=obs, ci_lo=lo, ci_hi=hi, null=float(null), n=len(P), sig=int(lo>null or hi<null)))
        auw = [d for d in auroc_all if d["mech"]==m]
        if auw: print(f"[{m:<8}] AUROC@{a.max_nt}nt={auw[-1]['auroc']:.3f}  AUPRC={auw[-1]['auprc']:.3f} (base {auw[-1]['base_rate']:.3f})", file=sys.stderr)

    den_curve = []
    for m in MECHS:
        for cls in ("Canonical","Competitor"):
            M = np.array(DEN[m][cls], float)
            for wi, w in enumerate(wins):
                den_curve.append(dict(mech=m, cls=cls, window=int(w), per_kb=float(M[:,wi].mean())/(w/1000.0)))
    pd.DataFrame(den_curve).to_csv(f"{a.out}_density_curve.tsv", sep="\t", index=False)
    pd.DataFrame(ratio_all).to_csv(f"{a.out}_ratio.tsv", sep="\t", index=False)
    pd.DataFrame(auroc_all).to_csv(f"{a.out}_auroc.tsv", sep="\t", index=False)
    pd.DataFrame(paired_all).to_csv(f"{a.out}_paired.tsv", sep="\t", index=False)

    den_auroc = []
    for m in MECHS:
        Dc = np.array(DEN[m]["Canonical"], float); Dk = np.array(DEN[m]["Competitor"], float)
        for wi, w in enumerate(wins):
            pc = Dc[:, wi]; kc = Dk[:, wi]
            if len(pc) < 20 or len(kc) < 20: continue
            au, ap_, base = auroc_auprc(pc, kc)
            den_auroc.append(dict(mech=m, window=int(w), auroc=au, auprc=ap_, base_rate=base))
    pd.DataFrame(den_auroc).to_csv(f"{a.out}_density_auroc.tsv", sep="\t", index=False)

    pd.DataFrame([dict(cls=c, n=int(npk[c])) for c in ("Canonical","Competitor")]
                 ).to_csv(f"{a.out}_class_counts.tsv", sep="\t", index=False)

    if any(DWIN[m]["Canonical"] or DWIN[m]["Competitor"] for m in DW_MECHS):
        _dwarr = {}
        for _m in DW_MECHS:
            cl = DWIN[_m]["Canonical"]; kl = DWIN[_m]["Competitor"]
            _dwarr[_m] = np.concatenate(
                [np.stack(cl) if cl else np.zeros((0,len(DW_THRS[_m]),nW),np.float32),
                 np.stack(kl) if kl else np.zeros((0,len(DW_THRS[_m]),nW),np.float32)], axis=0)
        _ybc = np.concatenate([np.ones(npk["Canonical"],np.int8), np.zeros(npk["Competitor"],np.int8)])
        BCA, BCR, _picks = bcshow_tables(
            _dwarr, DW_THRS, wins, _ybc, auroc_all,
            u1_thr=a.bcshow_u1_thr,
            donor_search=tuple(float(x) for x in a.bcshow_donor_search.split(",")),
            nybin=a.bcshow_nybin)
        BCA.to_csv(f"{a.out}_bcshow_auroc.tsv", sep="\t", index=False)
        BCR.to_csv(f"{a.out}_bcshow_ratio.tsv", sep="\t", index=False)
        print(f"[bcshow] wrote {a.out}_bcshow_auroc.tsv ({len(BCA)} rows) / "
              f"{a.out}_bcshow_ratio.tsv ({len(BCR)} cells)", file=sys.stderr)
        for _m in ("acceptor","u1","donor"):
            g = BCA[BCA.mech == _m]
            if not len(g): continue
            r = g.loc[g.auroc.idxmax()]
            print(f"    peak AUROC {_m:9s} {r.auroc:.4f} at {r.window:.0f}nt  "
                  f"(raw {r.auroc_raw:.4f}, predicts {r.predicts})", file=sys.stderr)
    for m in MECHS:
        subm = [d for d in den_auroc if d["mech"] == m]
        if subm: print(f"[density-auroc] {m:<8} AUROC@{a.max_nt}nt={subm[-1]['auroc']:.3f}", file=sys.stderr)

    def _pg_diff(key):
        cg2 = {}; kg2 = {}
        for d in IE["Canonical"]:
            v = d.get(key, np.nan)
            if np.isfinite(v): cg2.setdefault(d.get("gene"), []).append(v)
        for d in IE["Competitor"]:
            v = d.get(key, np.nan)
            if np.isfinite(v): kg2.setdefault(d.get("gene"), []).append(v)
        return np.array([np.mean(cg2[g]) - np.mean(kg2[g]) for g in set(cg2) & set(kg2)])
    for key, tag in (("acc", "acc"), ("interv_density5", "interv_donor"), ("interv_u1_density", "interv_u1"),
                     ("edef_best5", "edef_best5")):
        dd = _pg_diff(key)
        if len(dd):
            pd.DataFrame({"mech": tag, "diff": dd}).to_csv(f"{a.out}_paired_diff_{tag}.tsv", sep="\t", index=False)
            print(f"[paired-diff] {tag:<12} within-gene canon-comp median={np.median(dd):+.3f} "
                  f"({(dd < 0).mean():.0%} of genes competitor higher)", file=sys.stderr)

    def _arr(cls, key): return np.array([d.get(key, np.nan) for d in IE[cls]], float)
    ACd = {c: _arr(c,"acc") for c in ("Canonical","Competitor")}
    DDd = {c: _arr(c,"dn_donor") for c in ("Canonical","Competitor")}
    def _z(x, ref):
        mu, sd = np.nanmean(ref), np.nanstd(ref) + 1e-9
        return (x - mu) / sd
    acc_ref = np.concatenate([ACd["Canonical"], ACd["Competitor"]])
    don_ref = np.concatenate([DDd["Canonical"], DDd["Competitor"]])
    don_floor = np.nanmin(don_ref) if np.isfinite(don_ref).any() else 0.0
    ie_rows = []
    for c in ("Canonical","Competitor"):
        za = _z(ACd[c], acc_ref)
        dd = np.where(np.isfinite(DDd[c]), DDd[c], don_floor)
        zd = _z(dd, np.where(np.isfinite(don_ref), don_ref, don_floor))
        conj = np.minimum(za, zd)
        for i in range(len(za)):
            ie_rows.append(dict(cls=c, z_acceptor=float(za[i]) if np.isfinite(za[i]) else np.nan,
                                z_dn_donor=float(zd[i]), conjunction=float(conj[i]) if np.isfinite(conj[i]) else np.nan))
    IEdf = pd.DataFrame(ie_rows); IEdf.to_csv(f"{a.out}_conjunction.tsv", sep="\t", index=False)
    conj_auroc = []
    for name, col in (("acceptor","z_acceptor"), ("dn_donor","z_dn_donor"), ("conjunction","conjunction")):
        pc = IEdf.loc[IEdf.cls=="Canonical", col].dropna().values
        kc = IEdf.loc[IEdf.cls=="Competitor", col].dropna().values
        if len(pc) > 20 and len(kc) > 20:
            au, ap_, base = auroc_auprc(pc, kc)
            conj_auroc.append(dict(feature=name, auroc=au, auprc=ap_, base_rate=base))
    pd.DataFrame(conj_auroc).to_csv(f"{a.out}_conjunction_auroc.tsv", sep="\t", index=False)
    for r in conj_auroc:
        print(f"[conjunction] {r['feature']:<12} AUROC={r['auroc']:.3f}", file=sys.stderr)

    per_rows = []
    for c in ("Canonical","Competitor"):
        dist = _arr(c, "dn_dist"); dist = dist[np.isfinite(dist)]
        for v in dist: per_rows.append(dict(cls=c, dn_dist=float(v)))
    if per_rows:
        pd.DataFrame(per_rows).to_csv(f"{a.out}_periodicity.tsv", sep="\t", index=False)
        for c in ("Canonical","Competitor"):
            dd = _arr(c,"dn_dist"); dd = dd[np.isfinite(dd)]
            if len(dd): print(f"[periodicity] {c:<11} anchor->donor median={np.median(dd):.0f}nt "
                              f"(has donor in {len(dd)}/{len(IE[c])})", file=sys.stderr)

    def _paired_frac(cA, cK, gA, gK):
        cg={}; kg={}
        for g,v in zip(gA,cA):
            if np.isfinite(v): cg.setdefault(g,[]).append(v)
        for g,v in zip(gK,cK):
            if np.isfinite(v): kg.setdefault(g,[]).append(v)
        diffs=[np.mean(kg[g])-np.mean(cg[g]) for g in set(cg)&set(kg)]
        if not diffs: return np.nan, np.nan, 0
        diffs=np.array(diffs); return float((diffs>0).mean()), float(np.median(diffs)), len(diffs)
    iv_rows=[]; iv_dist=[]
    geneA={c:[d.get("gene") for d in IE[c]] for c in ("Canonical","Competitor")}
    for key,label in (("interv_sum5","sum 5'ss strength"),
                      ("interv_sum_cand","sum candidate 5'ss (>=me5_cand)"),
                      ("interv_n_cand","# candidate 5'ss (>=me5_cand)"),
                      ("n_strong_interv","# strong 5'ss"),
                      ("interv_density5","strong 5'ss / kb"),
                      ("interv_sum5_perkb","sum 5'ss / kb"),
                      ("interv_u1_sum","sum U1 strength"),
                      ("interv_u1_nstrong","# strong U1"),
                      ("interv_u1_density","strong U1 / kb"),
                      ("interv_u1_sum_perkb","sum U1 / kb")):
        pc=_arr("Canonical",key); kc=_arr("Competitor",key)
        pcf=pc[np.isfinite(pc)]; kcf=kc[np.isfinite(kc)]
        au=auroc_auprc(pcf,kcf)[0] if (len(pcf)>20 and len(kcf)>20) else np.nan
        fr,md,nn=_paired_frac(pc,kc,geneA["Canonical"],geneA["Competitor"])
        iv_rows.append(dict(signal=key, label=label, auroc=au,
                            frac_comp_higher=fr, median_comp_minus_canon=md, n=nn,
                            canon_mean=float(np.nanmean(pc)), comp_mean=float(np.nanmean(kc))))
        for c in ("Canonical","Competitor"):
            v=_arr(c,key)
            for x in v[np.isfinite(v)]: iv_dist.append(dict(signal=key, cls=c, value=float(x)))
    pd.DataFrame(iv_rows).to_csv(f"{a.out}_interv_density_auroc.tsv", sep="\t", index=False)
    pd.DataFrame(iv_dist).to_csv(f"{a.out}_interv_density_values.tsv", sep="\t", index=False)
    print("[interv-density] intervening 5'ss sum/count/density:", file=sys.stderr)
    for r in iv_rows:
        au=f"{r['auroc']:.3f}" if np.isfinite(r['auroc']) else "nan"
        print(f"    {r['label']:22} AUROC={au}  comp>canon in {r['frac_comp_higher']:.0%}  "
              f"(canon {r['canon_mean']:.2f} vs comp {r['comp_mean']:.2f})", file=sys.stderr)

    ndC=_arr("Canonical","nearest_dn"); ndK=_arr("Competitor","nearest_dn")
    presC=np.isfinite(ndC); presK=np.isfinite(ndK)
    FAR=2000.0
    ndCf=np.where(presC, ndC, FAR); ndKf=np.where(presK, ndK, FAR)
    au_nd=auroc_auprc(ndCf, ndKf)[0]
    nd_rows=[]
    for c,nd,pres in (("Canonical",ndC,presC),("Competitor",ndK,presK)):
        vals=nd[np.isfinite(nd)]
        nd_rows.append(dict(cls=c, n=len(nd), frac_has_nearby=float(pres.mean()),
                            median_dist=float(np.median(vals)) if len(vals) else np.nan))
    pd.DataFrame([dict(cls=c, dist=float(v)) for c,nd in (("Canonical",ndC),("Competitor",ndK))
                 for v in nd[np.isfinite(nd)]]).to_csv(f"{a.out}_nearest_dn.tsv", sep="\t", index=False)
    pd.DataFrame(nd_rows).to_csv(f"{a.out}_nearest_dn_summary.tsv", sep="\t", index=False)
    cg2={}; kg2={}
    for g,v in zip([d.get("gene") for d in IE["Canonical"]], ndCf): cg2.setdefault(g,[]).append(v)
    for g,v in zip([d.get("gene") for d in IE["Competitor"]], ndKf): kg2.setdefault(g,[]).append(v)
    pdiff=[np.mean(kg2[g])-np.mean(cg2[g]) for g in set(cg2)&set(kg2)]
    frac_comp_nearer=float((np.array(pdiff)<0).mean()) if pdiff else np.nan
    print(f"[nearest-dn] distance to NEAREST strong downstream 5'ss (>={a.nd_thr:.0f}):", file=sys.stderr)
    for r in nd_rows:
        print(f"    {r['cls']:11} median={r['median_dist']:.0f}nt  has one nearby: {r['frac_has_nearby']:.0%}", file=sys.stderr)
    print(f"    AUROC={au_nd:.3f}  within-gene competitor-nearer in {frac_comp_nearer:.0%} of genes", file=sys.stderr)

    for cls in ("Canonical","Competitor"):
        for d in IE[cls]:
            b5=d.get("edef_best5",np.nan); ac=d.get("acc",np.nan)
            d["donor_acc_ratio"]=float(b5/ac) if (np.isfinite(b5) and np.isfinite(ac) and ac>0) else np.nan
            d["donor_acc_diff"]=float(b5-ac) if (np.isfinite(b5) and np.isfinite(ac)) else np.nan
    ed_rows=[]; ed_vals=[]
    for key,label in (("acc","acceptor 3'ss"),
                      ("edef_pas","PAS strength"),
                      ("edef_best5","best 5'ss in window"),
                      ("edef_compete","5'ss - PAS competition"),
                      ("edef_coupling","acceptor x 5'ss coupling"),
                      ("donor_acc_ratio","donor/acceptor ratio"),
                      ("donor_acc_diff","donor - acceptor")):
        pc=_arr("Canonical",key); kc=_arr("Competitor",key)
        pcf=pc[np.isfinite(pc)]; kcf=kc[np.isfinite(kc)]
        au=auroc_auprc(pcf,kcf)[0] if (len(pcf)>20 and len(kcf)>20) else np.nan
        fr,md,nn=_paired_frac(pc,kc,geneA["Canonical"],geneA["Competitor"])
        ed_rows.append(dict(measure=key,label=label,auroc=au,
                            frac_comp_higher=fr, frac_canon_higher=(1-fr) if np.isfinite(fr) else np.nan,
                            median_comp_minus_canon=md,n=nn,
                            canon_mean=float(np.nanmean(pc)),comp_mean=float(np.nanmean(kc))))
        for c in ("Canonical","Competitor"):
            v=_arr(c,key)
            for x in v[np.isfinite(v)]: ed_vals.append(dict(measure=key,cls=c,value=float(x)))
    pd.DataFrame(ed_rows).to_csv(f"{a.out}_edef_auroc.tsv", sep="\t", index=False)
    pd.DataFrame(ed_vals).to_csv(f"{a.out}_edef_values.tsv", sep="\t", index=False)
    print("[exon-definition] acceptor / PAS / best-5'ss-in-window / competition / coupling:", file=sys.stderr)
    for r in ed_rows:
        au=f"{r['auroc']:.3f}" if np.isfinite(r['auroc']) else "nan"
        ch=f"{r['frac_canon_higher']:.0%}" if np.isfinite(r['frac_canon_higher']) else "nan"
        print(f"    {r['label']:26} AUROC={au}  canon>comp within-gene {ch}  "
              f"(canon {r['canon_mean']:.2f} vs comp {r['comp_mean']:.2f})", file=sys.stderr)

    accV = _arr("Canonical","acc"); accK = _arr("Competitor","acc")
    b5V  = _arr("Canonical","edef_best5"); b5K = _arr("Competitor","edef_best5")
    a_edges = np.arange(0, 14.0+1.5, 1.5)
    d_edges = np.arange(-5.0, 13.0+2.0, 2.0)
    base_rate = len(accV) / max(len(accV)+len(accK), 1)
    def _hist2d(av, dv):
        m = np.isfinite(av) & np.isfinite(dv)
        H,_,_ = np.histogram2d(av[m], dv[m], bins=[a_edges, d_edges]); return H
    Hc = _hist2d(accV, b5V); Hk = _hist2d(accK, b5K)
    ac_ctr=(a_edges[:-1]+a_edges[1:])/2; d_ctr=(d_edges[:-1]+d_edges[1:])/2
    int5_rows=[]
    for i in range(len(ac_ctr)):
        for j in range(len(d_ctr)):
            tot = Hc[i,j] + Hk[i,j]
            if tot < a.min_cell: continue
            pcanon = Hc[i,j] / tot
            int5_rows.append(dict(acceptor=float(ac_ctr[i]), best5=float(d_ctr[j]),
                                  n=int(tot), p_canonical=float(pcanon),
                                  log2_enrich=float(np.log2((pcanon+1e-3)/(base_rate+1e-3)))))
    pd.DataFrame(int5_rows).to_csv(f"{a.out}_acc_x_5ss.tsv", sep="\t", index=False)
    print(f"[acc-x-5ss] wrote acceptor x best-5'ss interaction (P canonical, base {base_rate:.3f}), "
          f"{len(int5_rows)} cells >= {a.min_cell}", file=sys.stderr)

    intV = _arr("Canonical","interv_donor"); intK = _arr("Competitor","interv_donor")
    distV = _arr("Canonical","interv_dist"); distK = _arr("Competitor","interv_dist")
    def _hist2(av, dv, ea, ed):
        m = np.isfinite(av) & np.isfinite(dv)
        H,_,_ = np.histogram2d(av[m], dv[m], bins=[ea, ed]); return H
    ai_edges = np.arange(0, 14.0+1.5, 1.5)
    di_edges = np.arange(-6.0, 13.0+1.5, 1.5)
    HcI = _hist2(accV, intV, ai_edges, di_edges); HkI = _hist2(accK, intK, ai_edges, di_edges)
    aic=(ai_edges[:-1]+ai_edges[1:])/2; dic=(di_edges[:-1]+di_edges[1:])/2
    axi_rows=[]
    for i in range(len(aic)):
        for j in range(len(dic)):
            tot=HcI[i,j]+HkI[i,j]
            if tot < a.min_cell: continue
            pc=HcI[i,j]/tot
            axi_rows.append(dict(acceptor=float(aic[i]), interv=float(dic[j]), n=int(tot),
                                 p_canonical=float(pc),
                                 log2_enrich=float(np.log2((pc+1e-3)/(base_rate+1e-3)))))
    pd.DataFrame(axi_rows).to_csv(f"{a.out}_acc_x_interv.tsv", sep="\t", index=False)
    print(f"[acc-x-interv] wrote acceptor x intervening-5'ss P(canonical) grid, "
          f"{len(axi_rows)} cells >= {a.min_cell}", file=sys.stderr)

    aE = np.concatenate([accV, accK]); iE = np.concatenate([intV, intK])
    yE = np.concatenate([np.ones(len(accV)), np.zeros(len(accK))])
    gdE = np.isfinite(aE) & np.isfinite(iE); aE = aE[gdE]; iE = iE[gdE]; yE = yE[gdE]
    if len(yE) > 50 and aE.std() > 0 and iE.std() > 0:
        Xe = np.column_stack([np.ones(len(aE)), (aE-aE.mean())/aE.std(), (iE-iE.mean())/iE.std()])
        be = np.zeros(3)
        for _ in range(50):
            eta = Xe @ be; pp = 1/(1+np.exp(-eta)); W = np.clip(pp*(1-pp), 1e-6, None)
            zz = eta + (yE-pp)/W
            bn = np.linalg.solve((Xe*W[:,None]).T @ Xe + 1e-6*np.eye(3), (Xe*W[:,None]).T @ zz)
            if np.max(np.abs(bn-be)) < 1e-8: be = bn; break
            be = bn
        eta = Xe @ be; pp = 1/(1+np.exp(-eta)); W = np.clip(pp*(1-pp), 1e-6, None)
        cov = np.linalg.inv((Xe*W[:,None]).T @ Xe + 1e-6*np.eye(3)); se = np.sqrt(np.diag(cov))
        bi_ = float(be[2]); si_ = float(se[2])
        pd.DataFrame([dict(term="intervening_5ss_given_acceptor", beta=bi_, se=si_,
                           OR=float(np.exp(bi_)), ci_lo=float(np.exp(bi_-1.96*si_)),
                           ci_hi=float(np.exp(bi_+1.96*si_)),
                           z=float(bi_/si_) if si_ > 0 else np.nan,
                           acc_beta=float(be[1]), n=int(len(yE)))]).to_csv(
            f"{a.out}_e_coef.tsv", sep="\t", index=False)
        print(f"[e-coef] intervening 5'ss | acceptor: beta={bi_:+.3f} "
              f"(OR={np.exp(bi_):.3f}, z={bi_/si_:+.1f}); acceptor beta={be[1]:+.3f}", file=sys.stderr)

    dd_edges = np.arange(0, 600.0+40.0, 40.0)
    sv_edges = np.arange(-6.0, 13.0+1.5, 1.5)
    HcD=_hist2(distV, intV, dd_edges, sv_edges); HkD=_hist2(distK, intK, dd_edges, sv_edges)
    ddc=(dd_edges[:-1]+dd_edges[1:])/2; svc=(sv_edges[:-1]+sv_edges[1:])/2
    idm_rows=[]
    for i in range(len(ddc)):
        for j in range(len(svc)):
            tot=HcD[i,j]+HkD[i,j]
            if tot < a.min_cell: continue
            pc=HcD[i,j]/tot
            idm_rows.append(dict(dist=float(ddc[i]), interv=float(svc[j]), n=int(tot),
                                 p_canonical=float(pc),
                                 log2_enrich=float(np.log2((pc+1e-3)/(base_rate+1e-3)))))
    pd.DataFrame(idm_rows).to_csv(f"{a.out}_interv_dist_map.tsv", sep="\t", index=False)
    print(f"[interv-map] wrote intervening-5'ss distance x strength grid, "
          f"{len(idm_rows)} cells >= {a.min_cell}", file=sys.stderr)

    ip_rows=[]
    for c in ("Canonical","Competitor"):
        di=_arr(c,"interv_dist"); st=_arr(c,"interv_donor")
        m=np.isfinite(di)&np.isfinite(st)
        for d_,s_ in zip(di[m], st[m]): ip_rows.append((c,float(d_),float(s_)))
    pd.DataFrame(ip_rows,columns=["cls","dist","score"]).to_csv(f"{a.out}_interv_points.tsv",sep="\t",index=False)
    print(f"[interv-points] wrote {len(ip_rows)} per-peak intervening donors (dist,score) by class", file=sys.stderr)

    iprof_ctr=(IPROF_EDGES[:-1]+IPROF_EDGES[1:])/2; binw=float(IPROF_EDGES[1]-IPROF_EDGES[0])
    ipr_rows=[]
    for c in ("Canonical","Competitor"):
        per_site=IPROF[c]/max(npk[c],1)
        for i in range(IPROF_NB):
            ipr_rows.append(dict(cls=c, dist=float(iprof_ctr[i]),
                                 mean_count=float(per_site[i]),
                                 per_kb=float(per_site[i]/(binw/1000.0))))
    pd.DataFrame(ipr_rows).to_csv(f"{a.out}_interv_profile.tsv", sep="\t", index=False)
    print("[interv-profile] wrote strong intervening 5'ss density vs distance, by class", file=sys.stderr)

    mg_ctr=(MG_EDGES[:-1]+MG_EDGES[1:])/2
    mg_rows=[]
    for c in ("Canonical","Competitor"):
        d5=MG5[c]/max(npk[c],1)
        du=np.where(MGUN[c]>0, MGUS[c]/np.maximum(MGUN[c],1), np.nan)
        for i in range(MG_NB):
            mg_rows.append(dict(cls=c, pos=float(mg_ctr[i]), n5_per_site=float(d5[i]), u1_mean=float(du[i])))
    pd.DataFrame(mg_rows).to_csv(f"{a.out}_metagene.tsv", sep="\t", index=False)
    print("[metagene] wrote strong-5'ss density + mean U1 vs distance upstream of the site, by class", file=sys.stderr)

    feat_rows=[]
    for c in ("Canonical","Competitor"):
        for d in IE[c]:
            feat_rows.append(dict(gene=d.get("gene"), is_canonical=1 if c=="Canonical" else 0,
                acc=d.get("acc",np.nan), pas=d.get("edef_pas",np.nan),
                interv_sum5=d.get("interv_sum5",np.nan), interv_density5=d.get("interv_density5",np.nan),
                interv_u1_density=d.get("interv_u1_density",np.nan), edef_best5=d.get("edef_best5",np.nan),
                nearest_dn=d.get("nearest_dn",np.nan),
                lone5_best=d.get("lone5_best",np.nan), unit5_best=d.get("unit5_best",np.nan),
                **{k: d.get(k, np.nan) for k in POSITIONAL_KEYS}))
    FEAT = pd.DataFrame(feat_rows)
    if UCUBE["Canonical"] or UCUBE["Competitor"]:
        _ucat = np.concatenate([np.stack([np.asarray(x,np.float32) for x in UCUBE[c]])
                                for c in ("Canonical","Competitor") if UCUBE[c]], axis=0)
        if _ucat.shape[0] == len(FEAT):
            for k, v in u1_winner_features(_ucat.astype(np.float64)).items():
                FEAT[k] = v
            print(f"[peak-features] added U1 licensing columns: "
                  f"{', '.join(u1_winner_features(_ucat[:1].astype(np.float64)).keys())}", file=sys.stderr)
        else:
            print(f"[peak-features] WARNING: cube rows {_ucat.shape[0]} != feature rows "
                  f"{len(FEAT)}; U1 columns NOT added", file=sys.stderr)
    FEAT.to_csv(f"{a.out}_peak_features.tsv", sep="\t", index=False)
    pos_rows = []
    for c_ in ("Canonical", "Competitor"):
        for d in IE[c_]:
            r = dict(gene=d.get("gene"), is_canonical=1 if c_ == "Canonical" else 0,
                     chrom=d.get("chrom"), center=d.get("center"), strand=d.get("strand"),
                     acc=d.get("acc", np.nan),
                     nearest_dn=d.get("nearest_dn", np.nan),
                     interv_donor=d.get("interv_donor", np.nan),
                     dn5_nearest_strong=d.get("dn5_nearest_strong", np.nan),
                     first5_from_anchor=d.get("first5_from_anchor", np.nan),
                     internal_exon_like=d.get("internal_exon_like", 0))
            r.update({k: d.get(k, np.nan) for k in POSITIONAL_KEYS})
            pos_rows.append(r)
    POS = pd.DataFrame(pos_rows)
    POS.to_csv(f"{a.out}_positional_features.tsv", sep="\t", index=False)
    print(f"[positional] wrote {a.out}_positional_features.tsv ({len(POS)} rows, "
          f"{POS.anchor_3ss_pos.notna().sum()} with an anchor)", file=sys.stderr)
    print(f"[peak-features] wrote {len(feat_rows)} peaks x features (clogit / matched-pair / mechanism)", file=sys.stderr)

    def _stack(store):
        cl = [np.asarray(x, np.float32) for x in store["Canonical"]]
        kl = [np.asarray(x, np.float32) for x in store["Competitor"]]
        return np.concatenate([np.stack(cl) if cl else np.zeros((0,UC_NB,UC_NF),np.float32),
                               np.stack(kl) if kl else np.zeros((0,UC_NB,UC_NF),np.float32)], axis=0)
    if UCUBE["Canonical"] or UCUBE["Competitor"]:
        cube = _stack(UCUBE); mcube = _stack(MCUBE)
        meta_rows = []
        for c in ("Canonical","Competitor"):
            for d in IE[c]:
                meta_rows.append(dict(gene=d.get("gene"), is_canonical=1 if c=="Canonical" else 0,
                    chrom=d.get("chrom"), center=d.get("center"), strand=d.get("strand"),
                    acc=d.get("acc",np.nan), edef_best5=d.get("edef_best5",np.nan),
                    nearest_dn=d.get("nearest_dn",np.nan), pas=d.get("edef_pas",np.nan),
                    interv_donor=d.get("interv_donor",np.nan),
                    u1_dw_up=d.get("u1_dw_up",np.nan), u1_dw_dn=d.get("u1_dw_dn",np.nan)))
        MET = pd.DataFrame(meta_rows)
        if len(MET) != cube.shape[0] or len(MET) != mcube.shape[0]:
            print(f"[u1-cube] FATAL: meta {len(MET)} vs u1 {cube.shape[0]} vs me5 {mcube.shape[0]}", file=sys.stderr)
            sys.exit(1)
        _dw = {}
        for _m in DW_MECHS:
            cl = DWIN[_m]["Canonical"]; kl = DWIN[_m]["Competitor"]
            if cl or kl:
                _dw[f"dw_{_m}"] = np.concatenate(
                    [np.stack(cl) if cl else np.zeros((0,len(DW_THRS[_m]),nW),np.float32),
                     np.stack(kl) if kl else np.zeros((0,len(DW_THRS[_m]),nW),np.float32)], axis=0)
                _dw[f"thr_{_m}"] = DW_THRS[_m]
        if _dw:
            _dw["wins"] = wins
            _dw["is_canonical"] = np.concatenate([np.ones(len(DWIN[DW_MECHS[0]]["Canonical"]),np.int8),
                                                  np.zeros(len(DWIN[DW_MECHS[0]]["Competitor"]),np.int8)])
            np.savez_compressed(f"{a.out}_denswin.npz", **_dw)
            print(f"[denswin] wrote {a.out}_denswin.npz  "
                  + "; ".join(f"{m}: {_dw[f'dw_{m}'].shape} thr={list(DW_THRS[m])}" for m in DW_MECHS
                              if f"dw_{m}" in _dw), file=sys.stderr)
        MET.to_csv(f"{a.out}_ucube_meta.tsv", sep="\t", index=False)
        np.savez_compressed(f"{a.out}_ucube.npz", cube=cube, me5cube=mcube, edges=UC_EDGES,
                            thrs=UC_THRS, me5_thrs=MC_THRS, u1_dn_span=np.array([a.u1_dn_span]))
        print(f"[u1-cube] wrote {a.out}_ucube.npz  cube={cube.shape} "
              f"(peaks x {UC_NB} bins of {int(UC_EDGES[1]-UC_EDGES[0])}nt over "
              f"[{int(UC_EDGES[0])},{int(UC_EDGES[-1])}] x [n,sum,max + {UC_NT} thr counts])", file=sys.stderr)
        if a.u1_dn_span < int(UC_EDGES[-1]):
            print(f"[u1-cube] WARNING: --u1_dn_span={a.u1_dn_span} < {int(UC_EDGES[-1])}; "
                  f"downstream bins past {a.u1_dn_span}nt are UNCOVERED (n=0), not empty.", file=sys.stderr)

    band=1.0
    gc={}; gk={}
    for c,store in (("Canonical",gc),("Competitor",gk)):
        for d in IE[c]:
            store.setdefault(d.get("gene"),[]).append((d.get("acc",np.nan),d.get("edef_pas",np.nan),d.get("interv_sum5",np.nan)))
    hits=0; tot=0; mp_diffs=[]
    for g in set(gc)&set(gk):
        for (aca,pasa,iva) in gc[g]:
            if not (np.isfinite(aca) and np.isfinite(iva) and np.isfinite(pasa)): continue
            cand=[iv for (ac,pas,iv) in gk[g] if np.isfinite(iv) and np.isfinite(pas) and pas==pasa and abs(ac-aca)<=band]
            if cand:
                tot+=1; mp_diffs.append(float(np.mean(cand)-iva))
                if np.mean(cand) > iva: hits+=1
    pd.DataFrame({"diff":mp_diffs}).to_csv(f"{a.out}_matched_pair_diffs.tsv", sep="\t", index=False)
    from math import sqrt, erf
    if tot:
        ph=hits/tot; z=1.96; den=1+z*z/tot
        lo=(ph+z*z/(2*tot)-z*sqrt(ph*(1-ph)/tot+z*z/(4*tot*tot)))/den
        hi=(ph+z*z/(2*tot)+z*sqrt(ph*(1-ph)/tot+z*z/(4*tot*tot)))/den
        zstat=(ph-0.5)/sqrt(0.25/tot); pval=2*(1-0.5*(1+erf(abs(zstat)/sqrt(2))))
    else: ph=lo=hi=pval=float("nan")
    pd.DataFrame([dict(n_pairs=tot, frac_comp_higher=ph, ci_lo=lo, ci_hi=hi, p=pval, band=band)]).to_csv(f"{a.out}_matched_pair.tsv", sep="\t", index=False)
    print(f"[matched-pair] competitor stronger intervening 5'ss in {ph:.0%} of {tot} PAS+acceptor-matched pairs (p={pval:.1e})", file=sys.stderr)

    def _au(key):
        pc=_arr("Canonical",key); kc=_arr("Competitor",key)
        pcf=pc[np.isfinite(pc)]; kcf=kc[np.isfinite(kc)]
        return (auroc_auprc(pcf,kcf)[0] if (len(pcf)>20 and len(kcf)>20) else np.nan), int(np.isfinite(pc).sum()), int(np.isfinite(kc).sum())
    mech_rows=[]
    for key,lab in (("lone5_best","lone 5'ss (no downstream 3'ss: telescripting)"),
                    ("unit5_best","complete unit (5'ss + downstream 3'ss)"),
                    ("edef_best5","any intervening 5'ss")):
        au,nc,nk=_au(key)
        mech_rows.append(dict(feature=key,label=lab,auroc=au,
                              disc=float(abs(au-0.5)) if np.isfinite(au) else np.nan,n_canon=nc,n_comp=nk))
    pd.DataFrame(mech_rows).to_csv(f"{a.out}_mechanism.tsv", sep="\t", index=False)
    print("[mechanism] lone vs complete-unit 5'ss discrimination (canonical=pos):", file=sys.stderr)
    for r in mech_rows:
        au=f"{r['auroc']:.3f}" if np.isfinite(r['auroc']) else "nan"
        print(f"    {r['label']:48} AUROC={au}  (n_canon {r['n_canon']}, n_comp {r['n_comp']})", file=sys.stderr)

    bd_rows=[]
    for key,lab in (("first5_from_anchor","nearest strong 5'ss from anchor (both sides)"),
                    ("dn5_best","best downstream 5'ss (past the site)"),
                    ("dn5_nearest_strong","nearest strong downstream 5'ss")):
        au,nc,nk=_au(key)
        vc=_arr("Canonical",key); vk=_arr("Competitor",key)
        bd_rows.append(dict(feature=key,label=lab,auroc=au,n_canon=nc,n_comp=nk,
                            canon_mean=float(np.nanmean(vc)) if np.isfinite(vc).any() else np.nan,
                            comp_mean=float(np.nanmean(vk)) if np.isfinite(vk).any() else np.nan))
    pd.DataFrame(bd_rows).to_csv(f"{a.out}_bidir_auroc.tsv", sep="\t", index=False)
    f5_rows=[]
    for c in ("Canonical","Competitor"):
        v=_arr(c,"first5_from_anchor")
        for x in v[np.isfinite(v)]: f5_rows.append(dict(cls=c, value=float(x)))
    pd.DataFrame(f5_rows).to_csv(f"{a.out}_first5_from_anchor.tsv", sep="\t", index=False)
    def _wil2(k,n,z=1.96):
        if n==0: return (np.nan,np.nan)
        ph=k/n; d=1+z*z/n
        return ((ph+z*z/(2*n)-z*np.sqrt(ph*(1-ph)/n+z*z/(4*n*n)))/d,
                (ph+z*z/(2*n)+z*np.sqrt(ph*(1-ph)/n+z*z/(4*n*n)))/d)
    iel_rows=[]
    for c in ("Canonical","Competitor"):
        v=np.array([d.get("internal_exon_like",0) for d in IE[c]], float)
        f=float(v.mean()); n=len(v); lo,hi=_wil2(int(round(f*n)),n)
        iel_rows.append(dict(cls=c, frac_internal_exon_like=f, lo=lo, hi=hi, n=n))
    pd.DataFrame(iel_rows).to_csv(f"{a.out}_internal_exon_like.tsv", sep="\t", index=False)
    print("[bidir-5ss] internal-exon 5'ss up/downstream of the site (canonical=pos):", file=sys.stderr)
    for r in bd_rows:
        au=f"{r['auroc']:.3f}" if np.isfinite(r['auroc']) else "nan"
        print(f"    {r['label']:44} AUROC={au}  (canon {r['canon_mean']:.0f} vs comp {r['comp_mean']:.0f} nt)", file=sys.stderr)
    ielc=[r for r in iel_rows if r['cls']=='Canonical'][0]; ielk=[r for r in iel_rows if r['cls']=='Competitor'][0]
    print(f"    internal-exon-like (strong 5'ss 50-300nt from anchor): canon {ielc['frac_internal_exon_like']:.0%} vs comp {ielk['frac_internal_exon_like']:.0%}", file=sys.stderr)

    if a.annot_5ss:
        ann_rows=[]
        for key,lab in (("annot_first5_from_anchor","nearest ANNOTATED 5'ss from anchor (both sides)"),
                        ("annot_dn5_dist","nearest ANNOTATED 5'ss downstream of the site")):
            au,nc,nk=_au(key)
            vc=_arr("Canonical",key); vk=_arr("Competitor",key)
            ann_rows.append(dict(feature=key,label=lab,auroc=au,n_canon=nc,n_comp=nk,
                                 canon_mean=float(np.nanmean(vc)) if np.isfinite(vc).any() else np.nan,
                                 comp_mean=float(np.nanmean(vk)) if np.isfinite(vk).any() else np.nan))
        pd.DataFrame(ann_rows).to_csv(f"{a.out}_annot_auroc.tsv", sep="\t", index=False)
        af_rows=[]
        for c in ("Canonical","Competitor"):
            v=_arr(c,"annot_first5_from_anchor")
            for x in v[np.isfinite(v)]: af_rows.append(dict(cls=c, value=float(x)))
        pd.DataFrame(af_rows).to_csv(f"{a.out}_annot_first5.tsv", sep="\t", index=False)
        aiel_rows=[]
        for c in ("Canonical","Competitor"):
            for fld,tag in (("annot_internal_exon_like","internal_exon_like"),("annot_dn5_within","dn5_within_400")):
                v=np.array([d.get(fld,0) for d in IE[c]], float)
                f=float(v.mean()); n=len(v); lo,hi=_wil2(int(round(f*n)),n)
                aiel_rows.append(dict(cls=c, measure=tag, frac=f, lo=lo, hi=hi, n=n))
        pd.DataFrame(aiel_rows).to_csv(f"{a.out}_annot_internal_exon_like.tsv", sep="\t", index=False)
        print("[annot-bidir] ANNOTATED internal-exon 5'ss (canonical=pos):", file=sys.stderr)
        for r in ann_rows:
            au=f"{r['auroc']:.3f}" if np.isfinite(r['auroc']) else "nan"
            print(f"    {r['label']:46} AUROC={au}  (canon {r['canon_mean']:.0f} vs comp {r['comp_mean']:.0f} nt)", file=sys.stderr)
        for tag in ("internal_exon_like","dn5_within_400"):
            cc=[r for r in aiel_rows if r['cls']=='Canonical' and r['measure']==tag][0]
            kk=[r for r in aiel_rows if r['cls']=='Competitor' and r['measure']==tag][0]
            print(f"    annotated {tag:20}: canon {cc['frac']:.0%} vs comp {kk['frac']:.0%}", file=sys.stderr)

    acc_thr=a.strong_acc
    relay_rows=[]
    for c in ("Canonical","Competitor"):
        accv=_arr(c,"acc"); iel=np.array([d.get("internal_exon_like",0) for d in IE[c]], float); fin=np.isfinite(accv)
        for s3 in (1,0):
            m3=fin & ((accv>=acc_thr).astype(int)==s3)
            for d5 in (1,0):
                relay_rows.append(dict(cls=c, strong_3ss=int(s3), has_dn5=int(d5), n=int((m3 & (iel==d5)).sum())))
    Rc=pd.DataFrame(relay_rows); Rc.to_csv(f"{a.out}_relay_conjunction.tsv", sep="\t", index=False)
    print(f"[u1-relay] conjunction (strong 3'ss>={acc_thr:.0f}, strong 5'ss>={a.nd_thr:.0f} at 50-300nt), P(canonical) per cell:", file=sys.stderr)
    for s3 in (1,0):
        for d5 in (1,0):
            sub=Rc[(Rc.strong_3ss==s3)&(Rc.has_dn5==d5)]
            nc=int(sub[sub.cls=="Canonical"].n.sum()); nk=int(sub[sub.cls=="Competitor"].n.sum()); tot=nc+nk
            p=nc/tot if tot else float("nan")
            print(f"    3'ss={'strong' if s3 else 'weak  '}  downstream_5'ss={'yes' if d5 else 'no '}  n={tot:>7}  P(canonical)={p:.3f}", file=sys.stderr)

    accP=np.concatenate([_arr("Canonical","acc"), _arr("Competitor","acc")])
    d5P =np.concatenate([_arr("Canonical","first5_from_anchor"), _arr("Competitor","first5_from_anchor")])
    yP  =np.concatenate([np.ones(len(IE["Canonical"])), np.zeros(len(IE["Competitor"]))])
    strong=accP>=acc_thr
    dr_edges=np.array([0,60,110,160,220,300,400,550,750,1100])
    rd_rows=[]
    has=strong & np.isfinite(d5P); yhas=yP[has]
    idx=np.clip(np.searchsorted(dr_edges,d5P[has],side="right")-1,0,len(dr_edges)-2)
    for j in range(len(dr_edges)-1):
        m=(idx==j); n=int(m.sum())
        if n<30: continue
        nc=int(yhas[m].sum()); nk=n-nc
        rd_rows.append(dict(order=j, bin=f"{int(dr_edges[j])}-{int(dr_edges[j+1])}",
                            dist=float((dr_edges[j]+dr_edges[j+1])/2), nc=nc, nk=nk))
    none_m=strong & ~np.isfinite(d5P)
    ncn=int(yP[none_m].sum()); nkn=int(none_m.sum()-ncn)
    rd_rows.append(dict(order=99, bin="none", dist=float("nan"), nc=ncn, nk=nkn))
    pd.DataFrame(rd_rows).to_csv(f"{a.out}_relay_dist.tsv", sep="\t", index=False)
    tc=sum(r["nc"] for r in rd_rows); tk=sum(r["nk"] for r in rd_rows)
    print("[u1-relay] nearest strong 5'ss distance (strong 3'ss), canon vs comp within-class %:", file=sys.stderr)
    for r in rd_rows:
        fc=r["nc"]/max(tc,1); fk=r["nk"]/max(tk,1); lr=np.log2((fc+1e-6)/(fk+1e-6))
        print(f"    {r['bin']:>10}nt  canon {fc:.1%}  comp {fk:.1%}  log2(canon/comp)={lr:+.2f}", file=sys.stderr)

    yC=yP==1; yK=yP==0
    sd_rows=[]
    for cval,ym in (("Canonical",yC),("Competitor",yK)):
        m=strong & ym & np.isfinite(d5P)
        for x in d5P[m]: sd_rows.append(dict(cls=cval, dist=float(x)))
    pd.DataFrame(sd_rows).to_csv(f"{a.out}_relay_strong_dist.tsv", sep="\t", index=False)
    iel_all=np.array([d.get("internal_exon_like",0) for d in IE["Canonical"]]
                     +[d.get("internal_exon_like",0) for d in IE["Competitor"]], float)
    fr_rows=[]
    for cval,ym in (("Canonical",yC),("Competitor",yK)):
        m=strong & ym; n=int(m.sum()); k=int(iel_all[m].sum()); f=(k/n) if n else float("nan")
        lo,hi=_wil2(k,n)
        fr_rows.append(dict(cls=cval, frac=f, lo=lo, hi=hi, n=n, k=k))
    pd.DataFrame(fr_rows).to_csv(f"{a.out}_relay_iel_frac.tsv", sep="\t", index=False)
    print("[u1-relay] internal-exon-like among strong-3'ss: " +
          "  ".join(f"{r['cls']} {r['frac']:.1%} (n={r['n']})" for r in fr_rows), file=sys.stderr)

    allacc = np.concatenate([accV[np.isfinite(accV)], accK[np.isfinite(accK)]])
    q1, q2 = np.quantile(allacc, [1/3, 2/3])
    def _stratum(x):
        return np.where(x < q1, "weak acceptor",
                np.where(x < q2, "medium acceptor", "strong acceptor"))
    rows_s = []
    for lab, av, bv in (("Canonical", accV, b5V), ("Competitor", accK, b5K)):
        m = np.isfinite(av) & np.isfinite(bv)
        for a_, b_ in zip(av[m], bv[m]):
            rows_s.append((float(a_), float(b_), 1 if lab=="Canonical" else 0))
    Sdf = pd.DataFrame(rows_s, columns=["acc","best5","canon"])
    Sdf["stratum"] = _stratum(Sdf.acc.values)
    b5_edges = np.arange(-5, 14+2, 2.0)
    b5_ctr = (b5_edges[:-1]+b5_edges[1:])/2
    strat_rows = []; slope_rows = []
    for st in ("weak acceptor","medium acceptor","strong acceptor"):
        sub = Sdf[Sdf.stratum==st]
        if len(sub) < 50: continue
        bi = np.clip(np.searchsorted(b5_edges, sub.best5.values, side="right")-1, 0, len(b5_ctr)-1)
        for j in range(len(b5_ctr)):
            cell = sub[bi==j]
            if len(cell) < a.min_cell: continue
            p = float(cell.canon.mean())
            strat_rows.append(dict(stratum=st, best5=float(b5_ctr[j]), p_canon=p, n=len(cell)))
        x = sub.best5.values; y = sub.canon.values.astype(float)
        if x.std() > 0:
            slope = float(np.polyfit(x, y, 1)[0])
            bs = []
            rng2 = np.random.default_rng(0)
            for _ in range(200):
                idx = rng2.integers(0, len(x), len(x))
                if x[idx].std() > 0: bs.append(np.polyfit(x[idx], y[idx], 1)[0])
            lo, hi = np.percentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan)
            acc_med = float(np.median(sub.acc.values))
            slope_rows.append(dict(stratum=st, acc_median=acc_med, slope=slope,
                                   slope_lo=float(lo), slope_hi=float(hi), n=len(sub)))
    pd.DataFrame(strat_rows).to_csv(f"{a.out}_strat_curves.tsv", sep="\t", index=False)
    pd.DataFrame(slope_rows).to_csv(f"{a.out}_strat_slopes.tsv", sep="\t", index=False)
    print("[stratified] P(canonical) vs best-5'ss slope within acceptor strata:", file=sys.stderr)
    for r in slope_rows:
        sig = "" if (r["slope_lo"]<=0<=r["slope_hi"]) else " *"
        print(f"    {r['stratum']:16} (acc~{r['acc_median']:.1f})  slope={r['slope']:+.4f} "
              f"[{r['slope_lo']:+.4f},{r['slope_hi']:+.4f}]{sig}", file=sys.stderr)

    m = np.isfinite(Sdf.acc) & np.isfinite(Sdf.best5)
    X = np.column_stack([np.ones(m.sum()),
                         (Sdf.acc[m]-Sdf.acc[m].mean())/Sdf.acc[m].std(),
                         (Sdf.best5[m]-Sdf.best5[m].mean())/Sdf.best5[m].std()])
    y = Sdf.canon[m].values.astype(float)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    print(f"[stratified] acceptor-controlled: canon ~ acc + best5 (standardized); "
          f"best5 partial slope = {beta[2]:+.4f} (acc slope {beta[1]:+.4f})", file=sys.stderr)

    accP = np.concatenate([accV, accK])
    b5P  = np.concatenate([b5V,  b5K])
    yP   = np.concatenate([np.ones(len(accV)), np.zeros(len(accK))])
    good = np.isfinite(accP) & np.isfinite(b5P)
    accP, b5P, yP = accP[good], b5P[good], yP[good]

    strat_rows = []
    qt = np.quantile(accP, [1/3, 2/3])
    acc_stratum = np.where(accP <= qt[0], "weak acceptor",
                    np.where(accP <= qt[1], "medium acceptor", "strong acceptor"))
    b5_edges = np.arange(-4, 13, 2.0)
    b5_ctr = (b5_edges[:-1] + b5_edges[1:]) / 2
    for stratum in ("weak acceptor","medium acceptor","strong acceptor"):
        sm = acc_stratum == stratum
        bidx = np.searchsorted(b5_edges, b5P[sm], side="right") - 1
        for j in range(len(b5_ctr)):
            cell = sm.copy(); cell[sm] = (bidx == j)
            n = int(cell.sum())
            if n < a.min_cell: continue
            pcanon = float(yP[cell].mean())
            strat_rows.append(dict(stratum=stratum, best5=float(b5_ctr[j]), n=n, p_canonical=pcanon))
    pd.DataFrame(strat_rows).to_csv(f"{a.out}_strat_curves.tsv", sep="\t", index=False)

    slope_rows = []
    aq = np.quantile(accP, np.linspace(0.05, 0.95, 10))
    for lo, hi in zip(aq[:-1], aq[1:]):
        m = (accP >= lo) & (accP < hi)
        if m.sum() < 200: continue
        x = b5P[m]; y = yP[m]
        X = np.column_stack([np.ones(m.sum()), (x - x.mean())/(x.std()+1e-9)])
        beta = np.zeros(2)
        for _ in range(25):
            eta = X @ beta; p = 1/(1+np.exp(-eta)); W = np.clip(p*(1-p),1e-6,None)
            z = eta + (y - p)/W
            beta_new = np.linalg.solve((X*W[:,None]).T @ X + 1e-6*np.eye(2), (X*W[:,None]).T @ z)
            if np.max(np.abs(beta_new-beta)) < 1e-6: beta = beta_new; break
            beta = beta_new
        slope_rows.append(dict(acceptor_mid=float((lo+hi)/2), slope_best5=float(beta[1]), n=int(m.sum())))
    pd.DataFrame(slope_rows).to_csv(f"{a.out}_slope_vs_acc.tsv", sep="\t", index=False)

    Xf = np.column_stack([np.ones(len(accP)),
                          (accP-accP.mean())/(accP.std()+1e-9),
                          (b5P-b5P.mean())/(b5P.std()+1e-9),
                          ((accP-accP.mean())/(accP.std()+1e-9))*((b5P-b5P.mean())/(b5P.std()+1e-9))])
    beta = np.zeros(4)
    for _ in range(50):
        eta = Xf @ beta; p = 1/(1+np.exp(-eta)); W = np.clip(p*(1-p),1e-6,None)
        z = eta + (yP - p)/W
        bn = np.linalg.solve((Xf*W[:,None]).T @ Xf + 1e-6*np.eye(4), (Xf*W[:,None]).T @ z)
        if np.max(np.abs(bn-beta)) < 1e-8: beta = bn; break
        beta = bn
    eta = Xf @ beta; p = 1/(1+np.exp(-eta)); W = np.clip(p*(1-p),1e-6,None)
    cov = np.linalg.inv((Xf*W[:,None]).T @ Xf + 1e-6*np.eye(4)); se = np.sqrt(np.diag(cov))
    names = ["intercept","acceptor","best5","acceptor:best5"]
    print("[interaction-logit] canonical ~ acceptor + best5 + acceptor:best5 (z-scored):", file=sys.stderr)
    for nm, b_, s_ in zip(names, beta, se):
        print(f"    {nm:16} beta={b_:+.3f}  OR={np.exp(b_):.3f}  z={b_/s_:+.1f}", file=sys.stderr)
    print("    (negative acceptor:best5 => 5'ss pushes toward COMPETITOR more when acceptor is STRONG)", file=sys.stderr)

    accA = {c: _arr(c,"acc") for c in ("Canonical","Competitor")}
    intA = {c: _arr(c,"interv_donor") for c in ("Canonical","Competitor")}
    ax_edges = np.arange(0, 14.0+1.0, 1.0)
    dx_edges = np.arange(-15.0, 14.0+1.5, 1.5)
    inter_rows=[]
    grids={}
    for c in ("Canonical","Competitor"):
        m = np.isfinite(accA[c]) & np.isfinite(intA[c])
        H,_,_ = np.histogram2d(accA[c][m], intA[c][m], bins=[ax_edges,dx_edges])
        grids[c] = H/ max(H.sum(),1)
    axc=(ax_edges[:-1]+ax_edges[1:])/2; dxc=(dx_edges[:-1]+dx_edges[1:])/2
    eps=1.0/min(np.isfinite(accA["Canonical"]).sum(), np.isfinite(accA["Competitor"]).sum(), 1)
    for i in range(len(axc)):
        for j in range(len(dxc)):
            tot = grids["Canonical"][i,j]+grids["Competitor"][i,j]
            if tot <= 0: continue
            lr = np.log2((grids["Canonical"][i,j]+eps)/(grids["Competitor"][i,j]+eps))
            inter_rows.append(dict(acceptor=float(axc[i]), interv_donor=float(dxc[j]), log2ratio=float(lr)))
    pd.DataFrame(inter_rows).to_csv(f"{a.out}_interaction.tsv", sep="\t", index=False)
    print(f"[interaction] wrote acceptor x intervening-donor heatmap", file=sys.stderr)

    asym_rows=[]; asym_pair=[]
    genesC={}; genesK={}
    for c,store in (("Canonical",genesC),("Competitor",genesK)):
        accv=_arr(c,"acc"); intv=_arr(c,"interv_donor"); gv=[d.get("gene") for d in IE[c]]
        for i in range(len(accv)):
            if np.isfinite(accv[i]) and np.isfinite(intv[i]):
                val=float(accv[i]-intv[i]); asym_rows.append(dict(cls=c, asym=val))
                store.setdefault(gv[i],[]).append(val)
    pd.DataFrame(asym_rows).to_csv(f"{a.out}_asymmetry.tsv", sep="\t", index=False)
    diffs=[np.mean(genesC[g])-np.mean(genesK[g]) for g in set(genesC)&set(genesK)]
    if diffs:
        diffs=np.array(diffs)
        pd.DataFrame({"diff":diffs}).to_csv(f"{a.out}_asymmetry_paired.tsv", sep="\t", index=False)
        pcf=np.array([r["asym"] for r in asym_rows if r["cls"]=="Canonical"])
        kcf=np.array([r["asym"] for r in asym_rows if r["cls"]=="Competitor"])
        au,_,_=auroc_auprc(pcf,kcf)
        print(f"[asymmetry] AUROC(acc-intervDonor)={au:.3f}  within-gene median canon-comp={np.median(diffs):+.2f} "
              f"({(diffs>0).mean():.0%} genes canon higher)", file=sys.stderr)

    u1f_rows=[]
    for c in ("Canonical","Competitor"):
        v=_arr(c,"interv_u1flank"); v=v[np.isfinite(v)]
        for x in v: u1f_rows.append(dict(cls=c, u1flank=float(x)))
    if u1f_rows:
        pd.DataFrame(u1f_rows).to_csv(f"{a.out}_u1flank.tsv", sep="\t", index=False)
        pc=np.array([r["u1flank"] for r in u1f_rows if r["cls"]=="Canonical"])
        kc=np.array([r["u1flank"] for r in u1f_rows if r["cls"]=="Competitor"])
        if len(pc)>20 and len(kc)>20:
            au,_,_=auroc_auprc(pc,kc)
            print(f"[u1flank] AUROC(U1 within 50nt of intervening donor)={au:.3f}", file=sys.stderr)

    nsi_rows=[]
    for c in ("Canonical","Competitor"):
        v=np.array([d.get("n_strong_interv",0) for d in IE[c]], float)
        for x in v: nsi_rows.append(dict(cls=c, n_strong=int(x)))
    if nsi_rows:
        pd.DataFrame(nsi_rows).to_csv(f"{a.out}_n_intervening.tsv", sep="\t", index=False)
        pc=np.array([r["n_strong"] for r in nsi_rows if r["cls"]=="Canonical"],float)
        kc=np.array([r["n_strong"] for r in nsi_rows if r["cls"]=="Competitor"],float)
        au,_,_=auroc_auprc(pc,kc)
        print(f"[n-intervening] mean strong donors: canon={pc.mean():.2f} comp={kc.mean():.2f} AUROC={au:.3f}", file=sys.stderr)

    uc = (U1DN_EDGES[:-1] + U1DN_EDGES[1:]) / 2
    u4 = []
    for c in ("Canonical","Competitor"):
        mean = np.where(u1dn_cnt[c] > 0, u1dn_sum[c]/np.maximum(u1dn_cnt[c],1), np.nan)
        for i in range(nU):
            if np.isfinite(mean[i]): u4.append(dict(cls=c, dist_downstream=float(uc[i]), mean_u1=float(mean[i]), n=int(u1dn_cnt[c][i])))
    pd.DataFrame(u4).to_csv(f"{a.out}_u1_downstream.tsv", sep="\t", index=False)
    print(f"[u1-downstream] wrote {a.out}_u1_downstream.tsv (U1 profiled 3' of each peak's own site)", file=sys.stderr)

    for gname, gvals, gthrs in (("u1", u1grid_vals, U1G_THRS), ("me5", me5grid_vals, ME5G_THRS)):
        if not (gvals["Canonical"] and gvals["Competitor"]): continue
        GC = np.array(gvals["Canonical"]); GK = np.array(gvals["Competitor"])
        nB, nT = GC.shape[1], GC.shape[2]
        gr_rows = []
        for bi_ in range(nB):
            for ti in range(nT):
                pc = GC[:, bi_, ti]; kc = GK[:, bi_, ti]
                pc = pc[np.isfinite(pc)]; kc = kc[np.isfinite(kc)]
                if len(pc) < 20 or len(kc) < 20: continue
                au = auroc_auprc(pc, kc)[0]
                lo, hi = U1G_BANDS[bi_]; thr = gthrs[ti]
                gr_rows.append(dict(signal=gname, band=f"{int(lo)}-{int(hi)}",
                                    band_lo=int(lo), band_hi=int(hi), thr=float(thr),
                                    auroc=au, disc=abs(au-0.5),
                                    canon_mean=float(pc.mean()), comp_mean=float(kc.mean())))
        GRD = pd.DataFrame(gr_rows)
        GRD.to_csv(f"{a.out}_{gname}_search.tsv", sep="\t", index=False)
        print(f"[{gname}-search] {gname} density (sites/kb) past anchor 3'ss, best cells by |AUROC-0.5|:", file=sys.stderr)
        for _, r in GRD.reindex(GRD.disc.sort_values(ascending=False).index).head(6).iterrows():
            print(f"    band {r['band']:>8}nt  thr>={r['thr']:+.1f}  AUROC={r['auroc']:.3f}  "
                  f"(canon {r['canon_mean']:.1f} vs comp {r['comp_mean']:.1f} sites/kb)", file=sys.stderr)

    from math import erf, sqrt as _sqrt
    accCg = {}; accKg = {}
    for d in IE["Canonical"]:
        v = d.get("acc", np.nan)
        if np.isfinite(v): accCg.setdefault(d.get("gene"), []).append(v)
    for d in IE["Competitor"]:
        v = d.get("acc", np.nan)
        if np.isfinite(v): accKg.setdefault(d.get("gene"), []).append(v)
    pair_genes = sorted(set(accCg) & set(accKg))
    rows_pair = []; gwin = []; wsum = 0.0; tsum = 0.0
    for gname in pair_genes:
        cv = np.array(accCg[gname], float); kv = np.array(accKg[gname], float)
        w = float((cv[:,None] > kv[None,:]).sum() + 0.5*(cv[:,None] == kv[None,:]).sum())
        t = cv.size * kv.size
        wsum += w; tsum += t; wr = w/t; gwin.append(wr)
        rows_pair.append(dict(gene=gname, canon_win_rate=wr, n_canon=cv.size, n_comp=kv.size))
    gwin = np.array(gwin, float); nG = len(gwin)
    if nG:
        within_auroc = float(gwin.mean())
        pooled_auroc = float(wsum/tsum)
        frac = float((gwin > 0.5).mean()); medwr = float(np.median(gwin))
        d = gwin - 0.5; nz = d[d != 0]; nn = len(nz)
        if nn:
            absd = np.abs(nz)
            uv, inv, cnt = np.unique(absd, return_inverse=True, return_counts=True)
            cs = np.cumsum(cnt); ranks = (((cs - cnt) + cs + 1)/2.0)[inv]
            Wp = float(ranks[nz > 0].sum()); muW = nn*(nn+1)/4.0; sdW = _sqrt(nn*(nn+1)*(2*nn+1)/24.0)
            p_wilcox = 2*(1 - 0.5*(1 + erf(abs((Wp - muW)/sdW)/_sqrt(2)))) if sdW > 0 else np.nan
            k = int((nz > 0).sum())
            p_sign = 2*(1 - 0.5*(1 + erf(abs((k - nn/2)/_sqrt(nn/4.0))/_sqrt(2))))
        else:
            p_wilcox = p_sign = np.nan
        pd.DataFrame(rows_pair).to_csv(f"{a.out}_acc_paired.tsv", sep="\t", index=False)
        pd.DataFrame([dict(n_genes=nG, within_gene_auroc=within_auroc, pooled_auroc=pooled_auroc,
                           frac_genes_canon=frac, median_winrate=medwr,
                           p_sign=p_sign, p_wilcox=p_wilcox)]).to_csv(
            f"{a.out}_acc_paired_summary.tsv", sep="\t", index=False)
        print(f"[acc-paired] within-gene acceptor AUROC {within_auroc:.3f} (gene-avg); true end "
              f"outranks competitor in {frac:.0%} of {nG} genes (median win-rate {medwr:.2f}, "
              f"Wilcoxon p={p_wilcox:.1e})", file=sys.stderr)

    NBOOT = 2; rng_b = np.random.default_rng(a.seed)
    gidxC = {}; gidxK = {}
    for i, gname in enumerate(GENE["Canonical"]): gidxC.setdefault(gname, []).append(i)
    for i, gname in enumerate(GENE["Competitor"]): gidxK.setdefault(gname, []).append(i)
    idxC_by_g = {g: np.array(v, int) for g, v in gidxC.items()}
    idxK_by_g = {g: np.array(v, int) for g, v in gidxK.items()}
    allg = sorted(set(gidxC) | set(gidxK))
    def _fast_auroc(pos, neg):
        pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]
        if len(pos) < 20 or len(neg) < 20: return np.nan
        return auroc_auprc(pos, neg)[0]
    ci_rows = []
    for m in MECHS:
        RMc = np.array(RM[m]["Canonical"], float); RMk = np.array(RM[m]["Competitor"], float)
        use_dens = (m != "acceptor")
        if use_dens:
            Dc = np.array(DEN[m]["Canonical"], float); Dk = np.array(DEN[m]["Competitor"], float)
        boot = np.full((NBOOT, nW), np.nan)
        for b in range(NBOOT):
            gs = [allg[j] for j in rng_b.integers(0, len(allg), len(allg))]
            rc = [idxC_by_g[g] for g in gs if g in idxC_by_g]
            rk = [idxK_by_g[g] for g in gs if g in idxK_by_g]
            if not rc or not rk: continue
            rc = np.concatenate(rc); rk = np.concatenate(rk)
            if len(rc) < 20 or len(rk) < 20: continue
            src_c = Dc if use_dens else RMc; src_k = Dk if use_dens else RMk
            for wi in range(nW):
                boot[b, wi] = _fast_auroc(src_c[rc, wi], src_k[rk, wi])
        lo = np.nanpercentile(boot, 2.5, axis=0); hi = np.nanpercentile(boot, 97.5, axis=0)
        md = np.nanmedian(boot, axis=0)
        for wi, w in enumerate(wins):
            ci_rows.append(dict(mech=m, window=int(w), lo=float(lo[wi]), hi=float(hi[wi]), med=float(md[wi])))
    pd.DataFrame(ci_rows).to_csv(f"{a.out}_auroc_ci.tsv", sep="\t", index=False)
    print(f"[auroc-ci] wrote gene-bootstrap AUROC band ({NBOOT} reps) -> {a.out}_auroc_ci.tsv", file=sys.stderr)

    print(f"[peaks] {npk}; {n_bad} unfetchable", file=sys.stderr)
    print(f"[out] base + conjunction + periodicity + u1_downstream", file=sys.stderr)

if __name__ == "__main__":
    main()


