# minimal pwms (log-odds-ish, transparent stand-ins)


# helpers


# interpretable, mechanism-grounded features for the cpa reranker
import numpy as np

_PYR = set("CTU")
def branchpoint_score(seq, upstream_nt, acc_pos=None, score3=None, m3=None,
                      bp_lo=18, bp_hi=45, win=7):
    if seq is None or len(seq) < upstream_nt: return np.nan
    if acc_pos is None:
        acc_pos = _find_acceptor(seq, upstream_nt, score3, m3)
    if acc_pos is None: return np.nan
    lo = max(0, acc_pos - bp_hi); hi = max(0, acc_pos - bp_lo)
    if hi - lo < win: return np.nan
    region = seq[lo:hi].upper()
    best = 0.0
    for i in range(len(region) - win + 1):
        w = region[i:i+win]
        s = 0.0
        for j in range(2, len(w)-1):
            if w[j] == "A":
                s_local = 1.0
                if w[j-1] == "T": s_local += 1.5
                if j-2 >= 0 and w[j-2] in _PYR: s_local += 0.5
                if j+1 < len(w) and w[j+1] in _PYR: s_local += 0.5
                best = max(best, s_local)
    return float(best)

def u2af65_ppt(seq, upstream_nt, acc_pos=None, score3=None, m3=None,
               lo_nt=5, hi_nt=40, win=20):
    if seq is None or len(seq) < upstream_nt: return np.nan
    if acc_pos is None:
        acc_pos = _find_acceptor(seq, upstream_nt, score3, m3)
    if acc_pos is None: return np.nan
    lo = max(0, acc_pos - hi_nt); hi = max(0, acc_pos - lo_nt)
    if hi - lo < win: return np.nan
    region = seq[lo:hi].upper()
    wts = np.fromiter(((1.0 if b in "TU" else 0.5 if b == "C" else 0.0) for b in region),
                      dtype=np.float32, count=len(region))
    if len(wts) < win: return np.nan
    conv = np.convolve(wts, np.ones(win, np.float32), mode="valid")
    return float(conv.max() / win)

def u2af35_ag(seq, upstream_nt, acc_pos=None, score3=None, m3=None):
    if seq is None or len(seq) < upstream_nt: return np.nan
    if acc_pos is None:
        acc_pos = _find_acceptor(seq, upstream_nt, score3, m3)
    if acc_pos is None or acc_pos < 3: return np.nan
    ag = seq[acc_pos-2:acc_pos].upper()
    s = 2.0 if ag == "AG" else 0.0
    if acc_pos-3 >= 0 and seq[acc_pos-3].upper() in _PYR: s += 0.5
    if acc_pos < len(seq) and seq[acc_pos].upper() == "G": s += 0.25
    return float(s)

def _donor_density(seq, lo, hi, thresh=0.6):
    lo=max(0,lo); hi=min(len(seq),hi)
    if hi-lo < 12: return np.nan
    region=seq[lo:hi].upper()
    arr=np.frombuffer(region.encode("ascii","ignore"),dtype=np.uint8); L=arr.size
    if L<10: return np.nan
    gt=(arr[3:L-5]==71)&(arr[4:L-4]==84)
    cand=np.nonzero(gt)[0]
    if cand.size==0: return 0.0
    hits=sum(1 for i in cand if _donor_score(region[i:i+9])>=thresh)
    return float(100.0*hits/max(L-9,1))

def u1_suppression_down(seq, upstream_nt, downstream_nt, win_nt=500, thresh=0.6):
    if seq is None or len(seq)<upstream_nt+12: return np.nan
    return _donor_density(seq, upstream_nt, upstream_nt+win_nt, thresh)

def u1_suppression_up(seq, upstream_nt, downstream_nt, win_nt=500, thresh=0.6):
    if seq is None or len(seq)<upstream_nt+12: return np.nan
    return _donor_density(seq, upstream_nt-win_nt, upstream_nt, thresh)

_DONOR = {0:{"A":.3,"C":.1,"G":.4,"T":.2},1:{"A":.6,"C":.1,"G":.1,"T":.2},
          2:{"A":.1,"C":.0,"G":.8,"T":.1},3:{"G":1.0},4:{"T":1.0},
          5:{"A":.6,"G":.3,"C":.05,"T":.05},6:{"A":.7,"G":.1,"C":.1,"T":.1},
          7:{"G":.8,"A":.1,"C":.05,"T":.05},8:{"T":.5,"C":.2,"A":.2,"G":.1}}
def _donor_score(k):
    if len(k) < 9 or k[3:5] != "GT": return 0.0
    return float(np.mean([_DONOR[i].get(k[i],0.0) for i in range(9)]))

def _find_acceptor(seq, upstream_nt, score3=None, m3=None, zone_lo=25, zone_hi=500):
    lo = max(0, upstream_nt - zone_hi); hi = max(0, upstream_nt - zone_lo)
    if hi - lo < 25: return None
    best_i, best_s = None, -1e9
    region = seq[lo:hi].upper()
    _ra = np.frombuffer(region.encode("ascii","replace"), dtype=np.uint8)
    _ag = np.where((_ra[:-1]==65) & (_ra[1:]==71))[0] + 2
    for i in _ag:
        i=int(i)
        if i<2 or i>len(region): continue
        if score3 is not None and m3 is not None:
            s0 = lo + i - 20
            if s0 < 0: continue
            try: s = score3(seq[s0:lo+i+3].upper(), m3)
            except Exception: s = _acc_heur(seq, lo+i)
        else:
            s = _acc_heur(seq, lo+i)
        if s > best_s: best_s, best_i = s, lo + i
    return best_i

def _acc_heur(seq, acc):
    ppt = seq[max(0,acc-25):acc-2].upper()
    return sum(1 for b in ppt if b in _PYR) / max(len(ppt),1)


def _scan_3ss(seq, lo, hi, score3=None, m3=None, step=3):
    lo=max(0,lo); hi=min(len(seq),hi)
    if hi-lo<25: return (np.nan,None)
    region=seq[lo:hi].upper()
    ra=np.frombuffer(region.encode("ascii","replace"),dtype=np.uint8)
    ag=np.where((ra[:-1]==65)&(ra[1:]==71))[0]+2
    best_s=-1e9; best_p=None
    for i in ag[::1]:
        i=int(i); p=lo+i
        if score3 is not None and m3 is not None:
            s0=p-20
            if s0<0: continue
            try: s=score3(seq[s0:p+3].upper(),m3)
            except Exception: s=_acc_heur(seq,p)
        else:
            s=_acc_heur(seq,p)
        if s>best_s: best_s=s; best_p=p
    return (float(best_s) if best_p is not None else np.nan, best_p)

def acceptor_competition(seq, upstream_nt, downstream_nt, score3=None, m3=None,
                         flank=2000):
    out=dict(term_3ss_abs=np.nan, comp_up_3ss=np.nan, comp_dn_3ss=np.nan,
             rel_3ss_up=np.nan, rel_3ss_dn=np.nan)
    if seq is None or len(seq)<upstream_nt: return out
    t_s,t_p=_scan_3ss(seq, max(0,upstream_nt-500), upstream_nt, score3, m3)
    out["term_3ss_abs"]=t_s
    up_s,_=_scan_3ss(seq, max(0,upstream_nt-flank), max(0,upstream_nt-500), score3, m3)
    dn_s,_=_scan_3ss(seq, upstream_nt+50, min(len(seq),upstream_nt+flank), score3, m3)
    out["comp_up_3ss"]=up_s; out["comp_dn_3ss"]=dn_s
    if np.isfinite(t_s) and np.isfinite(up_s): out["rel_3ss_up"]=t_s-up_s
    if np.isfinite(t_s) and np.isfinite(dn_s): out["rel_3ss_dn"]=t_s-dn_s
    return out

def donor_geometry(seq, upstream_nt, downstream_nt, thresh=0.6, flank=2000):
    out=dict(dist_nearest_5ss=np.nan, dist_up_5ss=np.nan, dist_dn_5ss=np.nan)
    if seq is None or len(seq)<upstream_nt+12: return out
    lo=max(0,upstream_nt-flank); hi=min(len(seq),upstream_nt+flank)
    region=seq[lo:hi].upper()
    arr=np.frombuffer(region.encode("ascii","ignore"),dtype=np.uint8); L=arr.size
    if L<10: return out
    gt=(arr[3:L-5]==71)&(arr[4:L-4]==84)
    cand=np.nonzero(gt)[0]
    strong=[lo+i for i in cand if _donor_score(region[i:i+9])>=thresh]
    if not strong: return out
    strong=np.array(strong)
    d=strong-upstream_nt
    out["dist_nearest_5ss"]=float(np.min(np.abs(d)))
    up=d[d<0]; dn=d[d>0]
    if len(up): out["dist_up_5ss"]=float(-up.max())
    if len(dn): out["dist_dn_5ss"]=float(dn.min())
    return out

def donor_counts(seq, upstream_nt, downstream_nt, thresh=0.6):
    out=dict(n_5ss_500=np.nan, n_5ss_1k=np.nan, sum_5ss=np.nan, mean_top3_5ss=np.nan)
    if seq is None or len(seq)<upstream_nt+12: return out
    L=len(seq)
    arr=np.frombuffer(seq.upper().encode("ascii","ignore"),dtype=np.uint8); n=arr.size
    if n<10: return out
    gt=(arr[3:n-5]==71)&(arr[4:n-4]==84)
    cand=np.nonzero(gt)[0]
    scores=[]; pos=[]
    for i in cand:
        s=_donor_score(seq[i:i+9].upper())
        if s>=thresh: scores.append(s); pos.append(i)
    if not scores:
        return dict(n_5ss_500=0.0,n_5ss_1k=0.0,sum_5ss=0.0,mean_top3_5ss=0.0)
    scores=np.array(scores); pos=np.array(pos)
    d=np.abs(pos-upstream_nt)
    out["n_5ss_500"]=float((d<=500).sum())
    out["n_5ss_1k"]=float((d<=1000).sum())
    out["sum_5ss"]=float(scores.sum())
    out["mean_top3_5ss"]=float(np.sort(scores)[::-1][:3].mean())
    return out


def _gtract_score(s):
    if not s: return (0.0,0.0)
    best=0; cur=0; n3=0; run=0
    for ch in s.upper():
        if ch=="G":
            cur+=1; run+=1; best=max(best,cur)
        else:
            if run>=3: n3+=1
            cur=0; run=0
    if run>=3: n3+=1
    return (float(best), float(n3))

def positioned_upstream_elements(seq, upstream_nt, downstream_nt,
                                 g_lo=-400, g_hi=-250, at_lo=-200, at_hi=-100):
    out=dict(up_g_frac=np.nan, up_g_longrun=np.nan, up_g_n3=np.nan,
             up_gc_frac=np.nan, up_at_frac_mid=np.nan, up_purine_skew=np.nan)
    if seq is None or len(seq) < upstream_nt: return out
    c=upstream_nt
    def grab(lo,hi):
        a=max(0,c+lo); b=min(len(seq),c+hi)
        return seq[a:b].upper() if b>a else ""
    gwin=grab(g_lo,g_hi); atwin=grab(at_lo,at_hi)
    if gwin:
        L=len(gwin)
        out["up_g_frac"]=gwin.count("G")/L
        out["up_gc_frac"]=(gwin.count("G")+gwin.count("C"))/L
        lr,n3=_gtract_score(gwin); out["up_g_longrun"]=lr; out["up_g_n3"]=n3
        gc=gwin.count("G")+gwin.count("C")
        out["up_purine_skew"]=(gwin.count("G")-gwin.count("C"))/gc if gc else 0.0
    if atwin:
        out["up_at_frac_mid"]=(atwin.count("A")+atwin.count("T"))/len(atwin)
    return out


def _gc_content(s):
    if not s: return np.nan
    s=s.upper(); n=len(s)
    return (s.count("G")+s.count("C"))/n if n else np.nan

def _g_content(s):
    if not s: return np.nan
    s=s.upper(); return s.count("G")/len(s) if s else np.nan

def _at_content(s):
    if not s: return np.nan
    s=s.upper(); return (s.count("A")+s.count("T"))/len(s) if s else np.nan

def _max_g_run(s):
    if not s: return 0
    s=s.upper(); best=cur=0
    for ch in s:
        if ch=="G": cur+=1; best=max(best,cur)
        else: cur=0
    return best

def _g4_score(s):
    if not s: return 0
    import re
    return len(re.findall(r"G{3,}", s.upper()))

def positioned_gtract(seq, upstream_nt, downstream_nt,
                      up_lo=400, up_hi=250, comp_lo=200, comp_hi=100):
    out=dict(up_g_content=np.nan, up_gc_content=np.nan, up_max_g_run=np.nan,
             up_g4_score=np.nan, comp_at_content=np.nan, comp_u_richness=np.nan,
             up_minus_comp_g=np.nan)
    if seq is None or len(seq) < upstream_nt: return out
    L=len(seq)
    a0=max(0, upstream_nt-up_lo); a1=max(0, upstream_nt-up_hi)
    c0=max(0, upstream_nt-comp_lo); c1=max(0, upstream_nt-comp_hi)
    up=seq[a0:a1]; comp=seq[c0:c1]
    if up:
        out["up_g_content"]=_g_content(up); out["up_gc_content"]=_gc_content(up)
        out["up_max_g_run"]=float(_max_g_run(up)); out["up_g4_score"]=float(_g4_score(up))
    if comp:
        out["comp_at_content"]=_at_content(comp)
        cu=comp.upper(); out["comp_u_richness"]=cu.count("T")/len(cu) if cu else np.nan
    if np.isfinite(out["up_g_content"]) and np.isfinite(out.get("comp_at_content",np.nan)):
        out["up_minus_comp_g"]=out["up_g_content"]-(1-out["comp_at_content"])/2
    return out

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
    out=dict(dist_to_max_3ss=np.nan)
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
    out["dist_to_max_3ss"]=float(dist)
    return out
