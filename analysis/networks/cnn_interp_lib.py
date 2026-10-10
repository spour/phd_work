#!/usr/bin/env python3
# the pure (model-free) helpers shared by cnn_interpret.py's logic and cnn_interpret2.py
import numpy as np
NT = "ACGT"

REF_MOTIFS = {
    "PAS_AATAAA": "AATAAA",
    "PAS_ATTAAA": "ATTAAA",
    "acceptor_3ss_pyAG": "TTTTTTTTTTNCAGG",
    "donor_5ss": "CAGGTAAGT",
    "CA_cleavage": "CA",
}

def consensus_pwm(s):
    m = np.full((len(s), 4), 0.05)
    for i, c in enumerate(s):
        if c == "N": m[i] = 0.25
        else: m[i] = 0.05; m[i, NT.index(c)] = 0.85
    return m / m.sum(1, keepdims=True)

def pwm_match(p, q, min_overlap=6):
    def sc(a, b):
        best = -1.0
        for off in range(-len(b) + min_overlap, len(a) - min_overlap + 1):
            i0, i1 = max(0, off), min(len(a), off + len(b))
            j0, j1 = i0 - off, i1 - off
            if i1 - i0 < min_overlap: continue
            u = a[i0:i1].ravel(); v = b[j0:j1].ravel()
            if u.std() < 1e-9 or v.std() < 1e-9: continue
            best = max(best, float(np.corrcoef(u, v)[0, 1]))
        return best
    rc = q[::-1, ::-1]
    f, r = sc(p, q), sc(p, rc)
    return (f, "+") if f >= r else (r, "-")

def match_threshold(refs, rng, n=300, w=19, q=0.95):
    return {k: float(np.quantile([pwm_match(rng.dirichlet(np.ones(4) * 0.35, w), ref)[0]
                                  for _ in range(n)], q)) for k, ref in refs.items()}

def write_meme(pwm, nsite, path, W=19):
    with open(path, "w") as fh:
        fh.write("MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\n"
                 "Background letter frequencies\nA 0.25 C 0.25 G 0.25 T 0.25\n\n")
        for f in range(len(pwm)):
            fh.write(f"MOTIF filter{f:02d}\nletter-probability matrix: alength= 4 w= {W} "
                     f"nsites= {max(int(nsite[f]),1)} E= 0\n")
            for r in pwm[f]: fh.write(" " + " ".join(f"{x:.6f}" for x in r) + "\n")
            fh.write("\n")

def dinuc_shuffle(tok, rng):
    t = tok[tok < 4]
    if len(t) < 3: return tok.copy()
    last = t[-1]; edges = [[] for _ in range(4)]
    for i in range(len(t) - 1): edges[t[i]].append(t[i + 1])
    for _ in range(100):
        pick = {}; okv = True
        for v in range(4):
            if not edges[v] or v == last: continue
            pick[v] = edges[v][rng.integers(len(edges[v]))]
        for v in list(pick):
            seen = set(); u = v
            while u != last and u in pick and u not in seen:
                seen.add(u); u = pick[u]
            if u != last: okv = False; break
        if okv: break
    else:
        return tok.copy()
    order = []
    for v in range(4):
        e = list(edges[v])
        if v in pick: e.remove(pick[v]); rng.shuffle(e); e.append(pick[v])
        else: rng.shuffle(e)
        order.append(e)
    ptr = [0, 0, 0, 0]; out = [t[0]]; u = t[0]
    for _ in range(len(t) - 1):
        if ptr[u] >= len(order[u]): break
        nxt = order[u][ptr[u]]; ptr[u] += 1; out.append(nxt); u = nxt
    out = np.array(out, tok.dtype)
    if len(out) != len(t): return tok.copy()
    res = tok.copy(); res[tok < 4] = out
    return res
