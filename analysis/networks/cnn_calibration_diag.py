#!/usr/bin/env python3
# why cnn_train.py reports temperature 1225.8
import sys
import numpy as np

def log(*a): print(*a, file=sys.stderr, flush=True)
S = np.load(sys.argv[1] if len(sys.argv) > 1 else "cnn_model_scores.npz", allow_pickle=True)
lg = S["logit"].astype(np.float64); y = S["y"].astype(np.float64); sp = S["split"]
log(f"stored temperature {float(S['temperature']):.4f}")

def bce(z, yy): return float(np.mean(np.logaddexp(0, np.clip(z, -60, 60)) - yy * z))

rows = []
for s in ("train", "val", "test"):
    m = sp == s
    rows.append((s, int(m.sum()), lg[m].mean(), lg[m].std(), lg[m].min(), lg[m].max(),
                 lg[m & (y == 1)].mean(), lg[m & (y == 0)].mean(), float((lg[m & (y == 0)] > 0).mean())))
    log(f"[scale] {s:5s} n {rows[-1][1]:6d}  mean {rows[-1][2]:+7.3f}  sd {rows[-1][3]:6.3f}  "
        f"range [{rows[-1][4]:+.2f},{rows[-1][5]:+.2f}]  real {rows[-1][6]:+.3f}  "
        f"competitor {rows[-1][7]:+.3f}  competitors with logit>0: {rows[-1][8]:.3f}")

sv = lg[sp == "val"]; yv = y[sp == "val"]; p = yv.mean()
log(f"\n[offset] val base rate {p:.5f}, logit(base) {np.log(p/(1-p)):+.4f}; "
    f"mean raw logit {sv.mean():+.4f}  ->  offset {sv.mean()-np.log(p/(1-p)):+.2f}")

Xd = np.column_stack([sv, np.ones(len(sv))]); b = np.array([0.5, -1.0]); cur = bce(Xd @ b, yv)
for _ in range(200):
    z = np.clip(Xd @ b, -30, 30); q = 1 / (1 + np.exp(-z)); Wq = np.maximum(q * (1 - q), 1e-6)
    g = Xd.T @ (q - yv) / len(yv); H = (Xd * Wq[:, None]).T @ Xd / len(yv) + 1e-8 * np.eye(2)
    d = np.linalg.solve(H, g); t = 1.0
    while t > 1e-10 and bce(Xd @ (b - t * d), yv) > cur - 1e-12: t *= 0.5
    b = b - t * d; new = bce(Xd @ b, yv)
    if cur - new < 1e-12: break
    cur = new

log("\n[calibration] unweighted BCE on val")
log(f"  raw, T=1                        {bce(sv, yv):.5f}")
for T in (2, 4, 8, 50, 1000, 1225.8):
    log(f"  temperature-only, T={T:<8g}    {bce(sv/T, yv):.5f}")
log(f"  temperature-only, T=infinity    {np.log(2):.5f}   <- all predictions 0.5")
log(f"  intercept only (base rate)      {-(p*np.log(p)+(1-p)*np.log(1-p)):.5f}")
log(f"  Platt, scale {b[0]:.4f} bias {b[1]:+.4f}   {cur:.5f}   <- T equivalent {1/b[0]:.3f}")
log("\n[verdict] the temperature runs away because temperature scaling has no bias term and "
    "cannot undo the pos_weight offset, not because the logits are large. Adding the bias "
    f"calibrates the model at T = {1/b[0]:.2f}. Ranking metrics and real/shuffled attribution "
    "ratios are invariant to this rescaling.")

with open("calibration_diagnosis.tsv", "w") as fh:
    fh.write("split\tn\tmean_logit\tsd_logit\tmin\tmax\tmean_real\tmean_competitor\tfrac_comp_pos\n")
    for r in rows:
        fh.write(f"{r[0]}\t{r[1]}\t{r[2]:.5f}\t{r[3]:.5f}\t{r[4]:.5f}\t{r[5]:.5f}\t"
                 f"{r[6]:.5f}\t{r[7]:.5f}\t{r[8]:.5f}\n")
    fh.write(f"calibration\ttemperature_only\t{float(S['temperature']):.5f}\t\t\t\t\t\t\n")
    fh.write(f"calibration\tplatt_scale\t{b[0]:.5f}\t\t\t\t\t\t\n")
    fh.write(f"calibration\tplatt_bias\t{b[1]:.5f}\t\t\t\t\t\t\n")
    fh.write(f"calibration\tplatt_equiv_T\t{1/b[0]:.5f}\t\t\t\t\t\t\n")
    fh.write(f"calibration\tbce_raw\t{bce(sv,yv):.5f}\t\t\t\t\t\t\n")
    fh.write(f"calibration\tbce_temp_only\t{bce(sv/float(S['temperature']),yv):.5f}\t\t\t\t\t\t\n")
    fh.write(f"calibration\tbce_platt\t{cur:.5f}\t\t\t\t\t\t\n")
    fh.write(f"calibration\tbce_base_rate\t{-(p*np.log(p)+(1-p)*np.log(1-p)):.5f}\t\t\t\t\t\t\n")
log("[out] calibration_diagnosis.tsv")
