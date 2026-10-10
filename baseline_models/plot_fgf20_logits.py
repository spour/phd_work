# re-score fgf20 with raw cnn logits and compare to the saved sigmoid probs
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from Bio import SeqIO

from cpa_cnn import CPACNN, one_hot_sequence


# paths (edit if needed)
CNN_DIR    = "../train/cnn_runs/cnn_20260517_083922_40273075"
PARQUET    = "../train/gene_scans/per_gene/ERP27.parquet"
GENES_BED  = "../train/data/pc_genes.bed"
GENOME     = "/scratch/spour98/remake_model/updated_HumanGeneEnds/reanal/hg38.fa"
GENE       = "ERP27"
OUT_PNG    = f"{GENE}_logits.png"


def load_cnn():
    meta = json.load(open(os.path.join(CNN_DIR, "metadata.json")))
    arch = meta["winner_meta"]["model_arch"].copy()
    arch.pop("n_params", None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CPACNN(**arch).to(device)
    ckpt = torch.load(os.path.join(CNN_DIR, "cnn_model.pt"),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded CNN on {device}, seq_len={arch['seq_len']}")
    return model, arch["seq_len"], device


def gene_bed_lookup(gene_name):
    with open(GENES_BED) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 4 and p[3] == gene_name:
                return {"chrom": p[0], "start": int(p[1]), "end": int(p[2]),
                        "strand": p[5] if len(p) > 5 else "+"}
    raise ValueError(f"{gene_name} not in {GENES_BED}")


def fetch_sequence(rec):
    import subprocess, tempfile
    bed = tempfile.NamedTemporaryFile(mode="w", suffix=".bed", delete=False)
    bed.write(f"{rec['chrom']}\t{rec['start']}\t{rec['end']}\t"
              f"{GENE}\t0\t{rec['strand']}\n")
    bed.close()
    fa = bed.name.replace(".bed", ".fa")
    subprocess.run(["bedtools", "getfasta", "-s", "-name",
                    "-fi", GENOME, "-bed", bed.name, "-fo", fa], check=True)
    for r in SeqIO.parse(fa, "fasta"):
        os.unlink(bed.name); os.unlink(fa)
        for ext in [".fai"]:
            p = GENOME + ext
            if os.path.exists(p) and os.path.getmtime(p) > os.path.getmtime(GENOME):
                pass
        return str(r.seq).upper()
    raise RuntimeError("bedtools returned nothing")


@torch.no_grad()
def score_logits(model, ohe, win_len, device, batch=512):
    L = ohe.shape[0]
    n_macro = L - win_len + 1
    s0, s1 = ohe.strides
    macros = np.lib.stride_tricks.as_strided(
        ohe, shape=(n_macro, win_len, 4),
        strides=(s0, s0, s1), writeable=False,
    )
    logits = np.empty(n_macro, dtype=np.float32)
    model.eval()
    for i in range(0, n_macro, batch):
        j = min(i + batch, n_macro)
        b = np.transpose(macros[i:j], (0, 2, 1)).astype(np.float32)
        logits[i:j] = model(torch.from_numpy(b).to(device)).cpu().numpy()
    return logits


# run
df = pd.read_parquet(PARQUET).sort_values("window_offset").reset_index(drop=True)
print(f"{GENE}: {len(df):,} windows in parquet")
canon_off = int(df[df.is_constitutive_cpa].window_offset.iloc[0])
print(f"Canonical window offset: {canon_off}")

rec = gene_bed_lookup(GENE)
print(f"Gene coords: {rec}")
seq = fetch_sequence(rec)
print(f"Fetched sequence length: {len(seq)}")

model, win_len, device = load_cnn()
ohe = one_hot_sequence(seq, len(seq)).T.astype(np.uint8)
print(f"OHE shape: {ohe.shape}")

logits = score_logits(model, ohe, win_len, device)
probs_recomputed = 1.0 / (1.0 + np.exp(-logits))
print(f"Re-scored {len(logits):,} windows")

saved_probs = df["cnn_prob"].values
diff = np.abs(saved_probs - probs_recomputed).max()
print(f"Max diff between saved CNN probs and recomputed: {diff:.6f}")

# stats
print(f"\nLogit stats:")
print(f"  range: [{logits.min():.2f}, {logits.max():.2f}]")
print(f"  saturated (sigmoid > 0.99): {(probs_recomputed > 0.99).sum():,} windows")
print(f"  >0.9999: {(probs_recomputed > 0.9999).sum():,}")
print(f"  >0.99999: {(probs_recomputed > 0.99999).sum():,}")
print(f"\nCanonical window:")
print(f"  logit: {logits[canon_off]:.2f}  prob: {probs_recomputed[canon_off]:.6f}")
print(f"\nTop window:")
top_idx = int(np.argmax(logits))
print(f"  offset: {top_idx}  logit: {logits[top_idx]:.2f}  "
      f"prob: {probs_recomputed[top_idx]:.6f}")
print(f"  distance from canonical: {abs(top_idx - canon_off):,} nt")
print(f"  logit gap (top - canon): {logits[top_idx] - logits[canon_off]:.2f}")
print(f"\nRank of canonical by logit: "
      f"{int((logits > logits[canon_off]).sum()) + 1}")

order = np.argsort(-logits)[:10]
print(f"\nTop 10 by logit:")
print(f"  {'rank':>4}  {'offset':>7}  {'logit':>8}  {'prob':>14}  "
      f"{'dist_from_canon':>15}")
for r, idx in enumerate(order, 1):
    print(f"  {r:>4}  {int(idx):>7}  {logits[idx]:>8.2f}  "
          f"{probs_recomputed[idx]:>14.6f}  {abs(int(idx)-canon_off):>15,d}")

# plot
fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

axes[0].plot(saved_probs, color="tab:green", linewidth=0.6, alpha=0.9)
axes[0].axvline(canon_off, color="red", linestyle="--", linewidth=1,
                alpha=0.7, label="canonical")
axes[0].set_ylabel("Sigmoid prob")
axes[0].set_ylim(-0.02, 1.02)
axes[0].legend(loc="upper left")
axes[0].grid(alpha=0.3)
axes[0].set_title(f"{GENE}: CNN sigmoid output (from parquet)")

axes[1].plot(logits, color="tab:purple", linewidth=0.6, alpha=0.9)
axes[1].axvline(canon_off, color="red", linestyle="--", linewidth=1, alpha=0.7)
axes[1].axhline(logits[canon_off], color="red", linestyle=":", linewidth=0.8,
                alpha=0.6, label=f"canonical logit ({logits[canon_off]:.1f})")
axes[1].set_ylabel("Raw logit")
axes[1].legend(loc="upper left")
axes[1].grid(alpha=0.3)
axes[1].set_title("CNN raw logits (shows true ranking)")

hi = logits.max()
zoom_low = max(hi - 10, np.percentile(logits, 95))
axes[2].plot(logits, color="tab:purple", linewidth=0.6, alpha=0.9)
axes[2].axvline(canon_off, color="red", linestyle="--", linewidth=1, alpha=0.7)
axes[2].axhline(logits[canon_off], color="red", linestyle=":", linewidth=0.8,
                alpha=0.6)
axes[2].set_ylim(zoom_low, hi + 1)
axes[2].set_ylabel("Logit (zoomed)")
axes[2].set_xlabel("Position in transcript (nt)")
axes[2].grid(alpha=0.3)
axes[2].set_title(f"Top-region zoom: logits in [{zoom_low:.1f}, {hi+1:.1f}]")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
print(f"\nSaved {OUT_PNG}")
