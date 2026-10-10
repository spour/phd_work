#!/usr/bin/env python3
# terminal-exon splice model + memory-light sequence dataset
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset
from Bio import SeqIO

_CODE = np.full(256, 4, np.uint8)
for _b, _i in {"A":0,"C":1,"G":2,"T":3,"U":3}.items():
    _CODE[ord(_b)] = _i
    _CODE[ord(_b.lower())] = _i


class TESpliceCNN(nn.Module):
    def __init__(self, seq_len=1500, n_filters_1=128, n_filters_2=256,
                 n_filters_3=256, n_hidden=128, dropout=0.3):
        super().__init__()
        self.seq_len = seq_len
        self.conv1 = nn.Conv1d(4, n_filters_1, kernel_size=8, padding="same")
        self.bn1   = nn.BatchNorm1d(n_filters_1)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_filters_1, n_filters_2, kernel_size=8, padding="same")
        self.bn2   = nn.BatchNorm1d(n_filters_2)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_filters_2, n_filters_3, kernel_size=4, padding="same")
        self.bn3   = nn.BatchNorm1d(n_filters_3)
        self.adaptive = nn.AdaptiveMaxPool1d(8)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(n_filters_3 * 8, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.adaptive(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, x):
        self.eval()
        return torch.sigmoid(self.forward(x))


class SeqUint8Dataset(Dataset):
    def __init__(self, pos_fa, neg_fa, seq_len, verbose=True):
        self.seq_len = seq_len
        codes, labels = [], []
        for label, path in [(1, pos_fa), (0, neg_fa)]:
            kept = skip = 0
            for rec in SeqIO.parse(path, "fasta"):
                s = str(rec.seq)
                if len(s) != seq_len:
                    skip += 1; continue
                arr = _CODE[np.frombuffer(s.encode("ascii", "replace"), dtype=np.uint8)]
                codes.append(arr); labels.append(label); kept += 1
            if verbose:
                print(f"  {'pos' if label else 'neg'}: kept {kept:,} skip {skip:,} ({path})")
        if not codes:
            raise ValueError("no sequences loaded - check FASTA paths and seq_len")
        self.codes = np.stack(codes)
        self.y = torch.tensor(labels, dtype=torch.float32)
        self.n_pos = int(self.y.sum().item()); self.n_neg = len(self.y) - self.n_pos
        if verbose:
            print(f"  total {len(self.y):,} ({self.n_pos:,} pos, {self.n_neg:,} neg, "
                  f"ratio 1:{self.n_neg / max(self.n_pos,1):.1f})")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        c = self.codes[i].astype(np.int64)
        oh = np.zeros((4, self.seq_len), np.float32)
        valid = c < 4
        oh[c[valid], np.arange(self.seq_len)[valid]] = 1.0
        return torch.from_numpy(oh), self.y[i]

    @property
    def pos_weight(self):
        return self.n_neg / max(self.n_pos, 1)
