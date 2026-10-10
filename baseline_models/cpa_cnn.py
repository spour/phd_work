# cpa-cnn: a convolutional neural net for cleavage and polyadenylation site prediction
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from Bio import SeqIO


# model

class CPACNN(nn.Module):
    def __init__(
        self,
        seq_len: int = 500,
        n_filters_1: int = 128,
        n_filters_2: int = 256,
        n_hidden: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.seq_len = seq_len

        self.conv1 = nn.Conv1d(4, n_filters_1, kernel_size=8, padding="same")
        self.bn1 = nn.BatchNorm1d(n_filters_1)
        self.pool1 = nn.MaxPool1d(4)

        self.conv2 = nn.Conv1d(n_filters_1, n_filters_2, kernel_size=8, padding="same")
        self.bn2 = nn.BatchNorm1d(n_filters_2)
        self.pool2 = nn.MaxPool1d(4)

        self.conv3 = nn.Conv1d(n_filters_2, n_filters_2, kernel_size=4, padding="same")
        self.bn3 = nn.BatchNorm1d(n_filters_2)

        self.adaptive_pool = nn.AdaptiveMaxPool1d(4)

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(n_filters_2 * 4, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self.forward(x))


# dataset

_BASE_IDX = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}


def one_hot_sequence(seq: str, seq_len: int) -> np.ndarray:
    arr = np.zeros((4, seq_len), dtype=np.float32)
    for i, ch in enumerate(seq.upper()):
        j = _BASE_IDX.get(ch)
        if j is not None:
            arr[j, i] = 1.0
    return arr


class CPASequenceDataset(Dataset):

    def __init__(self, pos_fasta: str, neg_fasta: str, seq_len: int = 500, verbose: bool = True):
        self.seq_len = seq_len
        seqs, labels = [], []

        for label, path in [(1, pos_fasta), (0, neg_fasta)]:
            n_skipped = 0
            n_kept = 0
            for rec in SeqIO.parse(path, "fasta"):
                s = str(rec.seq)
                if len(s) != seq_len:
                    n_skipped += 1
                    continue
                seqs.append(one_hot_sequence(s, seq_len))
                labels.append(label)
                n_kept += 1
            if verbose:
                kind = "pos" if label == 1 else "neg"
                print(f"  {kind}: kept {n_kept:,}, skipped {n_skipped:,} from {path}")

        if not seqs:
            raise ValueError("No sequences loaded — check FASTA paths and seq_len.")

        self.X = torch.from_numpy(np.stack(seqs))
        self.y = torch.tensor(labels, dtype=torch.float32)

        n_pos = int(self.y.sum().item())
        n_neg = len(self.y) - n_pos
        self.n_pos = n_pos
        self.n_neg = n_neg
        if verbose:
            print(f"  total: {len(self.y):,} ({n_pos:,} pos, {n_neg:,} neg, "
                  f"ratio 1:{n_neg / max(n_pos,1):.1f})")

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    @property
    def pos_weight(self) -> float:
        return self.n_neg / max(self.n_pos, 1)


if __name__ == "__main__":
    m = CPACNN()
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"CPACNN trainable params: {n_params:,}")
    x = torch.randn(2, 4, 500)
    out = m(x)
    print(f"Output shape: {out.shape}  (expected (2,))")