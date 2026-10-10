#!/usr/bin/env python3
# a bpnet-style, motif-friendly variant of the reported cnn
import torch, torch.nn as nn, torch.nn.functional as F

DIL = (2, 4, 8, 16, 32, 64)

class BPNet(nn.Module):
    def __init__(s, drop=0.4, nf=48, w=25, nbin=25, pool=100):
        super().__init__()
        s.nf, s.w, s.nbin, s.pool = nf, w, nbin, pool
        s.c1 = nn.Conv1d(4, nf, w, padding=w // 2, bias=True)
        nn.init.constant_(s.c1.bias, -1.0)
        s.bn_post = nn.BatchNorm1d(nf)
        s.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv1d(nf, nf, 3, dilation=d, padding=d, bias=False),
                          nn.BatchNorm1d(nf), nn.ReLU()) for d in DIL])
        s.do = nn.Dropout(drop)
        s.fc1 = nn.Linear(nf * nbin, 64); s.fc2 = nn.Linear(64, 1)

    def first(s, x):
        return torch.exp(torch.clamp(s.c1(x), min=-8.0, max=4.0))

    def forward(s, x, return_first=False):
        a = s.first(x)
        h = s.bn_post(a)
        for b in s.blocks: h = h + b(h)
        h = F.avg_pool1d(h, s.pool)
        h = s.do(h.flatten(1))
        out = s.fc2(s.do(F.relu(s.fc1(h)))).squeeze(1)
        return (out, a) if return_first else out

    @staticmethod
    def receptive_field(w=25, pool=100):
        return w + 2 * sum(DIL) + (pool - 1)
