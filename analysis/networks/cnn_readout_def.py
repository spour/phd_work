#!/usr/bin/env python3
# an interpretable readout grafted onto the current exponential model
import torch, torch.nn as nn, torch.nn.functional as F

NBINS, SEQLEN = 25, 2500
BINW = SEQLEN // NBINS

class NetReadout(nn.Module):
    def __init__(s, base, nbins=NBINS, seqlen=SEQLEN):
        super().__init__()
        s.base = base
        s.nbins, s.seqlen, s.binw = nbins, seqlen, seqlen // nbins
        nfilt = base.c1.out_channels
        s.pos_w = nn.Parameter(torch.ones(nfilt, nbins))
        s.attn = nn.Conv1d(nfilt, 1, 1)
        nn.init.normal_(s.attn.weight, std=0.05); nn.init.zeros_(s.attn.bias)
        s.beta = nn.Parameter(torch.zeros(1))
        for p in s.base.parameters(): p.requires_grad_(False)

    def freeze_base(s):
        s.base.eval()
        for p in s.base.parameters(): p.requires_grad_(False)

    def attention(s, a):
        return F.softmax(s.attn(a).squeeze(1), dim=1) * s.seqlen

    def modified_first(s, x):
        a = s.base.first(x)
        w = s.pos_w.repeat_interleave(s.binw, dim=1)[None]
        al = s.attention(a)
        return a * w * (1.0 + s.beta * (al[:, None, :] - 1.0)), al

    def forward(s, x, want_attn=False):
        a, al = s.modified_first(x)
        b = s.base
        h = F.max_pool1d(a, 10)
        h = F.max_pool1d(F.relu(b.b2(b.c2(h))), 5)
        h = F.max_pool1d(F.relu(b.b3(b.c3(h))), 2)
        h = b.do(h.flatten(1))
        o = b.fc2(b.do(F.relu(b.fc1(h)))).squeeze(1)
        return (o, al) if want_attn else o
