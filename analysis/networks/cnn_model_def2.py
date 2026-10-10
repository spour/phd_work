#!/usr/bin/env python3
# retrain variants of the interpretable cnn
import torch, torch.nn as nn, torch.nn.functional as F

class Net2(nn.Module):
    def __init__(s, drop=0.4, variant="exp", bn_head=True, logscale=False, clamp=8.0):
        super().__init__()
        s.variant, s.bn_head, s.use_logscale, s.clamp = variant, bn_head, logscale, clamp
        s.c1 = nn.Conv1d(4, 32, 19, padding=9, bias=False)
        s.b1 = nn.BatchNorm1d(32)
        if logscale:
            s.ls = nn.Parameter(torch.zeros(32)); s.lb = nn.Parameter(torch.zeros(32))
        s.c2 = nn.Conv1d(32, 64, 7, padding=3, bias=False); s.b2 = nn.BatchNorm1d(64)
        s.c3 = nn.Conv1d(64, 96, 5, padding=2, bias=False); s.b3 = nn.BatchNorm1d(96)
        s.do = nn.Dropout(drop)
        s.fc1 = nn.Linear(96 * 25, 64)
        s.bh = nn.BatchNorm1d(64) if bn_head else nn.Identity()
        s.fc2 = nn.Linear(64, 1)

    def first(s, x):
        z = s.b1(s.c1(x))
        if s.use_logscale:
            z = z * torch.exp(s.ls)[None, :, None] + s.lb[None, :, None]
        if s.variant == "exp":      return torch.exp(torch.clamp(z, max=s.clamp))
        if s.variant == "softplus": return F.softplus(z)
        return F.relu(z)

    def forward(s, x, want_act=False):
        a = s.first(x)
        h = F.max_pool1d(a, 10)
        h = F.max_pool1d(F.relu(s.b2(s.c2(h))), 5)
        h = F.max_pool1d(F.relu(s.b3(s.c3(h))), 2)
        h = s.do(h.flatten(1))
        o = s.fc2(s.bh(s.do(F.relu(s.fc1(h))))).squeeze(1)
        return (o, a) if want_act else o
