#!/usr/bin/env python3
# the network, shared by cnn_train.py and cnn_interpret.py so the interpretability pass
import torch, torch.nn as nn, torch.nn.functional as F

class Net(nn.Module):
    def __init__(s, drop=0.4):
        super().__init__()
        s.c1 = nn.Conv1d(4, 32, 19, padding=9, bias=False)
        s.b1 = nn.BatchNorm1d(32)
        s.c2 = nn.Conv1d(32, 64, 7, padding=3, bias=False); s.b2 = nn.BatchNorm1d(64)
        s.c3 = nn.Conv1d(64, 96, 5, padding=2, bias=False); s.b3 = nn.BatchNorm1d(96)
        s.do = nn.Dropout(drop)
        s.fc1 = nn.Linear(96 * 25, 64); s.fc2 = nn.Linear(64, 1)
    def first(s, x):
        return torch.exp(torch.clamp(s.b1(s.c1(x)), max=8.0))
    def forward(s, x):
        h = F.max_pool1d(s.first(x), 10)
        h = F.max_pool1d(F.relu(s.b2(s.c2(h))), 5)
        h = F.max_pool1d(F.relu(s.b3(s.c3(h))), 2)
        h = s.do(h.flatten(1))
        return s.fc2(s.do(F.relu(s.fc1(h)))).squeeze(1)
