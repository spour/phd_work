# puffincpa: a puffin-style sequence-only cnn for cleavage-and-polyadenylation site
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


UP_FLANK   = 3000
DOWN_FLANK = 1000
SEQ_LEN    = UP_FLANK + DOWN_FLANK


ALPHA = "ACGT"
A2I = {b: i for i, b in enumerate(ALPHA)}

PAS_HEXAMERS = [
    "AATAAA", "ATTAAA", "TATAAA", "AGTAAA", "AAGAAA", "AATATA",
    "AATACA", "CATAAA", "GATAAA", "AATGAA", "ACTAAA", "AATAGA",
]
USE_CONSENSUS  = "TTTTTTTT"
DSE_CONSENSUS  = "TGTGTGTG"
UGUA_CONSENSUS = "TTGTAT"

SS5_CONSENSUS   = "MAGGTRAGT"
SS3_CONSENSUS   = "YYYYYYAGG"
BP_CONSENSUS    = "YNYTRAC"
POLYY_CONSENSUS = "YYYYYYYY"

PAS_MOTIF_NAMES = [f"PAS_{h}" for h in PAS_HEXAMERS] + [
    "USE_Trich", "DSE_GUrich", "UGUA_CFIm25",
]
N_PAS_PWMS = len(PAS_MOTIF_NAMES)

SPLICE_MOTIF_NAMES = [
    "SS5_U1_binding",
    "SS3_consensus",
    "BP_consensus",
    "PolyY_tract",
]
N_SPLICE_PWMS = len(SPLICE_MOTIF_NAMES)


def _consensus_to_pwm(consensus, strength=2.0):
    IUPAC = {
        "A": "A", "C": "C", "G": "G", "T": "T", "U": "T",
        "R": "AG", "Y": "CT", "S": "GC", "W": "AT", "K": "GT", "M": "AC",
        "B": "CGT", "D": "AGT", "H": "ACT", "V": "ACG", "N": "ACGT",
    }
    L = len(consensus)
    eS = float(np.exp(strength))
    pwm = np.zeros((4, L), dtype=np.float32)
    for j, c in enumerate(consensus.upper()):
        bases = IUPAC.get(c, "ACGT")
        w = np.array([eS if b in bases else 1.0 for b in ALPHA])
        pwm[:, j] = w / w.sum()
    return pwm


def _pad_pwm(pwm, target_len):
    L = pwm.shape[1]
    if L >= target_len:
        s = (L - target_len) // 2
        return pwm[:, s:s + target_len]
    left  = (target_len - L) // 2
    right = target_len - L - left
    return np.pad(pwm, ((0, 0), (left, right)),
                   mode="constant", constant_values=0.25)


def build_pas_pwm_stack(motif_kernel, hex_strength=2.5, aux_strength=1.5):
    pwms = []
    for h in PAS_HEXAMERS:
        pwms.append(_consensus_to_pwm(h, strength=hex_strength))
    pwms.append(_consensus_to_pwm(USE_CONSENSUS,  strength=aux_strength))
    pwms.append(_consensus_to_pwm(DSE_CONSENSUS,  strength=aux_strength))
    pwms.append(_consensus_to_pwm(UGUA_CONSENSUS, strength=aux_strength))
    padded = [_pad_pwm(p, motif_kernel) for p in pwms]
    stacked_prob = np.stack(padded, axis=0).astype(np.float32)
    log_odds = np.log2(np.maximum(stacked_prob, 1e-9) / 0.25)
    return log_odds.astype(np.float32)


def build_splice_pwm_stack(motif_kernel, strength=2.5):
    pwms = [
        _consensus_to_pwm(SS5_CONSENSUS,   strength=strength),
        _consensus_to_pwm(SS3_CONSENSUS,   strength=strength),
        _consensus_to_pwm(BP_CONSENSUS,    strength=strength),
        _consensus_to_pwm(POLYY_CONSENSUS, strength=strength),
    ]
    padded = [_pad_pwm(p, motif_kernel) for p in pwms]
    stacked_prob = np.stack(padded, axis=0).astype(np.float32)
    log_odds = np.log2(np.maximum(stacked_prob, 1e-9) / 0.25)
    return log_odds.astype(np.float32)


class StructuredPool(nn.Module):

    def __init__(self, length, center, regions, topk=10):
        super().__init__()
        self.length = length
        self.center = center
        self.regions = regions
        self.topk = topk

        idx_ranges = []
        for name, s_off, e_off in regions:
            i0 = max(0,      center + s_off)
            i1 = min(length, center + e_off)
            assert i1 > i0, f"Empty bin {name}: {i0}..{i1}"
            idx_ranges.append((name, i0, i1))
        self.idx_ranges = idx_ranges

        self.feature_names = ["global_mean", "global_max", f"top{topk}_mean"]
        for name, _, _ in regions:
            self.feature_names.append(f"{name}_mean")
            self.feature_names.append(f"{name}_max")
        self.n_features = len(self.feature_names)

    def forward(self, per_pos):
        feats = []
        feats.append(per_pos.mean(dim=1))
        feats.append(per_pos.max(dim=1).values)
        k = min(self.topk, per_pos.shape[1])
        feats.append(per_pos.topk(k, dim=1).values.mean(dim=1))
        for _, i0, i1 in self.idx_ranges:
            sub = per_pos[:, i0:i1]
            feats.append(sub.mean(dim=1))
            feats.append(sub.max(dim=1).values)
        return torch.stack(feats, dim=1)


PAS_REGIONS = [
    ("immediate",   -25,    25),
    ("center50",    -50,    50),
    ("upstream",   -250,   -50),
    ("downstream",   50,   250),
]

SPLICE_REGIONS = [
    ("distal_up",     -3000, -1000),
    ("proximal_up",   -1000,  -250),
    ("pas_prox_up",    -250,     0),
    ("pas_prox_down",     0,   250),
    ("distal_down",     250,  1000),
]


class PuffinTrack(nn.Module):

    def __init__(self, length, center,
                 motif_kernel, effect_kernel, regions,
                 pwm_init=None, n_free=64, dropout=0.0, topk=10,
                 per_motif_heads=False):
        super().__init__()
        self.length          = length
        self.center          = center
        self.motif_kernel    = motif_kernel
        self.effect_kernel   = effect_kernel
        self.per_motif_heads = per_motif_heads

        if pwm_init is not None:
            n_pwm = pwm_init.shape[0]
        else:
            n_pwm = 0
        n_motifs = n_pwm + n_free
        self.n_pwm    = n_pwm
        self.n_free   = n_free
        self.n_motifs = n_motifs

        self.motif_conv = nn.Conv1d(
            4, n_motifs, motif_kernel,
            padding=(motif_kernel - 1) // 2, bias=True)
        if pwm_init is not None:
            with torch.no_grad():
                self.motif_conv.weight[:n_pwm] = torch.from_numpy(pwm_init.copy())
                self.motif_conv.bias[:n_pwm].fill_(0.0)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.pool = StructuredPool(length=length, center=center,
                                    regions=regions, topk=topk)
        self.n_features = self.pool.n_features

        if per_motif_heads:
            self.effect_conv = nn.Conv1d(
                n_motifs, n_motifs, effect_kernel,
                padding=(effect_kernel - 1) // 2, bias=False,
                groups=n_motifs)
            with torch.no_grad():
                self.effect_conv.weight.mul_(0.01)
            self.summary_head = nn.Parameter(
                torch.full((n_motifs, self.n_features),
                           1.0 / self.n_features))
        else:
            self.effect_conv = nn.Conv1d(
                n_motifs, 1, effect_kernel,
                padding=(effect_kernel - 1) // 2, bias=False)
            with torch.no_grad():
                self.effect_conv.weight.mul_(0.01)
            self.summary_head = nn.Linear(self.n_features, 1, bias=False)
            with torch.no_grad():
                self.summary_head.weight.fill_(1.0 / self.n_features)

    def motif_activations(self, seq):
        return F.softplus(self.motif_conv(seq))

    def forward(self, seq, return_intermediates=False):
        motif_act = self.motif_activations(seq)
        motif_act = self.dropout(motif_act)
        out = {}
        if self.per_motif_heads:
            per_pos_m = self.effect_conv(motif_act)
            B, n_motifs, L = per_pos_m.shape
            per_pos_flat = per_pos_m.reshape(B * n_motifs, L)
            summary_flat = self.pool(per_pos_flat)
            summary = summary_flat.reshape(B, n_motifs, self.n_features)
            motif_logits = (summary * self.summary_head.unsqueeze(0)).sum(dim=2)
            logit = motif_logits.sum(dim=1)
            out["motif_logits"] = motif_logits
            out["logit"]        = logit
            if return_intermediates:
                out["motif_act"]    = motif_act
                out["per_pos_m"]    = per_pos_m
                out["summary"]      = summary
        else:
            per_pos = self.effect_conv(motif_act).squeeze(1)
            summary = self.pool(per_pos)
            logit   = self.summary_head(summary).squeeze(-1)
            out["logit"] = logit
            if return_intermediates:
                out["motif_act"] = motif_act
                out["per_pos"]   = per_pos
                out["summary"]   = summary
        return out

    def set_pwm_frozen(self, frozen):
        if hasattr(self, "_grad_hook_handle") and self._grad_hook_handle:
            self._grad_hook_handle.remove()
            self._grad_hook_handle = None
        if frozen and self.n_pwm > 0:
            n_pwm = self.n_pwm
            def hook(grad):
                grad = grad.clone()
                grad[:n_pwm] = 0.0
                return grad
            self._grad_hook_handle = self.motif_conv.weight.register_hook(hook)
        self._pwm_frozen = frozen

    def is_pwm_frozen(self):
        return getattr(self, "_pwm_frozen", False)


@dataclass
class PuffinCPAConfig:
    pas_motif_kernel:    int  = 11
    pas_effect_kernel:   int  = 101
    pas_n_free:          int  = 0
    pas_crop_half:       int  = 250
    splice_motif_kernel: int  = 15
    splice_effect_kernel: int = 601
    splice_n_free:       int  = 0
    splice_per_motif_heads: bool = True
    dropout:             float = 0.0
    pwm_warm_start:      bool  = True


class PuffinCPA(nn.Module):

    def __init__(self, config: Optional[PuffinCPAConfig] = None):
        super().__init__()
        cfg = config or PuffinCPAConfig()
        self.config = cfg

        pas_pwm    = (build_pas_pwm_stack(cfg.pas_motif_kernel)
                       if cfg.pwm_warm_start else None)
        splice_pwm = (build_splice_pwm_stack(cfg.splice_motif_kernel)
                       if cfg.pwm_warm_start else None)
        self.pas_motif_names = (PAS_MOTIF_NAMES +
            [f"PAS_free_{i}" for i in range(cfg.pas_n_free)])
        self.splice_motif_names = (SPLICE_MOTIF_NAMES +
            [f"SPL_free_{i}" for i in range(cfg.splice_n_free)])

        pas_length = 2 * cfg.pas_crop_half
        pas_center = cfg.pas_crop_half
        self.pas_track = PuffinTrack(
            length=pas_length, center=pas_center,
            motif_kernel=cfg.pas_motif_kernel,
            effect_kernel=cfg.pas_effect_kernel,
            regions=PAS_REGIONS,
            pwm_init=pas_pwm, n_free=cfg.pas_n_free, dropout=cfg.dropout)
        self.splice_track = PuffinTrack(
            length=SEQ_LEN, center=UP_FLANK,
            motif_kernel=cfg.splice_motif_kernel,
            effect_kernel=cfg.splice_effect_kernel,
            regions=SPLICE_REGIONS,
            pwm_init=splice_pwm, n_free=cfg.splice_n_free, dropout=cfg.dropout,
            per_motif_heads=cfg.splice_per_motif_heads)

        self.pas_crop_half = cfg.pas_crop_half

        if cfg.splice_per_motif_heads:
            n_head_in = 1 + self.splice_track.n_motifs
        else:
            n_head_in = 2
        self.head = nn.Linear(n_head_in, 1, bias=True)
        with torch.no_grad():
            self.head.weight.fill_(1.0)
            self.head.bias.fill_(0.0)
        self._head_n_in = n_head_in

    def _crop_pas(self, seq):
        center = UP_FLANK
        return seq[:, :, center - self.pas_crop_half:center + self.pas_crop_half]

    def forward(self, seq, return_intermediates=False):
        seq_pas    = self._crop_pas(seq)
        pas_out    = self.pas_track(seq_pas,    return_intermediates=return_intermediates)
        splice_out = self.splice_track(seq,     return_intermediates=return_intermediates)
        a_pas = pas_out["logit"]
        if self.splice_track.per_motif_heads:
            splice_motif_logits = splice_out["motif_logits"]
            a_splice = splice_motif_logits.sum(dim=1)
            head_in = torch.cat([a_pas.unsqueeze(1), splice_motif_logits],
                                 dim=1)
        else:
            a_splice = splice_out["logit"]
            head_in = torch.stack([a_pas, a_splice], dim=1)
            splice_motif_logits = None
        logit = self.head(head_in).squeeze(-1)
        prob  = torch.sigmoid(logit)
        w_pas    = self.head.weight[0, 0]
        if self.splice_track.per_motif_heads:
            w_splice_motifs = self.head.weight[0, 1:]
            w_splice = w_splice_motifs.mean()
        else:
            w_splice_motifs = None
            w_splice = self.head.weight[0, 1]
        out = {
            "logit":           logit,
            "prob":            prob,
            "a_pas":           a_pas,
            "a_splice":        a_splice,
            "splice_motif_logits": splice_motif_logits,
            "w_pas":           w_pas,
            "w_splice":        w_splice,
            "w_splice_motifs": w_splice_motifs,
            "bias":            self.head.bias[0],
        }
        if return_intermediates:
            out["pas_motif_act"]    = pas_out["motif_act"]
            out["pas_per_pos"]      = pas_out["per_pos"]
            out["pas_summary"]      = pas_out["summary"]
            out["splice_motif_act"] = splice_out["motif_act"]
            if "per_pos" in splice_out:
                out["splice_per_pos"] = splice_out["per_pos"]
            if "per_pos_m" in splice_out:
                out["splice_per_pos_m"] = splice_out["per_pos_m"]
            out["splice_summary"]   = splice_out["summary"]
        return out

    # introspection helpers

    def unfreeze_pwms(self):
        self.pas_track.set_pwm_frozen(False)
        self.splice_track.set_pwm_frozen(False)

    def freeze_pwms(self):
        self.pas_track.set_pwm_frozen(True)
        self.splice_track.set_pwm_frozen(True)

    def attribution(self, seq):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            out = self.forward(seq, return_intermediates=True)
            w_pas    = out["w_pas"].item()
            w_splice = out["w_splice"].item()

            def per_motif_contribs(motif_act, track):
                pool = track.pool
                B, n_motifs, L = motif_act.shape
                ec = track.effect_conv
                pad = (ec.kernel_size[0] - 1) // 2

                if track.per_motif_heads:
                    per_pos_m = ec(motif_act)
                    head_w = track.summary_head
                else:
                    w_grouped = ec.weight.transpose(0, 1)
                    per_pos_m = F.conv1d(motif_act, w_grouped,
                                          padding=pad, groups=n_motifs)
                    head_w = track.summary_head.weight.squeeze(0).unsqueeze(0).expand(
                        n_motifs, -1)

                per_pos_total = per_pos_m.sum(dim=1)

                feature_attrs = []

                feature_attrs.append(per_pos_m.mean(dim=2))

                if track.per_motif_heads:
                    argmax_m = per_pos_m.argmax(dim=2, keepdim=True)
                    feature_attrs.append(
                        per_pos_m.gather(2, argmax_m).squeeze(2))
                else:
                    argmax = per_pos_total.argmax(dim=1)
                    idx = argmax.view(B, 1, 1).expand(B, n_motifs, 1)
                    feature_attrs.append(per_pos_m.gather(2, idx).squeeze(2))

                k = min(pool.topk, L)
                if track.per_motif_heads:
                    topk_idx_m = per_pos_m.topk(k, dim=2).indices
                    feature_attrs.append(
                        per_pos_m.gather(2, topk_idx_m).mean(dim=2))
                else:
                    topk_idx = per_pos_total.topk(k, dim=1).indices
                    idx = topk_idx.view(B, 1, k).expand(B, n_motifs, k)
                    feature_attrs.append(per_pos_m.gather(2, idx).mean(dim=2))

                for _, i0, i1 in pool.idx_ranges:
                    sub_m = per_pos_m[:, :, i0:i1]
                    feature_attrs.append(sub_m.mean(dim=2))
                    if track.per_motif_heads:
                        argmax_m = sub_m.argmax(dim=2, keepdim=True)
                        feature_attrs.append(
                            sub_m.gather(2, argmax_m).squeeze(2))
                    else:
                        sub_total = per_pos_total[:, i0:i1]
                        argmax = sub_total.argmax(dim=1)
                        idx = argmax.view(B, 1, 1).expand(B, n_motifs, 1)
                        feature_attrs.append(sub_m.gather(2, idx).squeeze(2))

                F_attr = torch.stack(feature_attrs, dim=2)
                contribs = (F_attr * head_w.unsqueeze(0)).sum(dim=2)
                return contribs

            pas_contribs = per_motif_contribs(
                out["pas_motif_act"], self.pas_track) * w_pas

            splice_intra = per_motif_contribs(
                out["splice_motif_act"], self.splice_track)
            if self.splice_track.per_motif_heads:
                w_motifs = self.head.weight[0, 1:].detach().cpu()
                splice_contribs = splice_intra * w_motifs.unsqueeze(0).to(splice_intra.device)
            else:
                splice_contribs = splice_intra * w_splice

        if was_training:
            self.train()
        return {
            "pas_contribs":     pas_contribs,
            "splice_contribs":  splice_contribs,
            "pas_names":        self.pas_motif_names,
            "splice_names":     self.splice_motif_names,
            "w_pas":            w_pas,
            "w_splice":         w_splice,
            "bias":             out["bias"].item(),
            "total_logit":      out["logit"],
        }

    def pwm_param_groups(self, base_lr=1e-3, weight_decay=1e-4):
        pwm_param_ids = set()
        pwm_params = []
        for track in (self.pas_track, self.splice_track):
            if track.n_pwm > 0:
                pwm_params.append(track.motif_conv.weight)
                pwm_param_ids.add(id(track.motif_conv.weight))
                pwm_params.append(track.motif_conv.bias)
                pwm_param_ids.add(id(track.motif_conv.bias))
        other_params = [p for p in self.parameters()
                         if id(p) not in pwm_param_ids]
        return [
            {"params": pwm_params,   "lr": base_lr, "weight_decay": 0.0},
            {"params": other_params, "lr": base_lr, "weight_decay": weight_decay},
        ]


def encode_sequence(seq, pad_len=SEQ_LEN):
    out = np.zeros((4, pad_len), dtype=np.float32)
    L = min(len(seq), pad_len)
    for i in range(L):
        idx = A2I.get(seq[i].upper())
        if idx is not None:
            out[idx, i] = 1.0
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    B = 4
    model = PuffinCPA()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"PuffinCPA parameter count: {n_params:,}")
    print(f"  PAS track:    {sum(p.numel() for p in model.pas_track.parameters()):,} "
          f"({model.pas_track.n_pwm} PWM + {model.pas_track.n_free} free filters, "
          f"effect kernel {model.pas_track.effect_kernel} nt)")
    print(f"  Splice track: {sum(p.numel() for p in model.splice_track.parameters()):,} "
          f"({model.splice_track.n_pwm} PWM + {model.splice_track.n_free} free filters, "
          f"effect kernel {model.splice_track.effect_kernel} nt, "
          f"per_motif_heads={model.splice_track.per_motif_heads})")
    print(f"  Final head input dim: {model._head_n_in} "
          f"(1 a_pas + {model.splice_track.n_motifs if model.splice_track.per_motif_heads else 1} splice)")

    seq = torch.randn(B, 4, SEQ_LEN)
    out = model(seq)
    print(f"\nForward output:")
    for k, v in out.items():
        if torch.is_tensor(v):
            print(f"  {k:25s}  shape={tuple(v.shape)}")

    attr = model.attribution(seq)
    print(f"\nAttribution: pas_contribs shape={tuple(attr['pas_contribs'].shape)} "
          f"splice_contribs shape={tuple(attr['splice_contribs'].shape)}")
    rec = (attr["pas_contribs"].sum(dim=1) + attr["splice_contribs"].sum(dim=1)
           + attr["bias"])
    print(f"  reconstructed logit (per-motif sum + bias) vs actual:")
    for b in range(B):
        print(f"    sample {b}: recon={rec[b].item():+.4f}  "
              f"actual={attr['total_logit'][b].item():+.4f}  "
              f"diff={abs(rec[b].item() - attr['total_logit'][b].item()):.2e}")

    if model.splice_track.per_motif_heads:
        print(f"\nFinal head weights:")
        print(f"  w_pas    = {model.head.weight[0, 0].item():+.4f}")
        for i, name in enumerate(model.splice_motif_names):
            print(f"  w_{name:18s} = {model.head.weight[0, 1+i].item():+.4f}")
        print(f"  bias     = {model.head.bias[0].item():+.4f}")

    np.random.seed(0)
    seq_str_pos = ''.join(np.random.choice(list('ACGT'), size=SEQ_LEN))
    seq_str_pos = seq_str_pos[:UP_FLANK-3] + 'AATAAA' + seq_str_pos[UP_FLANK+3:]
    seq_str_neg = seq_str_pos[:UP_FLANK-3] + 'GCGCGC' + seq_str_pos[UP_FLANK+3:]
    x_pos = torch.from_numpy(encode_sequence(seq_str_pos)).unsqueeze(0)
    x_neg = torch.from_numpy(encode_sequence(seq_str_neg)).unsqueeze(0)
    with torch.no_grad():
        out_pos = model(x_pos, return_intermediates=True)
        out_neg = model(x_neg, return_intermediates=True)
    aataaa_pos = out_pos["pas_motif_act"][0, 0, 250].item()
    aataaa_neg = out_neg["pas_motif_act"][0, 0, 250].item()
    print(f"\nAATAAA-filter activation at PAS center:")
    print(f"  with AATAAA:  {aataaa_pos:+.3f}")
    print(f"  with GCGCGC:  {aataaa_neg:+.3f}")
    assert aataaa_pos > aataaa_neg, "PWM warm-start broken"
    print(f"  -> PWM warm-start correctly biases toward consensus")

    groups = model.pwm_param_groups(base_lr=1e-3, weight_decay=1e-4)
    print(f"\nAdamW param groups:")
    print(f"  PWM group: n_params={sum(p.numel() for p in groups[0]['params']):,}, "
          f"wd={groups[0]['weight_decay']}")
    print(f"  other:     n_params={sum(p.numel() for p in groups[1]['params']):,}, "
          f"wd={groups[1]['weight_decay']}")

    print(f"\nAll smoke tests passed.")