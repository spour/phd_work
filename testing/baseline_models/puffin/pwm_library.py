# position-weight matrix (pwm) library for the puffincpa model
import numpy as np


ALPHA = "ACGT"
A2I = {b: i for i, b in enumerate(ALPHA)}


def consensus_to_pwm(consensus, strength=2.0):
    IUPAC = {
        "A": "A", "C": "C", "G": "G", "T": "T", "U": "T",
        "R": "AG", "Y": "CT", "S": "GC", "W": "AT", "K": "GT", "M": "AC",
        "B": "CGT", "D": "AGT", "H": "ACT", "V": "ACG", "N": "ACGT",
    }
    L = len(consensus)
    pwm = np.full((4, L), 1.0 / 4.0)
    eS = np.exp(strength)
    for j, c in enumerate(consensus.upper()):
        bases = IUPAC.get(c, "ACGT")
        weights = np.array([eS if b in bases else 1.0 for b in ALPHA])
        pwm[:, j] = weights / weights.sum()
    return np.log2(pwm / 0.25).astype(np.float32)


def pad_pwm(pwm, target_len):
    L = pwm.shape[1]
    if L >= target_len:
        start = (L - target_len) // 2
        return pwm[:, start:start + target_len]
    left  = (target_len - L) // 2
    right = target_len - L - left
    return np.pad(pwm, ((0, 0), (left, right)), mode="constant",
                  constant_values=0.0)


def stack_pwms(pwms, kernel_len):
    padded = [pad_pwm(p, kernel_len) for p in pwms]
    return np.stack(padded, axis=0).astype(np.float32)


PAS_HEXAMERS = [
    "AATAAA", "ATTAAA", "TATAAA", "AGTAAA", "AAGAAA", "AATATA",
    "AATACA", "CATAAA", "GATAAA", "AATGAA", "ACTAAA", "AATAGA",
]

USE_CONSENSUS = "TTTTTTTT"

DSE_CONSENSUS = "TGTGTGTG"

UGUA_CONSENSUS = "TTGTAT"


def build_pas_track_pwms(kernel_len=8, hexamer_strength=2.5,
                          aux_strength=1.5):
    pwms = []
    names = []
    for h in PAS_HEXAMERS:
        pwms.append(consensus_to_pwm(h, strength=hexamer_strength))
        names.append(f"PAS_{h}")
    pwms.append(consensus_to_pwm(USE_CONSENSUS, strength=aux_strength))
    names.append("USE_Trich")
    pwms.append(consensus_to_pwm(DSE_CONSENSUS, strength=aux_strength))
    names.append("DSE_GUrich")
    pwms.append(consensus_to_pwm(UGUA_CONSENSUS, strength=aux_strength))
    names.append("UGUA_CFIm25")
    weights = stack_pwms(pwms, kernel_len)
    return weights, names


SS5_CONSENSUS = "MAGGTRAGT"

SS3_CONSENSUS = "YYYYYYAGG"


def build_splice_track_pwms(kernel_len=9, strength=2.5):
    pwms = []
    names = []
    pwms.append(consensus_to_pwm(SS5_CONSENSUS, strength=strength))
    names.append("SS5_U1_binding")
    pwms.append(consensus_to_pwm(SS3_CONSENSUS, strength=strength))
    names.append("SS3_consensus")
    weights = stack_pwms(pwms, kernel_len)
    return weights, names


if __name__ == "__main__":
    pas_w, pas_n = build_pas_track_pwms()
    print(f"PAS track PWMs: shape={pas_w.shape}, dtype={pas_w.dtype}")
    print(f"  Filter names ({len(pas_n)}):")
    for n in pas_n:
        print(f"    {n}")
    print(f"  Weight stats: min={pas_w.min():.3f}, max={pas_w.max():.3f}, "
          f"mean={pas_w.mean():.3f}")

    splice_w, splice_n = build_splice_track_pwms()
    print(f"\nSplice track PWMs: shape={splice_w.shape}, dtype={splice_w.dtype}")
    print(f"  Filter names ({len(splice_n)}):")
    for n in splice_n:
        print(f"    {n}")

    def encode(seq):
        out = np.zeros((4, len(seq)), dtype=np.float32)
        for i, b in enumerate(seq.upper()):
            if b in A2I:
                out[A2I[b], i] = 1.0
        return out

    aataaa_filter = pas_w[0]
    print(f"\n  AATAAA filter (filter 0), shape {aataaa_filter.shape}:")
    print(f"  Per-position consensus base (argmax):")
    print(f"    {''.join(ALPHA[i] for i in aataaa_filter.argmax(axis=0))}")

    test_pos = encode("AATAAA")
    test_neg = encode("GCGCGC")
    test_pos_padded = np.zeros((4, 8), dtype=np.float32)
    test_pos_padded[:, 1:7] = test_pos
    test_neg_padded = np.zeros((4, 8), dtype=np.float32)
    test_neg_padded[:, 1:7] = test_neg
    act_pos = (aataaa_filter * test_pos_padded).sum()
    act_neg = (aataaa_filter * test_neg_padded).sum()
    print(f"  AATAAA filter activation on 'AATAAA': {act_pos:+.3f}")
    print(f"  AATAAA filter activation on 'GCGCGC': {act_neg:+.3f}")
    assert act_pos > act_neg, "PWM init seems broken: consensus < scramble"
    print(f"  ✓ Consensus activation > scramble activation")