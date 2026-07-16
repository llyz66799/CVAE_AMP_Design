"""Amino acid vocabulary and one-hot encoding utilities."""

import numpy as np

from cvae_amp.config.defaults import AA_LIST, VOCAB_SIZE, PAD_ID

w_to_id = {aa: i for i, aa in enumerate(AA_LIST)}
id_to_A = {i: aa for i, aa in enumerate(AA_LIST)}


def aa_to_onehot(idx: int) -> np.ndarray:
    vec = np.zeros(VOCAB_SIZE)
    vec[idx] = 1.0
    return vec


def seq_to_onehot(seq: list[str], max_len: int = 50) -> np.ndarray:
    seq = seq[:max_len]
    padded = seq + ["PAD"] * (max_len - len(seq))
    return np.array([aa_to_onehot(w_to_id[aa]) for aa in padded])


def onehot_to_seq(onehot: np.ndarray) -> str:
    seq = ""
    for pos in onehot:
        idx = int(np.argmax(pos))
        aa = id_to_A[idx]
        if aa == "PAD":
            break
        seq += aa
    return seq


def process_labels(labels_list: list) -> np.ndarray:
    processed = []
    for label_set in labels_list:
        clean = [0.5 if l == "-" else float(l) for l in label_set]
        processed.append(clean)
    return np.array(processed)
