"""Sequence sampling from trained generative models."""

import torch

from cvae_amp.config.defaults import MAX_LEN, Z_DIM, PAD_ID, AA_LIST
from cvae_amp.generation.models import DEVICE

ID_TO_AA = {i: aa for i, aa in enumerate(AA_LIST)}


@torch.no_grad()
def sample_vae(
    model,
    temperature: float = 1.2,
    top_k: int = 10,
    min_len: int = 10,
    num_samples: int = 1,
) -> list[str]:
    model.eval()
    z = torch.randn(num_samples, Z_DIM).to(DEVICE)
    logits = model.decoder(z) / temperature
    probs = torch.softmax(logits, dim=-1)
    return _decode_probs(probs, top_k, min_len)


@torch.no_grad()
def sample_cvae(
    model,
    labels: list[float],
    temperature: float = 1.2,
    top_k: int = 10,
    min_len: int = 10,
    num_samples: int = 1,
) -> list[str]:
    model.eval()
    labels_t = torch.FloatTensor([labels] * num_samples).to(DEVICE)
    z = torch.randn(num_samples, Z_DIM).to(DEVICE)
    logits = model.decoder(z, labels_t) / temperature
    probs = torch.softmax(logits, dim=-1)
    return _decode_probs(probs, top_k, min_len)


def _decode_probs(
    probs: torch.Tensor,
    top_k: int,
    min_len: int,
) -> list[str]:
    peptides = []
    for n in range(probs.size(0)):
        seq = []
        for pos in range(MAX_LEN):
            p = probs[n, pos].clone()
            if pos < min_len:
                p[PAD_ID] = 0
            p = p / p.sum()
            top_probs, top_idx = torch.topk(p, top_k)
            top_probs = top_probs / top_probs.sum()
            sampled_idx = torch.multinomial(top_probs, 1).item()
            aa_idx = top_idx[sampled_idx].item()
            if aa_idx == PAD_ID:
                break
            seq.append(ID_TO_AA[aa_idx])
        peptides.append("".join(seq))
    return peptides
