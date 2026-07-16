"""Regularization utilities: word dropout, label dropout, KL annealing."""

import torch

from cvae_amp.config.defaults import PAD_ID


def apply_word_dropout(x: torch.Tensor, rate: float = 0.05) -> torch.Tensor:
    if rate <= 0:
        return x
    mask = torch.rand(x.size(0), x.size(1), device=x.device) < rate
    x = x.clone()
    x[mask] = 0
    x[mask, PAD_ID] = 1
    return x


def apply_label_dropout(labels: torch.Tensor, rate: float = 0.1) -> torch.Tensor:
    if rate <= 0:
        return labels
    labels = labels.clone()
    mask = torch.rand_like(labels) < rate
    labels[mask] = 0.5
    return labels


def get_kl_weight(epoch: int, cycle: int = 100) -> float:
    phase = epoch % cycle
    beta = min(1.0, phase / (cycle * 0.5))
    return beta * 0.5
