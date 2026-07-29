"""Loss functions for VAE, CVAE, and CVAE+Pred."""

import torch
import torch.nn.functional as F

from cvae_amp.config.defaults import VOCAB_SIZE, PAD_ID


def vae_loss_fn(
    logits: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    pad_weight: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor]:
    targets = torch.argmax(x, dim=-1)
    weights = torch.ones(VOCAB_SIZE, device=logits.device)
    weights[PAD_ID] = pad_weight
    recon = F.cross_entropy(logits.transpose(1, 2), targets, weight=weights)
    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return recon, kl


cvae_loss_fn = vae_loss_fn  # identical formulation


def cvae_pred_loss_fn(
    logits: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    pred_labels: torch.Tensor,
    true_labels: torch.Tensor,
    pad_weight: float = 0.15,
    prop_weight: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    targets = torch.argmax(x, dim=-1)
    weights = torch.ones(VOCAB_SIZE, device=logits.device)
    weights[PAD_ID] = pad_weight
    recon = F.cross_entropy(logits.transpose(1, 2), targets, weight=weights)
    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    mask = (true_labels != 0.5).float()
    prop = F.binary_cross_entropy_with_logits(pred_labels, true_labels, reduction="none")
    prop = (prop * mask).sum() / (mask.sum() + 1e-6)

    return recon, kl, prop
