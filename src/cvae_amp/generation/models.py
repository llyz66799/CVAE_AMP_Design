"""
Generative models for antimicrobial peptide design.

  Model A: VAE — unconditional, no label input
  Model B: CVAE — conditional, label input, no property predictor
  Model C: CVAE+Pred — conditional, with auxiliary property predictor head

All share the same encoder/decoder core architecture (z_dim=32, hidden=256).
"""

import torch
import torch.nn as nn

from cvae_amp.config.defaults import (
    INPUT_DIM, Z_DIM, HIDDEN_DIM, MAX_LEN, VOCAB_SIZE, LABEL_DIM, PAD_ID,
    WORD_DROPOUT_RATE, LABEL_DROPOUT_RATE,
)
from cvae_amp.generation.regularizers import apply_word_dropout, apply_label_dropout

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════
#  Model A: VAE (unconditional)
# ═══════════════════════════════════════════════════════════════

class VAE_Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.15),
        )
        self.fc_mu = nn.Linear(HIDDEN_DIM, Z_DIM)
        self.fc_logvar = nn.Linear(HIDDEN_DIM, Z_DIM)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x.view(x.size(0), -1))
        return self.fc_mu(h), self.fc_logvar(h)


class VAE_Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(Z_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(HIDDEN_DIM, INPUT_DIM),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).view(-1, MAX_LEN, VOCAB_SIZE)


class VAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = VAE_Encoder()
        self.decoder = VAE_Decoder()

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


# ═══════════════════════════════════════════════════════════════
#  Model B: CVAE (conditional, no predictor)
# ═══════════════════════════════════════════════════════════════

class CVAE_Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM + LABEL_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.15),
        )
        self.fc_mu = nn.Linear(HIDDEN_DIM, Z_DIM)
        self.fc_logvar = nn.Linear(HIDDEN_DIM, Z_DIM)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.cat([x.view(x.size(0), -1), labels], dim=1)
        h = self.net(h)
        return self.fc_mu(h), self.fc_logvar(h)


class CVAE_Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(Z_DIM + LABEL_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(HIDDEN_DIM, INPUT_DIM),
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        h = torch.cat([z, labels], dim=1)
        return self.net(h).view(-1, MAX_LEN, VOCAB_SIZE)


class CVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = CVAE_Encoder()
        self.decoder = CVAE_Decoder()

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        labels = apply_label_dropout(labels, LABEL_DROPOUT_RATE)
        x = apply_word_dropout(x, WORD_DROPOUT_RATE)
        mu, logvar = self.encoder(x, labels)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z, labels), mu, logvar


# ═══════════════════════════════════════════════════════════════
#  Model C: CVAE with property predictor
# ═══════════════════════════════════════════════════════════════

class PropertyPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(Z_DIM, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, LABEL_DIM),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class CVAE_Pred(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = CVAE_Encoder()
        self.decoder = CVAE_Decoder()
        self.predictor = PropertyPredictor()

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        labels_dropped = apply_label_dropout(labels, LABEL_DROPOUT_RATE)
        x_dropped = apply_word_dropout(x, WORD_DROPOUT_RATE)
        mu, logvar = self.encoder(x_dropped, labels_dropped)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(z, labels_dropped)
        pred_labels = self.predictor(z)
        return logits, mu, logvar, pred_labels


# ── Model info ──

if __name__ == "__main__":
    for name, model_cls in [("VAE", VAE), ("CVAE", CVAE), ("CVAE+Pred", CVAE_Pred)]:
        m = model_cls()
        n = sum(p.numel() for p in m.parameters())
        print(f"{name}: {n:,} parameters")
