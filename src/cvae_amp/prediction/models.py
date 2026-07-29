"""BiLSTM + Attention model for antimicrobial peptide prediction.

Uses LayerNorm (instead of BatchNorm) and 2-layer LSTM for improved
training stability, as validated by Optuna hyperparameter optimization.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cvae_amp.config.defaults import AMP_INPUT_DIM, AMP_LSTM_HIDDEN, AMP_NUM_LAYERS, AMP_DROPOUT


class AMPAttentionModel(nn.Module):
    """BiLSTM with attention pooling for binary peptide classification."""

    def __init__(
        self,
        input_dim: int = AMP_INPUT_DIM,
        lstm_hidden: int = AMP_LSTM_HIDDEN,
        num_layers: int = AMP_NUM_LAYERS,
        dropout_rate: float = AMP_DROPOUT,
    ) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.attention_linear = nn.Linear(lstm_hidden * 2, 1)
        self.fc1 = nn.Linear(lstm_hidden * 2, 64)
        self.fc2 = nn.Linear(64, 10)
        self.out = nn.Linear(10, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        att_weights = F.softmax(self.attention_linear(lstm_out), dim=1)
        context = torch.sum(lstm_out * att_weights, dim=1)
        x = F.relu(self.fc1(context))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.out(x))
