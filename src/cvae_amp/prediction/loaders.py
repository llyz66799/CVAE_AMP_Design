"""Model checkpoint loading for AMP, AEP, and HP prediction.

AMP model architecture is auto-detected from the checkpoint state dict
so that models trained with different lstm_hidden sizes load correctly.
"""

from __future__ import annotations

import torch
import joblib

from cvae_amp.prediction.models import AMPAttentionModel
from cvae_amp.config.paths import MODELS_PRED


def _infer_amp_architecture(state_dict: dict) -> dict:
    """Infer AMPAttentionModel hyperparameters from a checkpoint state dict."""
    lstm_hidden = state_dict["fc1.weight"].shape[1] // 2
    input_dim = state_dict["lstm.weight_ih_l0"].shape[1]
    layer_keys = [k for k in state_dict if k.startswith("lstm.weight_ih_l")
                  and not k.endswith("_reverse")]
    num_layers = len(layer_keys)
    return {"input_dim": input_dim, "lstm_hidden": lstm_hidden, "num_layers": num_layers}


def load_amp(device: torch.device | None = None) -> AMPAttentionModel:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = MODELS_PRED / "AMP_Pc7.pth"
    state = torch.load(str(ckpt), map_location=device, weights_only=True)
    arch = _infer_amp_architecture(state)
    model = AMPAttentionModel(**arch)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_aep():
    return joblib.load(str(MODELS_PRED / "XGB_aep_model.pkl"))


def load_hp():
    return joblib.load(str(MODELS_PRED / "XGB_hp_model.pkl"))
