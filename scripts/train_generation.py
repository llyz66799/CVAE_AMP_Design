#!/usr/bin/env python3
"""Train VAE, CVAE, and CVAE+Pred generative models."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvae_amp.generation.training import train_all_models, train_one_model
from cvae_amp.generation.models import VAE, CVAE, CVAE_Pred
from cvae_amp.generation.losses import vae_loss_fn, cvae_loss_fn, cvae_pred_loss_fn
from cvae_amp.config.paths import DATA_DATASET


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CVAE generative models")
    parser.add_argument("--model", default="all",
                        choices=["vae", "cvae", "cvae_pred", "all"])
    parser.add_argument("--data", type=str, default=str(DATA_DATASET / "VAE_train_real.xlsx"))
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.model == "all":
        train_all_models(data_path=args.data, output_dir=args.output_dir)
    else:
        model_map = {
            "vae": (VAE(), vae_loss_fn, True, False),
            "cvae": (CVAE(), cvae_loss_fn, False, False),
            "cvae_pred": (CVAE_Pred(), cvae_pred_loss_fn, False, True),
        }
        model, loss_fn, is_vae, has_pred = model_map[args.model]
        train_one_model(model, loss_fn, args.model.upper(), args.data,
                        is_vae=is_vae, has_predictor=has_pred,
                        epochs=args.epochs, lr=args.lr)


if __name__ == "__main__":
    main()
