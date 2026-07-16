#!/usr/bin/env python3
"""Generate peptides using trained CVAE models."""

import argparse
import sys
from pathlib import Path

import torch
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvae_amp.generation.models import VAE, CVAE, CVAE_Pred, DEVICE
from cvae_amp.generation.sampling import sample_vae, sample_cvae
from cvae_amp.config.paths import MODELS_GEN, RESULTS_GEN
from cvae_amp.pipeline.filters import valid_sequence


MODEL_REGISTRY = {
    "vae": (VAE, "vae_model.pth"),
    "cvae": (CVAE, "cvae_model.pth"),
    "cvae_pred": (CVAE_Pred, "cvae_pred_model.pth"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AMP candidates")
    parser.add_argument("--model", default="cvae_pred", choices=["vae", "cvae", "cvae_pred"])
    parser.add_argument("--target", type=str, default="1.0,1.0,1.0",
                        help="Comma-separated: hemolysis, ecoli, amp")
    parser.add_argument("--num", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--min_len", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=40)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    target = [float(x) for x in args.target.split(",")]
    if len(target) != 3:
        sys.exit("Target must be 3 comma-separated values")

    model_cls, ckpt = MODEL_REGISTRY[args.model]
    ckpt_path = MODELS_GEN / ckpt
    model = model_cls().to(DEVICE)
    model.load_state_dict(torch.load(str(ckpt_path), map_location=DEVICE))
    model.eval()
    print(f"Loaded {args.model} from {ckpt_path}")

    is_vae = (args.model == "vae")
    unique: set[str] = set()
    attempts = 0

    while len(unique) < args.num and attempts < args.num * 20:
        n_batch = min(64, (args.num - len(unique)) * 2)
        if is_vae:
            peptides = sample_vae(model, temperature=args.temperature,
                                  top_k=args.top_k, min_len=args.min_len,
                                  num_samples=n_batch)
        else:
            peptides = sample_cvae(model, target, temperature=args.temperature,
                                   top_k=args.top_k, min_len=args.min_len,
                                   num_samples=n_batch)
        for p in peptides:
            if valid_sequence(p, args.min_len, args.max_len):
                unique.add(p)
        attempts += n_batch

    peptides = sorted(unique)[:args.num]
    out_path = args.output or str(RESULTS_GEN / f"{args.model}_generated.xlsx")
    df = pd.DataFrame({"sequence": peptides, "length": [len(p) for p in peptides]})
    df.to_excel(out_path, index=False)
    print(f"Generated {len(peptides)} valid peptides -> {out_path}")


if __name__ == "__main__":
    main()
