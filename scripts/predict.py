#!/usr/bin/env python3
"""Batch activity prediction (AMP, AEP, HP)."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvae_amp.prediction.features import FeatureEncoder
from cvae_amp.prediction.loaders import load_amp, load_aep, load_hp
from cvae_amp.config.paths import DATA_FEATURES


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict peptide activities")
    parser.add_argument("input", type=str, help="Input .xlsx file (auto-detects 'seq' column, otherwise column 1)")
    parser.add_argument("-o", "--output", type=str, default=None)
    parser.add_argument("-m", "--models", nargs="+", default=["amp", "aep", "hp"],
                        choices=["amp", "aep", "hp"])
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    output = args.output or args.input.replace(".xlsx", "_pred.xlsx")
    device_str = args.device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    df_in = pd.read_excel(args.input)
    # Auto-detect sequence column: use 'seq' if present, otherwise first column
    if "seq" in df_in.columns:
        sequences = df_in["seq"].tolist()
    else:
        sequences = df_in.iloc[:, 0].tolist()
    n = len(sequences)
    print(f"Loaded {n} sequences from {args.input}")

    # Save to temp file for FeatureEncoder (which reads .xlsx)
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp_path = tmp.name
        pd.DataFrame({"id": range(1, n+1), "seq": sequences}).to_excel(tmp_path, index=False)

    results = {"sequence": sequences, "length": [len(str(s)) for s in sequences]}

    if "amp" in args.models:
        print("  Loading AMP model...")
        model_amp = load_amp(device)
        enc_amp = FeatureEncoder(str(DATA_FEATURES / "20aa_AF7_feature.xlsx"))
        feat = enc_amp.encode_file(tmp_path)
        t = torch.tensor(feat, dtype=torch.float32).to(device)
        with torch.no_grad():
            results["amp_Pred"] = model_amp(t).cpu().numpy().flatten()
        print(f"  AMP done (mean={results['amp_Pred'].mean():.4f})")

    if "aep" in args.models:
        print("  Loading AEP model...")
        model_aep = load_aep()
        enc_aep = FeatureEncoder(str(DATA_FEATURES / "20aa_AF7_feature.xlsx"))
        feat = enc_aep.encode_file(tmp_path)
        n_s = len(feat)
        results["aep_Pred"] = model_aep.predict_proba(feat.reshape(n_s, -1))[:, 1]
        print(f"  AEP done (mean={results['aep_Pred'].mean():.4f})")

    if "hp" in args.models:
        print("  Loading HP model...")
        model_hp = load_hp()
        enc_hp = FeatureEncoder(str(DATA_FEATURES / "20aa_AF5_1_feature.xlsx"))
        feat = enc_hp.encode_file(tmp_path)
        n_s = len(feat)
        results["hp_Pred"] = model_hp.predict_proba(feat.reshape(n_s, -1))[:, 1]
        print(f"  HP done (mean={results['hp_Pred'].mean():.4f})")

    os.unlink(tmp_path)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(results)
    df_out.to_excel(output, index=False)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
