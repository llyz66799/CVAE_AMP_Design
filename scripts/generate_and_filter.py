#!/usr/bin/env python3
"""Generate 500 sequences from each generative model (VAE, CVAE, CVAE+Pred),
then filter by prediction models — keep only sequences where AMP/AEP/HP all > 0.9.
Target labels: [1.0, 1.0, 1.0] (low hemolysis, anti-E.coli, antimicrobial).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import openpyxl as xl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvae_amp.generation.models import VAE, CVAE, CVAE_Pred, DEVICE
from cvae_amp.generation.sampling import sample_vae, sample_cvae
from cvae_amp.prediction.features import FeatureEncoder
from cvae_amp.prediction.loaders import load_amp, load_aep, load_hp
from cvae_amp.pipeline.filters import valid_sequence
from cvae_amp.config.paths import MODELS_GEN, MODELS_PRED, DATA_FEATURES, RESULTS
from cvae_amp.config.defaults import DEFAULT_TEMPERATURE, DEFAULT_TOP_K, DEFAULT_MIN_LEN, DEFAULT_MAX_LEN

MODEL_CONFIGS = {
    "VAE":        (VAE,        "vae_model.pth",      True),
    "CVAE":       (CVAE,       "cvae_model.pth",     False),
    "CVAE+Pred":  (CVAE_Pred,  "cvae_pred_model.pth", False),
}

TARGET = [1.0, 1.0, 1.0]
NUM_PER_MODEL = 500
THRESHOLD = 0.9


def generate_until(model, model_name: str, is_vae: bool, n: int) -> list[str]:
    """Generate until we have `n` valid unique sequences."""
    unique: set[str] = set()
    attempts = 0
    max_attempts = n * 20
    batch_size = 64

    print(f"  Generating with {model_name} (target={TARGET})...")
    while len(unique) < n and attempts < max_attempts:
        n_batch = min(batch_size, (n - len(unique)) * 2)
        if is_vae:
            peptides = sample_vae(model, temperature=DEFAULT_TEMPERATURE,
                                  top_k=DEFAULT_TOP_K, min_len=DEFAULT_MIN_LEN,
                                  num_samples=n_batch)
        else:
            peptides = sample_cvae(model, TARGET, temperature=DEFAULT_TEMPERATURE,
                                   top_k=DEFAULT_TOP_K, min_len=DEFAULT_MIN_LEN,
                                   num_samples=n_batch)
        for p in peptides:
            if valid_sequence(p, DEFAULT_MIN_LEN, DEFAULT_MAX_LEN):
                unique.add(p)
                if len(unique) >= n:
                    break
        attempts += n_batch

    seqs = sorted(unique)[:n]
    print(f"    Got {len(seqs)} valid sequences ({attempts} attempts)")
    return seqs


def predict_all(sequences: list[str], model_amp, model_aep, model_hp,
                enc_amp: FeatureEncoder, enc_hp: FeatureEncoder,
                device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict AMP, AEP, HP scores for all sequences."""
    import tempfile
    import os

    # Write to temp file (id + seq format for FeatureEncoder column 2)
    n = len(sequences)
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp_path = tmp.name
        pd.DataFrame({"id": range(1, n + 1), "seq": sequences}).to_excel(tmp_path, index=False)

    # Encode
    feat_amp = enc_amp.encode_file(tmp_path)
    feat_hp = enc_hp.encode_file(tmp_path)

    # AMP
    t = torch.tensor(feat_amp, dtype=torch.float32).to(device)
    with torch.no_grad():
        amp_scores = model_amp(t).cpu().numpy().flatten()

    # AEP
    aep_scores = model_aep.predict_proba(feat_amp.reshape(n, -1))[:, 1]

    # HP
    hp_scores = model_hp.predict_proba(feat_hp.reshape(n, -1))[:, 1]

    os.unlink(tmp_path)
    return amp_scores, aep_scores, hp_scores


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load prediction models (loaded once, used for all)
    print("\nLoading prediction models...")
    model_amp = load_amp(device)
    model_aep = load_aep()
    model_hp = load_hp()
    enc_amp = FeatureEncoder(str(DATA_FEATURES / "20aa_AF7_feature.xlsx"))
    enc_hp = FeatureEncoder(str(DATA_FEATURES / "20aa_AF5_1_feature.xlsx"))
    print("  Done.\n")

    all_results = []
    summary_rows = []

    for name, (model_cls, ckpt_name, is_vae) in MODEL_CONFIGS.items():
        print(f"{'='*60}")
        print(f"Model: {name}")
        print(f"{'='*60}")

        # Load generative model
        ckpt_path = MODELS_GEN / ckpt_name
        gen_model = model_cls().to(device)
        gen_model.load_state_dict(torch.load(str(ckpt_path), map_location=device))
        gen_model.eval()
        p_count = sum(p.numel() for p in gen_model.parameters())
        print(f"  Loaded from {ckpt_path} ({p_count:,} params)")

        # Generate
        seqs = generate_until(gen_model, name, is_vae, NUM_PER_MODEL)

        # Predict
        print(f"  Predicting activities...")
        amp_scores, aep_scores, hp_scores = predict_all(
            seqs, model_amp, model_aep, model_hp, enc_amp, enc_hp, device,
        )

        # Build DataFrame
        df = pd.DataFrame({
            "sequence": seqs,
            "length": [len(s) for s in seqs],
            "amp_Pred": np.round(amp_scores, 4),
            "aep_Pred": np.round(aep_scores, 4),
            "hp_Pred": np.round(hp_scores, 4),
        })

        # Filter high-confidence (all > THRESHOLD)
        pass_mask = (amp_scores > THRESHOLD) & (aep_scores > THRESHOLD) & (hp_scores > THRESHOLD)
        high_conf = df[pass_mask].copy()
        high_conf["pass_all"] = "Y"

        n_pass = len(high_conf)
        amp_mean = amp_scores.mean()
        aep_mean = aep_scores.mean()
        hp_mean = hp_scores.mean()

        print(f"  Generated: {len(seqs)}")
        print(f"  Pass all > {THRESHOLD}: {n_pass} ({100*n_pass/len(seqs):.1f}%)")
        print(f"  Mean scores — AMP: {amp_mean:.4f}, AEP: {aep_mean:.4f}, HP: {hp_mean:.4f}")

        all_results.append(df)
        summary_rows.append({
            "Model": name,
            "Generated": len(seqs),
            f"Pass_all_>{THRESHOLD}": n_pass,
            "Pass_rate": f"{100*n_pass/len(seqs):.1f}%",
            "AMP_mean": round(amp_mean, 4),
            "AEP_mean": round(aep_mean, 4),
            "HP_mean": round(hp_mean, 4),
        })
        print()

    # Save to Excel
    out_path = str(RESULTS / "generation_and_filter_results.xlsx")
    RESULTS.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        # Summary sheet
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_excel(writer, sheet_name="summary", index=False)

        # Per-model sheets
        for name, df in zip(MODEL_CONFIGS.keys(), all_results):
            # All generated
            df.to_excel(writer, sheet_name=f"{name}_all_{len(df)}", index=False)
            # High confidence only
            pass_mask = (df["amp_Pred"] > THRESHOLD) & (df["aep_Pred"] > THRESHOLD) & (df["hp_Pred"] > THRESHOLD)
            hc = df[pass_mask]
            hc.to_excel(writer, sheet_name=f"{name}_pass_{len(hc)}", index=False)

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(summary_df.to_string(index=False))
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
