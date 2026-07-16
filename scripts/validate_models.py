#!/usr/bin/env python3
"""Validate AMP, AEP, HP prediction models on independent test datasets."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import openpyxl as xl
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvae_amp.prediction.features import FeatureEncoder
from cvae_amp.prediction.loaders import load_amp, load_aep, load_hp
from cvae_amp.config.paths import DATA_RAW, DATA_FEATURES, RESULTS


def _read_labels(path: Path) -> np.ndarray:
    wb = xl.load_workbook(str(path))
    ws = wb.active
    labels = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        labels.append(int(row[-1]))
    return np.array(labels)


def evaluate_one(model, x_test: np.ndarray, y_test: np.ndarray,
                 model_type: str, device: torch.device | None = None):
    """Evaluate a single model and return metrics dict."""
    if model_type == "amp":
        t = torch.tensor(x_test, dtype=torch.float32).to(device)
        with torch.no_grad():
            y_scores = model(t).cpu().numpy().flatten()
        y_pred = (y_scores > 0.5).astype(int)
    else:
        # XGBoost models: reshape to (n_samples, features)
        n = len(y_test)
        y_scores = model.predict_proba(x_test.reshape(n, -1))[:, 1]
        y_pred = model.predict(x_test.reshape(n, -1))

    return {
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "AUC": round(roc_auc_score(y_test, y_scores), 4),
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Define task config
    tasks = [
        {"name": "AMP", "test_file": "amp_test2149.xlsx",
         "feature_file": "20aa_pc7_feature.xlsx", "model_type": "amp"},
        {"name": "AEP", "test_file": "aep_test404.xlsx",
         "feature_file": "20aa_pc7_feature.xlsx", "model_type": "aep"},
        {"name": "HP",  "test_file": "hp_test416.xlsx",
         "feature_file": "20aa_pc5_1_feature.xlsx", "model_type": "hp"},
    ]

    # Load models
    print("\nLoading models...")
    model_amp = load_amp(device)
    model_aep = load_aep()
    model_hp = load_hp()
    models = {"amp": model_amp, "aep": model_aep, "hp": model_hp}
    print("  Done.")

    results = []

    for task in tasks:
        name = task["name"]
        print(f"\n{'='*50}")
        print(f"Evaluating {name}")
        print(f"{'='*50}")

        test_path = DATA_RAW / task["test_file"]
        feature_path = DATA_FEATURES / task["feature_file"]

        # Load and encode data
        y_test = _read_labels(test_path)
        encoder = FeatureEncoder(str(feature_path))
        x_test = encoder.encode_file(str(test_path))

        pos = (y_test == 1).sum()
        neg = (y_test == 0).sum()
        print(f"  Samples: {len(y_test)} (pos={pos}, neg={neg})")

        # Evaluate
        metrics = evaluate_one(
            models[task["model_type"]], x_test, y_test,
            task["model_type"], device,
        )
        metrics["Model"] = name
        metrics["Samples"] = len(y_test)
        results.append(metrics)

        for k, v in metrics.items():
            if k not in ("Model", "Samples"):
                print(f"  {k}: {v}")

    # Build result table
    df = pd.DataFrame(results)
    df = df[["Model", "Samples", "Accuracy", "Precision", "Recall", "AUC"]]

    # Print summary
    print(f"\n{'='*50}")
    print("Summary")
    print(f"{'='*50}")
    print(df.to_string(index=False))

    # Save
    out_path = str(RESULTS / "model_validation_results.xlsx")
    RESULTS.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="metrics", index=False)

        # Detailed predictions per model
        for task in tasks:
            name = task["name"]
            test_path = DATA_RAW / task["test_file"]
            feature_path = DATA_FEATURES / task["feature_file"]

            y_test = _read_labels(test_path)
            encoder = FeatureEncoder(str(feature_path))
            x_test = encoder.encode_file(str(test_path))

            model_type = task["model_type"]
            if model_type == "amp":
                t = torch.tensor(x_test, dtype=torch.float32).to(device)
                with torch.no_grad():
                    y_scores = models[model_type](t).cpu().numpy().flatten()
            else:
                n = len(y_test)
                y_scores = models[model_type].predict_proba(x_test.reshape(n, -1))[:, 1]

            detail_df = pd.DataFrame({
                "sample_id": range(1, len(y_test) + 1),
                "true_label": y_test,
                "pred_score": np.round(y_scores, 4),
                "pred_label": (y_scores > 0.5).astype(int),
            })
            detail_df.to_excel(writer, sheet_name=f"{name}_details", index=False)

    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
