"""Retrain all ensemble models with best-known hyperparameter configurations.

Refactored from retrain_all.py (folder 39). Trains XGBoost, RF, GBDT, and SVC
across all three tasks (AMP, AEP, HP) using pre-determined best configs from
prior GridSearchCV optimization. Uses sklearn.utils.shuffle to fix the label
de-sync bug present in the original.
"""

from __future__ import annotations

import os
import time
import argparse

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib

from cvae_amp.config.paths import DATA_RAW, DATA_FEATURES, MODELS_PRED, RESULTS
from cvae_amp.prediction.features import FeatureEncoder

FEATURE_PATHS: dict[str, str] = {
    "pc5_1": "20aa_pc5_1_feature.xlsx",
    "pc5_2": "20aa_pc5_2_feature.xlsx",
    "pc5_3": "20aa_pc5_3_feature.xlsx",
    "pc6_1": "20aa_pc6_1_feature.xlsx",
    "pc6_2": "20aa_pc6_2_feature.xlsx",
    "pc6_3": "20aa_pc6_3_feature.xlsx",
    "pc6": "20aa_pc6_feature.xlsx",
    "pc7": "20aa_pc7_feature.xlsx",
    "blosum62": "BLOSUM62_test_AtoY.xlsx",
}

DATASETS: dict[str, dict[str, str]] = {
    "aep": {"train": "aep_train1215.xlsx", "test": "aep_test404.xlsx"},
    "amp": {"train": "amp_train6450.xlsx", "test": "amp_test2149.xlsx"},
    "hp": {"train": "hp_train1250.xlsx", "test": "hp_test416.xlsx"},
}

BEST_CONFIGS: dict = {
    "XGBoost": {
        "aep": {"feature": "pc7", "n_estimators": 200, "learning_rate": 0.1, "subsample": 0.5},
        "amp": {"feature": "blosum62", "n_estimators": 200, "learning_rate": 0.1, "subsample": 0.5},
        "hp": {"feature": "pc5_1", "n_estimators": 100, "learning_rate": 0.1, "subsample": 0.6},
    },
    "RF": {
        "aep": {"feature": "pc7", "n_estimators": 200, "criterion": "gini", "max_depth": 50},
        "amp": {"feature": "pc5_3", "n_estimators": 100, "criterion": "gini", "max_depth": 50},
        "hp": {"feature": "pc5_2", "n_estimators": 200, "criterion": "entropy", "max_depth": 50},
    },
    "GBDT": {
        "aep": {"feature": "pc5_1", "n_estimators": 100, "learning_rate": 0.1, "subsample": 0.5},
        "amp": {"feature": "pc5_2", "n_estimators": 150, "learning_rate": 0.55, "subsample": 0.8},
        "hp": {"feature": "pc6_2", "n_estimators": 150, "learning_rate": 0.1, "subsample": 0.6},
    },
    "SVC": {
        "aep": {"feature": "pc6_2", "C": 1.5, "kernel": "rbf", "max_iter": -1},
        "amp": {"feature": "blosum62", "C": 1.5, "kernel": "rbf", "max_iter": -1},
        "hp": {"feature": "pc5_1", "C": 1.5, "kernel": "rbf", "max_iter": -1},
    },
}


def _read_labels(path) -> np.ndarray:
    import openpyxl as xl
    wb = xl.load_workbook(str(path))
    ws = wb.active
    labels = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        labels.append([int(row[-1])])
    return np.array(labels, dtype=np.float32).ravel()


def load_data(dataset_name: str, feature_name: str):
    train_path = DATA_RAW / DATASETS[dataset_name]["train"]
    test_path = DATA_RAW / DATASETS[dataset_name]["test"]
    feature_file = FEATURE_PATHS[feature_name]
    feature_path = DATA_FEATURES / feature_file

    encoder = FeatureEncoder(str(feature_path))
    x_train = encoder.encode_file(str(train_path))
    y_train = _read_labels(train_path)
    x_test = encoder.encode_file(str(test_path))
    y_test = _read_labels(test_path)

    n_train, n_test = len(y_train), len(y_test)
    x_train = x_train.reshape(n_train, -1)
    x_test = x_test.reshape(n_test, -1)

    # Fix: use sklearn.utils.shuffle to preserve label correspondence
    x_train, y_train = shuffle(x_train, y_train, random_state=16)
    x_test, y_test = shuffle(x_test, y_test, random_state=16)

    return x_train, y_train, x_test, y_test


def evaluate(model, model_name: str, x_test: np.ndarray, y_test: np.ndarray):
    if model_name == "SVC":
        y_scores = model.decision_function(x_test)
    else:
        y_scores = model.predict_proba(x_test)[:, 1]
    y_pred = model.predict(x_test)
    return (
        precision_score(y_test, y_pred, zero_division=0),
        recall_score(y_test, y_pred, zero_division=0),
        roc_auc_score(y_test, y_scores),
        accuracy_score(y_test, y_pred),
    )


def train_xgboost(cfg: dict, x_train: np.ndarray, y_train: np.ndarray):
    model = XGBClassifier(
        n_estimators=cfg["n_estimators"],
        learning_rate=cfg["learning_rate"],
        subsample=cfg["subsample"],
        random_state=16,
    )
    model.fit(x_train, y_train)
    return model


def train_rf(cfg: dict, x_train: np.ndarray, y_train: np.ndarray):
    model = RandomForestClassifier(
        n_estimators=cfg["n_estimators"],
        criterion=cfg["criterion"],
        max_depth=cfg["max_depth"],
        random_state=16,
    )
    model.fit(x_train, y_train)
    return model


def train_gbdt(cfg: dict, x_train: np.ndarray, y_train: np.ndarray):
    model = GradientBoostingClassifier(
        n_estimators=cfg["n_estimators"],
        learning_rate=cfg["learning_rate"],
        subsample=cfg["subsample"],
        random_state=16,
    )
    model.fit(x_train, y_train)
    return model


def train_svc(cfg: dict, x_train: np.ndarray, y_train: np.ndarray):
    model = SVC(
        C=cfg["C"],
        kernel=cfg["kernel"],
        max_iter=cfg.get("max_iter", -1),
    )
    model.fit(x_train, y_train)
    return model


TRAIN_FNS = {
    "XGBoost": train_xgboost,
    "RF": train_rf,
    "GBDT": train_gbdt,
    "SVC": train_svc,
}


def retrain_all(
    model_names: list[str] | None = None,
    dataset_names: list[str] | None = None,
) -> str:
    os.makedirs(str(MODELS_PRED), exist_ok=True)
    os.makedirs(str(RESULTS), exist_ok=True)

    if model_names is None:
        model_names = ["XGBoost", "RF", "GBDT", "SVC"]
    if dataset_names is None:
        dataset_names = ["aep", "amp", "hp"]

    results = []

    for model_name in model_names:
        for dataset_name in dataset_names:
            cfg = BEST_CONFIGS[model_name][dataset_name]
            feature_name = cfg["feature"]

            t0 = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] {model_name} on {dataset_name} "
                  f"(feature={feature_name}) ...", end=" ", flush=True)

            x_train, y_train, x_test, y_test = load_data(dataset_name, feature_name)
            model = TRAIN_FNS[model_name](cfg, x_train, y_train)

            save_path = MODELS_PRED / f"{model_name}_{dataset_name}.pkl"
            joblib.dump(model, str(save_path))

            pre, rec, auc, acc = evaluate(model, model_name, x_test, y_test)
            elapsed = time.time() - t0

            print(f"done ({elapsed:.0f}s) | acc={acc:.4f} prec={pre:.4f} rec={rec:.4f} auc={auc:.4f}")

            results.append({
                "Model": model_name, "Dataset": dataset_name, "Feature": feature_name,
                "Precision": round(pre, 4), "Recall": round(rec, 4),
                "AUC": round(auc, 4), "ACC": round(acc, 4),
            })

    df = pd.DataFrame(results)
    result_path = str(RESULTS / "retrain_results.xlsx")
    df.to_excel(result_path, index=False)
    print(f"\nResults saved to {result_path}")
    print(df.to_string(index=False))
    return result_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain ensemble models with best configs")
    parser.add_argument("--models", nargs="+", default=None,
                        choices=["XGBoost", "RF", "GBDT", "SVC"])
    parser.add_argument("--datasets", nargs="+", default=None,
                        choices=["amp", "aep", "hp"])
    args = parser.parse_args()
    retrain_all(args.models, args.datasets)
