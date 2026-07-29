"""Train ensemble models (XGBoost, RF, GBDT, SVC) with GridSearchCV.

Refactored from folder 39 training scripts.
Uses sklearn.utils.shuffle to fix the label de-sync bug.
"""

from __future__ import annotations

import os
import time
import argparse

import numpy as np
import openpyxl as xl
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib

from cvae_amp.config.paths import DATA_DATASET, DATA_FEATURES, MODELS_PRED, RESULTS
from cvae_amp.prediction.features import FeatureEncoder


TASK_FILES = {
    "amp": ("amp_train6450.xlsx", "amp_val2149.xlsx", "amp_test2149.xlsx"),
    "aep": ("aep_train1215.xlsx", "aep_val404.xlsx", "aep_test404.xlsx"),
    "hp": ("hp_train1250.xlsx", "hp_val416.xlsx", "hp_test416.xlsx"),
}

FEATURE_NAMES = [
    "AF5_1", "AF5_2", "AF5_3", "AF6_1", "AF6_2", "AF6_3", "AF6", "AF7",
]

PARAM_GRIDS = {
    "xgboost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.1, 0.55, 1.0],
        "subsample": [0.5, 0.6, 1.0],
    },
    "rf": {
        "n_estimators": [100, 200],
        "criterion": ["gini", "entropy"],
        "max_depth": [10, 50, None],
    },
    "gbdt": {
        "n_estimators": [100, 150],
        "learning_rate": [0.1, 0.55],
        "subsample": [0.5, 0.6, 0.8],
    },
    "svc": {
        "C": [0.1, 1.0, 1.5, 10.0],
        "kernel": ["rbf", "linear"],
    },
}


def _read_labels(path) -> np.ndarray:
    wb = xl.load_workbook(str(path))
    ws = wb.active
    labels = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        labels.append([int(row[-1])])
    return np.array(labels, dtype=np.float32).ravel()


def load_and_encode(task: str, feature: str):
    train_f, val_f, test_f = TASK_FILES[task]
    encoder = FeatureEncoder(str(DATA_FEATURES / f"20aa_{feature}_feature.xlsx"))

    x_train = encoder.encode_file(str(DATA_DATASET / train_f))
    y_train = _read_labels(DATA_DATASET / train_f)
    x_val = encoder.encode_file(str(DATA_DATASET / val_f))
    y_val = _read_labels(DATA_DATASET / val_f)
    x_test = encoder.encode_file(str(DATA_DATASET / test_f))
    y_test = _read_labels(DATA_DATASET / test_f)

    n_train, n_val, n_test = len(y_train), len(y_val), len(y_test)
    return (
        x_train.reshape(n_train, -1), y_train,
        x_val.reshape(n_val, -1), y_val,
        x_test.reshape(n_test, -1), y_test,
    )


def evaluate(y_true: np.ndarray, y_scores: np.ndarray, y_pred: np.ndarray):
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_scores)
    acc = accuracy_score(y_true, y_pred)
    return pre, rec, auc, acc


def train_xgboost(task: str, feature: str):
    x_train, y_train, x_val, y_val, x_test, y_test = load_and_encode(task, feature)

    # Fix: shuffle with sklearn to preserve label correspondence
    x_train, y_train = shuffle(x_train, y_train, random_state=16)
    x_val, y_val = shuffle(x_val, y_val, random_state=16)

    x_train_val = np.concatenate([x_train, x_val], axis=0)
    y_train_val = np.concatenate([y_train, y_val], axis=0)

    model = XGBClassifier(random_state=16)
    grid = GridSearchCV(model, PARAM_GRIDS["xgboost"], cv=10)
    grid.fit(x_train_val, y_train_val)

    best = XGBClassifier(**grid.best_params_, random_state=16)
    best.fit(x_train, y_train)

    y_pred = best.predict(x_test)
    y_scores = best.predict_proba(x_test)[:, 1]
    pre, rec, auc, acc = evaluate(y_test, y_scores, y_pred)

    save_path = MODELS_PRED / f"XGBoost_{task}_{feature}.pkl"
    joblib.dump(best, str(save_path))

    return {
        "model": "XGBoost", "task": task, "feature": feature,
        "best_params": grid.best_params_,
        "precision": pre, "recall": rec, "auc": auc, "acc": acc,
    }, str(save_path)


def train_rf(task: str, feature: str):
    x_train, y_train, x_val, y_val, x_test, y_test = load_and_encode(task, feature)
    x_train, y_train = shuffle(x_train, y_train, random_state=16)

    model = RandomForestClassifier(random_state=16)
    grid = GridSearchCV(model, PARAM_GRIDS["rf"], cv=10)
    x_train_val = np.concatenate([x_train, x_val], axis=0)
    y_train_val = np.concatenate([y_train, y_val], axis=0)
    grid.fit(x_train_val, y_train_val)

    best = RandomForestClassifier(**grid.best_params_, random_state=16)
    best.fit(x_train, y_train)

    y_pred = best.predict(x_test)
    y_scores = best.predict_proba(x_test)[:, 1]
    pre, rec, auc, acc = evaluate(y_test, y_scores, y_pred)

    save_path = MODELS_PRED / f"RF_{task}_{feature}.pkl"
    joblib.dump(best, str(save_path))

    return {
        "model": "RF", "task": task, "feature": feature,
        "best_params": grid.best_params_,
        "precision": pre, "recall": rec, "auc": auc, "acc": acc,
    }, str(save_path)


def train_gbdt(task: str, feature: str):
    x_train, y_train, x_val, y_val, x_test, y_test = load_and_encode(task, feature)
    x_train, y_train = shuffle(x_train, y_train, random_state=16)

    model = GradientBoostingClassifier(random_state=16)
    grid = GridSearchCV(model, PARAM_GRIDS["gbdt"], cv=10)
    x_train_val = np.concatenate([x_train, x_val], axis=0)
    y_train_val = np.concatenate([y_train, y_val], axis=0)
    grid.fit(x_train_val, y_train_val)

    best = GradientBoostingClassifier(**grid.best_params_, random_state=16)
    best.fit(x_train, y_train)

    y_pred = best.predict(x_test)
    y_scores = best.predict_proba(x_test)[:, 1]
    pre, rec, auc, acc = evaluate(y_test, y_scores, y_pred)

    save_path = MODELS_PRED / f"GBDT_{task}_{feature}.pkl"
    joblib.dump(best, str(save_path))

    return {
        "model": "GBDT", "task": task, "feature": feature,
        "best_params": grid.best_params_,
        "precision": pre, "recall": rec, "auc": auc, "acc": acc,
    }, str(save_path)


def train_svc(task: str, feature: str):
    x_train, y_train, x_val, y_val, x_test, y_test = load_and_encode(task, feature)
    x_train, y_train = shuffle(x_train, y_train, random_state=16)

    model = SVC(max_iter=2000)
    grid = GridSearchCV(model, PARAM_GRIDS["svc"], cv=10)
    x_train_val = np.concatenate([x_train, x_val], axis=0)
    y_train_val = np.concatenate([y_train, y_val], axis=0)
    grid.fit(x_train_val, y_train_val)

    best = SVC(**grid.best_params_, max_iter=2000)
    best.fit(x_train, y_train)

    y_pred = best.predict(x_test)
    y_scores = best.decision_function(x_test)
    pre, rec, auc, acc = evaluate(y_test, y_scores, y_pred)

    save_path = MODELS_PRED / f"SVC_{task}_{feature}.pkl"
    joblib.dump(best, str(save_path))

    return {
        "model": "SVC", "task": task, "feature": feature,
        "best_params": grid.best_params_,
        "precision": pre, "recall": rec, "auc": auc, "acc": acc,
    }, str(save_path)


TRAIN_FNS = {
    "xgboost": train_xgboost,
    "rf": train_rf,
    "gbdt": train_gbdt,
    "svc": train_svc,
}


def train_ensemble(
    method: str = "xgboost",
    task: str = "aep",
    feature: str = "AF7",
) -> str:
    os.makedirs(str(MODELS_PRED), exist_ok=True)
    os.makedirs(str(RESULTS), exist_ok=True)

    print(f"Training {method.upper()} on {task} with {feature}...")
    t0 = time.time()
    result, save_path = TRAIN_FNS[method](task, feature)
    elapsed = time.time() - t0

    print(f"  Done in {elapsed:.0f}s")
    print(f"  Best params: {result['best_params']}")
    print(f"  Test: Acc={result['acc']:.4f}, Pre={result['precision']:.4f}, "
          f"Rec={result['recall']:.4f}, AUC={result['auc']:.4f}")
    print(f"  Model saved to {save_path}")
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ensemble models with GridSearchCV")
    parser.add_argument("--method", default="xgboost", choices=["xgboost", "rf", "gbdt", "svc"])
    parser.add_argument("--task", default="aep", choices=["amp", "aep", "hp"])
    parser.add_argument("--feature", default="AF7")
    args = parser.parse_args()
    train_ensemble(args.method, args.task, args.feature)
