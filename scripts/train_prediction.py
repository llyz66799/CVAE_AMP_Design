#!/usr/bin/env python3
"""Train prediction models: BiLSTM+Attention or ensemble (XGBoost/RF/GBDT/SVC)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train activity prediction models")
    parser.add_argument("--method", default="attention",
                        choices=["attention", "xgboost", "rf", "gbdt", "svc", "all_ensemble"])
    parser.add_argument("--task", default="amp", choices=["amp", "aep", "hp", "all"])
    parser.add_argument("--feature", default="AF7")
    parser.add_argument("--trials", type=int, default=30,
                        help="Optuna trials (attention only)")
    args = parser.parse_args()

    if args.method == "attention":
        from cvae_amp.prediction.training.train_attention import train_attention
        train_attention(args.task, args.feature, args.trials)

    elif args.method == "all_ensemble":
        from cvae_amp.prediction.training.retrain_best import retrain_all
        tasks = ["amp", "aep", "hp"] if args.task == "all" else [args.task]
        retrain_all(dataset_names=tasks)

    else:
        from cvae_amp.prediction.training.train_ensemble import train_ensemble
        train_ensemble(args.method, args.task, args.feature)


if __name__ == "__main__":
    main()
