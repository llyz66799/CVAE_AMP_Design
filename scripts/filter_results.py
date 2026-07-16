#!/usr/bin/env python3
"""Filter prediction results by score threshold."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvae_amp.pipeline.filters import filter_by_threshold


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter predictions by threshold")
    parser.add_argument("input", type=str, help="Prediction results .xlsx")
    parser.add_argument("-t", "--threshold", type=float, default=0.9)
    parser.add_argument("-o", "--output", type=str, default=None)
    parser.add_argument("-c", "--columns", nargs="+", default=None,
                        help="Columns to filter on (default: all *_Pred columns)")
    args = parser.parse_args()

    filter_by_threshold(args.input, args.threshold, args.output, args.columns)


if __name__ == "__main__":
    main()
