"""Sequence validity and score threshold filters."""

from __future__ import annotations

import pandas as pd


def valid_sequence(seq: str, min_len: int = 8, max_len: int = 40) -> bool:
    if len(seq) < min_len or len(seq) > max_len:
        return False
    for aa in set(seq):
        if seq.count(aa) / len(seq) > 0.5:
            return False
    return True


def filter_by_threshold(
    input_path: str,
    threshold: float = 0.9,
    output_path: str | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    df = pd.read_excel(input_path)

    pred_cols = columns or [c for c in df.columns if c.endswith("_Pred")]
    mask = (df[pred_cols] > threshold).all(axis=1)
    filtered = df[mask].copy()

    if output_path is None:
        output_path = input_path.replace(".xlsx", f"_filtered_t{threshold}.xlsx")
    filtered.to_excel(output_path, index=False)
    print(f"Filtered: {len(filtered)} / {len(df)} sequences pass threshold {threshold}")
    print(f"Saved to {output_path}")
    return filtered
