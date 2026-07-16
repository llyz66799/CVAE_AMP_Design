"""Centralised path resolution for the project.

All filesystem paths are defined here so that changing the directory layout
only requires updating this single module.
"""

from __future__ import annotations

from pathlib import Path

# Project root (CVAE_AMP_Design/)
ROOT = Path(__file__).resolve().parents[3]

# Data
DATA_RAW = ROOT / "data" / "raw"
DATA_FEATURES = ROOT / "data" / "features"

# Model checkpoints
MODELS_GEN = ROOT / "models" / "generation"
MODELS_PRED = ROOT / "models" / "prediction"

# Output directories
RESULTS = ROOT / "results"
RESULTS_GEN = RESULTS / "generation"
RESULTS_PRED = RESULTS / "prediction"
RESULTS_PIPELINE = RESULTS / "pipeline"
RESULTS_FIGURES = RESULTS / "figures"

# Scripts
SCRIPTS = ROOT / "scripts"

# External tools (set via environment or use defaults)
import os

CDHIT_BIN = Path(os.environ.get("CDHIT_BIN", "/data/youzhuozhu/software/cd-hit-v4.8.1-2019-0228/cd-hit"))
BLAST_BIN = Path(os.environ.get("BLAST_BIN", "/data/youzhuozhu/software/ncbi-blast-2.17.0+/bin"))


def ensure_dirs() -> None:
    """Create all output directories if they do not exist."""
    for d in [RESULTS_GEN, RESULTS_PRED, RESULTS_PIPELINE, RESULTS_FIGURES]:
        d.mkdir(parents=True, exist_ok=True)
