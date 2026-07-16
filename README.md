# CVAE-AMP-Design

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

CVAE-based antimicrobial peptide (AMP) generation and multi-model activity prediction pipeline. This project integrates two computational stages for rational AMP design:

- **Stage 1 — Generation:** Conditional variational autoencoder (CVAE) with auxiliary property prediction loss, trained on 11,252 peptides to generate novel sequences with desired physicochemical profiles (low hemolysis, anti-E.coli, broad-spectrum antimicrobial).
- **Stage 2 — Prediction:** Ensemble of five complementary models (BiLSTM+Attention, XGBoost, Random Forest, GBDT, SVC) to predict antimicrobial, anti-endotoxin, and hemolytic activities, followed by CD-HIT deduplication and BLAST-based homology filtering.

## Project Structure

```
CVAE_AMP_Design/
├── README.md
├── requirements.txt
├── .gitignore
├── pyproject.toml
├── src/cvae_amp/                    # Importable Python package
│   ├── config/                      # Paths and hyperparameter defaults
│   ├── generation/                  # CVAE generative models
│   ├── prediction/                  # Activity prediction models & training
│   └── pipeline/                    # End-to-end orchestration
├── scripts/                         # CLI entry points
├── data/                            # Training datasets & feature tables
└──  models/                          # Trained model checkpoints
```

## Installation

```bash
cd CVAE_AMP_Design
pip install -e .
```

Or install dependencies only:

```bash
pip install -r requirements.txt
```

### External dependencies

The full pipeline requires [CD-HIT](https://github.com/weizhongli/cdhit) and [BLAST+](https://blast.ncbi.nlm.nih.gov/) for sequence deduplication and homology filtering. There are three ways to specify their paths (in priority order):

**1. CLI arguments (highest priority):**
```bash
python scripts/run_pipeline.py --cd-hit /path/to/cd-hit --blast-dir /path/to/ncbi-blast-plus/bin --num 100
```

**2. Environment variables:**
```bash
export CDHIT_BIN=/path/to/cd-hit
export BLAST_BIN=/path/to/ncbi-blast-plus/bin
```

**3. System PATH (lowest priority):**
If neither CLI args nor env vars are set, the pipeline looks for `cd-hit`, `makeblastdb`, and `blastp` on `$PATH`.

## Quick Start

### Generate Peptides (Stage 1)

```bash

# Generate 200 safe broad-spectrum AMP candidates
python scripts/generate.py --model cvae_pred --target 1.0,1.0,1.0 --num 200

# Compare all three generative models
python scripts/compare_models.py --num 1000
```

### Predict & Filter (Stage 2)

```bash
# Batch activity prediction
python scripts/predict.py data/dataset/amp_test2149.xlsx -o results/prediction/test_pred.xlsx

# Filter by threshold
python scripts/filter_results.py results/prediction/test_pred.xlsx -t 0.9
```

### Full Pipeline

```bash
# End-to-end: generate → CD-HIT → BLAST → predict → filter
python scripts/run_pipeline.py --model cvae_pred --target 1.0,1.0,1.0 --num 30000

# With explicit external tool paths
python scripts/run_pipeline.py --model cvae_pred --target 1.0,1.0,1.0 --num 30000 \
    --cd-hit /path/to/cd-hit --blast-dir /path/to/ncbi-blast-plus/bin
```

## Models

### Generation Models

| Model               | Parameters | Description                                    |
| ------------------- | ---------- | ---------------------------------------------- |
| VAE                 | 695,386    | Unconditional baseline                         |
| CVAE                | 696,922    | Label-conditioned with dropout                 |
| **CVAE+Pred** | 701,213    | Conditional + property predictor (recommended) |

### Prediction Models

| Model    | Type                         | Feature Set           |
| -------- | ---------------------------- | --------------------- |
| AMP      | BiLSTM + Attention (PyTorch) | AF7 (7 descriptors)   |
| AEP      | XGBoost                      | AF7 (7 descriptors)   |
| HP       | XGBoost                      | AF5_1 (5 descriptors) |
| Ensemble | XGBoost, RF, GBDT, SVC       | 9 feature sets        |

## Target Label Semantics

For conditional generation, three labels control desired peptide properties:

| Position | Property  | 1.0           | 0.0          | 0.5         |
| -------- | --------- | ------------- | ------------ | ----------- |
| 1st      | Hemolysis | Low (safe)    | High (toxic) | Unspecified |
| 2nd      | Ecoli     | Anti-E.coli   | No activity  | Unspecified |
| 3rd      | AMP       | Antimicrobial | No activity  | Unspecified |

Common targets: `1.0,1.0,1.0` (safe broad-spectrum AMP), `1.0,0.5,1.0` (safe AMP, no E.coli preference).

## Key Findings

- **CVAE+Pred** achieves 18/20 amino acid diversity and near-training-distribution charge (+6.93 vs +5.88), outperforming both VAE and pure CVAE.
- **Property prediction loss** stabilizes KL divergence (1.7–2.5 range vs 0–34 oscillation) and prevents amino acid collapse toward K/R-dominated sequences.
- **Label dropout** (10%) enables learning from the 82% of training samples with incomplete labels.

## Citation

If you use this code in your research, please cite the corresponding paper and this repository.

## License

MIT License. See [LICENSE](LICENSE) for details.
