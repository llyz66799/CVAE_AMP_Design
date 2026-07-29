"""Default hyperparameters and constants shared across all modules."""

from __future__ import annotations

# ── Reproducibility ──
SEED = 42

# ── Sequence encoding ──
MAX_LEN = 50
VOCAB_SIZE = 21  # 20 standard AAs + PAD token
PAD_ID = 20
LABEL_DIM = 3  # hemolysis, anti-E.coli, antimicrobial

# ── Generation model architecture ──
INPUT_DIM = MAX_LEN * VOCAB_SIZE  # 1050
Z_DIM = 32
HIDDEN_DIM = 256

# ── Generation training ──
GEN_EPOCHS = 500
GEN_BATCH_SIZE = 64
GEN_LR = 1e-4
KL_CYCLE = 100
KL_MAX_BETA = 0.5
PROP_WEIGHT = 10.0
WORD_DROPOUT_RATE = 0.05
LABEL_DROPOUT_RATE = 0.10

# ── Generation sampling ──
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_K = 5
DEFAULT_MIN_LEN = 8
DEFAULT_MAX_LEN = 40

# ── Prediction model architecture ──
AMP_INPUT_DIM = 7   # AF7 descriptors
AMP_LSTM_HIDDEN = 64
AMP_NUM_LAYERS = 2
AMP_DROPOUT = 0.4
HP_FEATURE_DIM = 5
AEP_FEATURE_DIM = 7

# ── Prediction training ──
PRED_EPOCHS = 300
PRED_BATCH_SIZE = 128
PRED_PATIENCE = 100
OPTUNA_TRIALS = 30

# ── Pipeline ──
PIPELINE_NUM_GEN = 30_000
PIPELINE_CDHIT_IDENTITY = 0.70
PIPELINE_BLAST_EVALUE = 1e-5
PIPELINE_BLAST_IDENTITY_CUTOFF = 30.0
PIPELINE_PRED_THRESHOLD = 0.9

# ── AA vocabulary ──
AA_LIST = [
    "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
    "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y", "PAD",
]
AA_TO_IDX: dict[str, int] = {aa: i for i, aa in enumerate(AA_LIST)}
AA_TO_IDX["B"] = 20  # alias: "B" used as padding token in prediction module
IDX_TO_AA: dict[int, str] = {i: aa for i, aa in enumerate(AA_LIST)}
