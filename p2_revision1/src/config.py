"""Revision 1 paths. Training scripts are retained pending separate repairs."""
import os

# --------------------------------------------------------------------------
# Repository layout
# --------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../P2_Submission
SRC = os.path.join(ROOT, "src")
DATA = os.path.join(ROOT, "data")
OUTPUT = os.path.join(ROOT, "outputs")
FIG = os.path.join(OUTPUT, "figures")
os.makedirs(DATA, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

# --------------------------------------------------------------------------
# Raw data locations (the assignment data on disk)
# --------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(ROOT))
SEC_DIR = os.path.join(PROJECT_ROOT, "SEC 10-K ITEM 7")
SEC_INDEX = os.path.join(SEC_DIR, "index.csv")
COMPUSTAT_ANNUAL_ZIP = os.path.join(PROJECT_ROOT, "Compustat", "Comp funda yr full 10_26.zip")

# --------------------------------------------------------------------------
# Analytical-sample design
# --------------------------------------------------------------------------
# We keep filings whose matched Compustat fiscal year allows a forward 1y AND
# 2y return to be computed, and we reserve the most recent years as a held-out
# test period. Filing years 2010..2019 -> train/validation, 2020..2022 -> test.
FILE_YEAR_MIN = 2016
FILE_YEAR_MAX = 2023          # last filing year with a computable fyear+1 return
TEST_FILE_YEAR_MIN = 2022     # held-out test window

# Text processing
MAX_CHARS = None             # cap Item-7 text length fed to models (≈ 2–3k words)
TFIDF_MAX_FEATURES = 20000
TFIDF_NGRAM = (1, 2)

# Transformer encoder
ENC_VOCAB_SIZE = 30000
ENC_MAX_LEN = 256             # token cap per document (was 128; 256 captures more
                              # of the Item-7 body at ~1.5-2x training cost)
ENC_EMBED = 96
ENC_HEADS = 4
ENC_LAYERS = 2
ENC_FF = 256
ENC_EPOCHS = 5
ENC_BATCH = 256
ENC_LR = 1e-3

# Random seed for reproducibility
SEED = 20261022

PREPROCESS_RUN = os.environ.get("P2_PREPROCESS_RUN")
METADATA_PATH = os.path.join(PROJECT_ROOT, "formal_project", "data", "external", "filing_metadata.csv")
