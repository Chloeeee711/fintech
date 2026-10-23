"""
Project 2 — Global configuration.
All paths are resolved relative to the repository root (parent of /src).
The pipeline is fully OFFLINE: it reads the SEC 10-K Item 7 plain-text corpus
and the WRDS Compustat Fundamentals files shipped with the assignment, and
derives every dependent variable locally (future returns, controls).
"""
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
SEC_DIR = "/Users/celeste/Downloads/fintech_gw/SEC 10-K ITEM 7"
SEC_INDEX = os.path.join(SEC_DIR, "index.csv")
COMPUSTAT_ANNUAL_ZIP = "/Users/celeste/Downloads/fintech_gw/Compustat/Comp funda yr full 10_26.zip"

# --------------------------------------------------------------------------
# Analytical-sample design
# --------------------------------------------------------------------------
# We keep filings whose matched Compustat fiscal year allows a forward 1y AND
# 2y return to be computed, and we reserve the most recent years as a held-out
# test period. Filing years 2010..2019 -> train/validation, 2020..2022 -> test.
FILE_YEAR_MIN = 2010
FILE_YEAR_MAX = 2022          # last filing year with a computable fyear+1 return
TEST_FILE_YEAR_MIN = 2020     # held-out test window

# Text processing
MAX_CHARS = 12000             # cap Item-7 text length fed to models (≈ 2–3k words)
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
