"""
Configurations handling for repository.
"""

from pathlib import Path

# Normalization Constants

MIN_DROPBACKS   = 50
MIN_RUSH_ATT_QB = 25
SUFFIXES        = {"jr", "sr", "ii", "iii", "iv", "v"}

# Data Directories

RAW_DIR       = Path("raw")
PROCESSED_DIR = Path("processed")

# QB PFR File Paths

PFR_QB_PASSING          = str(RAW_DIR / "qb/passing/passing_{year}.csv")
PFR_QB_ADVANCED_PASSING = str(RAW_DIR / "qb/advanced_passing/advanced_passing_{year}.csv")
PFR_QB_RUSHING          = str(RAW_DIR / "rb/rushing/rushing_{year}.csv")
PFR_QB_NFLFASTR         = str(RAW_DIR / "qb/nflfastr/pbp_{year}.parquet")

# Output File Path
OUTPUT_FILE             = str(PROCESSED_DIR / "qb_stats_{start}_{end}.csv")

# Years to load for QB dataset
YEARS = range(2018, 2026)