"""
Configurations handling for repository.
"""

from pathlib import Path


##### processors.py #####

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
QB_OUTPUT_DATA_FILE             = str(PROCESSED_DIR / "qb_stats_{start}_{end}.csv")

# Years to load for QB dataset
YEARS = range(2018, 2026)


##### features.py #####

MAX_NULL_RATE = 0.2  # Maximum allowed null rate for features
MAX_CORR      = 0.3  # Maximum allowed correlation with target variable for features after shuffling
MAX_GAP       = 2    # Maximum allowed gap in seasons for target variable

# Target column mapping
QB_TARGET_COLS = {
        "yds":          "target_yds",
        "td":           "target_td",
        "int":          "target_int",
        "att":          "target_att",
        "cmp":          "target_cmp",
        "g":            "target_g",
    }

# Rolling feature mapping (rolling column name to source stat)
QB_ROLLING_FEATURES_MAP = {
    "rolling_yds_{n}yr":            "yds",
    "rolling_td_{n}yr":             "td",
    "rolling_int_{n}yr":            "int",
    "rolling_cmp_pct_{n}yr":        "cmp_pct",
    "rolling_any_per_a_{n}yr":      "any_per_a",
    "rolling_qb_epa_mean_{n}yr":    "qb_epa_mean",
    "rolling_cpoe_mean_{n}yr":      "cpoe_mean",
    "rolling_rushing_yds_{n}yr":    "rushing_yds",
    "rolling_rushing_td_{n}yr":     "rushing_td",
    "rolling_gs_{n}yr":             "gs",
}

# Trend feature mapping (trend column name to source stat)
QB_ROLLING_TRENDS = {
    "trend_yds_{n}yr":          "yds",
    "trend_td_{n}yr":           "td",
    "trend_int_{n}yr":          "int",
    "trend_cmp_pct_{n}yr":      "cmp_pct",
    "trend_any_per_a_{n}yr":    "any_per_a",
    "trend_qb_epa_mean_{n}yr":  "qb_epa_mean",
    "trend_cpoe_mean_{n}yr":    "cpoe_mean",
    "trend_rushing_yds_{n}yr":  "rushing_yds",
    "trend_rushing_td_{n}yr":   "rushing_td",
    "trend_gs_{n}yr":           "gs",
}

# Consistency feature mapping (consistency column name to source stat)
QB_CONSISTENCY_FEATURES_MAP = {
    "epa_std_{n}yr": "qb_epa_mean",
    "cpoe_std_{n}yr": "cpoe_mean",
    "gs_std_{n}yr":   "gs",
}

QB_PRIME_AGE = 29
QB_YOUNG_AGE = 25
QB_DECLINE_AGE = 33

QB_OUTPUT_FEATURES_FILE = str(PROCESSED_DIR / "qb_features_{start}_{end}.csv")