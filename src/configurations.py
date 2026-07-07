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
    "rolling_yds_{n}yr":                "yds",
    "rolling_td_{n}yr":                 "td",
    "rolling_int_{n}yr":                "int",
    "rolling_cmp_pct_{n}yr":            "cmp_pct",
    "rolling_any_per_a_{n}yr":          "any_per_a",
    "rolling_qb_epa_mean_{n}yr":        "qb_epa_mean",
    "rolling_cpoe_mean_{n}yr":          "cpoe_mean",
    "rolling_rushing_yds_{n}yr":        "rushing_yds",
    "rolling_rushing_td_{n}yr":         "rushing_td",
    "rolling_gs_{n}yr":                 "gs",
    "rolling_epa_x_experience_{n}yr":   "epa_x_experience",
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


### base_models.py

ALPHA = 100.0 # For Ridge regression

### base_models.ipynb

# Set of feature columns to use in a training iteration
QB_FEATURE_COLS = [
    "gap_is_2",
    "any_per_a",
    "qb_epa_mean",
    "td",
    "peak_distance",
    "yoy_yds",
    "n_dropbacks",
    "att_per_g",
    "rolling_td_2yr",
    "yds",
    "dropback_confidence",
    "rolling_any_per_a_2yr",
    "int",
    "rolling_yds_2yr",
    "prime_years_remaining",
    "rolling_cpoe_mean_2yr",
    "att",
    "g",
    "rolling_gs_2yr",
    "rolling_int_2yr",
    "trend_qb_epa_mean_2yr",
    "trend_cmp_pct_2yr",
    "gs",
    "trend_cpoe_mean_2yr",
    "rolling_qb_epa_mean_2yr",
    "epa_x_experience",
    "yoy_epa",
    "cpoe_mean",
    "age",
    "rolling_epa_x_experience_2yr",
    "epa_std_2yr",
    "cpoe_std_2yr",
    "gs_std_2yr",
    "pressure_rate",
    "time_to_throw_mean"
]

QB_TARGET_FEATURE_COLS = {
    "target_yds": [
        "gap_is_2",
        "any_per_a",
        "qb_epa_mean",
        "td",
        "peak_distance",
        "yoy_yds",
        "n_dropbacks",
        "att_per_g",
        "rolling_td_2yr",
        "yds",
        "dropback_confidence",
        "rolling_any_per_a_2yr",
        "int",
        "rolling_yds_2yr",
        "prime_years_remaining",
        "rolling_cpoe_mean_2yr",
        "att",
        "g",
        "rolling_gs_2yr",
        "rolling_int_2yr",
        "trend_qb_epa_mean_2yr",
        "trend_cmp_pct_2yr",
        "gs",
        "trend_cpoe_mean_2yr",
        "rolling_qb_epa_mean_2yr",
        "epa_x_experience",
        "yoy_epa",
        "cpoe_mean",
        "age",
        "rolling_epa_x_experience_2yr",
        "epa_std_2yr",
        "cpoe_std_2yr",
        "gs_std_2yr",
        "pressure_rate",
        "time_to_throw_mean",
    ],
    "target_att": [
        "gap_is_2",
        "any_per_a",
        "qb_epa_mean",
        "td",
        "peak_distance",
        "yoy_yds",
        "n_dropbacks",
        "att_per_g",
        "rolling_td_2yr",
        "yds",
        "dropback_confidence",
        "rolling_any_per_a_2yr",
        "int",
        "rolling_yds_2yr",
        "prime_years_remaining",
        "rolling_cpoe_mean_2yr",
        "att",
        "g",
        "rolling_gs_2yr",
        "rolling_int_2yr",
        "trend_qb_epa_mean_2yr",
        "trend_cmp_pct_2yr",
        "gs",
        "trend_cpoe_mean_2yr",
        "rolling_qb_epa_mean_2yr",
        "epa_x_experience",
        "yoy_epa",
        "cpoe_mean",
        "age",
        "rolling_epa_x_experience_2yr",
        "epa_std_2yr",
        "cpoe_std_2yr",
        "gs_std_2yr",
        "pressure_rate",
        "time_to_throw_mean",
    ],
    "target_td": [
        "gap_is_2",
        "any_per_a",
        "qb_epa_mean",
        "td",
        "peak_distance",
        "yoy_yds",
        "n_dropbacks",
        "att_per_g",
        "rolling_td_2yr",
        "yds",
        "dropback_confidence",
        "rolling_any_per_a_2yr",
        "int",
        "rolling_yds_2yr",
        "prime_years_remaining",
        "rolling_cpoe_mean_2yr",
        "att",
        "g",
        "rolling_gs_2yr",
        "rolling_int_2yr",
        "trend_qb_epa_mean_2yr",
        "trend_cmp_pct_2yr",
        "gs",
        "trend_cpoe_mean_2yr",
        "rolling_qb_epa_mean_2yr",
        "epa_x_experience",
        "yoy_epa",
        "cpoe_mean",
        "age",
        "rolling_epa_x_experience_2yr",
        "epa_std_2yr",
        "cpoe_std_2yr",
        "gs_std_2yr",
        "pressure_rate",
        "time_to_throw_mean",
    ],
    "target_g": [
        "gap_is_2",
        "any_per_a",
        "qb_epa_mean",
        "td",
        "peak_distance",
        "yoy_yds",
        "n_dropbacks",
        "att_per_g",
        "rolling_td_2yr",
        "yds",
        "dropback_confidence",
        "rolling_any_per_a_2yr",
        "int",
        "rolling_yds_2yr",
        "prime_years_remaining",
        "rolling_cpoe_mean_2yr",
        "att",
        "g",
        "rolling_gs_2yr",
        "rolling_int_2yr",
        "trend_qb_epa_mean_2yr",
        "trend_cmp_pct_2yr",
        "gs",
        "trend_cpoe_mean_2yr",
        "rolling_qb_epa_mean_2yr",
        "epa_x_experience",
        "yoy_epa",
        "cpoe_mean",
        "age",
        "rolling_epa_x_experience_2yr",
        "epa_std_2yr",
        "cpoe_std_2yr",
        "gs_std_2yr",
        "pressure_rate",
        "time_to_throw_mean",
    ],
    "target_cmp": [
        "gap_is_2",
        "any_per_a",
        "qb_epa_mean",
        "td",
        "peak_distance",
        "yoy_yds",
        "n_dropbacks",
        "att_per_g",
        "rolling_td_2yr",
        "yds",
        "dropback_confidence",
        "rolling_any_per_a_2yr",
        "int",
        "rolling_yds_2yr",
        "prime_years_remaining",
        "rolling_cpoe_mean_2yr",
        "att",
        "g",
        "rolling_gs_2yr",
        "rolling_int_2yr",
        "trend_qb_epa_mean_2yr",
        "trend_cmp_pct_2yr",
        "gs",
        "trend_cpoe_mean_2yr",
        "rolling_qb_epa_mean_2yr",
        "epa_x_experience",
        "yoy_epa",
        "cpoe_mean",
        "age",
        "rolling_epa_x_experience_2yr",
        "epa_std_2yr",
        "cpoe_std_2yr",
        "gs_std_2yr",
        "pressure_rate",
        "time_to_throw_mean"
    ],
    "target_int": [
        "gap_is_2",
        "any_per_a",
        "qb_epa_mean",
        "td",
        "peak_distance",
        "yoy_yds",
        "n_dropbacks",
        "att_per_g",
        "rolling_td_2yr",
        "yds",
        "dropback_confidence",
        "int",
        "rolling_yds_2yr",
        "prime_years_remaining",
        "rolling_cpoe_mean_2yr",
        "rolling_gs_2yr",
        "rolling_int_2yr",
        "trend_cmp_pct_2yr",
        "gs",
        "rolling_qb_epa_mean_2yr",
        "epa_x_experience",
        "yoy_epa",
        "cpoe_mean",
        "age",
        "rolling_epa_x_experience_2yr",
        "epa_std_2yr",
        "cpoe_std_2yr",
        "pressure_rate",
        "time_to_throw_mean",
    ],
}

CHECKPOINT_DIR = Path(__file__).resolve().parent / "models" / "checkpoints"

##### trainer.py #####
QB_PHYSICAL_CONSTRAINTS = {
    "target_g": (0, 17),        # Games in a season
    "target_att": (0, None),    # Max QB attempts in a season
    "target_td": (0, None),     # Max QB passing TDs in a season
    "target_int": (0, None),    # Max QB interceptions in a season
    "target_yds": (0, None),    # Max QB passing yards in a season
}

##### qb_model.ipynb #####
QB_MODEL_PARAMS = {
    "n_estimators": 50,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.6,
    "colsample_bytree": 0.6,
    "min_child_weight": 1,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "random_state": 42,
}

QB_TARGET_MODEL_PARAMS = {
    "target_yds": {'max_depth': 3, 'min_child_weight': 5, 'n_estimators': 50, 'learning_rate': 0.05, 'subsample': 0.6, 'colsample_bytree': 0.4},
    "target_td": {'max_depth': 3, 'min_child_weight': 3, 'n_estimators': 40, 'learning_rate': 0.05, 'subsample': 0.6, 'colsample_bytree': 0.6},
    "target_int": {'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 25, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.6},
    "target_att": {'max_depth': 4, 'min_child_weight': 1, 'n_estimators': 50, 'learning_rate': 0.1, 'subsample': 0.6, 'colsample_bytree': 0.6},
    "target_cmp": {'max_depth': 3, 'min_child_weight': 3, 'n_estimators': 50, 'learning_rate': 0.05, 'subsample': 0.6, 'colsample_bytree': 0.6},
    "target_g": {'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 50, 'learning_rate': 0.05, 'subsample': 0.55, 'colsample_bytree': 0.5},
}

QB_RIDGE_SENSITIVITY = "ridge_alpha_sensitivity_best.csv"
QB_RIDGE_COEFFS = "ridge_coefficients_best.png"

##### predict.py #####
PREDICTIONS_DIR = Path(__file__).resolve().parent / "models" / "predictions"