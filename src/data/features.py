import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path
from typing import Self

sys.path.append(str(Path(__file__).resolve().parent.parent))
import configurations as cfg

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ─── General Helper Functions ───────────────────────────────────────────

def rolling_trend(series: pd.Series) -> float:
    """
    Compute the slope of a linear regression line over a rolling window

    Args:
        series: Series of values to compute the trend on

    Returns:
        Slope of the best fit line (trend) for the given series
    """

    if len(series) < 2:
        return 0
    
    x = np.arange(len(series))
    y = series.values

    return np.polyfit(x, y, 1)[0]


def add_age_features(self, prime, young, decline) -> Self:
        """
        Add age curve and experience features.

        ArgsL
            prime: Age considered the performance peak for the position
            young: Age below which players are considered "young"
            decline: Age at which performance typically starts to decline
        """

        # Calculate experience as count of prior seasons up to current season in dataset
        self.df["experience"] = self.df.groupby("pfr_id").cumcount() + 1
        self.df["is_first_season"] = (self.df["experience"] == 1).astype(int)

        # Use age squared to capture non-linear effects of aging
        self.df["age_squared"] = self.df["age"] ** 2

        # Calculate distance from empirical position peak age (prime)
        self.df["peak_distance"] = (self.df["age"] - prime).abs()

        # Flag young and old players based on typical performance curves
        self.df["is_young"] = (self.df["age"] <= young).astype(int)
        self.df["is_declining"] = (self.df["age"] >= decline).astype(int)

        # Estimate prime years remaining based on typical decline starting at decline
        self.df["prime_years_remaining"] = np.where(
            self.df["age"] < decline,
            decline - self.df["age"],  # Positive years remaining until decline
            0  # Already in decline phase
        )


class QBFeatures:
    """
    Transforms QB dataset into ML-ready features.
    Handles target construction, rolling features, age curves,
    and feature normalization
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.input_cols = df.columns.tolist() # Keep track of original columns for validation


    # ─── Target Construction ─────────────────────────────────────────────────

    def add_targets(self) -> "QBFeatures":
        """
        Shift next season's stats forward to create prediction targets.
        For each QB, creates target columns representing Year+1 performance
        """

        # Sort by QB and season
        self.df = self.df.sort_values(["pfr_id", "season"]).reset_index(drop=True)

        # Determine valid targets
        self.df["next_season"] = self.df.groupby("pfr_id")["season"].shift(-1)

        # Compute target gap
        self.df["gap"] = self.df["next_season"] - self.df["season"]

        # Define valid targets
        self.df["valid_target"] = self.df["gap"].between(1, cfg.MAX_GAP)
        self.df["gap_is_2"] = (self.df["gap"] == 2).astype(int)

        # Create target columns by shifting next season's stats
        for stat, target_col in cfg.QB_TARGET_COLS.items():
            self.df[target_col] = self.df.groupby("pfr_id")[stat].shift(-1)

            # Null targets for non-consecutive seasons
            self.df.loc[~self.df["valid_target"], target_col] = None

        # Drop helper columns
        self.df = self.df.drop(columns=["next_season", "gap"])


    # ─── Feature Computation ────────────────────────────────────────────────────

    def add_rolling_features(self, windows: list[int] = [2, 3]) -> "QBFeatures":
        """
        Compute rolling averages and trends over past N seasons
        for key performance metrics. Respects temporal order —
        only uses past data, never future data

        Args:
            windows: List of window sizes to compute rolling stats over
        """

        for n in windows:
            # Compute rolling features
            for rolling_col, stat in cfg.QB_ROLLING_FEATURES_MAP.items():
                self.df[rolling_col.format(n=n)] = (
                    self.df.groupby("pfr_id")[stat]     # Extract desired column
                    # Shift to avoid leakage and apply rolling mean
                    .transform(lambda s: s.shift(1).rolling(window=n, min_periods=1).mean())
                )

            for trend_col, stat in cfg.QB_ROLLING_TRENDS.items():
                self.df[trend_col.format(n=n)] = (
                    self.df.groupby("pfr_id")[stat]     # Extract desired column
                    # Shift to avoid leakage and apply rolling trend function
                    .transform(lambda s: s.shift(1).rolling(window=n, min_periods=1).apply(rolling_trend))
                )


    def add_consistency_features(self, windows: list[int] = [2, 3]) -> "QBFeatures":
        """
        Measure year-over-year consistency for key metrics.
        Consistent QBs are easier to predict and generally
        more valuable than volatile ones

        Args:
            windows: List of window sizes to compute consistency over
        """

        for n in windows:
            for consistency_col, stat in cfg.QB_CONSISTENCY_FEATURES_MAP.items():
                self.df[consistency_col.format(n=n)] = (
                    self.df.groupby("pfr_id")[stat]     # Extract desired column
                    # Shift to avoid leakage and apply rolling std
                    .transform(lambda s: s.shift(1).rolling(window=n, min_periods=1).std())
                )


    def add_confidence_features(self) -> "QBFeatures":
        """
        Add features that help the model weight predictions by
        sample size and data reliability
        """

        self.df["has_full_nflfastr"] = (
            self.df["passer_player_id"].notnull() &
            self.df["qb_epa_mean"].notnull() &
            self.df["cpoe_mean"].notnull() &
            self.df["air_yards_mean"].notnull() &
            self.df["pressure_rate"].notnull() &
            self.df["time_to_throw_mean"].notnull() &
            self.df["n_dropbacks"].notnull()
        ).astype(int)

        # Normalize n_dropbacks to 0-1 scale for confidence weighting
        self.df["dropback_confidence"] = (self.df["n_dropbacks"] / 600).clip(upper=1.0)


    # ─── Validation ──────────────────────────────────────────────────────────

    def validate_features(self) -> None:
        """
        Sanity check the feature engineered dataset before saving

        Raises:
            ValueError on critical failures
        """

        # Determine set of all feature columns
        feature_cols = set(self.df.columns) - set(self.input_cols) - set(cfg.QB_TARGET_COLS.values())

        # Check for data leakage
        leaky_features = set(cfg.QB_TARGET_COLS.keys()) & feature_cols
        if leaky_features:
            raise ValueError(f"Data leakage detected! Target columns {leaky_features} are also present as features")
        
        # Check for suspiciously strong predictors by shuffling targets and re-checking correlations
        shuffled_df = self.df.copy()
        for target_col in cfg.QB_TARGET_COLS.values():
            shuffled_df[target_col] = (
                shuffled_df[target_col].sample(frac=1).values
            )

        for col in feature_cols:
            if self.df[col].dtype.kind not in "bifc":  # Only check numeric features
                continue

            for target_col in cfg.QB_TARGET_COLS.values():
                corr = shuffled_df[col].corr(shuffled_df[target_col])

                if abs(corr) > cfg.MAX_CORR:
                    print(f"Warning: Feature '{col}' has high correlation with target '{target_col}' even after shuffling (corr={corr:.2f})")

        # Check that target columns exist and have reasonable null rates
        for col in cfg.QB_TARGET_COLS.values():
            if col not in self.df.columns:
                raise ValueError(f"Expected target column '{col}' not found in dataset")
            null_rate = self.df[col].isnull().mean()

            # Last season will always have null targets
            expected_null_rate = 1 / len(self.df["season"].unique())

            if null_rate > expected_null_rate * 1.1:  # Allow some buffer
                print(f"High null rate for target '{col}': {null_rate:.2%} (expected ~{expected_null_rate:.2%})")

        # Check that rolling features are NaN for first season (expected)
        first_season_mask = self.df["experience"] == 1
        for n in [2, 3]:
            for rolling_col in cfg.QB_ROLLING_FEATURES_MAP.keys():
                col = rolling_col.format(n=n)
                if col in self.df.columns:
                    if not self.df.loc[first_season_mask, col].isnull().all():
                        raise ValueError(f"Rolling feature '{col}' should be NaN for first season")
                    
        # Check for infinite values
        if np.isinf(self.df.select_dtypes(include=[np.number])).any().any():
            raise ValueError("Infinite values found in numeric features")
        
        # Check that rolling, trend and consistency features are only null for players' first season (expected)
        for n in [2, 3]:
            for rolling_col in cfg.QB_ROLLING_FEATURES_MAP.keys():
                col = rolling_col.format(n=n)
                if col in self.df.columns:
                    non_first_season_mask = self.df["experience"] > 1
                    if self.df.loc[non_first_season_mask, col].isnull().any():
                        raise ValueError(f"Rolling feature '{col}' should not be null for non-first seasons")
            
            for trend_col in cfg.QB_ROLLING_TRENDS.keys():
                col = trend_col.format(n=n)
                if col in self.df.columns:
                    non_first_season_mask = self.df["experience"] > 1
                    if self.df.loc[non_first_season_mask, col].isnull().any():
                        raise ValueError(f"Trend feature '{col}' should not be null for non-first seasons")

            for consistency_col in cfg.QB_CONSISTENCY_FEATURES_MAP.keys():
                col = consistency_col.format(n=n)
                if col in self.df.columns:
                    non_first_season_mask = self.df["experience"] > 1
                    if self.df.loc[non_first_season_mask, col].isnull().any():
                        raise ValueError(f"Consistency feature '{col}' should not be null for non-first seasons")
        
        # Determine number of rows with valid targets for training
        logger.info(f"Fully usable training rows: {len(self.df)}")
        logger.info(f"Seasons covered:\n{self.df['season'].value_counts().sort_index()}")
        

    # ─── Pipeline ────────────────────────────────────────────────────────────

    def build_features(self) -> pd.DataFrame:
        """
        Orchestrate the full feature engineering pipeline.
        Saves final feature set to data/processed/qb_features_{start}_{end}.csv

        Returns:
            df: Feature-engineered DataFrame ready for modeling
        """

        logger.info("Starting feature engineering pipeline...")

        # Add targets
        logger.info("Adding target columns...")
        self.add_targets()

        # Compute fumbles per game
        logger.info("Adding rate features...")
        self.df["fmb_per_g"] = self.df["fmb"] / self.df["gs"]

        # compute games percentage
        self.df["g_pct"] = self.df["g"] / 17

        # Add rolling features
        logger.info("Adding rolling features...")
        self.add_rolling_features()

        # Add age curve features
        logger.info("Adding age curve features...")
        add_age_features(
            self,
            prime=cfg.QB_PRIME_AGE,
            young=cfg.QB_YOUNG_AGE,
            decline=cfg.QB_DECLINE_AGE
        )

        # Add consistency features
        logger.info("Adding consistency features...")
        self.add_consistency_features()

        # Add confidence features
        logger.info("Adding confidence features...")
        self.add_confidence_features()

        # Drop rows corresponding to players with only one year in dataset
        self.df = self.df.groupby("pfr_id").filter(lambda x: len(x) > 1)

        # Drop unusable rows
        self.df = self.df[self.df["valid_target"] == True]

        # If its a player's second year in the dataset, set trend columns and consistency columns to 0
        trend_cols = [c for c in self.df.columns if c.startswith("trend_")]
        consistency_cols = [c for c in self.df.columns if "_std_" in c]
        second_year_mask = self.df["experience"] == 2
        for col in trend_cols:
            self.df.loc[second_year_mask, col] = 0
        for col in consistency_cols:
            self.df.loc[second_year_mask, col] = 0

        # If its a player's third year in the dataset, set 3yr trends/stds to 2yr trends/stds
        third_year_mask = self.df["experience"] == 3
        for col in cfg.QB_ROLLING_TRENDS.keys():
            col_3yr = col.format(n=3)
            col_2yr = col.format(n=2)
            if col_3yr in self.df.columns and col_2yr in self.df.columns:
                self.df.loc[third_year_mask, col_3yr] = self.df.loc[third_year_mask, col_2yr]
        for col in cfg.QB_CONSISTENCY_FEATURES_MAP.keys():
            col_3yr = col.format(n=3)
            col_2yr = col.format(n=2)
            if col_3yr in self.df.columns and col_2yr in self.df.columns:
                self.df.loc[third_year_mask, col_3yr] = self.df.loc[third_year_mask, col_2yr]

        # Validate final dataset
        logger.info("Validating final dataset...")
        self.validate_features()
        logger.info("Validation passed!")

        # Save final feature set
        output_path = cfg.QB_OUTPUT_FEATURES_FILE.format(start=cfg.YEARS.start, end=cfg.YEARS.stop - 1)
        self.df.to_csv(output_path, index=False)
        logger.info(f"Final dataset saved to {output_path}")

        return self.df
    

if __name__ == "__main__":
    # Load preprocessed QB dataset
    logger.info("Loading preprocessed QB dataset...")
    input_path = cfg.QB_OUTPUT_DATA_FILE.format(start=cfg.YEARS.start, end=cfg.YEARS.stop - 1)
    df_qb = pd.read_csv(input_path)
    features_qb = QBFeatures(df_qb)
    df_qb_features = features_qb.build_features()
    logger.info(df_qb_features.head())