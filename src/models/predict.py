import pandas as pd
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import configurations as cfg
import models
from models import QBXGBoostModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ─── Positionaly-Invarient Prediction Pipeline ───────────────────────────────────────────

def load_features(season: int, input_file: Path) -> pd.DataFrame:
    """
    Load the feature-engineered dataset and filter to the prediction season.

    Args:
        season: The season to generate predictions for
        input_file: Path to the input file

    Returns:
        DataFrame containing feature rows for the given season
    """

    logger.info(f"Loading features for season {season} from {input_file}...")

    # Load the dataset
    df_features = pd.read_csv(
        Path.cwd().parent.parent / "src" / "data" / 
        input_file
    )

    return df_features[df_features["season"] == season].reset_index(drop=True)


def load_model(model_name: str, model):
    """
    Load most-recently saved QBXGBoostModel from disk.

    Args:
        model_name: Name of the saved model (used to resolve the file path via cfg)

    Returns:
        Loaded Model instance
    """

    checkpoints = sorted(cfg.CHECKPOINT_DIR.glob(f"{model_name}_*/"), reverse=True)

    if not checkpoints:
        raise FileNotFoundError(f"No saved checkpoints found for model '{model_name}'")
    
    latest = checkpoints[0]
    logger.info(f"Loading model from {latest}")
    return models.load(model, latest)


def generate_predictions(model, df: pd.DataFrame) -> pd.DataFrame:
    """
    Run inference and return a formatted prediction summary.

    Args:
        model: Trained Model
        df: Feature DataFrame for the prediction season

    Returns:
        DataFrame with player, season, and predicted values per target
    """

    logger.info("Generating predictions...")

    # Run inference
    predictions = model.predict(df)

    # Create a DataFrame with the predictions
    df_predictions = df[["player", "season"]].copy()
    df_predictions["season"] = df_predictions["season"] + 1
    for target in model.target_cols:
        df_predictions[target] = predictions[target].values

    # Rename columns to exclude 'target' label
    df_predictions = df_predictions.rename(columns={
        col: col.replace("target_", "") for col in model.target_cols
    })

    return df_predictions


def save_predictions(df_predictions: pd.DataFrame, season: int, pos: str) -> None:
    """
    Save prediction output to CSV.

    Args:
        df_predictions: Output of generate_predictions()
        season: Prediction season (used in output filename)
        pos: Position (used in output filename)
    """

    # Save the predictions to a CSV file
    df_predictions.to_csv(
        cfg.PREDICTIONS_DIR / f"{pos}_predictions_{season}.csv",
        index=False,
    )

    logger.info(f"Predictions saved to {cfg.PREDICTIONS_DIR / f'{pos}_predictions_{season}.csv'}")


# ─── Positionally-Dependent Metric Computation Helpers ───────────────────────────────────────────

def infer_metrics_qb(df_predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Infer positionally-dependent metrics for QB predictions.

    Args:
        df_predictions: DataFrame containing QB predictions
    """

    logger.info("Inferring metrics for QB predictions...")

    # Rate-based metrics
    df_predictions["cmp_pct"] = df_predictions["cmp"] * 100 / df_predictions["att"]
    df_predictions["td_pct"] = df_predictions["td"] * 100 / df_predictions["att"]
    df_predictions["int_pct"] = df_predictions["int"] * 100 / df_predictions["att"]

    df_predictions["y_per_a"] = df_predictions["yds"] / df_predictions["att"]
    df_predictions["ay_per_a"] = (df_predictions["yds"] + 
                                  df_predictions["td"] * 20 - 
                                  df_predictions["int"] * 45) / df_predictions["att"]
    df_predictions["y_per_c"] = df_predictions["yds"] / df_predictions["cmp"]
    df_predictions["y_per_g"] = df_predictions["yds"] / df_predictions["g"]

    # Append position
    df_predictions["pos"] = "QB"

    return df_predictions


# ─── Data Merging Function ───────────────────────────────────────────

def merge_predictions(df_predictions: pd.DataFrame, df_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Merge predictions with historical data

    Args:
        df_predictions: DataFrame containing predictions
        df_stats: DataFrame containing historical stats

    Returns:
        Merged DataFrame
    """

    # Round each numerical column to 1 decimal place
    num_cols = df_predictions.select_dtypes(include="float").columns
    df_predictions[num_cols] = df_predictions[num_cols].astype(float).round(1)

    # Get each player's most recent season to pull pfr_id and infer 2026 age
    df_latest = (
        df_stats.sort_values("season")
        .groupby("player", as_index=False)
        .last()[["player", "season", "pfr_id", "age"]]
    )
    df_latest["age"] = df_latest["age"] + (2026 - df_latest["season"])

    # Merge predictions with historical data
    df_predictions = df_predictions.merge(
        df_latest[["player", "pfr_id", "age"]],
        how="left",
        on="player"
    )

    # Restrict historical data to columns present in predictions + identifiers
    shared_cols = ["player", "season", "pfr_id", "age"] + [
        col for col in df_predictions.columns
        if col not in ("player", "season", "pfr_id", "age")
        and col in df_stats.columns
    ]
    df_stats_subset = df_stats[shared_cols]

    return pd.concat([df_stats_subset, df_predictions], ignore_index=True)


if __name__ == "__main__":
    season = cfg.YEARS.stop - 1  # Most recent season (2025)

    # Load features for the prediction season
    df: pd.DataFrame = load_features(
        season, 
        cfg.QB_OUTPUT_FEATURES_FILE.replace("{start}", "2018").replace("{end}", "2025")
    )

    # Load the corresponding model
    model: QBXGBoostModel = load_model("qb_xgboost", QBXGBoostModel)

    # Generate predictions
    df_predictions: pd.DataFrame = generate_predictions(model, df)

    # Using predictions, infer a variety of positionally-dependent metrics
    df_predictions = infer_metrics_qb(df_predictions)

    # Load historical stats to merge with predictions
    df_stats: pd.DataFrame = pd.read_csv(
        Path.cwd().parent.parent / "src" / "data" / 
        cfg.QB_OUTPUT_DATA_FILE.replace("{start}", "2018").replace("{end}", "2025")
    )

    # Merge predictions with historical stats
    df_predictions = merge_predictions(df_predictions, df_stats)

    # Save predictions/metrics
    save_predictions(df_predictions, season + 1, "qb")