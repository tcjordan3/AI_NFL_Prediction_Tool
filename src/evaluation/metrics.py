import pandas as pd
import numpy as np
import sys
from pathlib import Path


# ─── Core Metrics ───────────────────────────────────────────────────────────

def rmse(actual: pd.Series, predicted: pd.Series) -> float:
    """
    Compute Root Mean Squared Error between actual and predicted values

    Args:
        actual:    Series of actual values
        predicted: Series of predicted values

    Returns:
        RMSE as a float
    """

    return np.sqrt(np.mean((actual - predicted) ** 2))


def mae(actual: pd.Series, predicted: pd.Series) -> float:
    """
    Compute Mean Absolute Error between actual and predicted values

    Args:
        actual:    Series of actual values
        predicted: Series of predicted values

    Returns:
        MAE as a float
    """

    return np.mean(np.abs(actual - predicted))


def r_squared(actual: pd.Series, predicted: pd.Series) -> float:
    """
    Compute R² (coefficient of determination) between actual and predicted

    Args:
        actual:    Series of actual values
        predicted: Series of predicted values

    Returns:
        R² as a float
    """

    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)

    if ss_tot == 0:
        return 0.0  # Avoid division by zero
    
    return 1 - (ss_res / ss_tot)


# ─── Evaluation ─────────────────────────────────────────────────────────────

def evaluate(
    df: pd.DataFrame,
    predictions: pd.DataFrame,
    target_cols: list[str]
) -> pd.DataFrame:
    """
    Compute RMSE, MAE, and R² for each target column

    Args:
        df:          DataFrame containing actual target values
        predictions: DataFrame containing predicted values
        target_cols: List of target column names to evaluate

    Returns:
        df_metrics: DataFrame with columns [target, rmse, mae, r2]
    """
    
    metrics = []

    for target in target_cols:
        actual = df[target]
        predicted = predictions[target]

        # Align actual and predicted on index, then compute metrics
        actual, predicted = actual.align(predicted, join='inner', axis=0)

        # Filter null values
        mask = actual.notnull() & predicted.notnull()
        actual = actual[mask]
        predicted = predicted[mask]

        metrics.append(
            {
                "target": target,
                "rmse": rmse(actual, predicted),
                "mae": mae(actual, predicted),
                "r2": r_squared(actual, predicted)
            }
        )

    return pd.DataFrame(metrics).sort_values("target").reset_index(drop=True)


# ─── Model Comparison ───────────────────────────────────────────────────────

def compare_models(
    metrics: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Compare metrics across multiple models side by side.

    Args:
        metrics: Dict mapping model name to its evaluate() output

    Returns:
        df_comparison: DataFrame with targets as rows, models as columns, metrics as values
    """

    # Build per-metric comparison DataFrames
    df_rmse = pd.DataFrame()
    df_mae = pd.DataFrame()
    df_r2 = pd.DataFrame()

    # Construct comparison DataFrames
    for model_name, model_metrics in metrics.items():
        df_rmse[model_name] = model_metrics.set_index("target")["rmse"]
        df_mae[model_name] = model_metrics.set_index("target")["mae"]
        df_r2[model_name] = model_metrics.set_index("target")["r2"]

    # Identify best model for each metric
    df_rmse["best_model"] = df_rmse.idxmin(axis=1)
    df_mae["best_model"] = df_mae.idxmin(axis=1)
    df_r2["best_model"] = df_r2.idxmax(axis=1)

    # Combine into a single comparison DataFrame
    df_rmse.columns = pd.MultiIndex.from_tuples(
        [("rmse", c) for c in df_rmse.columns]
    )
    df_mae.columns = pd.MultiIndex.from_tuples(
        [("mae", c) for c in df_mae.columns]
    )
    df_r2.columns = pd.MultiIndex.from_tuples(
        [("r2", c) for c in df_r2.columns]
    )

    df_comparison = pd.concat([df_rmse, df_mae, df_r2], axis=1)

    return df_comparison.reset_index()


def baseline_improvement(
    baseline_metrics: pd.DataFrame,
    model_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute percentage improvement of a model over a baseline for each metric

    Args:
        baseline_metrics: evaluate() output for the baseline model
        model_metrics:    evaluate() output for the model to compare

    Returns:
        DataFrame with improvement metrics
    """

    baseline_indexed = baseline_metrics.set_index("target")
    model_indexed = model_metrics.set_index("target")

    rows = []
    for target in baseline_indexed.index:
        for metric in ["rmse", "mae", "r2"]:
            baseline_value = baseline_indexed.loc[target, metric]
            model_value = model_indexed.loc[target, metric]

            if metric in ["rmse", "mae"]:
                improvement = f"{(baseline_value - model_value) / baseline_value * 100:.2f}%"
            else:  # R² improvement is positive if model is better
                improvement = model_value - baseline_value # Report raw difference to avoid division by zero

            rows.append(
                {
                    "target": target,
                    "metric": metric,
                    "baseline": baseline_value,
                    "model": model_value,
                    "improvement_pct": str(improvement)
                }
            )

    return(
        pd.DataFrame(rows)
        .sort_values(["metric", "improvement_pct"], ascending=[True, False])
        .reset_index(drop=True)
    )


# ─── Per-Player Analysis ─────────────────────────────────────────────────────

def per_player_errors(
    df: pd.DataFrame,
    predictions: pd.DataFrame,
    target_cols: list[str],
    player_col: str = "player"
) -> pd.DataFrame:
    """
    Compute per-player metrics across all target columns

    Args:
        df:          DataFrame containing actuals and player column
        predictions: DataFrame containing predicted values
        target_cols: List of target column names to evaluate
        player_col:  Name of the player identifier column

    Returns:
        DataFrame with player metrics for each target
    """

    # Reset indices before joining to avoid alignment issues
    df_reset   = df[[player_col] + target_cols].reset_index(drop=True)
    pred_reset = predictions[target_cols].reset_index(drop=True)

    # Join actuals and prediction on index
    df_combined = df_reset.join(pred_reset, rsuffix="_pred")

    rows = []
    for target in target_cols:
        actual_col = target
        predicted_col = f"{target}_pred"

        for player, group in df_combined.groupby(player_col):
            actual = group[actual_col]
            predicted = group[predicted_col]

            # Only compute if player has at least one valid actual
            if actual.notna().sum() == 0:
                continue

            rows.append(
                {
                    "player": player,
                    "target": target,
                    "rmse": rmse(actual, predicted),
                    "mae": mae(actual, predicted),
                    "r2": r_squared(actual, predicted),
                    "n": actual.notna().sum() # sample size per-player
                }
            )

    return(
        pd.DataFrame(rows)
        .sort_values("mae", ascending=False) # Sort by MAE to highlight prediction difficulty
        .reset_index(drop=True)
    )


def prediction_summary(
    df: pd.DataFrame,
    predictions: pd.DataFrame,
    target_cols: list[str],
    player_col: str = "player",
    season_col: str = "season"
) -> pd.DataFrame:
    """
    Produce a summary table of actual vs predicted values per player per season

    Args:
        df:          DataFrame containing actuals, player, and season columns
        predictions: DataFrame containing predicted values
        target_cols: List of target column names to include
        player_col:  Name of the player identifier column
        season_col:  Name of the season column

    Returns:
        DataFrame with per-player per-season actual vs predicted values and errors
    """

    # Join actuals and predictions on index
    df_combined = df[[player_col, season_col] + target_cols].join(
        predictions[target_cols],
        rsuffix="_pred",
    )

    rows = []
    for target in target_cols:
        actual_col = target
        predicted_col = f"{target}_pred"

        for _, row in df_combined.iterrows():
            actual = row[actual_col]
            predicted = row[predicted_col]

            # Skip rows with no actual value
            if pd.isna(actual):
                continue

            rows.append(
                {
                    player_col: row[player_col],
                    season_col: row[season_col],
                    "target": target,
                    "actual": actual,
                    "predicted": predicted,
                    "error": actual - predicted,
                    "abs_error": abs(actual - predicted)
                }
            )

    return(
        pd.DataFrame(rows)
        .sort_values([player_col, season_col, "target"])
        .reset_index(drop=True)
    )
