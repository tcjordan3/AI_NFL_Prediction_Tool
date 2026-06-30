import pandas as pd
import sys
import logging
from pathlib import Path
from sklearn.impute import SimpleImputer
import joblib
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parent.parent))
from evaluation.metrics import evaluate
import configurations as cfg

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ─── Train/Test Splitting ────────────────────────────────────────────────────

def time_series_split(
    df: pd.DataFrame,
    test_season: int,
    season_col: str = "season"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train and test sets by season

    Args:
        df:          Full DataFrame to split
        test_season: Season to hold out as test set
        season_col:  Name of the season column

    Returns:
        df_train, df_test: training and test DataFrames
    """

    logging.info(f"Performing time series split with test season {test_season}...")

    df_train = df[(df[season_col] < test_season) & (df["valid_target"] == True)].copy()
    df_test = df[(df[season_col] == test_season) & (df["valid_target"] == True)].copy()

    logging.info(
        f"Train: {df_train[season_col].min()}-{df_train[season_col].max()} "
        f"({len(df_train)} rows) | "
        f"Test: {test_season} ({len(df_test)} rows)"
    )

    return df_train, df_test


def get_cv_folds(
    df: pd.DataFrame,
    n_splits: int = 3,
    season_col: str = "season"
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generate expanding window cross validation folds split by season.
    Each fold adds one additional season to the training set

    Args:
        df:        Full DataFrame to split
        n_splits:  Number of CV folds
        season_col: Name of the season column

    Returns:
        List of (df_train, df_val) tuples, one per fold
    """

    logging.info(f"Generating {n_splits} CV folds with expanding window split...")

    folds = []
    seasons = sorted(df[season_col].unique())

    if n_splits >= len(seasons):
        raise ValueError(f"n_splits={n_splits} must be less than the number of seasons={len(seasons)}")

    val_seasons = seasons[-n_splits:] # Last n_splits seasons used for validation

    for val_season in val_seasons:

        df_train = df[(df[season_col] < val_season) & (df["valid_target"] == True)].copy()
        df_val = df[(df[season_col] == val_season) & (df["valid_target"] == True)].copy()

        folds.append((df_train, df_val))

        # Log fold details
        logger.info(
            f"Fold {val_seasons.index(val_season) + 1}: "
            f"train {df_train[season_col].min()}-{df_train[season_col].max()} "
            f"({len(df_train)} rows) -> "
            f"val {val_season} ({len(df_val)} rows)"
        )

    return folds


# ─── Feature Preparation ────────────────────────────────────────────────────

def prepare_features(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, SimpleImputer]:
    """
    Prepare features for modeling by fitting a median imputer on
    training data and applying it to both train and test

    Args:
        df_train:     Training DataFrame
        df_test:      Test DataFrame
        feature_cols: List of feature column names to prepare

    Returns:
        df_train_prepared, df_test_prepared: prepared training and test DataFrames
        imputer: Fitted imputer for saving alongside model artifacts
    """

    logging.info("Preparing features with median imputation...")

    # Fit imputer on training data
    imputer = SimpleImputer(strategy="median")

    # Transform both train and test
    df_train_prepared = pd.DataFrame(
        imputer.fit_transform(df_train[feature_cols]),
        columns=feature_cols,
        index=df_train.index
    )

    df_test_prepared = pd.DataFrame(
        imputer.transform(df_test[feature_cols]),
        columns=feature_cols,
        index=df_test.index
    )

    return df_train_prepared, df_test_prepared, imputer


# ─── Cross Validation ────────────────────────────────────────────────────────

def apply_physical_constraints(predictions: pd.DataFrame, constraints: dict = None) -> pd.DataFrame:
    """
    Apply known physical constraints to model predictions

    Args:
        predictions: DataFrame of raw model predictions
        constraints: Dictionary mapping target column names to (min, max) tuples for clipping.

    Returns:
        Predictions DataFrame with physical constraints applied
    """

    logging.info("Applying physical constraints to predictions...")

    if constraints is None:
        return predictions

    predictions = predictions.copy()

    for col, (lower, upper) in constraints.items():
        if col not in predictions.columns:
            continue
        predictions[col] = predictions[col].clip(lower=lower, upper=upper)

    # Completions cannot exceed attempts
    if "target_cmp" in predictions.columns and "target_att" in predictions.columns:
        predictions["target_cmp"] = predictions["target_cmp"].clip(lower=0, upper=predictions["target_att"])

    return predictions


def run_cv(
    df: pd.DataFrame,
    model_class,
    model_kwargs: dict,
    feature_cols: list[str],
    target_cols: list[str],
    n_splits: int = 3,
    season_col: str = "season",
    constraints: dict = None
) -> pd.DataFrame:
    """
    Run time series cross validation for a given model class.
    Fits and evaluates the model on each CV fold independently

    Args:
        df:           Full feature-engineered training DataFrame
        model_class:  Model class to instantiate
        model_kwargs: Keyword arguments passed to model_class constructor
        feature_cols: List of feature column names
        target_cols:  List of target column names
        n_splits:     Number of CV folds
        season_col:   Name of the season column
        constraints:  Dictionary mapping target column names to (min, max) tuples for clipping.
    Returns:
        df_cv_results: DataFrame with per-fold evaluation metrics
    """

    logging.info(f"Running cross validation with {n_splits} folds...")

    # Generate CV folds
    folds = get_cv_folds(df, n_splits=n_splits, season_col=season_col)

    rows = []
    for i, (df_fold_train, df_fold_val) in enumerate(folds):
        fold_num = i + 1
        val_season = df_fold_val[season_col].iloc[0]

        logging.info(f"Fold {fold_num}: validating on season {val_season}...")

        # Prepare features for this fold
        x_train, x_val, _ = prepare_features(df_fold_train, df_fold_val, feature_cols)
        y_train = df_fold_train[target_cols]
        df_fit = x_train.join(y_train)

        # Instantiate a fresh model for this fold
        model = model_class(**model_kwargs)

        # Fit model on fold training data
        model.fit(df_fit)

        # Generate predictions
        predictions = model.predict(x_val)
        predictions = apply_physical_constraints(predictions, constraints=constraints)

        # Evaluate predictions against actuals
        df_metrics = evaluate(df_fold_val, predictions, target_cols=target_cols)
        df_metrics["fold"] = fold_num
        df_metrics["val_season"] = val_season
        rows.append(df_metrics)

        logging.info(
            f"Fold {fold_num} complete: "
            f"mean RMSE={df_metrics['rmse'].mean():.3f}"
        )

    df_cv_results = pd.concat(rows, ignore_index=True)
    df_cv_results = df_cv_results[["fold", "val_season", "target", "rmse", "mae", "r2"]]

    logging.info("Cross validation complete")
    return df_cv_results


def summarize_cv(df_cv_results: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize cross validation results across all folds

    Args:
        df_cv_results: Output from run_cv()

    Returns:
        df_summary: DataFrame with columns [target, rmse_mean, rmse_std,
                    mae_mean, mae_std, r2_mean, r2_std]
    """

    logging.info("Summarizing cross validation results...")

    df_summary = (
        df_cv_results.groupby("target")
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std")
        )
    )

    return df_summary.reset_index().sort_values("rmse_mean", ascending=False)


# ─── Full Training Pipeline ──────────────────────────────────────────────────

def train_and_evaluate(
    df: pd.DataFrame,
    model_class,
    model_kwargs: dict,
    feature_cols: list[str],
    target_cols: list[str],
    test_season: int,
    n_cv_splits: int = 3,
    apply_constraints: bool = True,
    save_model: bool = False,
    model_name: str = "model",
    season_col: str = "season",
    constraints: dict = None
) -> dict:
    """
    Run full training pipeline

    Args:
        df:               Full feature-engineered DataFrame
        model_class:      Model class to train
        model_kwargs:     Constructor kwargs for model_class
        feature_cols:     List of feature column names
        target_cols:      List of target column names
        test_season:      Season to hold out for final evaluation
        n_cv_splits:      Number of CV folds to run
        apply_constraints: Whether to apply physical constraints to predictions
        save_model:       Whether to save fitted model to disk
        model_name:       Name used for saved model artifacts
        season_col:       Name of the season column
        constraints:      Dictionary mapping target column names to (min, max) tuples for clipping.

    Returns:
        results: Dict containing —
            "cv_results":    raw fold-level CV metrics DataFrame
            "cv_summary":    summarized CV metrics DataFrame
            "test_metrics":  final test set evaluate() output
            "predictions":   final test set predictions DataFrame
            "model":         fitted model instance
            "imputer":       fitted imputer instance
    """

    logging.info(
        f"Starting full training pipeline "
        f"for model {model_class.__name__} "
        f"with test season {test_season} "
        f"and CV folds {n_cv_splits}..."
    )

    # 1. Train/test split
    df_train, df_test = time_series_split(df, test_season=test_season, season_col=season_col)

    # 2. Cross validation on training data
    df_cv_results = run_cv(
        df=df_train,
        model_class=model_class,
        model_kwargs=model_kwargs,
        feature_cols=feature_cols,
        target_cols=target_cols,
        n_splits=n_cv_splits,
        season_col=season_col,
        constraints=constraints
    )
    df_cv_summary = summarize_cv(df_cv_results)

    # 3. Final model fit on full training set
    x_train, x_test, imputer = prepare_features(df_train, df_test, feature_cols)
    df_fit = x_train.join(df_train[target_cols])
    model = model_class(**model_kwargs)
    model.fit(df_fit)

    # 4. Generate test predictions
    predictions = model.predict(x_test)

    # 5. Optional physical constraint post-processing
    if apply_constraints:
        predictions = apply_physical_constraints(predictions, constraints=constraints)

    # 6. Evaluate on held-out test set
    test_metrics = evaluate(df_test, predictions, target_cols=target_cols)

    logging.info(
        f"Test set evaluation complete: "
        f"mean RMSE={test_metrics['rmse'].mean():.3f} "
        f"mean R²: {test_metrics['r2'].mean():.3f} "
    )

    # 7. Optional model artifact saving
    if save_model:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = Path(cfg.CHECKPOINT_DIR) / f"{model_name}_{now}.pkl"
        imputer_path = Path(cfg.CHECKPOINT_DIR) / f"{model_name}_imputer_{now}.pkl"
        cv_path = Path(cfg.CHECKPOINT_DIR) / f"{model_name}_cv_results_{now}.csv"

        joblib.dump(model, model_path)
        joblib.dump(imputer, imputer_path)
        df_cv_results.to_csv(cv_path, index=False)

        logging.info(f"Saved model to {model_path}")
        logging.info(f"Saved imputer to {imputer_path}")
        logging.info(f"Saved CV results to {cv_path}")

    return {
        "cv_results": df_cv_results,
        "cv_summary": df_cv_summary,
        "test_metrics": test_metrics,
        "predictions": predictions,
        "model": model,
        "imputer": imputer,
    }