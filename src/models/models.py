import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path
from typing import Self
from datetime import datetime
import json
import matplotlib.pyplot as plt

import xgboost as xgb
import shap

sys.path.append(str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


# ─── General Helper Functions ───────────────────────────────────────────────────

def validate_fitted(self) -> None:
    """
    Raise ValueError if model has not been fitted yet
    """

    if not self.is_fitted:
        raise ValueError(
            "Model must be fitted before calling this method. "
            "Call fit() first."
        )
    

def validate_columns(
    self,
    df: pd.DataFrame,
    context: str = ""
) -> None:
    """
    Verify all required feature columns are present in df

    Args:
        df:      DataFrame to validate
        context: Caller name for clearer error messages
    """

    # Collect all feature columns present among targets
    all_required = set(self.feature_cols)
    for features in self.target_feature_cols.values():
        all_required.update(features)

    # Determine if any required features are missing
    missing = [col for col in all_required if col not in df.columns]
    if missing:
        context_str = f" in {context}" if context else ""
        raise ValueError(f"Missing required columns{context_str}: {missing}")
    

def save(self, path: str | Path) -> None:
        """
        Save all fitted models and metadata to disk

        Args:
            path: Directory path to save model artifacts
        """

        # Validate model is fitted before saving
        validate_fitted(self)
        now = datetime.now().strftime('%Y%m%d_%H%M%S')

        path = Path(f"{path}_{now}")
        path.mkdir(parents=True, exist_ok=True)

        model: xgb.XGBRegressor = None  # Type hint
        logger.info(f"Saving XGBoostModel with targets {self.target_cols} at {now} to {path}")

        # Save per-target models as separate .json files
        for target_col, model in self.models.items():
            model.save_model(f"{path}/{target_col}.json")

        # Save metadata (feature/target columns) as a separate .json file
        metadata = {
            "target_cols": self.target_cols,
            "feature_cols": self.feature_cols,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "random_state": self.random_state,
            "target_params": self.target_params,
            "target_feature_cols": self.target_feature_cols,
        }

        with open(f"{path}/metadata.json", "w") as f:
            json.dump(metadata, f)

        logger.info(f"Model saving complete!")


@classmethod
def load(cls, path: str | Path) -> Self:
    """
    Load a previously saved XGBoostModel from disk

    Args:
        cls:  Class reference for XGBoostModel
        path: Directory path where model artifacts are saved

    Returns:
        Loaded XGBoostModel instance ready for prediction
    """

    # Convert path to Path object if it's a string
    path = Path(path)

    # Verify the specified path exists before attempting to load
    if not path.exists():
        raise FileNotFoundError(f"Specified path does not exist: {path}")

    # Load metadata to reconstruct model configuration
    with open(path / "metadata.json", "r") as f:
        metadata = json.load(f)

    # Load model metadata
    instance = cls(
        target_cols=metadata["target_cols"],
        feature_cols=metadata["feature_cols"],
        n_estimators=metadata["n_estimators"],
        max_depth=metadata["max_depth"],
        learning_rate=metadata["learning_rate"],
        subsample=metadata["subsample"],
        colsample_bytree=metadata["colsample_bytree"],
        min_child_weight=metadata["min_child_weight"],
        reg_alpha=metadata["reg_alpha"],
        reg_lambda=metadata["reg_lambda"],
        random_state=metadata["random_state"],
        target_params=metadata.get("target_params", {}),
        target_feature_cols=metadata.get("target_feature_cols", {})
    )

    # Load each target's model from its corresponding .json file
    for target_col in metadata["target_cols"]:
        model_path = path / f"{target_col}.json"

        # Verify model file exists before attempting to load
        if not model_path.exists():
            raise FileNotFoundError(f"Model file for target '{target_col}' not found at {model_path}")
        
        model = xgb.XGBRegressor()
        model.load_model(path / f"{target_col}.json")
        instance.models[target_col] = model
        logger.info(f"Loaded XGBoost model for target {target_col} from {model_path}")

    instance.is_fitted = True
    logger.info(f"Model loading complete. Model is fitted and ready for prediction")
    return instance


class QBXGBoostModel:
    """
    XGBoost model for QB performance prediction.
    Trains one XGBoost regressor per target column, following
    the same fit/predict interface as RidgeRegressionModel
    """

    def __init__(
        self,
        target_cols: list[str],
        feature_cols: list[str],
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 5,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        target_params: dict = None,
        target_feature_cols: dict = None,
    ):
        """
        Args:
            target_cols:            List of target column names to predict
            feature_cols:           List of feature column names to use as inputs
            n_estimators:           Number of boosting rounds
            max_depth:              Maximum tree depth
            learning_rate:          Step size shrinkage
            subsample:              Fraction of rows sampled per tree
            colsample_bytree:       Fraction of features sampled per tree
            min_child_weight:       Minimum sum of instance weight in a leaf
            reg_alpha:              L1 regularization on weights (Lasso)
            reg_lambda:             L2 regularization on weights (Ridge)
            random_state:           Random seed for reproducibility
            target_params:          Per-target hyperparameters
            target_feature_cols:    Per-target feature columns
        """

        self.target_cols = target_cols
        self.feature_cols = feature_cols
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.models = {}  # Dictionary to store fitted XGBoost models keyed by target name
        self.is_fitted = False
        self.target_params = target_params or {}
        self.target_feature_cols = target_feature_cols or {}

    # ─── Core Interface ──────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> Self:
        """
        Fit one XGBoost regressor per target column

        Args:
            df: Training DataFrame containing both features and targets
        """

        for target_col in self.target_cols:
            # Extract per-target features
            features = self.target_feature_cols.get(target_col, self.feature_cols)

            # Drop per-target rows where target is null
            df_target = df[df[target_col].notna()].copy()

            # Extract target-label rows for this target
            x = df_target[features]
            y = df_target[target_col].values

            # Merge shared params with any target-specific overrides
            shared_params = {
                "n_estimators":     self.n_estimators,
                "max_depth":        self.max_depth,
                "learning_rate":    self.learning_rate,
                "subsample":        self.subsample,
                "colsample_bytree": self.colsample_bytree,
                "min_child_weight": self.min_child_weight,
                "reg_alpha":        self.reg_alpha,
                "reg_lambda":       self.reg_lambda,
                "random_state":     self.random_state,
            }
            overrides = self.target_params.get(target_col, {})
            final_params = {**shared_params, **overrides}

            # Fit XGBoost regressor with specified hyperparameters
            model = xgb.XGBRegressor(**final_params)
            model.fit(x, y)
            self.models[target_col] = model
            logger.info(f"Fitted XGBoost model for target {target_col} on {len(df_target)} rows")

        self.is_fitted = True

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for each target using its fitted XGBoost model.

        Args:
            df: DataFrame containing feature columns

        Returns:
            df_predictions: DataFrame with one column per target,
                            indexed to match input df
        """

        # Validate model is fitted and has required feature columns before predicting
        validate_fitted(self)
        validate_columns(self, df, context="QBXGBoostModel.predict")

        df_predictions = pd.DataFrame(index=df.index)

        for target_col in self.target_cols:
            # Extract per-target feature columns
            features = self.target_feature_cols.get(target_col, self.feature_cols)

            # Generate predictions using the fitted model for this target
            x = df[features]
            df_predictions[target_col] = self.models[target_col].predict(x)

        return df_predictions

    # ─── Interpretability ────────────────────────────────────────────────────

    def get_feature_importance(
        self,
        importance_type: str = "gain"
    ) -> pd.DataFrame:
        """
        Return XGBoost feature importance for each fitted model

        Args:
            importance_type: One of 'gain', 'weight', 'cover'

        Returns:
            df_importance: DataFrame with features as rows, targets as columns,
                           sorted by mean importance descending
        """

        # Validate model is fitted before accessing feature importance
        validate_fitted(self)

        all_features = list(dict.fromkeys(
            self.feature_cols +
            [f for cols in self.target_feature_cols.values() for f in cols]
        ))  # union of all feature sets, preserving order

        df_importance = pd.DataFrame(index=all_features)

        for target_col in self.target_cols:
            # Extract model
            model: xgb.XGBRegressor = self.models[target_col]

            # Get importance scores from the model and align with feature columns
            importance_dict = model.get_booster().get_score(importance_type=importance_type)
            importance_series = pd.Series(importance_dict).reindex(all_features).fillna(0)
            df_importance[target_col] = importance_series

        df_importance["mean_importance"] = df_importance[self.target_cols].mean(axis=1)
        return (
            df_importance.sort_values("mean_importance", ascending=False)
            .drop(columns=["mean_importance"])
        )

    def get_shap_values(
        self,
        df: pd.DataFrame,
        target_col: str
    ) -> tuple[np.ndarray, shap.Explainer]:
        """
        Compute SHAP values for a specific target using its fitted model

        Args:
            df:         DataFrame containing feature columns to explain
            target_col: Which target model to explain

        Returns:
            explainer:   Fitted SHAP TreeExplainer for further analysis
            explanation: SHAP values for each feature and instance in df
        """
        # Validate model is fitted and has required feature columns before computing SHAP values
        validate_fitted(self)
        validate_columns(self, df, context="QBXGBoostModel.get_shap_values")

        # Extract per-target feature columns
        features = self.target_feature_cols.get(target_col, self.feature_cols)

        explainer: shap.Explainer = shap.TreeExplainer(self.models[target_col])
        explanation = explainer(df[features])

        return explainer, explanation

    def plot_shap_summary(
        self,
        df: pd.DataFrame,
        target_col: str,
        max_display: int = 20
    ) -> None:
        """
        Plot SHAP summary plot for a specific target

        Args:
            df:          DataFrame containing feature columns
            target_col:  Which target model to explain
            max_display: Maximum number of features to display
        """

        # Validate model is fitted and has required feature columns before plotting SHAP summary
        validate_fitted(self)
        validate_columns(self, df, context="plot_shap_summary")

        # Extract per-target feature columns
        features = self.target_feature_cols.get(target_col, self.feature_cols)

        # Extract SHAP values and explanation for the specified target
        _, explanation = self.get_shap_values(df, target_col)

        # Plot SHAP summary
        shap.summary_plot(
            explanation.values,
            features=df[features],
            feature_names=features,
            max_display=max_display,
            show=False
        )

        plt.title(f"SHAP Summary — {target_col.replace('target_', '').upper()}")
        plt.tight_layout()
        plt.show()
        

    def plot_shap_waterfall(
        self,
        df: pd.DataFrame,
        target_col: str,
        row_idx: int = 0
    ) -> None:
        """
        Plot SHAP waterfall plot for a single prediction

        Args:
            df:         DataFrame containing feature columns
            target_col: Which target model to explain
            row_idx:    Index of the row to explain (0-indexed within df)
        """

        # Validate model is fitted and has required feature columns before plotting SHAP summary
        validate_fitted(self)
        validate_columns(self, df, context="plot_shap_waterfall")

        # Extract SHAP values and explaination for the specified target
        _, explanation = self.get_shap_values(df, target_col)

        # Plot SHAP waterfall for the specified row
        shap.plots.waterfall(
            explanation[row_idx],
            max_display=20,
            show=False
        )

        plt.title(f"SHAP Waterfall — {target_col.replace('target_', '').upper()} | Row {row_idx}")
        plt.tight_layout()
        plt.show()