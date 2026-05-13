import pandas as pd
import numpy as np
import sys
from pathlib import Path

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent.parent))

import configurations as cfg


# ─── General Helper Function ───────────────────────────────────────────

def validate_columns(
    df: pd.DataFrame,
    required_cols: list[str],
    context: str = ""
) -> None:
    """
    Verify required columns are present in DataFrame

    Args:
        df:            DataFrame to validate
        required_cols: List of column names that must be present
        context:       String identifying the caller
    """

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        context_str = f" in {context}" if context else ""
        raise ValueError(
            f"Missing required columns{context_str}: {missing_cols}"
        )


# ─── Base Models ────────────────────────────────────────────────

class NaivePersistenceModel():
    """
    Baseline 1: Naive persistence model.
    Predicts next season's stats as equal to the current season's stats.
    """

    def __init__(
        self,
        target_cols: list[str],
        source_col_map: dict[str, str],
    ):
        
        self.target_cols = target_cols
        self.source_col_map = source_col_map


    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict next season's stats as equal to current season's stats

        Args:
            df: DataFrame containing source columns

        Returns:
            df_predictions: DataFrame containing predictions for each target column
        """

        # Validate required columns are present
        validate_columns(df, list(self.source_col_map.values()), "NaivePersistenceModel.predict")

        df_predictions = pd.DataFrame(index=df.index)

        for target_col in self.target_cols:
            source_col = self.source_col_map[target_col]
            df_predictions[target_col] = df[source_col]

        return df_predictions


class RidgeRegressionModel():
    """
    Baseline 2: Ridge regression on rolling averages.
    Trains a separate Ridge regression model per target column
    """

    def __init__(
        self,
        target_cols: list[str],
        feature_cols: list[str],
        alpha: float = 1.0, # Ridge regularization strength
    ):
        
        self.target_cols = target_cols
        self.feature_cols = feature_cols
        self.alpha = alpha
        self.models = {}  # Dictionary to store fitted Ridge models
        self.scalers = {} # Dictionary to store fitted scalers for each target column
        self.is_fitted = False


    def fit(self, df: pd.DataFrame):
        """
        Fit one Ridge regression model per target column

        Args:
            df: Training DataFrame containing features and targets
        """

        # Validate required columns are present
        validate_columns(df, self.feature_cols + self.target_cols, context="RidgeRegressionModel.fit")

        for target_col in self.target_cols:
            # Drop rows where target is null
            df_target = df[df[target_col].notna()].copy()

            x = df_target[self.feature_cols].values
            y = df_target[target_col].values

            # Scale features
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(x)

            model = Ridge(alpha=self.alpha)
            model.fit(x_scaled, y)

            # Store fitted model and scaler
            self.models[target_col] = model
            self.scalers[target_col] = scaler

        self.is_fitted = True


    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for each target using its fitted Ridge model.
        Handles missing feature values via imputation before predicting

        Args:
            df: DataFrame containing feature columns

        Returns:
            df_predictions: DataFrame containing predictions for each target column
        """

        # Ensure model has been fitted before predicting
        if not self.is_fitted:
            raise ValueError("RidgeRegressionModel must be fitted before predicting!")
        
        # Validate required feature columns are present
        validate_columns(df, self.feature_cols, context="RidgeRegressionModel.predict")

        df_predictions = pd.DataFrame(index=df.index)
        
        for target_col in self.target_cols:
            x = df[self.feature_cols].values
            
            # Verify that features have been through imputation
            if np.isnan(x).any():
                raise ValueError(
                    f"NaN found in target column: {target_col}. Ensure imputation is performed before predicting"
                )

            x_scaled = self.scalers[target_col].transform(x)  # Scale features using fitted scaler

            df_predictions[target_col] = self.models[target_col].predict(x_scaled)

            # Clip g to 17
            if target_col == "g":
                df_predictions[target_col] = df_predictions[target_col].clip(upper=17)

        return df_predictions


    def get_coefficients(self) -> pd.DataFrame:
        """
        Return feature coefficients for each fitted model

        Returns:
            df_coeffs: DataFrame with features as rows, targets as columns, and values as Ridge coefficients
        """

        if not self.is_fitted:
            raise ValueError("RidgeRegressionModel must be fitted before getting coefficients!")
        
        df_coeffs = pd.DataFrame(
            {
                target_col: model.coef_ 
                for target_col, model in self.models.items()
            },
            index=self.feature_cols
        )

        # Sort by absolute magnitude to see most important features for each target
        first_target = self.target_cols[0]
        return df_coeffs.reindex(df_coeffs[first_target].abs().sort_values(ascending=False).index)