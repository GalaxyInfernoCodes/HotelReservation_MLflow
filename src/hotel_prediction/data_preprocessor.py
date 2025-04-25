"""Data preprocessing utilities using pydantic‑backed configuration.

This module handles:
1. Encoding categorical variables for tree‑based models (ordinal/label encoding).
2. Preparing feature matrix **X** and target vector **y**.
3. Splitting the data into train/validation/test sets using configurable ratios.
4. Persisting splits to CSV files.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.hotel_prediction.config import FullConfig

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Encapsulates feature engineering and dataset splitting logic."""

    def __init__(self, config: FullConfig):
        self.config = config
        self.categorical_features: List[str] = self.config.features.categorical_features
        self.numerical_features: List[str] = self.config.features.numerical_features
        self.target_feature: str = self.config.features.target_feature[0]

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def _encode_categorical_features(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[Any, int]]]:
        """Ordinal‑encode categorical columns (except Booking_ID)."""
        logger.info("Encoding categorical features ...")
        df_encoded = df.copy()
        encoders: Dict[str, Dict[Any, int]] = {}

        features_to_encode = [f for f in self.categorical_features if f != "Booking_ID"]
        for feature in features_to_encode:
            unique_values = df[feature].unique()
            mapping = {val: idx for idx, val in enumerate(unique_values)}
            encoders[feature] = mapping
            df_encoded[feature] = df[feature].map(mapping)
            logger.debug(
                "Encoded %s with %d unique values", feature, len(unique_values)
            )

        return df_encoded, encoders

    # ------------------------------------------------------------------
    # Feature / target preparation
    # ------------------------------------------------------------------

    def _prepare_features_and_target(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Dict[Any, int]]]:
        df_encoded, encoders = self._encode_categorical_features(df)

        # Binary‑encode booking_status (Not_Canceled -> 1, else 0)
        y = (df_encoded[self.target_feature] == "Not_Canceled").astype(int)

        features_to_use = [
            col
            for col in df_encoded.columns
            if col not in ["Booking_ID", self.target_feature]
        ]
        X = df_encoded[features_to_use]
        logger.info("Feature matrix shape: %s, target shape: %s", X.shape, y.shape)

        return X, y, encoders

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, df: pd.DataFrame, save: bool = True) -> Dict[str, Dict[Any, int]]:
        """Full preprocessing pipeline returning encoders."""
        X, y, encoders = self._prepare_features_and_target(df)
        X_train, X_val, X_test, y_train, y_val, y_test = self._split_dataset(X, y)
        if save:
            self._save_splits(X_train, X_val, X_test, y_train, y_val, y_test)
        return encoders
