"""Data preprocessing utilities using pydantic‑backed configuration.

This module handles:
1. Encoding categorical variables for tree‑based models (ordinal/label encoding).
2. Preparing feature matrix **X** and target vector **y**.
3. Splitting the data into train/validation/test sets using configurable ratios.
4. Persisting splits to CSV files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

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
    # Splits and persistence
    # ------------------------------------------------------------------

    def _split_dataset(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series
    ]:
        val_size = self.config.project.val_size
        test_size = self.config.project.test_size
        random_state = self.config.project.random_seed

        logger.info(
            "Splitting dataset (val_size=%.2f, test_size=%.2f)", val_size, test_size
        )

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        adjusted_val_size = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=adjusted_val_size,
            random_state=random_state,
            stratify=y_temp,
        )
        logger.info(
            "Train: %s, Val: %s, Test: %s",
            X_train.shape,
            X_val.shape,
            X_test.shape,
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def _save_splits(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        y_test: pd.Series,
    ) -> None:
        output_dir = Path(self.config.project.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_df = X_train.copy()
        train_df["target"] = y_train

        val_df = X_val.copy()
        val_df["target"] = y_val

        test_df = X_test.copy()
        test_df["target"] = y_test

        train_df.to_csv(output_dir / "train_data.csv", index=False)
        val_df.to_csv(output_dir / "validation_data.csv", index=False)
        test_df.to_csv(output_dir / "test_data.csv", index=False)
        logger.info("Saved splits to %s", output_dir)

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
