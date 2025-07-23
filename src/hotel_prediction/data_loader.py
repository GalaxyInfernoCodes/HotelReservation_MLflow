from hotel_prediction.config import FullConfig

import duckdb
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, config: FullConfig):
        """
        Initialize a DataLoader instance.

        Args:
            config (FullConfig): Configuration object containing data loading parameters, including the path to the dataset.
        """
        self.config = config
        # Ensure the data path is absolute, relative to the project root
        data_path = Path(self.config.project.data_source_path)
        if not data_path.is_absolute():
            # Assume project root is two levels up from this file (src/hotel_prediction/)
            project_root = Path(__file__).resolve().parent.parent.parent
            data_path = project_root / data_path
        logger.info("Loading data from %s", data_path)
        self.hotel_dataframe = pd.read_csv(data_path)
        logger.info("Data loaded successfully")
        self.data_con = duckdb.connect(self.config.project.duckdb_data_path)

    def split_dataset(
        self, full_df: pd.DataFrame
    ) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series
    ]:
        """
        Split the dataset into training, validation, and test sets.

        Args:
            full_df (pd.DataFrame): Full dataframe containing features and target.

        Returns:
            Tuple:
                - X_train (pd.DataFrame): Training features
                - X_val (pd.DataFrame): Validation features
                - X_test (pd.DataFrame): Test features
                - y_train (pd.Series): Training targets
                - y_val (pd.Series): Validation targets
                - y_test (pd.Series): Test targets
        """
        target_column_name = self.config.features.target_feature[0]
        X = full_df.drop(target_column_name, axis=1)
        y = full_df[target_column_name]

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

    def save_splits(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        y_test: pd.Series,
    ) -> None:
        """
        Save the train, validation, and test splits to CSV files in the output directory.

        Args:
            X_train (pd.DataFrame): Training features.
            X_val (pd.DataFrame): Validation features.
            X_test (pd.DataFrame): Test features.
            y_train (pd.Series): Training targets.
            y_val (pd.Series): Validation targets.
            y_test (pd.Series): Test targets.
        """
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

