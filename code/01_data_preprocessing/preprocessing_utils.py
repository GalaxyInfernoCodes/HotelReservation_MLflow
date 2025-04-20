"""
Preprocessing utilities for the Hotel Reservation dataset.

This module provides functions for:
1. Loading data from CSV files
2. Loading feature configuration from YAML
3. Encoding categorical features for tree-based models
4. Splitting data into training, validation, and test sets
5. Saving processed datasets to CSV files
"""

import os
from typing import Dict, List, Tuple, Any
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, data_file: str, config_file: str, output_dir: str):
        self.data_file = data_file
        self.config_file = config_file
        self.output_dir = output_dir
    
    def load_data(self) -> pd.DataFrame:
        


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the hotel reservation dataset from CSV file

    Args:
        file_path (str): Path to the CSV file

    Returns:
        pd.DataFrame: Loaded dataset
    """
    logger.info("Loading data from %s" % file_path)
    try:
        df = pd.read_csv(file_path)
        logger.info("Successfully loaded data with shape %s" % str(df.shape))
        return df
    except Exception as e:
        logger.error("Error loading data: %s" % e)
        raise


def load_feature_config(config_file: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Load feature configuration from YAML file

    Args:
        config_file (str): Path to the YAML configuration file

    Returns:
        tuple: Lists of categorical, numerical, and target feature names
    """
    logger.info("Loading feature configuration from %s" % config_file)
    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        categorical_features = config["categorical_features"]
        numerical_features = config["numerical_features"]
        target_feature = config["target_feature"]

        logger.info(
            "Identified %d categorical features, %d numerical features, and %d target feature"
            % (len(categorical_features), len(numerical_features), len(target_feature))
        )

        return categorical_features, numerical_features, target_feature
    except Exception as e:
        logger.error("Error loading feature configuration: %s" % e)
        raise


def encode_categorical_features(
    df: pd.DataFrame, categorical_features: List[str]
) -> Tuple[pd.DataFrame, Dict[str, Dict[Any, int]]]:
    """
    Encode categorical features for use with tree-based models (Random Forest)
    For tree-based models, we use label encoding (ordinal encoding) instead of one-hot encoding

    Args:
        df (pd.DataFrame): Input dataset
        categorical_features (list): List of categorical feature names

    Returns:
        pd.DataFrame: Dataset with encoded categorical features
        dict: Dictionary of label encoders for each categorical feature
    """
    logger.info("Encoding categorical features")
    df_encoded = df.copy()
    encoders: Dict[str, Dict[Any, int]] = {}

    cat_features_to_encode = [f for f in categorical_features if f != "Booking_ID"]

    for feature in cat_features_to_encode:
        unique_values = df[feature].unique()
        value_to_int = {value: i for i, value in enumerate(unique_values)}

        encoders[feature] = value_to_int
        df_encoded[feature] = df[feature].map(value_to_int)

        logger.info("Encoded %s with %d unique values" % (feature, len(unique_values)))

    return df_encoded, encoders


def prepare_features_and_target(
    df: pd.DataFrame,
    categorical_features: List[str],
    numerical_features: List[str],
    target_feature: List[str],
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Dict[Any, int]]]:
    """
    Prepare features and target for model training

    Args:
        df (pd.DataFrame): Input dataset
        categorical_features (list): List of categorical feature names
        numerical_features (list): List of numerical feature names
        target_feature (list): Target feature name

    Returns:
        tuple: X (features DataFrame), y (target Series), encoders dictionary
    """
    logger.info("Preparing features and target")

    df_encoded, encoders = encode_categorical_features(df, categorical_features)

    target_col = target_feature[0]
    y = (df_encoded[target_col] == "Not_Canceled").astype(int)

    features_to_use = [
        col for col in df_encoded.columns if col not in ["Booking_ID", target_col]
    ]
    X = df_encoded[features_to_use]

    logger.info(
        "Prepared features with shape %s and target with shape %s"
        % (str(X.shape), str(y.shape))
    )

    return X, y, encoders


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split the dataset into training, validation, and test sets

    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target vector
        val_size (float): Proportion of the dataset to include in the validation split
        test_size (float): Proportion of the dataset to include in the test split
        random_state (int): Random seed for reproducibility

    Returns:
        tuple: X_train, X_val, X_test, y_train, y_val, y_test
    """
    logger.info(
        "Splitting dataset with val_size=%.2f, test_size=%.2f" % (val_size, test_size)
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
        "Training set shape: %s, Validation set shape: %s, Test set shape: %s"
        % (str(X_train.shape), str(X_val.shape), str(X_test.shape))
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def save_splits(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    output_dir: str,
) -> None:
    """
    Save the data splits to CSV files

    Args:
        X_train (pd.DataFrame): Training features
        X_val (pd.DataFrame): Validation features
        X_test (pd.DataFrame): Test features
        y_train (pd.Series): Training target
        y_val (pd.Series): Validation target
        y_test (pd.Series): Test target
        output_dir (str): Directory to save the CSV files
    """
    logger.info("Saving data splits to %s" % output_dir)

    os.makedirs(output_dir, exist_ok=True)

    train_df = X_train.copy()
    train_df["target"] = y_train

    val_df = X_val.copy()
    val_df["target"] = y_val

    test_df = X_test.copy()
    test_df["target"] = y_test

    train_df.to_csv(os.path.join(output_dir, "train_data.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "validation_data.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_data.csv"), index=False)

    logger.info("Successfully saved training, validation, and test data as CSV files")


def process_hotel_reservation_data(
    data_file: str, config_file: str, output_dir: str
) -> Dict[str, Dict[Any, int]]:
    """
    Process hotel reservation data: load, encode features, split, and save

    Args:
        data_file (str): Path to the CSV data file
        config_file (str): Path to the YAML configuration file
        output_dir (str): Directory to save the processed data

    Returns:
        Dict[str, Dict[Any, int]]: Dictionary of encoders for categorical features
    """
    df = load_data(data_file)
    categorical_features, numerical_features, target_feature = load_feature_config(
        config_file
    )
    X, y, encoders = prepare_features_and_target(
        df, categorical_features, numerical_features, target_feature
    )
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
    save_splits(X_train, X_val, X_test, y_train, y_val, y_test, output_dir)
    return encoders
