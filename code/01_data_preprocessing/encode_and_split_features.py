#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Encode and Split Features Module for Hotel Reservation Dataset

This script handles:
1. Loading the hotel reservation data
2. Identifying categorical and numerical features from config
3. Encoding categorical features for tree algorithms
4. Splitting data into training, validation, and test sets
5. Saving the splits to CSV files
"""

# %%
import os
import logging
import pickle

from preprocessing_utils import (
    load_data,
    load_feature_config,
    prepare_features_and_target,
    split_dataset,
    save_splits,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# %%

# Define file and directory paths
DATA_DIR = "../../data"
DATA_FILE = os.path.join(DATA_DIR, "Hotel_Reservations.csv")
OUTPUT_DIR = "../../output/processed_data"
CONFIG_DIR = "../../config"
FEATURE_CONFIG_FILE = os.path.join(CONFIG_DIR, "feature_config.yaml")


os.makedirs(OUTPUT_DIR, exist_ok=True)

# %%
logger.info("Step 1: Loading data")
df = load_data(DATA_FILE)

# %%
logger.info("Step 2: Loading feature configuration")
categorical_features, numerical_features, target_feature = load_feature_config(
    FEATURE_CONFIG_FILE
)

# %%
logger.info("Step 3: Encoding features")
X, y, encoders = prepare_features_and_target(
    df, categorical_features, numerical_features, target_feature
)

# %%
logger.info("Step 4: Splitting data")
X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
    X, y, val_size=0.15, test_size=0.15
)

# %%
logger.info("Step 5: Saving splits")
save_splits(X_train, X_val, X_test, y_train, y_val, y_test, OUTPUT_DIR)

# %%
logger.info("Step 6: Saving encoders")

with open(os.path.join(OUTPUT_DIR, "feature_encoders.pkl"), "wb") as f:
    pickle.dump(encoders, f)

logger.info(" Data encoding and splitting completed successfully")
logger.info(f"Training set: {X_train.shape[0]} samples")
logger.info(f"Validation set: {X_val.shape[0]} samples")
logger.info(f"Test set: {X_test.shape[0]} samples")
logger.info(f"All files saved to: {OUTPUT_DIR}")
