#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data Preprocessing Module for Hotel Reservation Dataset

This script handles the preprocessing of the hotel reservation dataset,
including feature identification, encoding, and preparation for model training.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DATA_FILE = os.path.join(DATA_DIR, "Hotel_Reservations.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(file_path):
    """
    Load the hotel reservation dataset from CSV file
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    logger.info(f"Loading data from {file_path}")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded data with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def identify_features(df):
    """
    Identify categorical, numerical, and target features
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        tuple: Lists of categorical, numerical, and target feature names
    """
    logger.info("Identifying feature types")
    
    target_feature = ['booking_status']
    
    categorical_features = [
        'Booking_ID',  # This is an ID column and should be dropped later
        'type_of_meal_plan',
        'required_car_parking_space',  # Binary but treated as categorical
        'room_type_reserved',
        'market_segment_type',
        'repeated_guest',  # Binary but treated as categorical
        'arrival_month'  # Month can be treated as categorical
    ]
    
    numerical_features = [
        'no_of_adults',
        'no_of_children',
        'no_of_weekend_nights',
        'no_of_week_nights',
        'lead_time',
        'arrival_year',
        'arrival_date',
        'no_of_previous_cancellations',
        'no_of_previous_bookings_not_canceled',
        'avg_price_per_room',
        'no_of_special_requests'
    ]
    
    logger.info(f"Identified {len(categorical_features)} categorical features, "
                f"{len(numerical_features)} numerical features, and "
                f"{len(target_feature)} target feature")
    
    return categorical_features, numerical_features, target_feature

def exploratory_data_analysis(df, categorical_features, numerical_features, target_feature):
    """
    Perform exploratory data analysis on the dataset
    
    Args:
        df (pd.DataFrame): Input dataset
        categorical_features (list): List of categorical feature names
        numerical_features (list): List of numerical feature names
        target_feature (list): List containing the target feature name
    """
    logger.info("Performing exploratory data analysis")
    
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Dataset columns: {df.columns.tolist()}")
    
    missing_values = df.isnull().sum()
    logger.info(f"Missing values per column:\n{missing_values}")
    
    logger.info("Basic statistics for numerical features:")
    logger.info(df[numerical_features].describe())
    
    for feature in categorical_features:
        if feature != 'Booking_ID':  # Skip ID column
            logger.info(f"Distribution of {feature}:")
            logger.info(df[feature].value_counts())
    
    logger.info(f"Distribution of target variable ({target_feature[0]}):")
    logger.info(df[target_feature[0]].value_counts())
    
    plt.figure(figsize=(10, 6))
    sns.countplot(x=target_feature[0], data=df)
    plt.title('Distribution of Booking Status')
    plt.savefig(os.path.join(OUTPUT_DIR, 'booking_status_distribution.png'))
    
    plt.figure(figsize=(12, 10))
    correlation_matrix = df[numerical_features].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_matrix.png'))
    
    logger.info("EDA completed and visualizations saved")

def preprocess_data(df, categorical_features, numerical_features, target_feature):
    """
    Preprocess the dataset for model training
    
    Args:
        df (pd.DataFrame): Input dataset
        categorical_features (list): List of categorical feature names
        numerical_features (list): List of numerical feature names
        target_feature (list): List containing the target feature name
        
    Returns:
        tuple: Preprocessed features and target, column transformer, feature names
    """
    logger.info("Preprocessing data")
    
    features_to_use = [col for col in df.columns if col not in ['Booking_ID']]
    
    X = df[features_to_use]
    y = df[target_feature[0]]
    
    # Convert target to binary (0 for Canceled, 1 for Not_Canceled)
    y = (y == 'Not_Canceled').astype(int)
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, [col for col in X.columns if col in categorical_features]),
            ('num', numerical_transformer, [col for col in X.columns if col in numerical_features])
        ])
    
    cat_cols = [col for col in X.columns if col in categorical_features]
    num_cols = [col for col in X.columns if col in numerical_features]
    
    logger.info("Data preprocessing completed")
    
    return X, y, preprocessor, cat_cols, num_cols

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target vector
        test_size (float): Proportion of the dataset to include in the test split
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    logger.info(f"Splitting data with test_size={test_size}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def main():
    """Main function to execute the data preprocessing pipeline"""
    with mlflow.start_run(run_name="data_preprocessing"):
        df = load_data(DATA_FILE)
        
        mlflow.log_param("dataset_shape", str(df.shape))
        mlflow.log_param("dataset_columns", str(df.columns.tolist()))
        
        categorical_features, numerical_features, target_feature = identify_features(df)
        
        mlflow.log_param("categorical_features", str(categorical_features))
        mlflow.log_param("numerical_features", str(numerical_features))
        mlflow.log_param("target_feature", str(target_feature))
        
        exploratory_data_analysis(df, categorical_features, numerical_features, target_feature)
        
        mlflow.log_artifact(os.path.join(OUTPUT_DIR, 'booking_status_distribution.png'))
        mlflow.log_artifact(os.path.join(OUTPUT_DIR, 'correlation_matrix.png'))
        
        X, y, preprocessor, cat_cols, num_cols = preprocess_data(
            df, categorical_features, numerical_features, target_feature
        )
        
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        mlflow.log_param("train_class_distribution", str(y_train.value_counts().to_dict()))
        mlflow.log_param("test_class_distribution", str(y_test.value_counts().to_dict()))
        
        logger.info("Data preprocessing pipeline completed successfully")
        
        np.save(os.path.join(OUTPUT_DIR, 'X_train.npy'), X_train)
        np.save(os.path.join(OUTPUT_DIR, 'X_test.npy'), X_test)
        np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'), y_train)
        np.save(os.path.join(OUTPUT_DIR, 'y_test.npy'), y_test)
        
        import joblib
        joblib.dump(preprocessor, os.path.join(OUTPUT_DIR, 'preprocessor.pkl'))
        
        logger.info("Preprocessed data and transformer saved to output directory")

if __name__ == "__main__":
    main()