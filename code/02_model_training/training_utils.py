"""
Training utilities for the Hotel Reservation dataset.

This module provides functions for:
1. Loading training and validation data
2. Training models with cross-validation
3. Tracking experiments with MLflow
4. Evaluating model performance
"""

import os
import logging
import pandas as pd
from typing import Dict, Tuple, List, Any, Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import mlflow
import mlflow.sklearn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_train_val_data(
    data_dir: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load training and validation data from the processed data directory

    Args:
        data_dir (str): Directory containing processed data files

    Returns:
        Tuple: X_train, y_train, X_val, y_val
    """
    logger.info(f"Loading training and validation data from {data_dir}")

    train_data = pd.read_csv(os.path.join(data_dir, "train_data.csv"))
    y_train = train_data["target"]
    X_train = train_data.drop("target", axis=1)

    val_data = pd.read_csv(os.path.join(data_dir, "validation_data.csv"))
    y_val = val_data["target"]
    X_val = val_data.drop("target", axis=1)

    logger.info(
        f"Loaded training data with shape {X_train.shape} and validation data with shape {X_val.shape}"
    )

    return X_train, y_train, X_val, y_val


def train_random_forest_with_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Dict[str, List[Any]],
    cv: int = 5,
    experiment_name: str = "hotel_reservation_model",
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """
    Train a Random Forest classifier with cross-validation grid search and MLflow tracking

    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        param_grid (Dict[str, List[Any]]): Parameter grid for grid search
        cv (int): Number of cross-validation folds
        experiment_name (str): MLflow experiment name

    Returns:
        Tuple: Best model, best parameters
    """
    logger.info("Training Random Forest with cross-validation grid search")

    # Note: MLflow experiment should be set by the caller

    rf = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=cv,
        scoring="f1",
        return_train_score=True,
        verbose=1,
        n_jobs=-1,
    )

    logger.info("Starting grid search with cross-validation")
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best CV score: {grid_search.best_score_:.4f}")

    # Log CV results to MLflow if a run is active
    try:
        for param, value in best_params.items():
            mlflow.log_param(f"cv_{param}", value)
        mlflow.log_metric("cv_f1_score", grid_search.best_score_)

        # Log CV results DataFrame
        cv_results = pd.DataFrame(grid_search.cv_results_)
        cv_results.to_csv("cv_results.csv", index=False)
        mlflow.log_artifact("cv_results.csv")
    except Exception as e:
        logger.warning(f"Could not log to MLflow: {e}")

    return best_model, best_params


def evaluate_model(
    model: RandomForestClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    run_id: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluate model on validation data and log results to MLflow

    Args:
        model (RandomForestClassifier): Trained model
        X_val (pd.DataFrame): Validation features
        y_val (pd.Series): Validation target
        run_id (Optional[str]): MLflow run ID to log metrics to

    Returns:
        Dict[str, float]: Dictionary of evaluation metrics
    """
    logger.info("Evaluating model on validation data")

    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "f1": f1_score(y_val, y_pred),
        "roc_auc": roc_auc_score(y_val, y_pred_proba),
    }

    logger.info(f"Validation metrics: {metrics}")

    # We don't need to create a new run here - metrics will be logged in the main script
    # Instead, log the run_id for reference if provided
    if run_id:
        logger.info(f"Run ID for metrics: {run_id}")

    cm = confusion_matrix(y_val, y_pred)
    logger.info(f"Confusion matrix:\n{cm}")

    cr = classification_report(y_val, y_pred)
    logger.info(f"Classification report:\n{cr}")

    return metrics


def log_model_to_mlflow(
    model: RandomForestClassifier,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    feature_names: List[str],
    experiment_name: str = "hotel_reservation_model",
    run_name: str = "random_forest_cv",
) -> str:
    """
    Log model, parameters, and metrics to MLflow

    Args:
        model (RandomForestClassifier): Trained model
        params (Dict[str, Any]): Model parameters
        metrics (Dict[str, float]): Evaluation metrics
        feature_names (List[str]): List of feature names
        experiment_name (str): MLflow experiment name
        run_name (str): MLflow run name

    Returns:
        str: MLflow run ID
    """
    logger.info(f"Logging model to MLflow experiment '{experiment_name}'")

    # Check if there's an active run
    active_run = mlflow.active_run()

    if active_run:
        logger.info(f"Using active run with ID: {active_run.info.run_id}")
        run_id = active_run.info.run_id
        # Log within the current run context
        mlflow.log_params(params)

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="hotel_reservation_rf_model",
            input_example=pd.DataFrame(0, index=[0], columns=feature_names),
            signature=mlflow.models.infer_signature(
                pd.DataFrame(0, index=[0], columns=feature_names),
                model.predict(pd.DataFrame(0, index=[0], columns=feature_names)),
            ),
        )

        feature_importance = pd.DataFrame(
            {"feature": feature_names, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)

        feature_importance.to_csv("feature_importance.csv", index=False)
        mlflow.log_artifact("feature_importance.csv")

        return run_id
    else:
        # Start a new run if none is active
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(params)

            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name="hotel_reservation_rf_model",
                input_example=pd.DataFrame(0, index=[0], columns=feature_names),
                signature=mlflow.models.infer_signature(
                    pd.DataFrame(0, index=[0], columns=feature_names),
                    model.predict(pd.DataFrame(0, index=[0], columns=feature_names)),
                ),
            )

            feature_importance = pd.DataFrame(
                {"feature": feature_names, "importance": model.feature_importances_}
            ).sort_values("importance", ascending=False)

            feature_importance.to_csv("feature_importance.csv", index=False)
            mlflow.log_artifact("feature_importance.csv")

            logger.info(f"Model logged to MLflow with run ID: {run.info.run_id}")

            return run.info.run_id
