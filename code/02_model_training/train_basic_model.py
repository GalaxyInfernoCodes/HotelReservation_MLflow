"""
Basic Random Forest model training for the Hotel Reservation dataset.

This script:
1. Loads preprocessed training and validation data
2. Trains a Random Forest model with cross-validation
3. Tracks experiments with MLflow
4. Evaluates the best model on validation data
5. Saves and logs the model to MLflow
"""

# %%
import logging
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from training_utils import (
    load_train_val_data,
    evaluate_model,
    log_model_to_mlflow,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# %%
DATA_DIR = "../../output/processed_data"
EXPERIMENT_NAME = "hotel_reservation_model"
RUN_NAME = "random_forest_basic_cv"

# %%
logger.info("Step 1: Loading training and validation data")
X_train, y_train, X_val, y_val = load_train_val_data(DATA_DIR)

logger.info(f"Training data shape: {X_train.shape}")
logger.info(f"Validation data shape: {X_val.shape}")

train_target_dist = pd.DataFrame(y_train.value_counts(normalize=True)).reset_index()
train_target_dist.columns = ["target", "proportion"]
logger.info(f"Training target distribution: \n{train_target_dist}")

val_target_dist = pd.DataFrame(y_val.value_counts(normalize=True)).reset_index()
val_target_dist.columns = ["target", "proportion"]
logger.info(f"Validation target distribution: \n{val_target_dist}")

# %%
logger.info("Step 2: Defining parameter grid for Random Forest")
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}

# %%
logger.info("Step 3: Training Random Forest with cross-validation")
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name=RUN_NAME) as run:
    # Perform grid search directly in the main script
    rf = RandomForestClassifier(random_state=42)
    
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
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
    
    # Log CV results to MLflow
    for param, value in best_params.items():
        mlflow.log_param(f"cv_{param}", value)
    mlflow.log_metric("cv_f1_score", grid_search.best_score_)
    
    # Log CV results DataFrame
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results.to_csv("cv_results.csv", index=False)
    mlflow.log_artifact("cv_results.csv")

    # Log all parameters to MLflow
    mlflow.log_params(best_params)

    logger.info("Step 4: Evaluating model on validation data")
    metrics = evaluate_model(
        model=best_model, X_val=X_val, y_val=y_val, run_id=run.info.run_id
    )
    
    # Log validation metrics
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(f"val_{metric_name}", metric_value)

    logger.info("Step 5: Logging model to MLflow")
    # Use the log_model_to_mlflow function but rely on the active run
    log_model_to_mlflow(
        model=best_model,
        params=best_params,
        metrics=metrics,
        feature_names=X_train.columns.tolist(),
        experiment_name=EXPERIMENT_NAME,
        run_name=RUN_NAME,
    )

    logger.info(f"MLflow run ID: {run.info.run_id}")
    logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

    # %%
    logger.info("Step 6: Analyzing feature importance")
    feature_importance = pd.DataFrame(
        {"feature": X_train.columns, "importance": best_model.feature_importances_}
    ).sort_values("importance", ascending=False)

    logger.info(f"Top 10 important features:\n{feature_importance.head(10)}")

# %%
logger.info("Model training and evaluation complete")
logger.info(f"Best parameters: {best_params}")
logger.info(f"Validation metrics: {metrics}")
