import logging
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("Loading data...")
    df = pd.read_csv("../data/Hotel_Reservations.csv")
    logger.info(f"Data shape: {df.shape}")

    logger.info("Data info:")
    logger.info(df.info())
    logger.info("Data description:")
    logger.info(df.describe())
    logger.info("Target distribution:")
    logger.info(df["booking_status"].value_counts())

    # --- 1. Feature Engineering ---
    logger.info("Feature engineering...")
    df["arrival_date_full"] = pd.to_datetime(
        df["arrival_year"].astype(str)
        + "-"
        + df["arrival_month"].astype(str)
        + "-"
        + df["arrival_date"].astype(str),
        errors="coerce",
    )
    df["day_of_week"] = df["arrival_date_full"].dt.dayofweek
    df["is_weekend"] = (df["arrival_date_full"].dt.dayofweek >= 5).astype(int)
    df["month"] = df["arrival_date_full"].dt.month

    # Drop original date columns and Booking_ID
    df_processed = df.drop(
        columns=[
            "Booking_ID",
            "arrival_year",
            "arrival_month",
            "arrival_date",
            "arrival_date_full",
        ]
    )

    # Encode the target variable
    df_processed["booking_status"] = df_processed["booking_status"].map(
        {"Canceled": 1, "Not_Canceled": 0}
    )

    # --- 2. Define Features and Target ---
    X = df_processed.drop("booking_status", axis=1)
    y = df_processed["booking_status"]

    categorical_features = [
        "type_of_meal_plan",
        "room_type_reserved",
        "market_segment_type",
    ]
    numerical_features = [col for col in X.columns if col not in categorical_features]

    # --- 3. Create Preprocessing Pipelines ---
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # --- 4. Combine Preprocessing Steps with ColumnTransformer ---
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # --- 5. Split Data ---
    logger.info("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # --- 6. Create the Full Model Pipeline ---
    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(class_weight="balanced", random_state=42),
            ),
        ]
    )

    # --- 7. Define Hyperparameter Search Space ---
    param_dist = {
        "classifier__n_estimators": [100, 300, 500],
        "classifier__max_depth": [None, 10, 30],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 5],
        "classifier__max_features": ["sqrt", "log2", None],
    }

    # --- 8. Setup and Run Hyperparameter Search ---
    logger.info("Starting hyperparameter search...")
    search = GridSearchCV(
        model_pipeline,
        param_grid=param_dist,
        cv=3,
        verbose=1,
        n_jobs=-1,
        scoring="f1_macro",
    )
    search.fit(X_train, y_train)
    logger.info(f"Best parameters: {search.best_params_}")

    # --- 9. Evaluate the Best Model ---
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)

    logger.info("--- Classification Report ---")
    logger.info("\n" + classification_report(y_test, y_pred))

    logger.info("--- Confusion Matrix ---")
    ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
    plt.show()

    # --- 10. Feature Importance ---
    logger.info("Plotting feature importances...")
    classifier = best_model.named_steps["classifier"]
    ohe_feature_names = (
        preprocessor.named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out(categorical_features)
    )
    all_feature_names = np.concatenate([numerical_features, ohe_feature_names])
    importances = pd.Series(classifier.feature_importances_, index=all_feature_names)

    plt.figure(figsize=(12, 8))
    importances.sort_values().plot(kind="barh", title="Feature Importances")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
