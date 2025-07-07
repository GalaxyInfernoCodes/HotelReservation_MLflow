import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, ConfusionMatrixDisplay
    from sklearn.model_selection import GridSearchCV
    from scipy.stats import randint
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    import numpy as np
    return (
        ColumnTransformer,
        ConfusionMatrixDisplay,
        GridSearchCV,
        OneHotEncoder,
        Pipeline,
        RandomForestClassifier,
        SimpleImputer,
        StandardScaler,
        classification_report,
        np,
        pd,
        plt,
        train_test_split,
    )


@app.cell
def _(mo):
    mo.md(r"""We're loading the data directly from disk inside the notebook, which is fine for now. Later, we’ll move this logic into reusable scripts.""")
    return


@app.cell
def _(pd):
    df = pd.read_csv("./data/Hotel_Reservations.csv")
    df.head()
    return (df,)


@app.cell
def _(mo):
    mo.md(r"""For quick prototypes, use visual inspection. In production, you’d automate profiling (e.g., with pandas-profiling or ydata-profiling)""")
    return


@app.cell
def _(df, plt):
    df.info()
    df.describe()
    df['booking_status'].value_counts().plot(kind='bar', title='Target Distribution')
    plt.show()
    return


@app.cell
def _(
    ColumnTransformer,
    OneHotEncoder,
    Pipeline,
    SimpleImputer,
    StandardScaler,
    df,
    pd,
):
    # --- 1. Feature Engineering ---
    # Convert to datetime and extract features
    df['arrival_date_full'] = pd.to_datetime(
        df['arrival_year'].astype(str)
        + '-'
        + df['arrival_month'].astype(str)
        + '-'
        + df['arrival_date'].astype(str),
        errors='coerce',
    )
    df['day_of_week'] = df['arrival_date_full'].dt.dayofweek
    df['is_weekend'] = (df['arrival_date_full'].dt.dayofweek >= 5).astype(int)
    df['month'] = df['arrival_date_full'].dt.month

    # Drop original date columns and Booking_ID
    df_processed = df.drop(
        columns=[
            'Booking_ID',
            'arrival_year',
            'arrival_month',
            'arrival_date',
            'arrival_date_full',
        ]
    )

    # Encode the target variable
    df_processed['booking_status'] = df_processed['booking_status'].map(
        {'Canceled': 1, 'Not_Canceled': 0}
    )

    # --- 2. Define Features and Target ---
    X = df_processed.drop('booking_status', axis=1)
    y = df_processed['booking_status']

    # Define categorical and numerical features
    categorical_features = [
        'type_of_meal_plan',
        'room_type_reserved',
        'market_segment_type',
    ]
    numerical_features = [
        col for col in X.columns if col not in categorical_features
    ]

    # --- 3. Create Preprocessing Pipelines ---
    numeric_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore')),
        ]
    )

    # --- 4. Combine Preprocessing Steps with ColumnTransformer ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features),
        ]
    )

    return X, categorical_features, numerical_features, preprocessor, y


@app.cell
def _(X, train_test_split, y):
    # --- 5. Split Data ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    return X_test, X_train, y_test, y_train


@app.cell
def _(Pipeline, RandomForestClassifier, preprocessor):
    # --- 6. Create the Full Model Pipeline ---
    model_pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            (
                'classifier',
                RandomForestClassifier(class_weight='balanced', random_state=42),
            ),
        ]
    )
    return (model_pipeline,)


@app.cell
def _():
    # --- 7. Define Hyperparameter Search Space ---
    # Note the 'classifier__' prefix for pipeline parameters
    param_dist = {
        'classifier__n_estimators': [100, 300, 500],
        'classifier__max_depth': [None, 10, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 5],
        'classifier__max_features': ['sqrt', 'log2', None],
    }
    return (param_dist,)


@app.cell
def _(GridSearchCV, X_train, model_pipeline, param_dist, y_train):
    # --- 8. Setup and Run Hyperparameter Search ---
    search = GridSearchCV(
        model_pipeline,
        param_grid=param_dist,
        cv=3,
        verbose=1,
        n_jobs=-1,
        scoring='f1_macro',
    )
    search.fit(X_train, y_train)
    return (search,)


@app.cell
def _(search):
    print(search.best_params_)
    return


@app.cell
def _(mo):
    mo.md(r"""In the next phases, we’ll want to track these metrics and results automatically using tools like MLflow.""")
    return


@app.cell
def _(
    ConfusionMatrixDisplay,
    X_test,
    classification_report,
    plt,
    search,
    y_test,
):
    # --- 9. Evaluate the Best Model ---
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)

    print('--- Classification Report ---')
    print(classification_report(y_test, y_pred))

    print('\n--- Confusion Matrix ---')
    ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
    plt.show()
    return (best_model,)


@app.cell
def _(
    best_model,
    categorical_features,
    np,
    numerical_features,
    pd,
    plt,
    preprocessor,
):
    # --- 10. Feature Importance ---

    # Extract classifier from the pipeline
    classifier = best_model.named_steps['classifier']

    # Get feature names after one-hot encoding
    ohe_feature_names = preprocessor.named_transformers_['cat'] \
        .named_steps['onehot'] \
        .get_feature_names_out(categorical_features)

    # Combine with numerical feature names
    all_feature_names = np.concatenate([numerical_features, ohe_feature_names])

    # Create a series for easy plotting
    importances = pd.Series(
        classifier.feature_importances_, index=all_feature_names
    )

    # Plot
    plt.figure(figsize=(12, 8))
    importances.sort_values().plot(kind='barh', title='Feature Importances')
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
