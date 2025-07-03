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
    return (
        ConfusionMatrixDisplay,
        RandomForestClassifier,
        classification_report,
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
def _(df, pd):
    # Drop ID column
    usable_df = df.drop(columns=['Booking_ID'])

    # Drop date features for now
    usable_df = usable_df.drop(columns=['arrival_year', 'arrival_month', 'arrival_date'])
    # drop parking space because binary, drop previous cancellations
    usable_df = usable_df.drop(columns=['required_car_parking_space', 'no_of_previous_cancellations'])

    # Drop rows with missing values (Phase 1 shortcut)
    usable_df = usable_df.dropna()

    # Encode the target
    usable_df['booking_status'] = usable_df['booking_status'].map({'Canceled': 1, 'Not_Canceled': 0})

    # Optional: print to double-check
    print(usable_df['booking_status'].value_counts())

    # One-hot encode low-cardinality categoricals
    categorical_cols = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
    usable_df = pd.get_dummies(usable_df, columns=categorical_cols)

    # Define X and y
    X = usable_df.drop(columns='booking_status')
    y = usable_df['booking_status']

    return X, y


@app.cell
def _(X):
    X
    return


@app.cell
def _(X, train_test_split, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    return X_test, X_train, y_test, y_train


@app.cell
def _(RandomForestClassifier, X_train, y_train):
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    return (clf,)


@app.cell
def _(mo):
    mo.md(r"""In the next phases, we’ll want to track these metrics and results automatically using tools like MLflow.""")
    return


@app.cell
def _(ConfusionMatrixDisplay, X_test, classification_report, clf, plt, y_test):
    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred))
    ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
