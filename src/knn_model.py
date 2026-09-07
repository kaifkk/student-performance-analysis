import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score


def build_knn(preprocessor, n_neighbors=5):

    knn = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                KNeighborsClassifier(
                    n_neighbors=n_neighbors
                )
            )
        ]
    )

    return knn


def select_best_k(
    preprocessor,
    X_train,
    y_train,
    k_values=None,
    cv_splits=5,
    random_state=42
):
    """
    Runs stratified cross-validation on the training data for a range
    of k values and returns the mean/std macro-F1 for each k, along
    with the best-performing k.

    Macro-F1 is used (rather than accuracy) because the performance
    classes are imbalanced (Low/Medium/High are not equally sized),
    so accuracy alone could hide poor performance on minority classes.
    """

    if k_values is None:
        k_values = [3, 5, 7, 9, 11]

    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state
    )

    records = []

    for k in k_values:
        model = build_knn(preprocessor, n_neighbors=k)

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="f1_macro"
        )

        records.append(
            {
                "k": k,
                "mean_f1_macro": scores.mean(),
                "std_f1_macro": scores.std()
            }
        )

    results = pd.DataFrame(records)

    best_row = results.loc[results["mean_f1_macro"].idxmax()]
    best_k = int(best_row["k"])

    return results, best_k
