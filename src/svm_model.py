import pandas as pd

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV


def build_svm(preprocessor, C=1.0, kernel="rbf", gamma="scale"):

    svm = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                SVC(
                    C=C,
                    kernel=kernel,
                    gamma=gamma,
                    random_state=42
                )
            )
        ]
    )

    return svm


def tune_svm(
    preprocessor,
    X_train,
    y_train,
    cv_splits=5,
    random_state=42
):
    """
    Runs a grid search with stratified cross-validation on the training
    data to select SVM hyperparameters (kernel, C, gamma).

    Scoring is macro-F1, for the same reason as in select_best_k:
    the performance classes are imbalanced, so we don't want the
    search to favor a model that only predicts the majority class well.
    """

    pipeline = build_svm(preprocessor)

    param_grid = [
        {
            "classifier__kernel": ["rbf"],
            "classifier__C": [0.1, 1, 10, 100],
            "classifier__gamma": ["scale", 0.01, 0.1, 1]
        },
        {
            "classifier__kernel": ["linear"],
            "classifier__C": [0.1, 1, 10, 100]
        }
    ]

    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    cv_results = pd.DataFrame(grid_search.cv_results_)

    return grid_search, cv_results
