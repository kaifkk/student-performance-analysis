from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent


EXCLUDED_COLUMNS = [
    "G3"
]


NUMERICAL_FEATURES = [
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
    "absences",
    "G1",
    "G2"
]


CATEGORICAL_FEATURES = [
    "school",
    "sex",
    "address",
    "famsize",
    "Pstatus",
    "Mjob",
    "Fjob",
    "reason",
    "guardian",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "romantic"
]


def create_target(df):

    df = df.copy()

    df["performance_class"] = pd.cut(
        df["G3"],
        bins=[-1, 9, 14, 20],
        labels=["Low", "Medium", "High"]
    )

    return df


def create_features_and_target(df):

    X = df[
        NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    ].copy()

    y = df["performance_class"].copy()

    return X, y


def build_preprocessor():
    numerical_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_FEATURES
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES
            )
        ]
    )

    return preprocessor