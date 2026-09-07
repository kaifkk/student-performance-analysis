from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline


def build_baseline(preprocessor):

    baseline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                DummyClassifier(
                    strategy="most_frequent"
                )
            )
        ]
    )

    return baseline