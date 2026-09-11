from sklearn.ensemble import RandomForestClassifier
import joblib

class RandomForestModel:
    def __init__(self, n_estimators=200, max_depth=None, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight='balanced'
        )

    def fit(self, X_train, y_train):
        """Train Random Forest using preprocessed data from Member 2."""
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """Predict test labels for evaluation (Member 3’s test split)."""
        return self.model.predict(X_test)

    def save_model(self, path='results/random_forest_model.pkl'):
        """Save trained model for reproducibility."""
        joblib.dump(self.model, path)
