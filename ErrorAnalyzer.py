import pandas as pd

class ErrorAnalyzer:
    def __init__(self, X_test, y_true, y_pred):
        self.X_test = X_test.copy()
        self.y_true = y_true
        self.y_pred = y_pred

    def analyze_errors(self):
        """Identify misclassified samples for deeper inspection."""
        errors = self.X_test[self.y_true != self.y_pred]
        errors['true_label'] = self.y_true[self.y_true != self.y_pred]
        errors['pred_label'] = self.y_pred[self.y_true != self.y_pred]
        return errors

    def save_errors(self, path='results/error_analysis.csv'):
        """Save misclassified samples for Member 1’s EDA follow-up."""
        errors = self.analyze_errors()
        errors.to_csv(path, index=False)
