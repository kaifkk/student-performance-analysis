from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class ModelEvaluator:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def evaluate(self):
        """Generate precision, recall, F1, and accuracy metrics."""
        report = classification_report(self.y_true, self.y_pred, output_dict=True)
        return pd.DataFrame(report).transpose()

    def plot_confusion_matrix(self, save_path='figures/confusion_matrix_rf.png'):
        """Visualize confusion matrix for Random Forest predictions."""
        cm = confusion_matrix(self.y_true, self.y_pred)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - Random Forest')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
