import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class FeatureInterpreter:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names

    def plot_importance(self, save_path='figures/feature_importance_rf.png'):
        """Visualize feature importance from Random Forest."""
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(8,6))
        sns.barplot(x=importances[indices], y=np.array(self.feature_names)[indices])
        plt.title('Feature Importance - Random Forest')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
