import sys
import os

# ✅ Ensure project root is in path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from src.train import train_model
from src.evaluate import evaluate_model
from src.visualize import (
    plot_confusion_matrix,
    plot_class_distribution,
    plot_prediction_pie
)

import pandas as pd


def main():
    print("🚀 Starting full pipeline...")

    # 🔹 Train model
    print("📌 Training model...")
    model, X_test, y_test = train_model()

    # 🔹 Evaluate model
    print("📊 Evaluating model...")
    cm, y_pred = evaluate_model(model, X_test, y_test)

    # 🔹 Labels for confusion matrix
    labels = sorted(pd.unique(y_test))

    # 🔹 Visualizations
    print("📈 Generating visualizations...")
    plot_confusion_matrix(cm, labels)
    plot_class_distribution(y_test)
    plot_prediction_pie(y_pred)

    print("✅ All tasks completed successfully!")


# ✅ Proper entry point
if __name__ == "__main__":
    main()