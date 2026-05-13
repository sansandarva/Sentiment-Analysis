import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

os.makedirs("results/plots", exist_ok=True)

# 1️⃣ Confusion Matrix (Heatmap)
def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("results/plots/confusion_matrix.png")
    plt.show()


# 2️⃣ Bar Chart – Class Distribution
def plot_class_distribution(y_test):
    values = y_test.value_counts()

    plt.figure(figsize=(6, 4))
    values.plot(kind="bar")
    plt.title("Class Distribution in Test Data")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("results/plots/class_distribution.png")
    plt.show()


# 3️⃣ Pie Chart – Prediction Distribution
def plot_prediction_pie(y_pred):
    values = np.unique(y_pred, return_counts=True)

    plt.figure(figsize=(6, 6))
    plt.pie(
        values[1],
        labels=values[0],
        autopct="%1.1f%%",
        startangle=90
    )
    plt.title("Prediction Distribution")
    plt.tight_layout()
    plt.savefig("results/plots/prediction_pie.png")
    plt.show()


# 4️⃣ Accuracy Bar Chart (Future multiple models)
def plot_model_accuracy(results_dict):
    names = list(results_dict.keys())
    scores = list(results_dict.values())

    plt.figure(figsize=(8, 5))
    plt.bar(names, scores)
    plt.xlabel("Models")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("results/plots/model_accuracy.png")
    plt.show()
