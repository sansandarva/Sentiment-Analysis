import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.preprocess import clean_text


def train_model():
    print("🚀 Starting training...")

    # 📥 Load dataset
    df = pd.read_csv(
        "data/training.1600000.processed.noemoticon.csv",
        encoding="latin-1",
        header=None,
        low_memory=False
    )

    # 🏷️ Assign column names
    df.columns = ["target", "id", "date", "flag", "user", "text"]

    # Keep only required columns
    df = df[["target", "text"]]

    # 🔥 FIX LABELS (CRITICAL)
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df["target"] = df["target"].replace(4, 1)

    # Remove invalid rows
    df = df.dropna()

    # 🧹 Clean text
    print("🧹 Cleaning text...")
    df["clean_text"] = df["text"].apply(clean_text)

    # (Optional) Reduce dataset for faster training
    df = df.sample(50000, random_state=42)

    X = df["clean_text"]
    y = df["target"]

    print("📊 Total samples:", len(df))
    print("📊 Label distribution:\n", y.value_counts())

    # ✂️ Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 🤖 Model pipeline
    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

    # 🏋️ Train model
    print("🏋️ Training model...")
    model.fit(X_train, y_train)

    # 💾 Save model
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/sentiment_model.pkl")

    print("✅ Model trained and saved successfully!")

    return model, X_test, y_test


# 🔥 IMPORTANT: This ensures training runs
if __name__ == "__main__":
    train_model()