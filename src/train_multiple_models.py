import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from preprocess import clean_text
import os

def train_and_evaluate_models():
    df = pd.read_csv(
        "data/training.1600000.processed.noemoticon.csv",
        encoding="latin1",
        header=None
    )

    df.columns = ["sentiment", "id", "date", "query", "user", "text"]
    df = df[["sentiment", "text"]]
    df["sentiment"] = df["sentiment"].map({0: "negative", 4: "positive"})
    df = df.dropna()

    df["clean_text"] = df["text"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["sentiment"], test_size=0.2, random_state=42
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear SVM": LinearSVC(),
        "Naive Bayes": MultinomialNB()
    }

    results = []

    for name, clf in models.items():
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000)),
            ("clf", clf)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        results.append({"Model": name, "Accuracy": acc})

        print(f"{name} Accuracy: {acc:.4f}")

    os.makedirs("model", exist_ok=True)
    pd.DataFrame(results).to_csv("model/model_comparison.csv", index=False)

if __name__ == "__main__":
    train_and_evaluate_models()
