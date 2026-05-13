import sys
import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# Absolute project paths (FIXES your error forever)
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # src/
PROJECT_DIR = os.path.dirname(BASE_DIR)                 # TEST1/
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODEL_DIR = os.path.join(PROJECT_DIR, "model")

sys.path.append(BASE_DIR)

from preprocess import clean_text
from mbart_model import predict_mbert


# --------------------------------------------------
# Load dataset
# --------------------------------------------------
csv_path = os.path.join(
    DATA_DIR,
    "training.1600000.processed.noemoticon.csv"
)

df = pd.read_csv(
    csv_path,
    encoding="latin-1",
    header=None,
    low_memory=False
)

df.columns = ["target", "id", "date", "flag", "user", "text"]

# Binary labels
df["target"] = df["target"].replace(4, 1)

# Sample for speed
df = df.sample(5000, random_state=42)

X = df["text"].apply(clean_text)
y = df["target"]


# --------------------------------------------------
# Load trained ML model
# --------------------------------------------------
model_path = os.path.join(MODEL_DIR, "sentiment_model.pkl")
ml_model = joblib.load(model_path)

results = {}

# Classical ML accuracy
ml_preds = ml_model.predict(X)
results["Traditional ML Model"] = accuracy_score(y, ml_preds)


# ---------------------------------------------
# mBERT accuracy (FIXED LABEL CONVERSION)
# ---------------------------------------------
bert_preds_raw = X.apply(predict_mbert)

# Convert mBERT text output → numeric
bert_preds = bert_preds_raw.map(
    lambda x: 1 if str(x).lower() == "positive" else 0
)

# Ensure same datatype
y = y.astype(int)
bert_preds = bert_preds.astype(int)

results["mBERT"] = accuracy_score(y, bert_preds)




# --------------------------------------------------
# Output
# --------------------------------------------------
print("\n📊 Model Accuracy Comparison")
print("-" * 35)
for k, v in results.items():
    print(f"{k}: {v:.4f}")
