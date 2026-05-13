from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import joblib
import os
import sys
import pandas as pd
import io

# 🔧 Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocess import clean_text

# 🚀 Initialize app
app = FastAPI()

# 🔓 Enable CORS (important for React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📁 Load model (dynamic path)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_path = os.path.join(BASE_DIR, "model", "sentiment_model.pkl")

print("Loading model from:", model_path)

model = joblib.load(model_path)

# 🏠 Root
@app.get("/")
def home():
    return {"message": "API Running 🚀"}

# 🔹 TEXT SENTIMENT ANALYSIS
@app.post("/predict")
def predict(data: dict):
    try:
        text = clean_text(data["text"])
        pred = model.predict([text])[0]

        return {
            "sentiment": "positive" if pred == 1 else "negative"
        }

    except Exception as e:
        print("TEXT ERROR:", str(e))
        return {"error": str(e)}

# 🔹 FILE UPLOAD SENTIMENT ANALYSIS
@app.post("/analyze-file")
async def analyze_file(file: UploadFile = File(...)):
    try:
        print("📂 FILE RECEIVED:", file.filename)

        content = await file.read()

        # Safe decoding (handles most files)
        decoded = content.decode("latin-1")

        # Try reading as CSV
        try:
            df = pd.read_csv(io.StringIO(decoded), header=None)
            texts = df.iloc[:, -1].astype(str).tolist()
            print("CSV detected")
        except:
            # Fallback → treat as TXT
            texts = decoded.split("\n")
            print("TXT detected")

        positive = 0
        negative = 0

        for t in texts:
            t = t.strip()
            if not t:
                continue

            cleaned = clean_text(t)
            pred = model.predict([cleaned])[0]

            if pred == 1:
                positive += 1
            else:
                negative += 1

        result = {
            "positive": positive,
            "negative": negative,
            "total": positive + negative
        }

        print("RESULT:", result)

        return result

    except Exception as e:
        print("❌ FILE ERROR:", str(e))
        return {"error": str(e)}
