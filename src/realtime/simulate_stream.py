import pandas as pd
import time
import joblib
from src.preprocess import clean_text

def simulate_real_time(file_path, delay=2):
    model = joblib.load("model/sentiment_model.pkl")
    df = pd.read_csv(file_path)

    print("🔴 Simulated Real-Time Stream Started...\n")

    for i, row in df.iterrows():
        text = row["text"]
        clean = clean_text(text)
        prediction = model.predict([clean])[0]

        print(f"Tweet {i+1}: {text}")
        print(f"Predicted Sentiment: {prediction}")
        print("-" * 50)

        time.sleep(delay)
