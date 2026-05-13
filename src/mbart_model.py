from transformers import BertTokenizer, BertForSequenceClassification
import torch

MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME)

def predict_mbert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    scores = torch.softmax(outputs.logits, dim=1)
    label = torch.argmax(scores).item()

    if label <= 1:
        return "negative"
    elif label == 2:
        return "neutral"
    else:
        return "positive"
