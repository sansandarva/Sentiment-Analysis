import re
import string
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()   # <-- SIMPLE SPLIT (NO punkt)
    words = [w for w in words if w not in stop_words]
    return " ".join(words)
